import argparse
import csv
import json
import os
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import gymnasium as gym

warnings.filterwarnings("ignore")

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

import sumo_rl
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
NET_FILE    = str(BASE / "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.net.xml")
ROUTE_FILE  = str(BASE / "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.rou.xml")
METRICS_DIR = BASE / "metrics" / "single"
MODELS_DIR  = BASE / "models"
SEED = 42

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Metrics writer ────────────────────────────────────────────────────────────
class MetricsWriter:
    def __init__(self, algo: str, total_steps: int):
        self.algo = algo
        self.total_steps = total_steps
        self.episode = 0
        self.start = time.time()
        self.csv_path    = METRICS_DIR / f"{algo}_metrics.csv"
        self.status_path = METRICS_DIR / f"{algo}_status.json"

        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "timestep", "reward", "waiting_time", "mean_speed", "elapsed_s"]
            )
        self._status("running", 0)

    def log(self, timestep: int, reward: float, waiting_time: float, mean_speed: float):
        self.episode += 1
        elapsed = round(time.time() - self.start, 2)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode, int(timestep),
                round(float(reward), 4), round(float(waiting_time), 4),
                round(float(mean_speed), 4), elapsed,
            ])
        progress = min(100, int(timestep / self.total_steps * 100))
        self._status("running", progress, timestep, reward)
        print(f"  [{self.algo}] ep={self.episode:4d} | step={timestep:7d} | "
              f"wait={waiting_time:6.2f}s | reward={reward:8.2f}")

    def done(self):
        self._status("done", 100, self.total_steps)

    def error(self, msg: str):
        self._status("error", 0, message=msg)

    def _status(self, state, progress, timestep=0, reward=0.0, message=""):
        payload = {
            "algo": self.algo, "state": state, "progress": int(progress),
            "timestep": int(timestep), "total": self.total_steps,
            "episode": self.episode, "reward": round(float(reward), 4),
            "message": message, "updated_at": time.time(),
        }
        tmp = self.status_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(self.status_path)


# ── Environment factory ───────────────────────────────────────────────────────
def make_env() -> gym.Env:
    env = sumo_rl.SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        num_seconds=3600,
        delta_time=5,
        yellow_time=2,
        reward_fn="diff-waiting-time",
        sumo_seed=SEED,
        single_agent=True,
        sumo_warnings=False,
    )
    return Monitor(env)


# ── SB3 callback ─────────────────────────────────────────────────────────────
class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, writer: MetricsWriter):
        super().__init__()
        self.writer = writer
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            self.writer.log(
                self.num_timesteps,
                self._ep_reward,
                info.get("system_mean_waiting_time", 0.0),
                info.get("system_mean_speed", 0.0),
            )
            self._ep_reward = 0.0
        return True


# ── Fixed-cycle baseline ──────────────────────────────────────────────────────
def train_fixed_cycle(writer: MetricsWriter):
    print("[fixed] Running fixed-cycle baseline (1 episode)")
    env = make_env()
    obs, _ = env.reset()

    n_actions     = env.action_space.n
    phase_steps   = 6
    phase         = 0
    step_in_phase = 0
    ep_reward     = 0.0
    timestep      = 0

    done = False
    while not done:
        action = phase
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        timestep  += 1
        done = terminated or truncated

        step_in_phase += 1
        if step_in_phase >= phase_steps:
            phase = (phase + 1) % n_actions
            step_in_phase = 0

    writer.log(timestep, ep_reward,
               info.get("system_mean_waiting_time", 0.0),
               info.get("system_mean_speed", 0.0))
    env.close()
    print("[fixed] Done.")


# ── Tabular Q-Learning ────────────────────────────────────────────────────────
class TabularQLearning:
    def __init__(self, n_actions: int, bins: int = 5,
                 alpha: float = 0.1, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05):
        self.n_actions = n_actions
        self.bins      = bins
        self.alpha     = alpha
        self.gamma     = gamma
        self.eps_start = eps_start
        self.eps_end   = eps_end
        self.q: dict   = defaultdict(lambda: np.zeros(n_actions))

    def _discretize(self, obs: np.ndarray) -> tuple:
        phase  = int(np.argmax(obs[:self.n_actions]))
        queues = tuple(int(min(v, self.bins - 1)) for v in obs[self.n_actions:])
        return (phase, *queues)

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        state = self._discretize(obs)
        return int(np.argmax(self.q[state]))

    def update(self, obs, action, reward, next_obs, done):
        s  = self._discretize(obs)
        s_ = self._discretize(next_obs)
        best_next = 0.0 if done else np.max(self.q[s_])
        self.q[s][action] += self.alpha * (
            reward + self.gamma * best_next - self.q[s][action]
        )

    def train(self, total_steps: int, writer: MetricsWriter):
        print(f"[qlearning] Training for {total_steps:,} steps")
        env = make_env()
        self.n_actions = env.action_space.n
        timestep  = 0
        eps_decay = (self.eps_start - self.eps_end) / total_steps

        while timestep < total_steps:
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            epsilon = max(self.eps_end, self.eps_start - eps_decay * timestep)

            while not done:
                action = self.act(obs, epsilon)
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.update(obs, action, reward, next_obs, terminated or truncated)
                obs        = next_obs
                ep_reward += reward
                timestep  += 1
                done = terminated or truncated

            writer.log(timestep, ep_reward,
                       info.get("system_mean_waiting_time", 0.0),
                       info.get("system_mean_speed", 0.0))

        env.close()
        print("[qlearning] Done.")


# ── DQN ───────────────────────────────────────────────────────────────────────
def train_dqn(total_steps: int, writer: MetricsWriter):
    print(f"[dqn] Training for {total_steps:,} steps")
    env = make_env()
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [64, 64]},
        seed=SEED,
        verbose=0,
    )
    model.learn(total_steps, callback=EpisodeMetricsCallback(writer))
    model.save(str(MODELS_DIR / "dqn_model"))
    env.close()
    print("[dqn] Model saved.")


# ── PPO ───────────────────────────────────────────────────────────────────────
def train_ppo(total_steps: int, writer: MetricsWriter):
    print(f"[ppo] Training for {total_steps:,} steps")
    env = make_env()
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=720,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        seed=SEED,
        verbose=0,
    )
    model.learn(total_steps, callback=EpisodeMetricsCallback(writer))
    model.save(str(MODELS_DIR / "ppo_model"))
    env.close()
    print("[ppo] Model saved.")


# ════════════════════════════════════════════════════════════════════════════
# ── AQPPO: Adaptive Q-augmented PPO ─────────────────────────────────────────
#
#  Key ideas that help beat vanilla PPO:
#  1. Deeper shared network (128-128 vs 64-64) — more representational power
#  2. Q-value memory (replay buffer) — DQN-style auxiliary loss so the actor
#     never forgets good state-action values between PPO updates
#  3. GAE with λ=0.97 (higher than PPO's default 0.95) — less bias, smoother
#     advantage estimates for long traffic episodes
#  4. Adaptive entropy coefficient — starts high (explore) and decays based
#     on recent reward improvement so it exploits when things are going well
#  5. Congestion-aware reward shaping — penalises the agent extra when queue
#     length exceeds a threshold, pushing it to act before jams form
# ════════════════════════════════════════════════════════════════════════════

class AQPPONetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128),    nn.LayerNorm(128), nn.ReLU(),
        )
        self.actor  = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
        self.q_head = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor):
        f = self.shared(x)
        return self.actor(f), self.critic(f), self.q_head(f)

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        x = torch.FloatTensor(obs).unsqueeze(0)
        logits, value, q = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_t, actions_t):
        logits, values, q = self(obs_t)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions_t), dist.entropy(), values.squeeze(-1), q


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idxs]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (torch.FloatTensor(np.array(obs)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_obs)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buf)


def aqppo_update(policy, optimizer, rollout, replay_buf,
                 ent_coef, gamma=0.99, lam=0.97,
                 clip=0.2, n_epochs=6, batch_size=128,
                 q_coef=0.3):

    obs_t     = torch.FloatTensor(np.array(rollout["obs"]))
    actions_t = torch.LongTensor(rollout["actions"])
    old_lps_t = torch.FloatTensor(rollout["log_probs"])
    rewards   = rollout["rewards"]
    values    = rollout["values"]
    dones     = rollout["dones"]

    # GAE with λ=0.97
    # next_val=0 when done (terminal), else bootstrap from values[t+1]
    advantages, returns = [], []
    adv = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_val = 0.0
        elif t + 1 < len(values):
            next_val = values[t + 1]
        else:
            next_val = 0.0
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        adv   = delta + gamma * lam * (1.0 - dones[t]) * adv
        advantages.insert(0, adv)
        returns.insert(0, adv + values[t])

    adv_t = torch.FloatTensor(advantages)
    ret_t = torch.FloatTensor(returns)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(obs_t)
    for _ in range(n_epochs):
        idxs = torch.randperm(n)
        for start in range(0, n, batch_size):
            b = idxs[start:start + batch_size]
            lp, ent, val, _ = policy.evaluate(obs_t[b], actions_t[b])

            ratio = (lp - old_lps_t[b]).exp()
            obj   = torch.min(ratio * adv_t[b],
                              torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b])
            ppo_loss = -obj.mean() + 0.5 * (ret_t[b] - val).pow(2).mean() - ent_coef * ent.mean()

            # Q-auxiliary loss from replay buffer
            q_loss = torch.tensor(0.0)
            if len(replay_buf) >= batch_size:
                rb_obs, rb_act, rb_rew, rb_next, rb_done = replay_buf.sample(batch_size)
                _, _, _, q_vals = policy.evaluate(rb_obs, rb_act)
                with torch.no_grad():
                    _logits, next_val_rb, _q = policy.forward(rb_next)
                    next_val_rb = next_val_rb.squeeze(-1)
                td_target = rb_rew + gamma * next_val_rb * (1 - rb_done)
                q_current = q_vals.gather(1, rb_act.unsqueeze(1)).squeeze(1)
                q_loss    = (td_target - q_current).pow(2).mean()

            loss = ppo_loss + q_coef * q_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


def train_aqppo(total_steps: int, writer: MetricsWriter):
    print(f"[aqppo] Training for {total_steps:,} steps")
    env = make_env()
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy    = AQPPONetwork(obs_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=2.5e-4)
    replay    = ReplayBuffer(capacity=20_000)

    # Adaptive entropy: start at 0.02, decay toward 0.003
    ent_coef     = 0.02
    ent_min      = 0.003
    ent_decay    = 0.995
    recent_rewards = deque(maxlen=10)
    best_mean_reward = -np.inf

    rollout = {"obs": [], "actions": [], "log_probs": [],
               "rewards": [], "values": [], "dones": []}

    timestep = 0
    obs, _   = env.reset()
    obs      = np.array(obs, dtype=np.float32)

    while timestep < total_steps:
        ep_reward = 0.0
        done      = False

        while not done:
            action, lp, val = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = np.array(next_obs, dtype=np.float32)
            done = terminated or truncated

            # Congestion shaping: extra penalty if queue > threshold
            queue_obs  = obs[n_actions:]
            congestion = float(np.sum(queue_obs > 0.6))
            shaped_reward = reward - 0.05 * congestion

            rollout["obs"].append(obs.copy())
            rollout["actions"].append(action)
            rollout["log_probs"].append(lp)
            rollout["rewards"].append(shaped_reward)
            rollout["values"].append(val)
            rollout["dones"].append(float(done))

            replay.push(obs.copy(), action, shaped_reward, next_obs.copy(), float(done))

            obs        = next_obs
            ep_reward += reward
            timestep  += 1

        recent_rewards.append(ep_reward)
        mean_rew = np.mean(recent_rewards)

        # Decay entropy faster when improving, slower when stuck
        if mean_rew > best_mean_reward:
            best_mean_reward = mean_rew
            ent_coef = max(ent_min, ent_coef * ent_decay)
        else:
            ent_coef = min(0.02, ent_coef * 1.002)

        writer.log(timestep, ep_reward,
                   info.get("system_mean_waiting_time", 0.0),
                   info.get("system_mean_speed", 0.0))

        aqppo_update(policy, optimizer, rollout, replay, ent_coef)
        rollout = {k: [] for k in rollout}
        obs, _ = env.reset()
        obs    = np.array(obs, dtype=np.float32)

    torch.save(policy.state_dict(), str(MODELS_DIR / "aqppo_model.pt"))
    env.close()
    print("[aqppo] Model saved.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        choices=["fixed", "qlearning", "dqn", "ppo", "aqppo"])
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    writer = MetricsWriter(args.algo, args.steps)
    try:
        if args.algo == "fixed":
            train_fixed_cycle(writer)
        elif args.algo == "qlearning":
            agent = TabularQLearning(n_actions=2)
            agent.train(args.steps, writer)
        elif args.algo == "dqn":
            train_dqn(args.steps, writer)
        elif args.algo == "ppo":
            train_ppo(args.steps, writer)
        elif args.algo == "aqppo":
            train_aqppo(args.steps, writer)
        writer.done()
    except Exception as e:
        writer.error(str(e))
        raise


if __name__ == "__main__":
    main()
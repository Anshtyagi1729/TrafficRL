import argparse
import csv
import json
import os
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

warnings.filterwarnings("ignore")

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

import sumo_rl

BASE        = Path(__file__).parent
NET_FILE    = str(BASE / "sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml")
ROUTE_FILE  = str(BASE / "sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml")
METRICS_DIR = BASE / "metrics" / "marl_aqppo"
MODELS_DIR  = BASE / "models" / "marl_aqppo"
SEED = 42

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ════════════════════════════════════════════════════════════════════════════
# AQPPO Network
#  128-128 with LayerNorm + actor, critic, Q heads
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
        logits, value, _ = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_t: torch.Tensor, actions_t: torch.Tensor):
        logits, values, q = self(obs_t)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions_t), dist.entropy(), values.squeeze(-1), q


# ════════════════════════════════════════════════════════════════════════════
# Per-agent replay buffer
# ════════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs.copy(), int(action), float(reward),
                         next_obs.copy(), float(done)))

    def sample(self, batch_size: int):
        idxs  = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idxs]
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (torch.FloatTensor(np.array(obs)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_obs)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buf)


# ════════════════════════════════════════════════════════════════════════════
# Congestion shaping — safe for any obs layout
#
# sumo-rl obs = [phase_one_hot (n_actions bits)] + [density per lane (floats)]
# Density values are in [0, 1]. We use the LAST half of the non-phase part
# and a softer threshold (0.8) so it only fires on truly congested lanes.
# Penalty weight is 0.02 (was 0.05) — much smaller relative to base reward.
# ════════════════════════════════════════════════════════════════════════════
def congestion_penalty(obs: np.ndarray, n_actions: int) -> float:
    density_obs = obs[n_actions:]          # all density/queue features
    if len(density_obs) == 0:
        return 0.0
    # Use a high threshold so only severely congested lanes trigger
    severe = float(np.sum(density_obs > 0.8))
    return 0.02 * severe                   # max ~0.1 per step vs reward ~[-2, +2]


# ════════════════════════════════════════════════════════════════════════════
# AQPPO update
#  q_coef ramps from 0 → 0.2 over first 20 episodes to avoid destabilising
#  early policy gradient when replay buffer has only random transitions
# ════════════════════════════════════════════════════════════════════════════
def aqppo_update(policy, optimizer, rollout, replay_buf,
                 ent_coef, episode, gamma=0.99, lam=0.97,
                 clip=0.2, n_epochs=4, batch_size=64):

    if len(rollout["obs"]) == 0:
        return

    # Gradually enable Q-auxiliary loss — ramp over first 20 episodes
    q_coef = min(0.2, episode / 20.0 * 0.2)

    obs_t     = torch.FloatTensor(np.array(rollout["obs"]))
    actions_t = torch.LongTensor(rollout["actions"])
    old_lps_t = torch.FloatTensor(rollout["log_probs"])
    rewards   = rollout["rewards"]
    values    = rollout["values"]
    dones     = rollout["dones"]

    # GAE λ=0.97
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
            if len(b) < 2:
                continue

            lp, ent, val, _ = policy.evaluate(obs_t[b], actions_t[b])
            ratio    = (lp - old_lps_t[b]).exp()
            obj      = torch.min(ratio * adv_t[b],
                                 torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b])
            ppo_loss = (-obj.mean()
                        + 0.5 * (ret_t[b] - val).pow(2).mean()
                        - ent_coef * ent.mean())

            # Q-auxiliary loss — only active after warmup
            q_loss = torch.tensor(0.0)
            if q_coef > 0 and len(replay_buf) >= batch_size:
                rb_obs, rb_act, rb_rew, rb_next, rb_done = replay_buf.sample(batch_size)
                _, _, _, q_vals = policy.evaluate(rb_obs, rb_act)
                with torch.no_grad():
                    _logits, next_val_rb, _q = policy.forward(rb_next)
                    next_val_rb = next_val_rb.squeeze(-1)
                td_target = rb_rew + gamma * next_val_rb * (1.0 - rb_done)
                q_current = q_vals.gather(1, rb_act.unsqueeze(1)).squeeze(1)
                q_loss    = (td_target - q_current).pow(2).mean()

            loss = ppo_loss + q_coef * q_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


# ════════════════════════════════════════════════════════════════════════════
# Metrics writer
# ════════════════════════════════════════════════════════════════════════════
class AgentMetricsWriter:
    def __init__(self, agent_id: str, total_steps: int):
        self.agent_id    = agent_id
        self.total_steps = total_steps
        self.episode     = 0
        self.start       = time.time()
        self.csv_path    = METRICS_DIR / f"{agent_id}_metrics.csv"

        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "timestep", "reward", "waiting_time", "mean_speed", "elapsed_s"]
            )

    def log(self, timestep, reward, waiting_time, mean_speed):
        self.episode += 1
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode, int(timestep),
                round(float(reward), 4), round(float(waiting_time), 4),
                round(float(mean_speed), 4),
                round(time.time() - self.start, 2),
            ])


def write_status(episode, total_episodes, agent_rewards, state="running"):
    payload = {
        "state":          state,
        "episode":        episode,
        "total_episodes": total_episodes,
        "agent_rewards":  {k: round(float(v), 4) for k, v in agent_rewards.items()},
        "updated_at":     time.time(),
    }
    path = METRICS_DIR / "marl_aqppo_status.json"
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)


# ════════════════════════════════════════════════════════════════════════════
# IAQPPO training loop
# ════════════════════════════════════════════════════════════════════════════
def train_iaqppo(total_steps: int):
    print(f"[IAQPPO] Starting 2×2 MARL AQPPO training — {total_steps:,} steps")

    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        num_seconds=3600,
        delta_time=5,
        yellow_time=2,
        reward_fn="diff-waiting-time",
        sumo_seed=SEED,
        sumo_warnings=False,
    )

    obs_dict, _ = env.reset()
    agent_ids   = list(obs_dict.keys())
    obs_dim     = env.observation_space(agent_ids[0]).shape[0]
    n_actions   = env.action_space(agent_ids[0]).n

    print(f"[IAQPPO] Agents: {agent_ids} | obs_dim={obs_dim} | n_actions={n_actions}")

    policies   = {a: AQPPONetwork(obs_dim, n_actions)                for a in agent_ids}
    optimizers = {a: optim.Adam(policies[a].parameters(), lr=3e-4)   for a in agent_ids}
    replays    = {a: ReplayBuffer(capacity=20_000)                    for a in agent_ids}
    writers    = {a: AgentMetricsWriter(a, total_steps)               for a in agent_ids}

    # Per-agent adaptive entropy
    ent_coefs      = {a: 0.02    for a in agent_ids}
    best_mean_rews = {a: -np.inf for a in agent_ids}
    recent_rews    = {a: deque(maxlen=10) for a in agent_ids}
    ent_min, ent_decay = 0.005, 0.995   # higher floor than single-agent

    rollouts = {a: {"obs": [], "actions": [], "log_probs": [],
                    "rewards": [], "values": [], "dones": []}
                for a in agent_ids}

    timestep        = 0
    episode         = 0
    total_episodes  = total_steps // 720
    last_ep_rewards = {a: 0.0 for a in agent_ids}

    obs_dict = {a: np.array(o, dtype=np.float32) for a, o in obs_dict.items()}

    while timestep < total_steps:

        # ── collect one step ──────────────────────────────────────────────
        actions, log_probs, values = {}, {}, {}
        for a in env.agents:
            act, lp, val = policies[a].act(obs_dict[a])
            actions[a]   = act
            log_probs[a] = lp
            values[a]    = val

        next_obs_dict, rewards, terminations, truncations, infos = env.step(actions)
        next_obs_dict = {a: np.array(o, dtype=np.float32)
                         for a, o in next_obs_dict.items()}

        for a in agent_ids:
            done       = terminations.get(a, False) or truncations.get(a, False)
            obs        = obs_dict.get(a, np.zeros(obs_dim, dtype=np.float32))
            nobs       = next_obs_dict.get(a, np.zeros(obs_dim, dtype=np.float32))
            raw_reward = rewards.get(a, 0.0)

            # Soft congestion shaping — safe slice, high threshold, small weight
            penalty       = congestion_penalty(obs, n_actions)
            shaped_reward = raw_reward - penalty

            rollouts[a]["obs"].append(obs.copy())
            rollouts[a]["actions"].append(actions.get(a, 0))
            rollouts[a]["log_probs"].append(log_probs.get(a, 0.0))
            rollouts[a]["rewards"].append(shaped_reward)
            rollouts[a]["values"].append(values.get(a, 0.0))
            rollouts[a]["dones"].append(float(done))

            replays[a].push(obs, actions.get(a, 0), shaped_reward, nobs, float(done))

        obs_dict  = next_obs_dict
        timestep += 1

        # ── episode ended ─────────────────────────────────────────────────
        if not env.agents:
            episode += 1
            ep_rewards = {}

            for a in agent_ids:
                ep_rew             = sum(rollouts[a]["rewards"])
                ep_rewards[a]      = ep_rew
                last_ep_rewards[a] = ep_rew
                info = infos.get(a, {})
                writers[a].log(
                    timestep, ep_rew,
                    info.get("system_mean_waiting_time", 0.0),
                    info.get("system_mean_speed", 0.0),
                )

                # Adaptive entropy per agent
                recent_rews[a].append(ep_rew)
                mean_rew = float(np.mean(recent_rews[a]))
                if mean_rew > best_mean_rews[a]:
                    best_mean_rews[a] = mean_rew
                    ent_coefs[a] = max(ent_min, ent_coefs[a] * ent_decay)
                else:
                    ent_coefs[a] = min(0.02, ent_coefs[a] * 1.002)

                # AQPPO update — pass episode for q_coef warmup
                aqppo_update(policies[a], optimizers[a],
                             rollouts[a], replays[a],
                             ent_coefs[a], episode)
                rollouts[a] = {k: [] for k in rollouts[a]}

            write_status(episode, total_episodes, ep_rewards)
            print(f"  [IAQPPO] ep={episode:4d} | step={timestep:7d} | "
                  f"rewards: { {a: round(v, 1) for a, v in ep_rewards.items()} }")

            obs_dict, _ = env.reset()
            obs_dict = {a: np.array(o, dtype=np.float32) for a, o in obs_dict.items()}

    # ── save models ───────────────────────────────────────────────────────
    for a in agent_ids:
        safe = a.replace(" ", "_").replace("/", "_")
        torch.save(policies[a].state_dict(), MODELS_DIR / f"iaqppo_{safe}.pt")
    print(f"[IAQPPO] Models saved to {MODELS_DIR}")
    env.close()
    write_status(episode, total_episodes, last_ep_rewards, state="done")
    print("[IAQPPO] Done.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()
    train_iaqppo(args.steps)


if __name__ == "__main__":
    main()
import argparse
import csv
import json
import os
import time
import warnings
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

BASE       = Path(__file__).parent
NET_FILE   = str(BASE / "sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml")
ROUTE_FILE = str(BASE / "sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml")
METRICS_DIR = BASE / "metrics" / "marl"
MODELS_DIR  = BASE / "models" / "marl"
SEED = 42

METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Actor-Critic network ──────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
        )
        self.actor  = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        f = self.shared(x)
        return self.actor(f), self.critic(f)

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        x = torch.FloatTensor(obs).unsqueeze(0)
        logits, value = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs_t, actions_t):
        logits, values = self(obs_t)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions_t), dist.entropy(), values.squeeze(-1)


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(policy, optimizer, rollout, gamma=0.99, lam=0.95,
               clip=0.2, ent_coef=0.01, n_epochs=4, batch_size=64):
    obs_t     = torch.FloatTensor(np.array(rollout["obs"]))
    actions_t = torch.LongTensor(rollout["actions"])
    old_lps_t = torch.FloatTensor(rollout["log_probs"])
    rewards   = rollout["rewards"]
    values    = rollout["values"]
    dones     = rollout["dones"]

    # GAE advantages
    advantages, returns = [], []
    adv, last_val = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        next_val = 0.0 if dones[t] else values[t + 1] if t + 1 < len(values) else last_val
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        adv   = delta + gamma * lam * (1 - dones[t]) * adv
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
            lp, ent, val = policy.evaluate(obs_t[b], actions_t[b])
            ratio  = (lp - old_lps_t[b]).exp()
            obj    = torch.min(ratio * adv_t[b],
                               torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[b])
            loss   = -obj.mean() + 0.5 * (ret_t[b] - val).pow(2).mean() - ent_coef * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()


# ── Metrics writer ────────────────────────────────────────────────────────────
class AgentMetricsWriter:
    def __init__(self, agent_id: str, total_steps: int):
        self.agent_id = agent_id
        self.total_steps = total_steps
        self.episode = 0
        self.start = time.time()
        self.csv_path = METRICS_DIR / f"{agent_id}_metrics.csv"

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
                round(float(mean_speed), 4), round(time.time() - self.start, 2),
            ])


def write_marl_status(episode, total_episodes, agent_rewards):
    status = {
        "state": "running",
        "episode": episode,
        "total_episodes": total_episodes,
        "agent_rewards": {k: round(float(v), 4) for k, v in agent_rewards.items()},
        "updated_at": time.time(),
    }
    path = METRICS_DIR / "marl_status.json"
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f)
    tmp.replace(path)


# ── IPPO training loop ────────────────────────────────────────────────────────
def train_ippo(total_steps: int):
    print(f"[IPPO] Starting 2x2 MARL training — {total_steps:,} steps")

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

    print(f"[IPPO] Agents: {agent_ids} | obs_dim={obs_dim} | n_actions={n_actions}")

    policies   = {a: ActorCritic(obs_dim, n_actions) for a in agent_ids}
    optimizers = {a: optim.Adam(policies[a].parameters(), lr=3e-4) for a in agent_ids}
    writers    = {a: AgentMetricsWriter(a, total_steps) for a in agent_ids}

    rollouts = {a: {"obs": [], "actions": [], "log_probs": [],
                    "rewards": [], "values": [], "dones": []} for a in agent_ids}

    timestep   = 0
    episode    = 0
    step_since_update = 0
    UPDATE_EVERY = 720  # 1 full episode

    obs_dict, _ = env.reset()

    while timestep < total_steps:
        # collect one step across all agents
        actions, log_probs, values = {}, {}, {}
        for a in env.agents:
            act, lp, val = policies[a].act(obs_dict[a])
            actions[a]   = act
            log_probs[a] = lp
            values[a]    = val

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        for a in agent_ids:
            done = terminations.get(a, False) or truncations.get(a, False)
            rollouts[a]["obs"].append(obs_dict.get(a, np.zeros(obs_dim)))
            rollouts[a]["actions"].append(actions.get(a, 0))
            rollouts[a]["log_probs"].append(log_probs.get(a, 0.0))
            rollouts[a]["rewards"].append(rewards.get(a, 0.0))
            rollouts[a]["values"].append(values.get(a, 0.0))
            rollouts[a]["dones"].append(float(done))

        obs_dict = next_obs
        timestep += 1
        step_since_update += 1

        # episode ended
        if not env.agents:
            episode += 1
            ep_rewards = {}

            for a in agent_ids:
                ep_rew  = sum(rollouts[a]["rewards"])
                ep_rewards[a] = ep_rew
                info = infos.get(a, {})
                writers[a].log(
                    timestep, ep_rew,
                    info.get(f"{a}_accumulated_waiting_time", 0.0),
                    info.get(f"{a}_average_speed", 0.0),
                )

            write_marl_status(episode, total_steps // 720, ep_rewards)
            print(f"  [IPPO] ep={episode:4d} | step={timestep:7d} | "
                  f"rewards: { {a: round(v,1) for a,v in ep_rewards.items()} }")

            # PPO update
            for a in agent_ids:
                if len(rollouts[a]["obs"]) > 0:
                    ppo_update(policies[a], optimizers[a], rollouts[a])
                rollouts[a] = {k: [] for k in rollouts[a]}

            obs_dict, _ = env.reset()
            step_since_update = 0

    # save models
    for a in agent_ids:
        safe_name = a.replace(" ", "_").replace("/", "_")
        torch.save(policies[a].state_dict(), MODELS_DIR / f"ippo_{safe_name}.pt")
    print(f"[IPPO] Models saved to {MODELS_DIR}")
    env.close()

    # mark done
    final_rewards = {a: round(float(sum(rollouts[a]["rewards"]) if rollouts[a]["rewards"] else 0.0), 4) for a in agent_ids}
    path = METRICS_DIR / "marl_status.json"
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"state": "done", "episode": episode, "total_episodes": total_steps // 720,
                   "agent_rewards": final_rewards, "updated_at": time.time()}, f)
    tmp.replace(path)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()
    train_ippo(args.steps)


if __name__ == "__main__":
    main()

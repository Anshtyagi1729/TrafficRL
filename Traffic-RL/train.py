import argparse
import csv
import json
import os
import time
import warnings
from collections import defaultdict
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

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
SEED = 42

_NETS = {
    "single": {
        "net":     str(BASE / "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.net.xml"),
        "route":   str(BASE / "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.rou.xml"),
        "metrics": BASE / "metrics" / "single",
        "models":  BASE / "models",
    },
    "jiit": {
        "net":     str(BASE / "jiit-intersection/jiit.net.xml"),
        "route":   str(BASE / "jiit-intersection/jiit.rou.xml"),
        "metrics": BASE / "metrics" / "jiit",
        "models":  BASE / "models" / "jiit",
    },
}

# These are overwritten by main() based on --net; kept as globals so make_env()
# and MetricsWriter pick up the right paths at call time.
NET_FILE    = _NETS["single"]["net"]
ROUTE_FILE  = _NETS["single"]["route"]
METRICS_DIR = _NETS["single"]["metrics"]
MODELS_DIR  = _NETS["single"]["models"]


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

    n_actions   = env.action_space.n
    phase_steps = 6          # 6 × 5s = 30s per phase
    phase       = 0
    step_in_phase = 0
    ep_reward   = 0.0
    timestep    = 0

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

    writer.log(
        timestep, ep_reward,
        info.get("system_mean_waiting_time", 0.0),
        info.get("system_mean_speed", 0.0),
    )
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
        self.n_actions = env.action_space.n  # override hardcoded value with actual env
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

            writer.log(
                timestep, ep_reward,
                info.get("system_mean_waiting_time", 0.0),
                info.get("system_mean_speed", 0.0),
            )

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


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        choices=["fixed", "qlearning", "dqn", "ppo"])
    parser.add_argument("--net", choices=["single", "jiit"], default="single",
                        help="Which intersection network to use")
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    global NET_FILE, ROUTE_FILE, METRICS_DIR, MODELS_DIR
    cfg = _NETS[args.net]
    NET_FILE    = cfg["net"]
    ROUTE_FILE  = cfg["route"]
    METRICS_DIR = cfg["metrics"]
    MODELS_DIR  = cfg["models"]
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    writer = MetricsWriter(args.algo, args.steps)
    try:
        if args.algo == "fixed":
            train_fixed_cycle(writer)
        elif args.algo == "qlearning":
            agent = TabularQLearning(n_actions=2)  # placeholder; overridden by env in train()
            agent.train(args.steps, writer)
        elif args.algo == "dqn":
            train_dqn(args.steps, writer)
        elif args.algo == "ppo":
            train_ppo(args.steps, writer)
        writer.done()
    except Exception as e:
        writer.error(str(e))
        raise


if __name__ == "__main__":
    main()

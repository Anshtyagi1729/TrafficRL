import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

warnings.filterwarnings("ignore")

if "SUMO_HOME" not in os.environ:
    # Use raw string (r"") for Windows paths to handle backslashes properly
    os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"

import sumo_rl
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# ── Paths
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)

NET_FILE = "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.net.xml"
ROUTE_FILE = "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.rou.xml"
SEED = 42


# Metrics writer

class MetricsWriter:
    """Writes episode metrics to CSV and status to JSON after every episode."""

    def __init__(self, algo: str, total_timesteps: int):
        self.algo = algo
        self.total_timesteps = total_timesteps
        self.csv_path = METRICS_DIR / f"{algo}_metrics.csv"
        self.status_path = METRICS_DIR / f"{algo}_status.json"
        self.episode = 0
        self.start_time = time.time()

        # Write CSV header
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "timestep",
                    "reward",
                    "waiting_time",
                    "mean_speed",
                    "elapsed_s",
                ]
            )

        self._write_status("running", 0)

    def log(self, timestep: int, reward: float, info: dict):
        self.episode += 1
        elapsed = time.time() - self.start_time
        row = [
            self.episode,
            timestep,
            round(reward, 4),
            round(info.get("system_mean_waiting_time", 0), 4),
            round(info.get("system_mean_speed", 0), 4),
            round(elapsed, 2),
        ]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        progress = min(100, int(timestep / self.total_timesteps * 100))
        self._write_status("running", progress, timestep, reward)

    def done(self):
        self._write_status("done", 100, self.total_timesteps)

    def error(self, msg: str):
        self._write_status("error", 0, message=msg)

    def _write_status(
        self,
        state: str,
        progress: int,
        timestep: int = 0,
        reward: float = 0.0,
        message: str = "",
    ):
        payload = {
            "algo": self.algo,
            "state": state,  # running | done | error
            "progress": progress,
            "timestep": timestep,
            "total": self.total_timesteps,
            "episode": self.episode,
            "reward": round(reward, 4),
            "message": message,
            "updated_at": time.time(),
        }
        # Atomic write via temp file
        tmp = self.status_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(self.status_path)



# Environment factory
def make_env() -> gym.Env:
    env = sumo_rl.SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=True,
        num_seconds=3600,
        delta_time=5,
        yellow_time=2,
        reward_fn="diff-waiting-time",
        sumo_seed=SEED,
        single_agent=True
    )
    return Monitor(env)
if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    
    print("Environment loaded! Check the SUMO GUI.")
    
    done = False
    while not done:
        # Take a random traffic light action to keep time moving
        action = env.action_space.sample() 
        
        # Step the simulation forward
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("Simulation finished!")
    env.close()
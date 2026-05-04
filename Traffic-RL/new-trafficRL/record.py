import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

import traci
import sumo_rl
from stable_baselines3 import DQN, PPO

BASE        = Path(__file__).parent
NET_SINGLE  = str(BASE / "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.net.xml")
ROU_SINGLE  = str(BASE / "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.rou.xml")
NET_GRID    = str(BASE / "sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml")
ROU_GRID    = str(BASE / "sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml")
VIEW_FILE   = str(BASE / "viewsettings.xml")
VIDEOS_DIR  = BASE / "videos"
MODELS_DIR  = BASE / "models"
SEED = 42

VIDEOS_DIR.mkdir(exist_ok=True)


# ── same ActorCritic definition as train_marl.py ──────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),     nn.ReLU(),
        )
        self.actor  = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        f = self.shared(x)
        return self.actor(f), self.critic(f)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.FloatTensor(obs).unsqueeze(0)
        logits, _ = self(x)
        return int(logits.argmax(dim=-1).item())


def setup_gui_single():
    try:
        traci.gui.setSchema("View #0", "dark-traffic")
        traci.gui.setZoom("View #0", 150)
        traci.gui.setOffset("View #0", 200, 150)
    except Exception as e:
        print(f"[GUI] {e}")


def setup_gui_grid():
    try:
        traci.gui.setSchema("View #0", "dark-traffic")
        traci.gui.setZoom("View #0", 120)
        traci.gui.setOffset("View #0", 375, 525)
    except Exception as e:
        print(f"[GUI] {e}")


def stitch(tmpdir: str, out_path: Path, framerate: int = 30):
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(framerate),
        "-i", str(Path(tmpdir) / "frame_%05d.png"),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(out_path),
    ], check=True)
    print(f"Saved: {out_path}")


# ── Single-intersection recording ─────────────────────────────────────────────
def record_single(mode: str, max_steps: int, out_path: Path):
    env = sumo_rl.SumoEnvironment(
        net_file=NET_SINGLE, route_file=ROU_SINGLE,
        use_gui=True, num_seconds=3600, delta_time=5, yellow_time=2,
        reward_fn="diff-waiting-time", sumo_seed=SEED,
        single_agent=True, sumo_warnings=False,
        additional_sumo_cmd=f"--gui-settings-file {VIEW_FILE}",
    )
    obs, _ = env.reset()
    setup_gui_single()

    if mode == "ppo":
        model = PPO.load(str(MODELS_DIR / "ppo_model"))
    elif mode == "dqn":
        model = DQN.load(str(MODELS_DIR / "dqn_model"))
    else:
        model = None

    n_actions = env.action_space.n
    phase, step_in_phase = 0, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        frame, done = 0, False
        while not done and frame < max_steps:
            if mode == "random":
                action = env.action_space.sample()
            elif mode == "fixed":
                action = phase
            else:
                action, _ = model.predict(obs, deterministic=True)

            try:
                traci.gui.screenshot("View #0", str(Path(tmpdir) / f"frame_{frame:05d}.png"))
            except Exception:
                pass

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame += 1

            if mode == "fixed":
                step_in_phase += 1
                if step_in_phase >= 6:
                    phase = (phase + 1) % n_actions
                    step_in_phase = 0

        env.close()
        print(f"Captured {frame} frames — stitching...")
        stitch(tmpdir, out_path)


# ── 2×2 MARL recording ────────────────────────────────────────────────────────
def record_marl(mode: str, max_steps: int, out_path: Path):
    env = sumo_rl.parallel_env(
        net_file=NET_GRID, route_file=ROU_GRID,
        use_gui=True, num_seconds=3600, delta_time=5, yellow_time=2,
        reward_fn="diff-waiting-time", sumo_seed=SEED,
        sumo_warnings=False,
        additional_sumo_cmd=f"--gui-settings-file {VIEW_FILE}",
    )
    obs_dict, _ = env.reset()
    setup_gui_grid()

    agent_ids = list(obs_dict.keys())

    policies = {}
    if mode == "marl_ippo":
        obs_dim   = env.observation_space(agent_ids[0]).shape[0]
        n_actions = env.action_space(agent_ids[0]).n
        for aid in agent_ids:
            safe = aid.replace(" ", "_").replace("/", "_")
            net = ActorCritic(obs_dim, n_actions)
            net.load_state_dict(torch.load(
                str(MODELS_DIR / "marl" / f"ippo_{safe}.pt"),
                map_location="cpu",
            ))
            net.eval()
            policies[aid] = net

    with tempfile.TemporaryDirectory() as tmpdir:
        frame = 0
        while env.agents and frame < max_steps:
            actions = {}
            for aid in env.agents:
                if mode == "marl_random":
                    actions[aid] = env.action_space(aid).sample()
                else:
                    actions[aid] = policies[aid].act(obs_dict[aid])

            try:
                traci.gui.screenshot("View #0", str(Path(tmpdir) / f"frame_{frame:05d}.png"))
            except Exception:
                pass

            obs_dict, _, _, _, _ = env.step(actions)
            frame += 1

        env.close()
        print(f"Captured {frame} frames — stitching...")
        stitch(tmpdir, out_path)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["random", "fixed", "ppo", "dqn",
                                 "marl_random", "marl_ippo"])
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()

    if args.mode.startswith("marl"):
        out_path = VIDEOS_DIR / f"{args.mode}.mp4"
        record_marl(args.mode, args.steps, out_path)
    else:
        out_path = VIDEOS_DIR / f"single_{args.mode}.mp4"
        record_single(args.mode, args.steps, out_path)


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

import traci
import sumo_rl
from stable_baselines3 import DQN, PPO

BASE       = Path(__file__).parent
NET_FILE   = str(BASE / "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.net.xml")
ROUTE_FILE = str(BASE / "sumo-rl/sumo_rl/nets/single-intersection/single-intersection.rou.xml")
VIEW_FILE  = str(BASE / "viewsettings.xml")
VIDEOS_DIR = BASE / "videos"
MODELS_DIR = BASE / "models"
SEED = 42

VIDEOS_DIR.mkdir(exist_ok=True)


def make_env():
    return sumo_rl.SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=True,
        num_seconds=3600,
        delta_time=5,
        yellow_time=2,
        reward_fn="diff-waiting-time",
        sumo_seed=SEED,
        single_agent=True,
        sumo_warnings=False,
        additional_sumo_cmd=f"--gui-settings-file {VIEW_FILE}",
    )


def setup_gui():
    try:
        traci.gui.setSchema("View #0", "dark-traffic")
        traci.gui.setZoom("View #0", 150)
        traci.gui.setOffset("View #0", 200, 150)
    except Exception as e:
        print(f"[GUI] setup warning: {e}")


def record(mode: str, max_steps: int, out_path: Path):
    env = make_env()
    obs, _ = env.reset()
    setup_gui()

    if mode == "ppo":
        model = PPO.load(str(MODELS_DIR / "ppo_model"))
    elif mode == "dqn":
        model = DQN.load(str(MODELS_DIR / "dqn_model"))
    else:
        model = None

    n_actions = env.action_space.n
    phase, step_in_phase = 0, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        frame = 0
        done = False
        while not done and frame < max_steps:
            if mode == "random":
                action = env.action_space.sample()
            elif mode == "fixed":
                action = phase
            else:
                action, _ = model.predict(obs, deterministic=True)

            img = str(Path(tmpdir) / f"frame_{frame:05d}.png")
            try:
                traci.gui.screenshot("View #0", img)
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
        print(f"Captured {frame} frames — stitching video...")

        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", str(Path(tmpdir) / "frame_%05d.png"),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            str(out_path),
        ], check=True)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "fixed", "ppo", "dqn"], required=True)
    parser.add_argument("--steps", type=int, default=720, help="Max simulation steps to record")
    args = parser.parse_args()

    out_path = VIDEOS_DIR / f"single_{args.mode}.mp4"
    record(args.mode, args.steps, out_path)


if __name__ == "__main__":
    main()

import os
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = "/usr/share/sumo"

import traci
import sumo_rl

BASE = Path(__file__).parent
NET_FILE  = str(BASE / "sumo-rl/sumo_rl/nets/2x2grid/2x2.net.xml")
ROUTE_FILE = str(BASE / "sumo-rl/sumo_rl/nets/2x2grid/2x2.rou.xml")
VIEW_FILE  = str(BASE / "viewsettings.xml")

env = sumo_rl.parallel_env(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    use_gui=True,
    num_seconds=3600,
    delta_time=5,
    yellow_time=2,
    reward_fn="diff-waiting-time",
    sumo_seed=42,
    additional_sumo_cmd=f"--gui-settings-file {VIEW_FILE}",
)

observations, infos = env.reset()

# Dark theme + zoom centered on the 2x2 grid
try:
    traci.gui.setSchema("View #0", "dark-traffic")
    traci.gui.setZoom("View #0", 120)
    traci.gui.setOffset("View #0", 375, 525)
    print("[GUI] 2x2 dark theme applied")
except Exception as e:
    print(f"[GUI] setup failed: {e}")

print(f"Agents: {env.agents}")
print("Running 2x2 grid with random actions...")

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

print("Done.")
env.close()

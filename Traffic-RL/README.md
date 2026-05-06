# Traffic-RL

Reinforcement learning for adaptive traffic signal control, built on [SUMO](https://sumo.dlr.de) and [sumo-rl](https://github.com/LucasAlegre/sumo-rl).

JIIT Noida — Minor Project 2 — 2025–26

---

## What it does

Trains and benchmarks several RL algorithms to control traffic signals at:

- **Single intersection** — the standard sumo-rl benchmark intersection
- **JIIT intersection** — the real signal-controlled junction at JIIT Noida Sector 62, built from OpenStreetMap data
- **Custom intersections** — any intersection in the world, picked by clicking a map in the dashboard

A live Streamlit dashboard visualises training progress, compares algorithms, and lets you run/record simulations.

---

## Algorithms

### Single-agent (single intersection & JIIT)

| Algorithm | File | Notes |
|---|---|---|
| Fixed-cycle | `train.py` | Baseline — cycles through phases on a fixed 30 s timer |
| Q-Learning | `train.py` | Tabular, discretised observation space |
| DQN | `train.py` | Stable-Baselines3, 64-64 MLP |
| PPO | `train.py` | Stable-Baselines3, 64-64 MLP, best single-agent performer |

### Multi-agent (2×2 grid — 4 intersections)

| Algorithm | File | Notes |
|---|---|---|
| IPPO | `train_marl.py` | Independent PPO — each agent trains its own 64-64 actor-critic |
| IAQPPO | `new-trafficRL/train_marl_aqppo.py` | **Novel contribution** — see below |

### AQPPO / IAQPPO (novel)

AQPPO augments the standard PPO actor-critic with a Q-value head on the same shared backbone:

- **128-128 shared trunk** with LayerNorm (vs 64-64 in IPPO)
- **Three output heads:** policy logits (actor), state value (critic), Q-values per action
- The Q-head provides a direct action-quality signal that complements the advantage estimate from the critic
- Applied independently per agent in the MARL setting → **Independent AQPPO (IAQPPO)**

---

## Project structure

```
Traffic-RL/
├── train.py                  # Single-agent training (Fixed / Q-Learning / DQN / PPO)
├── train_marl.py             # MARL training — IPPO on 2×2 grid
├── custom_intersection.py    # OSM → SUMO pipeline for any intersection
├── dashboard.py              # Streamlit dashboard
├── environ.py                # Shared environment helpers
├── record.py                 # Record simulation videos
├── view_2x2.py               # Launch SUMO GUI for 2×2 grid
│
├── jiit-intersection/        # Real JIIT Noida intersection
│   ├── area.osm              # Raw OSM export
│   ├── jiit.net.xml          # SUMO network
│   ├── jiit.rou.xml          # Vehicle routes (1800 vehicles / hour)
│   └── explain.md            # Full step-by-step build notes
│
├── metrics/                  # Training logs (CSV + JSON status)
│   ├── single/               # Single-intersection results
│   ├── jiit/                 # JIIT intersection results
│   └── marl/                 # MARL results
│
├── models/                   # Saved model weights
│   ├── ppo_model.zip
│   ├── dqn_model.zip
│   ├── jiit/ppo_model.zip
│   └── marl/ippo_*.pt
│
├── videos/                   # Recorded simulation videos
├── custom-intersections/     # Generated custom intersection data
└── new-trafficRL/            # Extended version with AQPPO + single-agent AQPPO
```

---

## Setup

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv), SUMO

```bash
# Install SUMO (Ubuntu/Debian)
sudo apt install sumo sumo-tools

# Install Python dependencies
uv sync
```

---

## Training

### Single intersection

```bash
uv run train.py --net single --algo fixed
uv run train.py --net single --algo qlearning --steps 50000
uv run train.py --net single --algo dqn       --steps 100000
uv run train.py --net single --algo ppo       --steps 100000
```

### JIIT intersection

```bash
uv run train.py --net jiit --algo fixed
uv run train.py --net jiit --algo ppo --steps 100000
```

### MARL — 2×2 grid (IPPO)

```bash
uv run train_marl.py --steps 200000
```

### MARL — 2×2 grid (IAQPPO)

```bash
cd new-trafficRL
uv run train_marl_aqppo.py --steps 200000
```

---

## Dashboard

```bash
uv run streamlit run dashboard.py
```

Tabs:
- **Overview** — benchmark summary table
- **Training Curves** — reward and waiting time per episode
- **Comparison** — side-by-side algorithm comparison
- **MARL — 2×2 Grid** — per-agent IPPO training curves
- **Simulation** — play back recorded videos
- **JIIT Training** — PPO training on the real JIIT network
- **Real World — JIIT** — satellite map view of the JIIT intersection
- **Custom Intersection** — click any intersection on the map to simulate it

---

## JIIT Intersection

The JIIT intersection was built from real OpenStreetMap data:

1. OSM export → `netconvert` → `jiit.net.xml`
2. `randomTrips.py` + `duarouter` → 1800 vehicles / hour
3. 1 TLS with 5 green phases (10-phase SUMO signal including yellows), action space = 5
4. Observation: 5 one-hot phase bits + 15 normalised lane queue lengths (dim = 20)
5. Reward: `diff-waiting-time` — penalises increase in total vehicle wait

See [`jiit-intersection/explain.md`](jiit-intersection/explain.md) for the full build walkthrough.

---

## Environment

| Parameter | Value |
|---|---|
| Simulation duration | 3600 s (1 hour) |
| Agent decision interval | 5 s |
| Yellow phase | 2 s (automatic) |
| Reward function | diff-waiting-time |
| Seed | 42 |

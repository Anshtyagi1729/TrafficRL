# Real-World Intersection — JIIT Noida: How We Built It

This document explains every step taken to convert the real JIIT Noida intersection into a SUMO simulation that the RL agent can train on.

---

## 1. Choosing the Intersection

The intersection chosen is the main signal-controlled junction adjacent to JIIT (Jaypee Institute of Information Technology), Sector 62, Noida.

**Why this intersection:**
- It is a real, complex 4-approach junction with multiple turning movements
- It has an existing traffic signal (captured in OpenStreetMap data)
- It is directly relevant to the project's real-world motivation

**Approximate centre:** 28°31′50″ N, 77°21′44″ E

**OSM bounding box used for download:**

| | Latitude | Longitude |
|---|---|---|
| South-West | 28.5284940 | 77.3601060 |
| North-East | 28.5328940 | 77.3645060 |

This covers roughly a 500 m × 500 m area around the junction, enough to capture all approach roads without pulling in the wider Sector 62 road network.

---

## 2. Downloading OSM Data

The raw map data was downloaded from [OpenStreetMap](https://www.openstreetmap.org) as an `.osm` XML file covering the bounding box above.

**Output file:** `area.osm`

The `.osm` file contains all OSM primitives in the area — nodes (lat/lon points), ways (roads, footpaths), and relations — along with road tags such as `highway`, `lanes`, `oneway`, and crucially `highway=traffic_signals` which marks signalised junctions.

---

## 3. Converting OSM → SUMO Network (`netconvert`)

OpenStreetMap data cannot be used directly by SUMO. It was converted to a SUMO network file using `netconvert`, SUMO's network builder.

**Command used:**
```bash
netconvert \
    --osm-files area.osm \
    --output-file jiit.net.xml \
    --proj.utm \
    --geometry.remove \
    --roundabouts.guess \
    --tls.guess \
    --tls.guess-signals \
    --tls.set cluster_13253772280_13253772282_826539582_826539631_#3more
```

**What each flag does:**

| Flag | Purpose |
|---|---|
| `--osm-files area.osm` | Input the downloaded OSM data |
| `--output-file jiit.net.xml` | Output the SUMO network |
| `--proj.utm` | Project geographic coordinates (lat/lon) to a flat UTM metric coordinate system (Zone 43N for this area) — SUMO works in metres |
| `--geometry.remove` | Merge short intermediate edge segments into single edges, reduces model complexity |
| `--roundabouts.guess` | Auto-detect roundabout patterns in the OSM topology |
| `--tls.guess` | Automatically detect which junctions should be traffic-light controlled |
| `--tls.guess-signals` | Use OSM `traffic_signals` node tags to confirm TLS placement |
| `--tls.set ...` | Force the specific cluster of OSM junction nodes (the JIIT signal cluster) to be one combined TLS |

**Output:** `jiit.net.xml` — a SUMO network with edges, lanes, junctions, and one traffic light logic.

### Why one TLS?

The JIIT junction is topologically a cluster of nearby nodes (`#3more` in the ID means 3+ extra nodes were merged). `netconvert` clustered them into a single combined signal controller with ID:

```
cluster_13253772280_13253772282_826539582_826539631_#3more
```

This is standard SUMO behaviour for real-world intersections where OSM has multiple close nodes for a single physical junction.

---

## 4. Traffic Signal Phases

The combined TLS has **10 phases** — 5 green phases interleaved with 5 yellow clearance phases:

| Phase | Duration | State (16 lanes) | Type |
|---|---|---|---|
| 0 | 17 s | `GGrrrrGGGGrrrrrr` | Green (main through movements) |
| 1 | 5 s  | `yyrrrryyyyrrrrrr` | Yellow |
| 2 | 6 s  | `rrGrrrrrrrGrrrrr` | Green (left turns) |
| 3 | 5 s  | `rryrrrrrrryrrrrr` | Yellow |
| 4 | 18 s | `rrrGGGGrrrrrrrrg` | Green (cross movements) |
| 5 | 5 s  | `rrryyyyrrrrrrrrg` | Yellow |
| 6 | 6 s  | `rrrrrrrrrrrrrrrG` | Green (minor approach) |
| 7 | 5 s  | `rrrrrrrrrrrrrrry` | Yellow |
| 8 | 18 s | `rrrrrrrrrrrGGGGr` | Green (opposite cross) |
| 9 | 5 s  | `rrrrrrrrrrryyyyr` | Yellow |

`G` = green, `y` = yellow, `r` = red, `g` = lower-priority green (yield). Each character represents one of the 16 lanes entering the junction.

**sumo-rl exposes only the 5 green phases as actions** (yellow phases run automatically for 2 s after any phase switch). The RL agent therefore has an **action space of 5**.

---

## 5. Downloading Map Tiles (Satellite Imagery)

To make the SUMO GUI show the real satellite/map background, map tiles were downloaded using SUMO's `tileGet.py` utility.

**Command used:**
```bash
python /usr/share/sumo/tools/tileGet.py \
    -n jiit.net.xml \
    -t 16 \
    -d . \
    -s map_decals.xml \
    -m cartolight
```

**What each flag does:**

| Flag | Purpose |
|---|---|
| `-n jiit.net.xml` | Use the network to determine the geographic area to cover |
| `-t 16` | Download at zoom level 16 (tile size ~600 m per tile at this latitude) |
| `-d .` | Save tiles to the current directory |
| `-s map_decals.xml` | Output a SUMO viewsettings file with tile positions |
| `-m cartolight` | Use CartoDB Light map style (clean grey basemap, good contrast with vehicles) |

**How tile naming works:**

Tiles follow the OpenStreetMap slippy-map convention: `{zoom}_{x}_{y}.jpeg`. For example `tile46851_27345.jpeg` is the tile at zoom=17, x=46851, y=27345. The x/y are integer grid coordinates derived from the lat/lon bounding box at the given zoom level.

A second higher-resolution set was downloaded at zoom=17 for the `jiit_view.xml` (used in the SUMO GUI and dashboard), giving finer detail at the intersection centre.

**Output files:**
- `tile23424_*.jpeg` to `tile23426_*.jpeg` — zoom 16 tiles (3×4 grid)
- `tile46848_*.jpeg` to `tile46853_*.jpeg` — zoom 17 tiles (6×7 grid)
- `map_decals.xml` — viewsettings referencing zoom-16 tiles
- `jiit_view.xml` — viewsettings referencing zoom-17 tiles, plus vehicle colour scheme and viewport

Each tile is positioned in SUMO's metric coordinate system using the UTM-projected `centerX`/`centerY` values computed by `tileGet.py` from the tile's geographic boundaries.

---

## 6. Generating Vehicle Routes

SUMO needs vehicles with pre-assigned routes to run a simulation. Two tools were used sequentially:

### Step 1 — `randomTrips.py` (trip generation)

```bash
python /usr/share/sumo/tools/randomTrips.py \
    -n jiit.net.xml \
    -o jiit.trips.xml \
    --route-file jiit.rou.xml \
    --period 2.0 \
    --end 3600
```

| Parameter | Value | Meaning |
|---|---|---|
| `--period 2.0` | 2.0 s | One vehicle departs every 2 seconds |
| `--end 3600` | 3600 s | Trips span a 1-hour simulation window |
| Total vehicles | **1800** | 3600 ÷ 2.0 |

`randomTrips.py` randomly picks a source edge and a destination edge for each vehicle, creating a trip (origin → destination without a specific path). The trips are written to `jiit.trips.xml`.

**Initial calibration issue:** The first route file used `--period 0.8` (4500 vehicles). This caused severe gridlock (mean waiting time 400–750 s) because the small real-world network was saturated from the first episode and the agent could not distinguish its actions' effects from background congestion. Period was increased to 2.0 to give ~1800 vehicles — a congested but learnable traffic level.

### Step 2 — `duarouter` (route assignment)

`randomTrips.py` with `--validate` (or `--route-file`) internally calls `duarouter` to convert the origin-destination trips into full edge-by-edge routes using Dijkstra shortest-path routing:

```bash
duarouter \
    -n jiit.net.xml \
    -r jiit.trips.xml \
    -o jiit.rou.xml \
    --ignore-errors
```

`--ignore-errors` skips trips whose origin or destination is unreachable (can happen with disconnected OSM edges). The output `jiit.rou.xml` contains each vehicle with its complete sequence of SUMO edges.

---

## 7. RL Environment Setup

With network and routes ready, the environment is created using `sumo-rl`:

```python
sumo_rl.SumoEnvironment(
    net_file  = "jiit-intersection/jiit.net.xml",
    route_file= "jiit-intersection/jiit.rou.xml",
    use_gui   = False,
    num_seconds  = 3600,      # 1-hour episode
    delta_time   = 5,         # agent decides every 5 s
    yellow_time  = 2,         # 2 s yellow after each phase switch
    reward_fn    = "diff-waiting-time",
    single_agent = True,
)
```

**Observation space (dim = 20):**
- 5 one-hot values — current active green phase
- 15 values — normalised queue length for each of the 15 incoming lanes

**Action space:** 5 discrete actions — one per green phase

**Reward:** `diff-waiting-time` = −(change in total vehicle waiting time since last step). The agent is rewarded for reducing waiting and penalised for letting it grow.

**Algorithm chosen — PPO:**
PPO was selected over DQN and Q-Learning for the JIIT intersection because:
- It was the best performer in the single-intersection benchmark
- It handles the larger 5-action space more stably than DQN
- Q-Learning cannot scale here — the state space (5 phases × 15 discretised queue lengths) is too large for a tabular approach

---

## 8. File Summary

| File | What it is |
|---|---|
| `area.osm` | Raw OpenStreetMap export for the bounding box |
| `jiit.net.xml` | SUMO network (roads, lanes, junctions, TLS) |
| `jiit.trips.xml` | Origin-destination trips (before routing) |
| `jiit.rou.xml` | Full vehicle routes (after duarouter) |
| `jiit_view.xml` | SUMO GUI viewsettings — zoom-17 tiles + vehicle colours |
| `map_decals.xml` | SUMO GUI viewsettings — zoom-16 tiles |
| `map_decals_hd.xml` | SUMO GUI viewsettings — HD variant |
| `tile46848_*.jpeg` … `tile46853_*.jpeg` | Zoom-17 CartoDB Light map tiles |
| `tile23424_*.jpeg` … `tile23426_*.jpeg` | Zoom-16 CartoDB Light map tiles |

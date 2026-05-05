"""
Custom Intersection Generator
==============================
Click any intersection on the map → pipeline downloads OSM, builds SUMO
network, fetches map tiles, assigns routes → launch simulation.

To remove this feature:
  1. Delete this file
  2. Remove `tab_custom` from st.tabs() in dashboard.py
  3. Remove the `with tab_custom:` block at the bottom of dashboard.py
"""

import json
import math
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import folium
import requests
import streamlit as st
from pyproj import Proj
from streamlit_folium import st_folium

# ── Config ────────────────────────────────────────────────────────────────────
SUMO_HOME    = os.environ.get("SUMO_HOME", "/usr/share/sumo")
TOOLS        = Path(SUMO_HOME) / "tools"
OUT_BASE     = Path(__file__).parent / "custom-intersections"
OUT_BASE.mkdir(exist_ok=True)

OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
CARTO_URL     = "https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
DEFAULT_LAT   = 28.5314
DEFAULT_LON   = 77.3624


# ── Geometry ──────────────────────────────────────────────────────────────────
def bbox_from_center(lat, lon, radius_km):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


# ── Tile math ─────────────────────────────────────────────────────────────────
def _deg2tile(lat, lon, zoom):
    n  = 2 ** zoom
    x  = int((lon + 180) / 360 * n)
    lr = math.radians(lat)
    y  = int((1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n)
    return x, y


def _tile2latlon(tx, ty, zoom):
    """Return (lat, lon) of the NW corner of tile (tx, ty)."""
    n   = 2 ** zoom
    lon = tx / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    return lat, lon


def _zoom_for_radius(radius_km: float) -> int:
    """Pick a zoom level so tiles are ~100-200 m wide at this radius."""
    if radius_km <= 0.4:
        return 18
    if radius_km <= 0.8:
        return 17
    return 16


# ── Custom tile fetcher (replaces tileGet.py) ─────────────────────────────────
def fetch_tiles(net_file: Path, south, west, north, east,
                out_dir: Path, view_file: Path, radius_km: float) -> bool:
    """
    Download CartoDB Light tiles covering the bbox and write a SUMO view.xml.
    Uses the network projection to map tile lat/lon → SUMO XY precisely.
    """
    # --- parse network projection & offset -----------------------------------
    try:
        root = ET.parse(net_file).getroot()
        loc  = root.find(".//location")
        proj_str = loc.get("projParameter",
                           "+proj=utm +zone=43 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        ox, oy = map(float, loc.get("netOffset", "0,0").split(","))
    except Exception as e:
        st.warning(f"Could not parse network projection ({e}) — tiles skipped.")
        return False

    proj = Proj(proj_str)

    def latlon_to_sumo(lat, lon):
        ux, uy = proj(lon, lat)          # lon first (pyproj always_xy convention)
        return ux + ox, uy + oy

    # --- choose zoom and compute tile range ----------------------------------
    zoom   = _zoom_for_radius(radius_km)
    # add 1-tile padding on every side so roads near the edge have background
    tx_min, ty_max = _deg2tile(south, west, zoom)
    tx_max, ty_min = _deg2tile(north, east, zoom)
    tx_min -= 1;  tx_max += 1
    ty_min -= 1;  ty_max += 1

    total  = (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
    st.write(f"  Downloading {total} tiles at zoom {zoom}…")

    decals  = []
    session = requests.Session()
    session.headers.update({"User-Agent": "TrafficRL-SUMO/1.0"})

    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            url      = CARTO_URL.format(z=zoom, x=tx, y=ty)
            filename = f"tile{tx}_{ty}.png"
            dst      = out_dir / filename
            try:
                r = session.get(url, timeout=15)
                if r.status_code != 200:
                    continue
                dst.write_bytes(r.content)
            except Exception:
                continue

            # tile NW and SE corners → SUMO XY
            nw_lat, nw_lon = _tile2latlon(tx,     ty,     zoom)
            se_lat, se_lon = _tile2latlon(tx + 1, ty + 1, zoom)

            nw_x, nw_y = latlon_to_sumo(nw_lat, nw_lon)
            se_x, se_y = latlon_to_sumo(se_lat, se_lon)

            cx = (nw_x + se_x) / 2
            cy = (nw_y + se_y) / 2
            w  = abs(se_x - nw_x)
            h  = abs(nw_y - se_y)

            decals.append((filename, cx, cy, w, h))

    if not decals:
        st.warning("No tiles downloaded — simulation will open without map background.")
        return False

    # --- write view.xml ------------------------------------------------------
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<viewsettings>"]
    for fname, cx, cy, w, h in decals:
        lines.append(
            f'  <decal file="{fname}" centerX="{cx:.4f}" centerY="{cy:.4f}"'
            f' width="{w:.4f}" height="{h:.4f}" layer="0"/>'
        )
    lines += [
        '  <vehicles vehicleMode="8" vehicleQuality="2"'
        ' vehicle_minSize="4.0" vehicle_exaggeration="5.0">',
        '    <colorScheme name="by speed" interpolated="1">',
        '      <entry color="255,30,30"  threshold="0.00"/>',
        '      <entry color="255,180,0"  threshold="5.00"/>',
        '      <entry color="50,200,50"  threshold="10.00"/>',
        '      <entry color="0,160,255"  threshold="20.00"/>',
        '    </colorScheme>',
        '  </vehicles>',
        '  <background backgroundColor="50,50,50" showGrid="0"/>',
        "</viewsettings>",
    ]
    view_file.write_text("\n".join(lines))
    return True


# ── Pipeline ──────────────────────────────────────────────────────────────────
def download_osm(south, west, north, east, out_file: Path) -> bool:
    bbox  = f"{south},{west},{north},{east}"
    query = (
        f'[out:xml][timeout:90];'
        f'(node({bbox});way["highway"]({bbox});'
        f'relation["type"="restriction"]({bbox}););'
        f'out body;>;out skel qt;'
    )
    try:
        r = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=90,
            headers={"User-Agent": "TrafficRL-SUMO/1.0"},
        )
        r.raise_for_status()
        out_file.write_bytes(r.content)
        return True
    except Exception as e:
        st.error(f"OSM download failed: {e}")
        return False


def ensure_tls(net_file: Path) -> tuple[bool, str]:
    """
    Check whether the network has at least one TLS.
    If not, find the junction with the most incoming edges and rerun
    netconvert forcing a TLS there. Returns (had_tls, tls_id).
    """
    root = ET.parse(net_file).getroot()

    # Check existing TLS
    tls_ids = [tl.get("id") for tl in root.findall(".//tlLogic")]
    if tls_ids:
        return True, tls_ids[0]

    # No TLS found — pick the junction with most incoming connections
    junction_incoming: dict[str, int] = {}
    for conn in root.findall(".//connection"):
        jid = conn.get("to") or conn.get("via")
        if jid:
            junction_incoming[jid] = junction_incoming.get(jid, 0) + 1

    # Also count via junction shape
    for junc in root.findall(".//junction"):
        jid  = junc.get("id", "")
        jtyp = junc.get("type", "")
        if jtyp in ("priority", "right_before_left", "allway_stop"):
            inc = junc.get("incLanes", "")
            cnt = len(inc.split()) if inc else 0
            junction_incoming[jid] = max(junction_incoming.get(jid, 0), cnt)

    if not junction_incoming:
        return False, ""

    best = max(junction_incoming, key=junction_incoming.get)

    # Re-run netconvert forcing TLS at that junction
    result = subprocess.run([
        "netconvert",
        "--sumo-net-file", str(net_file),
        "--output-file",   str(net_file),
        "--tls.set",       best,
        "--no-warnings",
    ], capture_output=True, text=True)

    return result.returncode == 0, best


def run_cmd(cmd, cwd=None, label="") -> bool:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        st.error(f"**{label}** failed:\n```\n{res.stderr[-800:]}\n```")
        return False
    return True


def generate(lat, lon, radius_km, name, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    south, west, north, east = bbox_from_center(lat, lon, radius_km)

    osm_file    = out_dir / "area.osm"
    net_file    = out_dir / "net.xml"
    trips_file  = out_dir / "trips.xml"
    routes_file = out_dir / "routes.xml"
    view_file   = out_dir / "view.xml"

    with st.status("Generating intersection…", expanded=True) as status:

        st.write("Downloading OSM data…")
        if not download_osm(south, west, north, east, osm_file):
            status.update(label="Failed — OSM download", state="error")
            return False

        st.write("Building SUMO road network…")
        if not run_cmd([
            "netconvert",
            "--osm-files", str(osm_file),
            "--output-file", str(net_file),
            "--proj.utm", "--geometry.remove",
            "--roundabouts.guess",
            "--tls.guess", "--tls.guess-signals",
            "--no-warnings",
        ], label="netconvert") or not net_file.exists():
            status.update(label="Failed — netconvert", state="error")
            return False

        # Guarantee at least one TLS exists for the RL agent to control
        had_tls, tls_id = ensure_tls(net_file)
        if had_tls:
            st.write(f"Traffic light found: `{tls_id}`")
        else:
            st.warning(
                "No traffic lights in OSM data — a signal was forced at the "
                "most complex junction so the RL agent has something to control."
            )

        st.write("Fetching map tiles…")
        fetch_tiles(net_file, south, west, north, east, out_dir, view_file, radius_km)

        st.write("Generating vehicle trips…")
        if not run_cmd([
            "python", str(TOOLS / "randomTrips.py"),
            "-n", str(net_file),
            "-o", str(trips_file),
            "-e", "3600", "-p", "2.0",
        ], label="randomTrips") or not trips_file.exists():
            status.update(label="Failed — randomTrips", state="error")
            return False

        st.write("Routing vehicles…")
        if not run_cmd([
            "duarouter",
            "-n", str(net_file),
            "-r", str(trips_file),
            "-o", str(routes_file),
            "--ignore-errors", "--no-warnings",
        ], label="duarouter") or not routes_file.exists():
            status.update(label="Failed — duarouter", state="error")
            return False

        (out_dir / "meta.json").write_text(json.dumps({
            "name": name, "lat": lat, "lon": lon,
            "radius_km": radius_km,
            "bbox": [south, west, north, east],
            "created_at": time.time(),
        }, indent=2))

        status.update(label=f"Ready — {name}", state="complete")
    return True


# ── Saved intersections ───────────────────────────────────────────────────────
def list_saved():
    return sorted(
        [d for d in OUT_BASE.iterdir()
         if d.is_dir() and (d / "meta.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def launch_sumo(d: Path):
    cmd = [
        "sumo-gui",
        "-n", str(d / "net.xml"),
        "-r", str(d / "routes.xml"),
        "--start", "--delay", "80",
    ]
    if (d / "view.xml").exists():
        cmd += ["--gui-settings-file", str(d / "view.xml")]
    subprocess.Popen(cmd, cwd=str(d))


# ── UI ────────────────────────────────────────────────────────────────────────
def render_custom_tab():
    st.markdown("### Custom Intersection Generator")
    st.markdown(
        "Click any intersection on the map, set a capture radius, "
        "then hit **Generate** — OSM data, road network, map tiles, "
        "and vehicle routes are all built automatically."
    )
    st.caption(
        "Tip: find your intersection in the map below and click it. "
        "Or paste coordinates from Google Maps (right-click → copy coordinates)."
    )
    st.markdown("---")

    if "ci_lat" not in st.session_state:
        st.session_state.ci_lat    = DEFAULT_LAT
        st.session_state.ci_lon    = DEFAULT_LON
        st.session_state.ci_picked = False

    col_map, col_panel = st.columns([3, 2], gap="large")

    with col_map:
        radius_km = st.select_slider(
            "Capture radius",
            options=[0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5],
            value=0.4,
            format_func=lambda v: f"{v} km",
            help="Larger = more roads, more tiles, slower to generate",
        )

        lat = st.session_state.ci_lat
        lon = st.session_state.ci_lon

        m = folium.Map(location=[lat, lon], zoom_start=15, tiles="CartoDB positron")

        if st.session_state.ci_picked:
            folium.Marker(
                [lat, lon],
                tooltip=f"{lat:.5f}, {lon:.5f}",
                icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
            ).add_to(m)
            folium.Circle(
                [lat, lon],
                radius=radius_km * 1000,
                color="#3b82f6",
                fill=True,
                fill_opacity=0.10,
                tooltip=f"{radius_km} km capture area",
            ).add_to(m)

        result = st_folium(m, height=420, use_container_width=True)

        clicked = result.get("last_clicked")
        if clicked:
            st.session_state.ci_lat    = clicked["lat"]
            st.session_state.ci_lon    = clicked["lng"]
            st.session_state.ci_picked = True
            st.rerun()

        if st.session_state.ci_picked:
            st.caption(
                f"Selected: **{st.session_state.ci_lat:.5f}° N, "
                f"{st.session_state.ci_lon:.5f}° E**  ·  radius {radius_km} km"
            )
        else:
            st.caption("Click anywhere on the map to select an intersection.")

    with col_panel:
        st.markdown("#### Generate")

        name = st.text_input("Label", placeholder="e.g. Sector 18 Noida Signal")

        can_generate = st.session_state.ci_picked and bool(name.strip())

        if st.button(
            "Generate Intersection",
            type="primary",
            use_container_width=True,
            disabled=not can_generate,
        ):
            slug    = f"{name.strip().replace(' ', '_')}_{int(time.time())}"
            out_dir = OUT_BASE / slug
            ok = generate(
                st.session_state.ci_lat,
                st.session_state.ci_lon,
                radius_km,
                name.strip(),
                out_dir,
            )
            if ok:
                st.rerun()

        if not st.session_state.ci_picked:
            st.caption("Click the map first to pick a location.")
        elif not name.strip():
            st.caption("Enter a label to enable generation.")

        st.markdown("---")
        st.markdown("#### Saved Intersections")

        saved = list_saved()
        if not saved:
            st.info("Nothing generated yet.")
        else:
            for d in saved:
                try:
                    meta = json.loads((d / "meta.json").read_text())
                except Exception:
                    continue

                net_ok    = (d / "net.xml").exists()
                routes_ok = (d / "routes.xml").exists()
                tiles_ok  = (d / "view.xml").exists()
                complete  = net_ok and routes_ok
                created   = time.strftime("%d %b %H:%M",
                                          time.localtime(meta.get("created_at", 0)))

                with st.expander(f"**{meta['name']}** · {created}"):
                    st.caption(
                        f"{meta['lat']:.4f}°N {meta['lon']:.4f}°E · "
                        f"{meta['radius_km']} km"
                    )
                    st.markdown(
                        ("🟢 " if net_ok    else "🔴 ") + "Network &nbsp;"
                        + ("🟢 " if routes_ok else "🔴 ") + "Routes &nbsp;"
                        + ("🟢 " if tiles_ok  else "⚪ ") + "Tiles",
                        unsafe_allow_html=True,
                    )
                    if complete:
                        if st.button("Launch in SUMO GUI",
                                     key=f"launch_{d.name}",
                                     use_container_width=True,
                                     type="primary"):
                            launch_sumo(d)
                            st.success("SUMO GUI launched — check your taskbar.")
                    else:
                        st.warning("Incomplete — regenerate.")

                    if st.button("Delete", key=f"del_{d.name}",
                                 use_container_width=True):
                        shutil.rmtree(d)
                        st.rerun()

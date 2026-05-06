import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic RL — Benchmark Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
METRICS_DIR = BASE / "metrics"
MODELS_DIR = BASE / "models"
VIDEOS_DIR = BASE / "videos"

ALGO_META = {
    "fixed":     {"label": "Fixed-Cycle", "color": "#64748b"},
    "qlearning": {"label": "Q-Learning",  "color": "#f59e0b"},
    "dqn":       {"label": "DQN",          "color": "#3b82f6"},
    "ppo":       {"label": "PPO",          "color": "#10b981"},
    "aqppo":     {"label": "AQPPO ★",     "color": "#e879f9"},
}

MARL_META = {
    "1": {"label": "Agent 1", "color": "#8b5cf6"},
    "2": {"label": "Agent 2", "color": "#ec4899"},
    "5": {"label": "Agent 5", "color": "#06b6d4"},
    "6": {"label": "Agent 6", "color": "#f97316"},
}

MARL_AQPPO_META = {
    "1": {"label": "AQPPO Agent 1", "color": "#c084fc"},
    "2": {"label": "AQPPO Agent 2", "color": "#f472b6"},
    "5": {"label": "AQPPO Agent 5", "color": "#22d3ee"},
    "6": {"label": "AQPPO Agent 6", "color": "#fb923c"},
}

CHART = dict(
    paper_bgcolor="#0f1117",
    plot_bgcolor="#0f1117",
    font=dict(family="Inter, system-ui, sans-serif", color="#94a3b8", size=12),
    xaxis=dict(gridcolor="#1e2130", linecolor="#2d3450", zeroline=False),
    yaxis=dict(gridcolor="#1e2130", linecolor="#2d3450", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    margin=dict(t=48, b=48, l=64, r=24),
    hoverlabel=dict(bgcolor="#1e2130", bordercolor="#2d3450", font_color="#e2e8f0"),
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
.main .block-container { padding: 2rem 2.5rem; max-width: 1280px; }

[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2d3450;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
}
[data-testid="stMetricLabel"] > div {
    color: #64748b !important; font-size: 0.7rem !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] > div {
    color: #f1f5f9 !important; font-size: 1.8rem !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] > div { font-size: 0.78rem !important; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent; border-bottom: 1px solid #2d3450; gap: 0; padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748b;
    border-bottom: 2px solid transparent; border-radius: 0;
    padding: 0.7rem 1.4rem; font-size: 0.875rem; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: transparent !important; color: #f1f5f9 !important;
    border-bottom: 2px solid #3b82f6 !important;
}

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
}
.badge-done    { background: #052e16; color: #34d399; border: 1px solid #065f46; }
.badge-running { background: #1e3a5f; color: #60a5fa; border: 1px solid #1e40af; }
.badge-pending { background: #1c1917; color: #78716c; border: 1px solid #292524; }
.badge-new     { background: #2d1657; color: #e879f9; border: 1px solid #7c3aed; }

hr { border-color: #2d3450; margin: 1.5rem 0; }
[data-testid="stDataFrame"] { border: 1px solid #2d3450; border-radius: 8px; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d3450; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def load_single(algo: str) -> pd.DataFrame | None:
    path = METRICS_DIR / "single" / f"{algo}_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


@st.cache_data(ttl=10)
def load_marl(agent_id: str) -> pd.DataFrame | None:
    path = METRICS_DIR / "marl" / f"{agent_id}_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


@st.cache_data(ttl=10)
def load_marl_aqppo(agent_id: str) -> pd.DataFrame | None:
    path = METRICS_DIR / "marl_aqppo" / f"{agent_id}_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


@st.cache_data(ttl=10)
def load_status(algo: str, mode: str = "single") -> dict:
    path = METRICS_DIR / mode / f"{algo}_status.json"
    if not path.exists():
        return {"state": "pending", "progress": 0, "episode": 0}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"state": "pending", "progress": 0, "episode": 0}


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def final_stats(df: pd.DataFrame, n: int = 10) -> dict:
    tail = df.tail(n)
    return {
        "wait":     round(tail["waiting_time"].mean(), 2),
        "speed":    round(tail["mean_speed"].mean(), 3),
        "reward":   round(tail["reward"].mean(), 2),
        "episodes": int(df["episode"].max()),
    }


def algo_state(algo: str) -> str:
    return load_status(algo).get("state", "pending")


def badge(state: str, is_aqppo: bool = False) -> str:
    if is_aqppo and state == "pending":
        return '<span class="badge badge-new">New</span>'
    if state == "done":
        return '<span class="badge badge-done">Done</span>'
    if state == "running":
        return '<span class="badge badge-running">Training</span>'
    return '<span class="badge badge-pending">Pending</span>'


# ── Chart helpers ─────────────────────────────────────────────────────────────
def line_chart(title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font_color="#e2e8f0", font_size=14),
        xaxis_title="Episode", yaxis_title=y_label, **CHART,
    )
    return fig


def add_trace(fig, df, x_col, y_col, name, color, window, dash="solid", width=2):
    y = smooth(df[y_col], window)
    fig.add_trace(go.Scatter(
        x=df[x_col], y=y, name=name, mode="lines",
        line=dict(color=color, width=width, dash=dash),
        hovertemplate=f"<b>{name}</b><br>Episode: %{{x}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
    ))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Traffic RL")
    st.markdown("Benchmarking RL algorithms for adaptive traffic signal control.")
    st.markdown("---")

    st.markdown("**Single Intersection**")
    for algo, meta in ALGO_META.items():
        state = algo_state(algo)
        is_aqppo = algo == "aqppo"
        st.markdown(
            f"{meta['label']} &nbsp; {badge(state, is_aqppo)}",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**2×2 MARL Grid**")
    marl_state = load_status("marl", "marl").get("state", "pending")
    st.markdown(f"IPPO (4 agents) &nbsp; {badge(marl_state)}", unsafe_allow_html=True)

    marl_aqppo_status_path = METRICS_DIR / "marl_aqppo" / "marl_aqppo_status.json"
    if marl_aqppo_status_path.exists():
        try:
            marl_aqppo_state = json.loads(marl_aqppo_status_path.read_text()).get("state", "pending")
        except Exception:
            marl_aqppo_state = "pending"
    else:
        marl_aqppo_state = "pending"
    st.markdown(f"IAQPPO ★ (4 agents) &nbsp; {badge(marl_aqppo_state, is_aqppo=True)}", unsafe_allow_html=True)

    st.markdown("---")
    smoothing = st.slider("Smoothing window", 1, 20, 7, help="Rolling average over N episodes")

    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("JIIT Noida — Minor Project 2 — 2025–26")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_curves, tab_compare, tab_aqppo, tab_marl, tab_marl_aqppo, tab_video, tab_realworld = st.tabs(
    ["Overview", "Training Curves", "Comparison", "AQPPO", "MARL — IPPO", "MARL — IAQPPO ★", "Simulation", "Real World — JIIT"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### Benchmark Summary")
    st.markdown("Single signalised intersection — SUMO simulation — 3600 s per episode")
    st.markdown("---")

    all_stats = {}
    for algo in ALGO_META:
        df = load_single(algo)
        if df is not None:
            all_stats[algo] = final_stats(df)

    if not all_stats:
        st.info("No training data yet. Run `uv run python train.py --algo <algo>` to start.")
    else:
        best_algo  = min(all_stats, key=lambda a: all_stats[a]["wait"])
        fixed_wait = all_stats.get("fixed", {}).get("wait", None)
        best_wait  = all_stats[best_algo]["wait"]
        improvement = (
            round((fixed_wait - best_wait) / fixed_wait * 100, 1)
            if fixed_wait and fixed_wait > 0 else None
        )
        total_eps = sum(s["episodes"] for s in all_stats.values())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Best Algorithm", ALGO_META[best_algo]["label"])
        with c2:
            st.metric("Lowest Mean Wait", f"{best_wait:.1f} s")
        with c3:
            st.metric(
                "Improvement over Fixed-Cycle",
                f"{improvement:.1f}%" if improvement else "—",
                delta=f"−{improvement:.1f}%" if improvement else None,
                delta_color="normal",
            )
        with c4:
            st.metric("Total Episodes Trained", f"{total_eps:,}")

        st.markdown("---")

        algos_sorted = sorted(all_stats, key=lambda a: all_stats[a]["wait"], reverse=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[ALGO_META[a]["label"] for a in algos_sorted],
            y=[all_stats[a]["wait"] for a in algos_sorted],
            marker_color=[ALGO_META[a]["color"] for a in algos_sorted],
            text=[f"{all_stats[a]['wait']}" for a in algos_sorted],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=12),
            hovertemplate="<b>%{x}</b><br>Mean Wait: %{y} s<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Final Mean Vehicle Waiting Time — lower is better",
                        font_color="#e2e8f0", font_size=14),
            yaxis_title="Mean Waiting Time (s)", showlegend=False, **CHART,
        )
        fig.update_xaxes(tickfont=dict(size=13, color="#e2e8f0"))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Summary Table")
        rows = []
        for algo in ALGO_META:
            if algo in all_stats:
                s   = all_stats[algo]
                imp = (
                    f"{(fixed_wait - s['wait']) / fixed_wait * 100:.1f}%"
                    if fixed_wait and algo != "fixed" else "—"
                )
                rows.append({
                    "Algorithm":      ALGO_META[algo]["label"],
                    "Mean Wait (s)":  s["wait"],
                    "Mean Speed (m/s)": s["speed"],
                    "Mean Reward/ep": s["reward"],
                    "vs Fixed-Cycle": imp,
                    "Episodes":       s["episodes"],
                })
        st.dataframe(pd.DataFrame(rows).set_index("Algorithm"), width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING CURVES
# ════════════════════════════════════════════════════════════════════════════
with tab_curves:
    st.markdown("### Training Curves — Single Intersection")

    available = {a: load_single(a) for a in ALGO_META if load_single(a) is not None}
    if not available:
        st.info("No training data yet.")
    else:
        col_sel, _ = st.columns([2, 3])
        with col_sel:
            selected = st.multiselect(
                "Algorithms", options=list(available.keys()),
                default=list(available.keys()),
                format_func=lambda a: ALGO_META[a]["label"],
            )

        if selected:
            fig_r = line_chart("Cumulative Reward per Episode", "Reward")
            fig_w = line_chart("Mean Vehicle Waiting Time per Episode", "Waiting Time (s)")

            for algo in selected:
                df  = available[algo]
                col = ALGO_META[algo]["color"]
                lbl = ALGO_META[algo]["label"]
                w   = 3 if algo == "aqppo" else 2
                add_trace(fig_r, df, "episode", "reward", lbl, col, smoothing, width=w)
                add_trace(fig_w, df, "episode", "waiting_time", lbl, col, smoothing, width=w)

            st.plotly_chart(fig_r, width="stretch")
            st.plotly_chart(fig_w, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### Algorithm Comparison — Final Performance")

    all_stats_cmp = {}
    for algo in ALGO_META:
        df = load_single(algo)
        if df is not None:
            all_stats_cmp[algo] = final_stats(df)

    if not all_stats_cmp:
        st.info("No training data yet.")
    else:
        col_sel, _ = st.columns([3, 2])
        with col_sel:
            selected_cmp = st.multiselect(
                "Select algorithms to compare",
                options=list(all_stats_cmp.keys()),
                default=list(all_stats_cmp.keys()),
                format_func=lambda a: ALGO_META[a]["label"],
            )

        filtered = {a: all_stats_cmp[a] for a in selected_cmp if a in all_stats_cmp}

        if not filtered:
            st.warning("Select at least one algorithm above.")
        else:
            labels  = [ALGO_META[a]["label"] for a in filtered]
            colors  = [ALGO_META[a]["color"] for a in filtered]
            waits   = [filtered[a]["wait"]   for a in filtered]
            speeds  = [filtered[a]["speed"]  for a in filtered]
            rewards = [filtered[a]["reward"] for a in filtered]

            c_left, c_right = st.columns(2)
            with c_left:
                fig = go.Figure(go.Bar(
                    x=labels, y=waits, marker_color=colors,
                    text=[f"{w}" for w in waits], textposition="outside",
                    textfont=dict(color="#94a3b8"),
                    hovertemplate="<b>%{x}</b><br>%{y} s<extra></extra>",
                ))
                fig.update_layout(title=dict(text="Mean Waiting Time (s)", font_color="#e2e8f0", font_size=14),
                                  yaxis_title="Seconds", showlegend=False, **CHART)
                st.plotly_chart(fig, width="stretch")

            with c_right:
                fig = go.Figure(go.Bar(
                    x=labels, y=speeds, marker_color=colors,
                    text=[f"{s}" for s in speeds], textposition="outside",
                    textfont=dict(color="#94a3b8"),
                    hovertemplate="<b>%{x}</b><br>%{y} m/s<extra></extra>",
                ))
                fig.update_layout(title=dict(text="Mean Vehicle Speed (m/s)", font_color="#e2e8f0", font_size=14),
                                  yaxis_title="m/s", showlegend=False, **CHART)
                st.plotly_chart(fig, width="stretch")

            fig = go.Figure(go.Bar(
                x=labels, y=rewards, marker_color=colors,
                text=[f"{r}" for r in rewards], textposition="outside",
                textfont=dict(color="#94a3b8"),
                hovertemplate="<b>%{x}</b><br>Reward: %{y}<extra></extra>",
            ))
            fig.update_layout(title=dict(text="Mean Reward per Episode (higher is better)",
                                          font_color="#e2e8f0", font_size=14),
                              yaxis_title="Reward", showlegend=False, **CHART)
            st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — AQPPO
# ════════════════════════════════════════════════════════════════════════════
with tab_aqppo:
    st.markdown("### AQPPO — Adaptive Q-Augmented PPO")
    st.markdown("---")

    aqppo_df  = load_single("aqppo")
    ppo_df    = load_single("ppo")
    fixed_df  = load_single("fixed")

    if aqppo_df is None:
        st.info("AQPPO not trained yet. Run:")
        st.code("uv run python train.py --algo aqppo --steps 50000")
    else:
        aqppo_stats = final_stats(aqppo_df)
        ppo_stats   = final_stats(ppo_df)   if ppo_df   is not None else None
        fixed_stats = final_stats(fixed_df) if fixed_df is not None else None

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("AQPPO Mean Wait", f"{aqppo_stats['wait']} s")
        with c2:
            delta_ppo = (
                round(ppo_stats["wait"] - aqppo_stats["wait"], 2) if ppo_stats else None
            )
            if delta_ppo is not None:
                label = "better" if delta_ppo > 0 else "worse"
                d_color = "normal" if delta_ppo > 0 else "inverse"
                st.metric("vs PPO", f"{delta_ppo:+} s", delta=label, delta_color=d_color)
            else:
                st.metric("vs PPO", "—")
        with c3:
            delta_fixed = (
                round((fixed_stats["wait"] - aqppo_stats["wait"]) / fixed_stats["wait"] * 100, 1)
                if fixed_stats else None
            )
            if delta_fixed is not None:
                st.metric("vs Fixed-Cycle", f"{delta_fixed:.1f}% better")
            else:
                st.metric("vs Fixed-Cycle", "—")
        with c4:
            st.metric("Episodes Trained", f"{aqppo_stats['episodes']:,}")

        st.markdown("---")

        # AQPPO vs PPO training curves
        st.markdown("#### AQPPO vs PPO — Training Curves")
        fig_r = line_chart("Reward per Episode", "Reward")
        fig_w = line_chart("Mean Waiting Time per Episode", "Waiting Time (s)")

        add_trace(fig_r, aqppo_df, "episode", "reward", "AQPPO ★", "#e879f9", smoothing, width=3)
        add_trace(fig_w, aqppo_df, "episode", "waiting_time", "AQPPO ★", "#e879f9", smoothing, width=3)

        if ppo_df is not None:
            add_trace(fig_r, ppo_df, "episode", "reward", "PPO", "#10b981", smoothing)
            add_trace(fig_w, ppo_df, "episode", "waiting_time", "PPO", "#10b981", smoothing)

        st.plotly_chart(fig_r, width="stretch")
        st.plotly_chart(fig_w, width="stretch")

        # Head-to-head bar chart
        st.markdown("#### Head-to-Head: All Algorithms")
        compare_algos = ["fixed", "qlearning", "dqn", "ppo", "aqppo"]
        compare_data  = {a: final_stats(load_single(a)) for a in compare_algos if load_single(a) is not None}

        if compare_data:
            sorted_algos = sorted(compare_data, key=lambda a: compare_data[a]["wait"], reverse=True)
            fig = go.Figure(go.Bar(
                x=[ALGO_META[a]["label"] for a in sorted_algos],
                y=[compare_data[a]["wait"] for a in sorted_algos],
                marker_color=[ALGO_META[a]["color"] for a in sorted_algos],
                text=[f"{compare_data[a]['wait']}" for a in sorted_algos],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=12),
            ))
            fig.update_layout(
                title=dict(text="Mean Waiting Time — All Algorithms (lower is better)",
                            font_color="#e2e8f0", font_size=14),
                yaxis_title="Mean Waiting Time (s)", showlegend=False, **CHART,
            )
            fig.update_xaxes(tickfont=dict(size=13, color="#e2e8f0"))
            st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — MARL
# ════════════════════════════════════════════════════════════════════════════
with tab_marl:
    st.markdown("### Multi-Agent RL — 2×2 Intersection Grid (IPPO Baseline)")
    st.markdown("Independent PPO (IPPO) — one agent per traffic light junction")
    st.markdown("---")

    marl_data = {aid: load_marl(aid) for aid in MARL_META if load_marl(aid) is not None}

    if not marl_data:
        st.info("No MARL training data yet. Run `uv run python train_marl.py` to start.")
    else:
        cols = st.columns(4)
        for i, (aid, meta) in enumerate(MARL_META.items()):
            df = marl_data.get(aid)
            with cols[i]:
                if df is not None:
                    s = final_stats(df)
                    st.metric(meta["label"], f"{s['wait']} s wait",
                              delta=f"ep {s['episodes']}", delta_color="off")
                else:
                    st.metric(meta["label"], "No data", delta="pending", delta_color="off")

        st.markdown("---")

        fig_r = line_chart("Agent Reward per Episode", "Reward")
        fig_w = line_chart("Agent Mean Waiting Time per Episode", "Waiting Time (s)")

        for aid, meta in MARL_META.items():
            df = marl_data.get(aid)
            if df is not None:
                add_trace(fig_r, df, "episode", "reward", meta["label"], meta["color"], smoothing)
                add_trace(fig_w, df, "episode", "waiting_time", meta["label"], meta["color"], smoothing)

        st.plotly_chart(fig_r, width="stretch")
        st.plotly_chart(fig_w, width="stretch")

        st.markdown("---")
        st.markdown("#### MARL vs Baseline")

        fixed_df   = load_single("fixed")
        fixed_wait = final_stats(fixed_df)["wait"] if fixed_df is not None else None
        marl_wait  = float(np.mean([final_stats(df)["wait"] for df in marl_data.values()]))

        compare_labels = ["Fixed-Cycle", "MARL IPPO"]
        compare_vals   = [fixed_wait or 0, marl_wait]
        compare_colors = ["#64748b", "#8b5cf6"]

        fig = go.Figure(go.Bar(
            x=compare_labels, y=compare_vals, marker_color=compare_colors,
            text=[f"{v}" for v in compare_vals], textposition="outside",
            textfont=dict(color="#94a3b8"),
        ))
        fig.update_layout(
            title=dict(text="Mean Waiting Time: MARL vs Baselines",
                        font_color="#e2e8f0", font_size=14),
            yaxis_title="Mean Waiting Time (s)", showlegend=False, **CHART,
        )
        st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — MARL AQPPO
# ════════════════════════════════════════════════════════════════════════════
with tab_marl_aqppo:
    st.markdown("### IAQPPO — Independent Adaptive Q-Augmented PPO (2×2 Grid)")
    st.markdown("---")

    marl_aqppo_data = {aid: load_marl_aqppo(aid)
                       for aid in MARL_AQPPO_META
                       if load_marl_aqppo(aid) is not None}

    if not marl_aqppo_data:
        st.info("IAQPPO not trained yet. Run:")
        st.code("uv run python train_marl_aqppo.py --steps 100000")
    else:
        # Agent metric cards
        cols = st.columns(4)
        for i, (aid, meta) in enumerate(MARL_AQPPO_META.items()):
            df = marl_aqppo_data.get(aid)
            with cols[i]:
                if df is not None:
                    s = final_stats(df)
                    st.metric(meta["label"], f"{s['wait']} s wait",
                              delta=f"ep {s['episodes']}", delta_color="off")
                else:
                    st.metric(meta["label"], "No data", delta="pending", delta_color="off")

        st.markdown("---")

        # Training curves per agent
        st.markdown("#### Per-Agent Training Curves")
        fig_r = line_chart("Agent Reward per Episode", "Reward")
        fig_w = line_chart("Agent Mean Waiting Time per Episode", "Waiting Time (s)")

        for aid, meta in MARL_AQPPO_META.items():
            df = marl_aqppo_data.get(aid)
            if df is not None:
                add_trace(fig_r, df, "episode", "reward",
                          meta["label"], meta["color"], smoothing, width=2)
                add_trace(fig_w, df, "episode", "waiting_time",
                          meta["label"], meta["color"], smoothing, width=2)

        st.plotly_chart(fig_r, width="stretch")
        st.plotly_chart(fig_w, width="stretch")

        # IAQPPO vs IPPO agent comparison
        marl_ippo_data = {aid: load_marl(aid)
                          for aid in MARL_META
                          if load_marl(aid) is not None}

        if marl_ippo_data:
            st.markdown("---")
            st.markdown("#### IAQPPO vs IPPO — Per-Agent Mean Wait")

            agent_labels, ippo_waits, iaqppo_waits = [], [], []
            for aid in ["1", "2", "5", "6"]:
                ippo_df   = marl_ippo_data.get(aid)
                iaqppo_df = marl_aqppo_data.get(aid)
                if ippo_df is not None and iaqppo_df is not None:
                    agent_labels.append(f"Agent {aid}")
                    ippo_waits.append(final_stats(ippo_df)["wait"])
                    iaqppo_waits.append(final_stats(iaqppo_df)["wait"])

            if agent_labels:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="IPPO", x=agent_labels, y=ippo_waits,
                    marker_color="#8b5cf6",
                    text=[f"{v}" for v in ippo_waits],
                    textposition="outside", textfont=dict(color="#94a3b8"),
                ))
                fig.add_trace(go.Bar(
                    name="IAQPPO ★", x=agent_labels, y=iaqppo_waits,
                    marker_color="#e879f9",
                    text=[f"{v}" for v in iaqppo_waits],
                    textposition="outside", textfont=dict(color="#94a3b8"),
                ))
                fig.update_layout(
                    title=dict(text="Per-Agent Mean Waiting Time: IAQPPO vs IPPO (lower is better)",
                               font_color="#e2e8f0", font_size=14),
                    yaxis_title="Mean Waiting Time (s)",
                    barmode="group", **CHART,
                )
                st.plotly_chart(fig, width="stretch")

        # Full MARL comparison
        st.markdown("---")
        st.markdown("#### Full Comparison — All MARL Approaches")

        if marl_ippo_data:
            ippo_mean_wait = float(np.mean([final_stats(df)["wait"]
                                            for df in marl_ippo_data.values()]))
        else:
            ippo_mean_wait = None

        iaqppo_mean_wait = float(np.mean([final_stats(df)["wait"]
                                           for df in marl_aqppo_data.values()]))

        entries = [
            ("MARL IPPO",     "#8b5cf6", ippo_mean_wait),
            ("MARL IAQPPO ★", "#c084fc", iaqppo_mean_wait),
        ]
        entries = [(lbl, col, val) for lbl, col, val in entries if val is not None]

        if entries:
            labels_all  = [e[0] for e in entries]
            colors_all  = [e[1] for e in entries]
            vals_all    = [e[2] for e in entries]

            fig = go.Figure(go.Bar(
                x=labels_all, y=vals_all, marker_color=colors_all,
                text=[f"{v}" for v in vals_all],
                textposition="outside", textfont=dict(color="#94a3b8", size=12),
                hovertemplate="<b>%{x}</b><br>Mean Wait: %{y} s<extra></extra>",
            ))
            fig.update_layout(
                title=dict(text="Mean Waiting Time — MARL Comparison (lower is better)",
                           font_color="#e2e8f0", font_size=14),
                yaxis_title="Mean Waiting Time (s)", showlegend=False, **CHART,
            )
            fig.update_xaxes(tickfont=dict(size=12, color="#e2e8f0"))
            st.plotly_chart(fig, width="stretch")


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — SIMULATION VIDEO
# ════════════════════════════════════════════════════════════════════════════
_VIDEO_PLACEHOLDER = """
<div style='background:#1a1d27;border:1px solid #2d3450;border-radius:10px;
     padding:3.5rem 2rem;text-align:center;'>
    <div style='display:inline-flex;align-items:center;justify-content:center;
         width:56px;height:56px;background:#0f1117;border:1px solid #2d3450;
         border-radius:50%;margin-bottom:1.25rem;'>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
             stroke="#3b82f6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
    </div>
    <p style='color:#94a3b8;font-size:0.95rem;font-weight:600;margin:0 0 0.4rem;'>{title}</p>
    <p style='color:#475569;font-size:0.8rem;margin:0 0 1.2rem;'>{desc}</p>
    <code style='background:#0f1117;border:1px solid #2d3450;border-radius:5px;
          padding:0.3rem 0.75rem;color:#60a5fa;font-size:0.78rem;'>{cmd}</code>
</div>
"""

with tab_video:
    st.markdown("### Simulation — Policy Comparison")
    st.markdown("Side-by-side SUMO recordings comparing an untrained random policy against the trained PPO agent.")
    st.markdown("---")

    v_random = VIDEOS_DIR / "single_random.mp4"
    v_ppo    = VIDEOS_DIR / "single_ppo.mp4"

    st.markdown("#### Three-Way Policy Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<p style='color:#64748b;font-size:0.75rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.75rem;'>"
                    "Baseline — Random / Fixed-Cycle</p>", unsafe_allow_html=True)
        if v_random.exists():
            st.video(str(v_random))
        else:
            st.markdown(_VIDEO_PLACEHOLDER.format(
                title="No recording yet",
                desc="Random policy — high congestion, long queues",
                cmd="uv run python record.py --mode random",
            ), unsafe_allow_html=True)

    with col2:
        st.markdown("<p style='color:#10b981;font-size:0.75rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.75rem;'>"
                    "PPO Agent</p>", unsafe_allow_html=True)
        if v_ppo.exists():
            st.video(str(v_ppo))
        else:
            st.markdown(_VIDEO_PLACEHOLDER.format(
                title="No recording yet",
                desc="PPO agent — adaptive signal timing, reduced wait",
                cmd="uv run python record.py --mode ppo",
            ), unsafe_allow_html=True)

    v_aqppo = VIDEOS_DIR / "single_aqppo.mp4"
    with col3:
        st.markdown("<p style='color:#e879f9;font-size:0.75rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.75rem;'>"
                    "AQPPO ★</p>", unsafe_allow_html=True)
        if v_aqppo.exists():
            st.video(str(v_aqppo))
        else:
            st.markdown(_VIDEO_PLACEHOLDER.format(
                title="No recording yet",
                desc="AQPPO — Q-augmented PPO with adaptive entropy and congestion shaping",
                cmd="uv run python record.py --mode aqppo",
            ), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 2×2 Grid — MARL Comparison")
    col_ippo, col_iaqppo = st.columns(2)

    v_marl = VIDEOS_DIR / "marl_ippo.mp4"
    with col_ippo:
        st.markdown("<p style='color:#8b5cf6;font-size:0.75rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.75rem;'>"
                    "MARL IPPO — 4 Agents</p>", unsafe_allow_html=True)
        if v_marl.exists():
            st.video(str(v_marl))
        else:
            st.markdown(_VIDEO_PLACEHOLDER.format(
                title="No recording yet",
                desc="4 independent PPO agents coordinating across a 2×2 intersection grid",
                cmd="uv run python record.py --mode marl_ippo",
            ), unsafe_allow_html=True)

    v_marl_aqppo = VIDEOS_DIR / "marl_iaqppo.mp4"
    with col_iaqppo:
        st.markdown("<p style='color:#e879f9;font-size:0.75rem;font-weight:600;"
                    "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.75rem;'>"
                    "MARL IAQPPO ★</p>", unsafe_allow_html=True)
        if v_marl_aqppo.exists():
            st.video(str(v_marl_aqppo))
        else:
            st.markdown(_VIDEO_PLACEHOLDER.format(
                title="No recording yet",
                desc="4 AQPPO agents — Q-memory, adaptive entropy, congestion shaping per junction",
                cmd="uv run python record.py --mode marl_iaqppo",
            ), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#475569;font-size:0.82rem;'>"
                "Videos recorded with SUMO GUI — dark theme. "
                "Vehicle colour encodes speed: "
                "<span style='color:#008000;font-weight:600;'>green = fast</span>, "
                "<span style='color:#ef4444;font-weight:600;'>red = stopped</span>."
                "</p>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — REAL WORLD JIIT
# ════════════════════════════════════════════════════════════════════════════
JIIT_DIR = BASE / "jiit-intersection"

with tab_realworld:
    st.markdown("### Real-World Simulation — JIIT Noida Intersection")
    st.markdown("---")

    col_info, col_map = st.columns([1, 1])

    with col_info:
        st.markdown("""
        <div style='background:#1a1d27;border:1px solid #2d3450;border-radius:10px;padding:1.5rem;'>
        <p style='color:#64748b;font-size:0.7rem;font-weight:600;text-transform:uppercase;
           letter-spacing:0.08em;margin:0 0 0.75rem;'>Network Details</p>
        <table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>
        <tr><td style='color:#64748b;padding:0.3rem 0;'>Source</td>
            <td style='color:#e2e8f0;text-align:right;'>OpenStreetMap</td></tr>
        <tr><td style='color:#64748b;padding:0.3rem 0;'>Location</td>
            <td style='color:#e2e8f0;text-align:right;'>Sector 62, Noida</td></tr>
        <tr><td style='color:#64748b;padding:0.3rem 0;'>Coordinates</td>
            <td style='color:#e2e8f0;text-align:right;'>28°31′50″N 77°21′44″E</td></tr>
        <tr><td style='color:#64748b;padding:0.3rem 0;'>Signal Phases</td>
            <td style='color:#e2e8f0;text-align:right;'>5 green phases</td></tr>
        <tr><td style='color:#64748b;padding:0.3rem 0;'>Vehicles/hour</td>
            <td style='color:#e2e8f0;text-align:right;'>~1800</td></tr>
        <tr><td style='color:#64748b;padding:0.3rem 0;'>Simulation time</td>
            <td style='color:#e2e8f0;text-align:right;'>3600 s</td></tr>
        </table>
        <p style='color:#475569;font-size:0.78rem;margin:1rem 0 0;'>
        Network generated via <code>netconvert</code> from OSM data.
        Map tiles from Carto Light (zoom 17).
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        net_ok   = (JIIT_DIR / "jiit.net.xml").exists()
        route_ok = (JIIT_DIR / "jiit.rou.xml").exists()
        view_ok  = (JIIT_DIR / "jiit_view.xml").exists()
        ready    = net_ok and route_ok and view_ok

        if ready:
            if st.button("Launch SUMO Simulation", type="primary", width="stretch"):
                subprocess.Popen(
                    ["sumo-gui", "-n", str(JIIT_DIR / "jiit.net.xml"),
                     "-r", str(JIIT_DIR / "jiit.rou.xml"),
                     "--gui-settings-file", str(JIIT_DIR / "jiit_view.xml"),
                     "--start", "--delay", "80"],
                    cwd=str(JIIT_DIR),
                )
                st.success("SUMO GUI launched — check your taskbar.")
        else:
            missing = []
            if not net_ok:   missing.append("jiit.net.xml")
            if not route_ok: missing.append("jiit.rou.xml")
            if not view_ok:   missing.append("jiit_view.xml")
            st.error(f"Missing files: {', '.join(missing)}")

    with col_map:
        tile = JIIT_DIR / "tile46851_27345.jpeg"
        if tile.exists():
            st.image(str(tile), caption="Carto Light map tile — intersection area", width="stretch")
        else:
            st.markdown(
                "<div style='background:#1a1d27;border:1px solid #2d3450;border-radius:10px;"
                "padding:4rem;text-align:center;color:#475569;'>Map tile preview unavailable</div>",
                unsafe_allow_html=True,
            )
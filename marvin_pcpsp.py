"""
Marvin Mine — Precedence Constrained Production Scheduling Problem (PCPSP)
===========================================================================

Mathematical formulation (from Espinoza et al., MineLib 2012):

    (PCPSP)  max   ∑_b ∑_d ∑_t  p̃_bdt · y_bdt
    subject to:
        (7)  ∑_{τ≤t} x_bτ  ≤  ∑_{τ≤t} x_{b'τ}     ∀b∈B, b'∈B_b, t∈T   [precedence]
        (8)  x_bt            =  ∑_{d∈D} y_bdt         ∀b∈B, t∈T           [flow conservation]
        (9)  ∑_{t∈T} x_bt   ≤  1                      ∀b∈B                 [extract at most once]
       (10)  R_rt  ≤  ∑_b ∑_d q̂_brd · y_bdt  ≤  R̄_rt  ∀r∈R, t∈T         [resource bounds]
       (11)  a  ≤  Ay  ≤  ā                                                  [side constraints]
       (12)  y_bdt ∈ [0,1]                            ∀b∈B, d∈D, t∈T
       (13)  x_bt  ∈ {0,1}                            ∀b∈B, t∈T

Variables:
    x_bt  = 1 if block b is extracted in period t, 0 otherwise
    y_bdt = fraction of block b sent to destination d in period t

Parameters:
    p̃_bdt = p̌_bd / (1+α)^t        discounted profit (α = discount rate)
    D = {0=waste, 1=process}        destinations
    R_rt, R̄_rt                      min/max resource r in period t
    q̂_brd                           tonnes of resource r when block b → dest d

Marvin instance (MineLib):
    Blocks:       53,271
    Periods:       20
    Destinations:   2  (waste, mill)
    Resources:      2  (extraction ≤ 60 Mt/period, processing ≤ 20 Mt/period)
    Discount rate:  10%
    Known best:    $885,968,070  (LP GAP 2.8%, TopoSort heuristic by G. Muñoz)

Usage:
    python marvin_pcpsp.py --csv Marvin.csv --pcpsp marvin.pcpsp --prec Marvin_Prece.xlsx

Dependencies:
    pip install pandas numpy plotly openpyxl
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Install with: pip install plotly")

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("Warning: openpyxl not installed. Install with: pip install openpyxl")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_block_model(csv_path: str) -> pd.DataFrame:
    """Load block model from CSV. Expected columns: IX, IY, IZ, CU, AU,
    density, economic value process, economic value waste."""
    df = pd.read_csv(csv_path)
    df.index = range(len(df))
    print(f"  Loaded {len(df):,} blocks  "
          f"(IX: {df['IX'].min()}–{df['IX'].max()}, "
          f"IY: {df['IY'].min()}–{df['IY'].max()}, "
          f"IZ: {df['IZ'].min()}–{df['IZ'].max()})")
    return df


def load_precedences(prec_path: str, n_blocks: int) -> tuple[dict, dict]:
    """Load precedence relationships from Excel.
    Columns: BlockID, N_prede, prede1, prede2, ...
    Returns (predecessors, successors) dicts."""
    if not HAS_OPENPYXL:
        raise ImportError("openpyxl required to read precedence xlsx")

    wb = openpyxl.load_workbook(prec_path, read_only=True)
    ws = wb.active

    predecessors: dict[int, list[int]] = {}
    successors: dict[int, list[int]] = {b: [] for b in range(n_blocks)}

    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        b = int(row[0])
        n_pred = int(row[1]) if row[1] else 0
        preds = [int(row[2 + i]) for i in range(n_pred) if row[2 + i] is not None]
        predecessors[b] = preds
        for p in preds:
            if 0 <= p < n_blocks:
                successors[p].append(b)

    print(f"  Loaded precedences for {len(predecessors):,} blocks")
    return predecessors, successors


def load_pcpsp_file(pcpsp_path: str) -> tuple[dict, dict, dict, dict]:
    """Parse MineLib .pcpsp file.

    Returns:
        header: dict of NAME, TYPE, NBLOCKS, NPERIODS, ...
        obj_coeffs: {block: [waste_profit, process_profit]}  (undiscounted)
        res_coeffs: {(block, dest, resource): tonnes}
        res_limits: {(resource, period): (lower, upper)}
    """
    header = {}
    obj_coeffs: dict[int, list[float]] = {}
    res_coeffs: dict[tuple, float] = {}
    res_limits: dict[tuple, tuple] = {}

    section = None
    with open(pcpsp_path) as f:
        for line in f:
            line = line.strip()
            if not line or line == "EOF" or line.startswith("%"):
                continue

            if line == "OBJECTIVE_FUNCTION:":
                section = "obj"
                continue
            elif line == "RESOURCE_CONSTRAINT_COEFFICIENTS:":
                section = "res"
                continue
            elif line == "RESOURCE_CONSTRAINT_LIMITS:":
                section = "lim"
                continue
            elif ":" in line and not line[0].isdigit():
                # Header line
                key, _, val = line.partition(":")
                header[key.strip()] = val.strip()
                section = None
                continue

            parts = line.split()
            if section == "obj":
                b = int(parts[0])
                obj_coeffs[b] = [float(parts[1]), float(parts[2])]
            elif section == "res":
                b, d, r, v = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
                res_coeffs[(b, d, r)] = v
            elif section == "lim":
                r, t, constraint_type = int(parts[0]), int(parts[1]), parts[2]
                if constraint_type == "L":
                    res_limits[(r, t)] = (0.0, float(parts[3]))
                elif constraint_type == "G":
                    res_limits[(r, t)] = (float(parts[3]), float("inf"))
                elif constraint_type == "I":
                    res_limits[(r, t)] = (float(parts[3]), float(parts[4]))

    print(f"  Parsed {len(obj_coeffs):,} objective coefficients, "
          f"{len(res_coeffs):,} resource coefficients")
    return header, obj_coeffs, res_coeffs, res_limits


# ─────────────────────────────────────────────────────────────────────────────
# PCPSP Solver (TopoSort Greedy Heuristic)
# ─────────────────────────────────────────────────────────────────────────────

def solve_pcpsp_greedy(
    n_blocks: int,
    predecessors: dict,
    successors: dict,
    obj_coeffs: dict,
    res_coeffs: dict,
    n_periods: int = 20,
    discount: float = 0.10,
    r_max: dict | None = None,
    res_limits: dict | None = None,
) -> dict:
    """
    TopoSort greedy heuristic for PCPSP.

    Strategy:
    - Maintain a priority queue of "available" blocks (all predecessors extracted or
      extracted earlier in the same period).
    - In each period t, greedily schedule highest-discounted-value blocks until
      resource capacities are exhausted.
    - Respects cumulative precedence constraint (7): block b can be extracted
      in the same period as its predecessor b' (PCPSP uses cumulative sums).

    Reference: Modified TopoSort heuristic (Muñoz, 2012), as used to generate
    the best-known solution for Marvin PCPSP: $885,968,070.

    Parameters:
        r_max: {resource_id: capacity_per_period} or {period: {resource: cap}}
               default: {0: 60e6, 1: 20e6}
        res_limits: {(resource, period): (lower, upper)} parsed from .pcpsp.

    Returns dict with 'extracted', 'destination', 'total_npv', 'period_stats'.
    """
    # Resolve resource capacities by period.
    if res_limits is not None:
        cap_by_period: dict[int, dict[int, float]] = {}
        for (r, t), (_, hi) in res_limits.items():
            hi = 1e15 if hi == float("inf") else hi
            cap_by_period.setdefault(t, {})[r] = hi
    elif r_max is not None and all(isinstance(v, dict) for v in r_max.values()):
        cap_by_period = r_max
    else:
        if r_max is None:
            r_max = {0: 60_000_000.0, 1: 20_000_000.0}
        cap_by_period = {t: r_max for t in range(n_periods)}

    T = n_periods
    n_res = max((max(caps.keys()) for caps in cap_by_period.values()), default=-1) + 1

    # Precompute block tonnage for each (dest, resource)
    block_tons: list[dict] = []
    for b in range(n_blocks):
        dt = {}
        for d in range(2):
            dt[d] = {r: res_coeffs.get((b, d, r), 0.0) for r in range(n_res)}
        block_tons.append(dt)

    # Convert objective costs to profit if needed.
    # Marvin .pcpsp files provide negative values, so profit = -cost.
    profit_coeffs = [
        [-(obj_coeffs[b][0]), -(obj_coeffs[b][1])] for b in range(n_blocks)
    ]

    # Best destination per block (undiscounted profit)
    block_best_dest = [
        1 if profit_coeffs[b][1] >= profit_coeffs[b][0] else 0
        for b in range(n_blocks)
    ]

    # Initialise in-degree counter and available set
    in_degree = [len(predecessors.get(b, [])) for b in range(n_blocks)]
    available: set[int] = {b for b in range(n_blocks) if in_degree[b] == 0}

    extracted: dict[int, int] = {}    # block → period (0-indexed)
    destination: dict[int, int] = {}  # block → dest
    period_usage: list[dict] = [{r: 0.0 for r in range(n_res)} for _ in range(T)]
    total_npv = 0.0
    period_stats = []

    for t in range(T):
        disc = (1.0 + discount) ** (t + 1)
        caps = cap_by_period.get(t, {})

        # Sort available by discounted profit of best destination.
        def block_priority(b: int) -> float:
            d = block_best_dest[b]
            return profit_coeffs[b][d] / disc

        newly_extracted: list[int] = []
        while available:
            placed = False
            for b in sorted(available, key=block_priority, reverse=True):
                # Check extraction capacity first (resource 0 is always extraction)
                if period_usage[t][0] >= caps.get(0, float("inf")):
                    break

                for try_d in [block_best_dest[b], 1 - block_best_dest[b]]:
                    profit = profit_coeffs[b][try_d]
                    if profit <= 0:
                        continue

                    t_b = block_tons[b][try_d]
                    if all(period_usage[t][r] + t_b[r] <= caps.get(r, float("inf"))
                           for r in range(n_res)):
                        extracted[b] = t
                        destination[b] = try_d
                        for r in range(n_res):
                            period_usage[t][r] += t_b[r]
                        npv = profit / disc
                        total_npv += npv
                        newly_extracted.append(b)
                        available.remove(b)
                        for s in successors.get(b, []):
                            in_degree[s] -= 1
                            if in_degree[s] == 0:
                                available.add(s)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                break

        # Collect period statistics
        proc_b = [b for b in newly_extracted if destination[b] == 1]
        waste_b = [b for b in newly_extracted if destination[b] == 0]
        npv_t = sum(
            profit_coeffs[b][destination[b]] / disc for b in newly_extracted
        )
        period_stats.append({
            "period": t + 1,
            "blocks": len(newly_extracted),
            "process_blocks": len(proc_b),
            "waste_blocks": len(waste_b),
            "extraction_Mt": period_usage[t][0] / 1e6,
            "processing_Mt": period_usage[t].get(1, 0) / 1e6,
            "npv": npv_t,
            "cumulative_npv": 0.0,  # filled below
        })

    # Cumulative NPV
    cum = 0.0
    for ps in period_stats:
        cum += ps["npv"]
        ps["cumulative_npv"] = cum

    return {
        "extracted": extracted,
        "destination": destination,
        "total_npv": total_npv,
        "period_stats": period_stats,
        "n_scheduled": len(extracted),
        "n_blocks": n_blocks,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG = "rgb(8, 10, 22)"
PANEL_BG = "rgb(14, 16, 36)"
GRID_COLOR = "rgba(255,255,255,0.08)"
ACCENT = "#00d4ff"
GOLD = "#ffd700"
EMERALD = "#00e5a0"
CRIMSON = "#ff4560"

PERIOD_COLORSCALE = px.colors.sequential.Plasma if HAS_PLOTLY else None

LAYOUT_BASE = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=PANEL_BG,
    font=dict(family="'Share Tech Mono', 'Courier New', monospace", color="#c8d0e0"),
    margin=dict(l=10, r=10, t=80, b=10),
)


def _scene_axes() -> dict:
    axis = dict(
        backgroundcolor=DARK_BG,
        gridcolor=GRID_COLOR,
        showbackground=True,
        linecolor=GRID_COLOR,
        tickfont=dict(size=9),
    )
    return {
        "xaxis": dict(**axis, title="IX (Easting)"),
        "yaxis": dict(**axis, title="IY (Northing)"),
        "zaxis": dict(**axis, title="IZ (Depth level)"),
        "aspectmode": "data",
    }


def save_html(fig, path: str) -> None:
    fig.write_html(path, include_plotlyjs="cdn", auto_open=True)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – 3-D pit coloured by extraction period
# ─────────────────────────────────────────────────────────────────────────────

def plot_pit_by_period(df: pd.DataFrame, result: dict, filename: str) -> None:
    """3-D scatter coloured by mining period (1–20). Unmined blocks shown faintly."""
    if not HAS_PLOTLY:
        print("plotly not available — skipping plot")
        return

    extracted = result["extracted"]
    destination = result["destination"]
    T = max(result["period_stats"], key=lambda p: p["period"])["period"] if result["period_stats"] else 20

    # Assign per-block metadata
    df = df.copy()
    df["period"] = df.index.map(lambda b: extracted.get(b, -1) + 1)  # 1-indexed; 0 = unmined
    df["dest"] = df.index.map(lambda b: destination.get(b, -1))
    df["dest_label"] = df["dest"].map({1: "Process", 0: "Waste", -1: "Not mined"})

    mined = df[df["period"] > 0]
    unmined = df[df["period"] == 0]

    fig = go.Figure()

    # Unmined blocks (ghost)
    if not unmined.empty:
        fig.add_trace(go.Scatter3d(
            x=unmined["IX"], y=unmined["IY"], z=unmined["IZ"],
            mode="markers",
            marker=dict(size=1.5, color="rgba(80,90,120,0.15)"),
            hoverinfo="skip",
            name="Not mined",
            showlegend=True,
        ))

    # Mined blocks coloured by period
    fig.add_trace(go.Scatter3d(
        x=mined["IX"], y=mined["IY"], z=mined["IZ"],
        mode="markers",
        marker=dict(
            size=3.5,
            color=mined["period"],
            colorscale="Plasma",
            cmin=1, cmax=T,
            opacity=0.85,
            colorbar=dict(
                title="Period",
                tickvals=list(range(1, T + 1, 2)),
                thickness=12,
                len=0.6,
                x=1.02,
            ),
        ),
        text=mined["dest_label"],
        customdata=mined[["CU", "AU", "density",
                           "economic value process", "economic value waste"]].values,
        hovertemplate=(
            "<b>Block</b> IX=%{x} IY=%{y} IZ=%{z}<br>"
            "Period: %{marker.color}<br>"
            "Destination: %{text}<br>"
            "CU: %{customdata[0]:.3f}% | AU: %{customdata[1]:.3f} g/t<br>"
            "Density: %{customdata[2]:.3f} t/m³<br>"
            "Value (process): $%{customdata[3]:,.0f}<br>"
            "Value (waste):   $%{customdata[4]:,.0f}"
            "<extra></extra>"
        ),
        name="Mined blocks",
    ))

    npv = result["total_npv"]
    n_s = result["n_scheduled"]
    n_b = result["n_blocks"]

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=(
                f"Marvin PCPSP — 3-D Pit (coloured by period)<br>"
                f"<sup>Heuristic NPV: ${npv:,.0f} | "
                f"Known best: $885,968,070 | "
                f"Blocks mined: {n_s:,}/{n_b:,}</sup>"
            ),
            font=dict(size=15, color=ACCENT),
        ),
        scene=_scene_axes(),
        height=820,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.4)"),
    )

    save_html(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – 3-D pit coloured by destination (process vs waste)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pit_by_destination(df: pd.DataFrame, result: dict, filename: str) -> None:
    if not HAS_PLOTLY:
        return

    extracted = result["extracted"]
    destination = result["destination"]

    df = df.copy()
    df["period"] = df.index.map(lambda b: extracted.get(b, -1))
    df["dest"] = df.index.map(lambda b: destination.get(b, -1))

    process_df = df[(df["period"] >= 0) & (df["dest"] == 1)]
    waste_df = df[(df["period"] >= 0) & (df["dest"] == 0)]
    unmined_df = df[df["period"] < 0]

    fig = go.Figure()

    if not unmined_df.empty:
        fig.add_trace(go.Scatter3d(
            x=unmined_df["IX"], y=unmined_df["IY"], z=unmined_df["IZ"],
            mode="markers",
            marker=dict(size=1.2, color="rgba(60,70,100,0.12)"),
            hoverinfo="skip", name="Not mined",
        ))

    if not waste_df.empty:
        fig.add_trace(go.Scatter3d(
            x=waste_df["IX"], y=waste_df["IY"], z=waste_df["IZ"],
            mode="markers",
            marker=dict(size=2.5, color=CRIMSON, opacity=0.5),
            customdata=waste_df[["CU", "AU", "economic value waste",
                                  "period"]].assign(p=waste_df["period"] + 1).values,
            hovertemplate=(
                "<b>WASTE block</b><br>"
                "IX=%{x} IY=%{y} IZ=%{z}<br>"
                "Period: %{customdata[3]:.0f}<br>"
                "CU: %{customdata[0]:.3f}% AU: %{customdata[1]:.3f} g/t<br>"
                "Value: $%{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Waste dump",
        ))

    if not process_df.empty:
        fig.add_trace(go.Scatter3d(
            x=process_df["IX"], y=process_df["IY"], z=process_df["IZ"],
            mode="markers",
            marker=dict(
                size=4,
                color=process_df["economic value process"],
                colorscale="Viridis",
                opacity=0.9,
                colorbar=dict(title="Process value ($)", thickness=12, len=0.55, x=1.02),
            ),
            customdata=process_df[["CU", "AU", "economic value process",
                                    "period"]].assign(p=process_df["period"] + 1).values,
            hovertemplate=(
                "<b>PROCESS block</b><br>"
                "IX=%{x} IY=%{y} IZ=%{z}<br>"
                "Period: %{customdata[3]:.0f}<br>"
                "CU: %{customdata[0]:.3f}% AU: %{customdata[1]:.3f} g/t<br>"
                "Value: $%{customdata[2]:,.0f}<extra></extra>"
            ),
            name="Process (mill)",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=(
                "Marvin PCPSP — Extraction by Destination<br>"
                f"<sup>Process blocks: {len(process_df):,} | "
                f"Waste blocks: {len(waste_df):,} | "
                f"Not mined: {len(unmined_df):,}</sup>"
            ),
            font=dict(size=15, color=EMERALD),
        ),
        scene=_scene_axes(),
        height=820,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.4)"),
    )

    save_html(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – Period-by-period production dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_production_dashboard(result: dict, filename: str) -> None:
    if not HAS_PLOTLY:
        return

    ps = result["period_stats"]
    periods = [p["period"] for p in ps]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Blocks mined per period",
            "Resource usage (Mt/period)",
            "NPV contribution per period ($M)",
            "Cumulative NPV ($M)",
        ],
        vertical_spacing=0.16,
        horizontal_spacing=0.10,
    )

    # ── Blocks mined (stacked: process + waste)
    fig.add_trace(go.Bar(
        x=periods,
        y=[p["process_blocks"] for p in ps],
        name="Process",
        marker_color=EMERALD,
        opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=periods,
        y=[p["waste_blocks"] for p in ps],
        name="Waste",
        marker_color=CRIMSON,
        opacity=0.6,
    ), row=1, col=1)

    # ── Resource usage
    fig.add_trace(go.Bar(
        x=periods,
        y=[p["extraction_Mt"] for p in ps],
        name="Extraction (r=0)",
        marker_color=ACCENT,
        opacity=0.7,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=periods,
        y=[p["processing_Mt"] for p in ps],
        name="Processing (r=1)",
        marker_color=GOLD,
        opacity=0.8,
    ), row=1, col=2)
    # Capacity lines
    fig.add_hline(y=60, row=1, col=2, line=dict(color=ACCENT, dash="dot", width=1),
                  annotation_text="Extraction cap 60 Mt")
    fig.add_hline(y=20, row=1, col=2, line=dict(color=GOLD, dash="dot", width=1),
                  annotation_text="Processing cap 20 Mt")

    # ── NPV per period
    npv_m = [p["npv"] / 1e6 for p in ps]
    colors = [EMERALD if v >= 0 else CRIMSON for v in npv_m]
    fig.add_trace(go.Bar(
        x=periods, y=npv_m,
        name="Period NPV",
        marker_color=colors,
        opacity=0.85,
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line=dict(color="white", dash="dash", width=0.8))

    # ── Cumulative NPV
    cum_m = [p["cumulative_npv"] / 1e6 for p in ps]
    fig.add_trace(go.Scatter(
        x=periods, y=cum_m,
        mode="lines+markers",
        name="Cumulative NPV",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=6, color=GOLD),
        showlegend=False,
    ), row=2, col=2)
    # Known best reference
    fig.add_hline(y=885.968, row=2, col=2,
                  line=dict(color="rgba(255,255,255,0.4)", dash="dot", width=1.5),
                  annotation_text="Known best $885.9 M",
                  annotation_font_color="rgba(255,255,255,0.6)")

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=(
                "Marvin PCPSP — Production Schedule Dashboard<br>"
                f"<sup>20 periods | α=10% | Extraction ≤ 60 Mt | Processing ≤ 20 Mt</sup>"
            ),
            font=dict(size=14, color=ACCENT),
        ),
        height=700,
        barmode="stack",
        legend=dict(x=0.01, y=1.05, orientation="h", bgcolor="rgba(0,0,0,0.4)"),
    )

    for annotation in fig.layout.annotations:
        annotation.font.color = "#8a9bbf"
        annotation.font.size = 11

    for axis_name in dir(fig.layout):
        if axis_name.startswith(("xaxis", "yaxis")):
            axis = getattr(fig.layout, axis_name)
            if hasattr(axis, "gridcolor"):
                axis.gridcolor = GRID_COLOR
                axis.linecolor = GRID_COLOR
                axis.tickfont = dict(size=9)

    save_html(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 – 3-D blocks coloured by economic value (process)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pit_by_value(df: pd.DataFrame, result: dict, filename: str) -> None:
    if not HAS_PLOTLY:
        return

    extracted = result["extracted"]
    df = df.copy()
    df["mined"] = df.index.map(lambda b: b in extracted)
    df["period"] = df.index.map(lambda b: extracted.get(b, -1) + 1)

    mined = df[df["mined"]]
    unmined = df[~df["mined"]]

    fig = go.Figure()

    if not unmined.empty:
        fig.add_trace(go.Scatter3d(
            x=unmined["IX"], y=unmined["IY"], z=unmined["IZ"],
            mode="markers",
            marker=dict(size=1.2, color="rgba(40,50,80,0.12)"),
            hoverinfo="skip", name="Not mined",
        ))

    if not mined.empty:
        fig.add_trace(go.Scatter3d(
            x=mined["IX"], y=mined["IY"], z=mined["IZ"],
            mode="markers",
            marker=dict(
                size=3.8,
                color=mined["economic value process"],
                colorscale="RdYlGn",
                cmid=0,
                opacity=0.88,
                colorbar=dict(
                    title="Process value ($)",
                    thickness=12, len=0.65, x=1.02,
                ),
            ),
            customdata=mined[["CU", "AU", "density",
                               "economic value process",
                               "economic value waste", "period"]].values,
            hovertemplate=(
                "<b>Block</b> IX=%{x} IY=%{y} IZ=%{z}<br>"
                "Period: %{customdata[5]:.0f}<br>"
                "CU: %{customdata[0]:.3f}%  AU: %{customdata[1]:.3f} g/t<br>"
                "Density: %{customdata[2]:.3f} t/m³<br>"
                "Value (process): $%{customdata[3]:,.0f}<br>"
                "Value (waste):   $%{customdata[4]:,.0f}"
                "<extra></extra>"
            ),
            name="Mined",
        ))

    pos = mined[mined["economic value process"] > 0]
    neg = mined[mined["economic value process"] <= 0]

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=(
                "Marvin PCPSP — Blocks Coloured by Process Value<br>"
                f"<sup>Green = profitable ore | Red = waste | "
                f"Ore blocks: {len(pos):,} | Waste blocks: {len(neg):,}</sup>"
            ),
            font=dict(size=14, color=GOLD),
        ),
        scene=_scene_axes(),
        height=820,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.4)"),
    )

    save_html(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – Cross-section (level slice)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_section(df: pd.DataFrame, result: dict,
                       iz_level: int = 9, filename: str = "cross_section.html") -> None:
    """2-D cross-section at a given IZ level coloured by ore grade (CU)."""
    if not HAS_PLOTLY:
        return

    extracted = result["extracted"]
    destination = result["destination"]

    df = df.copy()
    df["period"] = df.index.map(lambda b: extracted.get(b, -1) + 1)
    df["dest"] = df.index.map(lambda b: destination.get(b, -1))

    level = df[df["IZ"] == iz_level].copy()
    if level.empty:
        print(f"  No blocks at IZ={iz_level}")
        return

    level["dest_label"] = level["dest"].map({1: "Process", 0: "Waste", -1: "Not mined"})
    level["marker_size"] = level["dest"].map({1: 14, 0: 8, -1: 4})

    fig = go.Figure()

    for dest, label, color, size in [
        (-1, "Not mined", "rgba(60,70,120,0.3)", 5),
        (0,  "Waste",     CRIMSON,               8),
        (1,  "Process",   EMERALD,               12),
    ]:
        sub = level[level["dest"] == dest]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["IX"], y=sub["IY"],
            mode="markers",
            marker=dict(
                size=size,
                color=sub["CU"] if dest == 1 else color,
                colorscale="YlOrRd" if dest == 1 else None,
                showscale=(dest == 1),
                colorbar=dict(title="CU (%)", thickness=12, x=1.02) if dest == 1 else None,
                opacity=0.9 if dest >= 0 else 0.3,
                symbol="square",
            ),
            customdata=sub[["CU", "AU", "period", "economic value process"]].values,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "IX=%{x}  IY=%{y}<br>"
                "CU: %{customdata[0]:.3f}%  AU: %{customdata[1]:.3f} g/t<br>"
                "Period: %{customdata[2]:.0f}<br>"
                "Value (process): $%{customdata[3]:,.0f}"
                "<extra></extra>"
            ),
            name=label,
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=(
                f"Marvin PCPSP — Cross-section at IZ = {iz_level}<br>"
                f"<sup>Coloured by Cu grade for process blocks</sup>"
            ),
            font=dict(size=14, color=GOLD),
        ),
        xaxis=dict(title="IX (Easting)", gridcolor=GRID_COLOR),
        yaxis=dict(title="IY (Northing)", gridcolor=GRID_COLOR, scaleanchor="x"),
        height=620,
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(0,0,0,0.4)"),
    )

    save_html(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 – Pit surface (max extracted depth per XY cell)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pit_surface(df: pd.DataFrame, result: dict, filename: str) -> None:
    """3-D surface showing the deepest mined block per XY cell."""
    if not HAS_PLOTLY:
        return

    extracted = result["extracted"]
    mined = df[df.index.map(lambda b: b in extracted)].copy()

    if mined.empty:
        return

    ix_vals = sorted(mined["IX"].unique())
    iy_vals = sorted(mined["IY"].unique())
    ix_idx = {v: i for i, v in enumerate(ix_vals)}
    iy_idx = {v: i for i, v in enumerate(iy_vals)}

    surf = np.full((len(ix_vals), len(iy_vals)), np.nan)
    for _, row in mined.iterrows():
        xi, yi = ix_idx[row["IX"]], iy_idx[row["IY"]]
        # deepest = minimum IZ (IZ increases downward in some conventions, 
        # but for Marvin IZ=1 is bottom, IZ=17 is top — so max IZ = surface)
        if np.isnan(surf[xi, yi]) or row["IZ"] > surf[xi, yi]:
            surf[xi, yi] = row["IZ"]

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=ix_vals,
        y=iy_vals,
        z=surf.T,
        colorscale="Turbo",
        opacity=0.82,
        colorbar=dict(title="Max IZ mined", thickness=12),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ),
        hovertemplate="IX=%{x}  IY=%{y}<br>Deepest level: %{z}<extra></extra>",
        name="Pit surface",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=(
                "Marvin PCPSP — Pit Envelope Surface<br>"
                "<sup>Maximum extraction depth per XY cell</sup>"
            ),
            font=dict(size=14, color=ACCENT),
        ),
        scene=dict(
            **_scene_axes(),
            zaxis_title="IZ (Depth level)",
        ),
        height=760,
    )

    save_html(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Main PCPSP class
# ─────────────────────────────────────────────────────────────────────────────

class MarvinPCPSP:
    """
    Encapsulates the Marvin PCPSP instance:
    - Data loading (block model, precedences, .pcpsp parameters)
    - Greedy heuristic solver
    - Plotly-based visualisations
    """

    def __init__(
        self,
        csv_path: str,
        pcpsp_path: str,
        prec_path: str,
        output_dir: str = ".",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n══ Marvin PCPSP ══════════════════════════════════════════")
        print("Loading block model …")
        self.df = load_block_model(csv_path)
        self.n_blocks = len(self.df)

        print("Loading precedences …")
        self.predecessors, self.successors = load_precedences(prec_path, self.n_blocks)

        print("Parsing PCPSP parameters …")
        self.header, self.obj_coeffs, self.res_coeffs, self.res_limits = \
            load_pcpsp_file(pcpsp_path)

        self.n_periods = int(self.header.get("NPERIODS", 20))
        self.discount = float(self.header.get("DISCOUNT_RATE", 0.10))
        self.result: dict | None = None

    def _derive_resource_caps(self) -> dict[int, dict[int, float]]:
        """Derive per-period upper bounds from PCPSP resource limits."""
        caps: dict[int, dict[int, float]] = {}
        for (r, t), (_, hi) in self.res_limits.items():
            hi = 1e15 if hi == float("inf") else hi
            caps.setdefault(t, {})[r] = hi
        return caps

    def solve(self) -> dict:
        """Run the TopoSort greedy heuristic and return the result dict."""
        print("\nSolving PCPSP (TopoSort greedy heuristic) …")
        t0 = time.time()
        r_max = self._derive_resource_caps() if self.res_limits else None
        self.result = solve_pcpsp_greedy(
            n_blocks=self.n_blocks,
            predecessors=self.predecessors,
            successors=self.successors,
            obj_coeffs=self.obj_coeffs,
            res_coeffs=self.res_coeffs,
            n_periods=self.n_periods,
            discount=self.discount,
            r_max=r_max,
            res_limits=self.res_limits if self.res_limits else None,
        )
        elapsed = time.time() - t0

        r = self.result
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Blocks scheduled : {r['n_scheduled']:,} / {r['n_blocks']:,}")
        print(f"  Heuristic NPV    : ${r['total_npv']:,.0f}")
        print(f"  Known best NPV   : $885,968,070  (LP gap 2.8%)")

        print("\n  Period-by-period summary:")
        print(f"  {'Per':>4}  {'Blocks':>7}  {'Process':>8}  "
              f"{'Waste':>8}  {'Extr Mt':>8}  {'Proc Mt':>8}  {'NPV $M':>10}")
        for ps in r["period_stats"]:
            print(f"  {ps['period']:>4}  {ps['blocks']:>7,}  "
                  f"{ps['process_blocks']:>8,}  {ps['waste_blocks']:>8,}  "
                  f"{ps['extraction_Mt']:>8.1f}  {ps['processing_Mt']:>8.1f}  "
                  f"{ps['npv']/1e6:>10.2f}")

        return self.result

    def save_schedule(self, path: str | None = None) -> None:
        """Save result dict to JSON."""
        if self.result is None:
            raise RuntimeError("Run solve() first.")
        out = path or str(self.output_dir / "marvin_pcpsp_schedule.json")
        with open(out, "w") as f:
            json.dump({
                "extracted": {str(k): v for k, v in self.result["extracted"].items()},
                "destination": {str(k): v for k, v in self.result["destination"].items()},
                "total_npv": self.result["total_npv"],
                "period_stats": self.result["period_stats"],
            }, f, indent=2)
        print(f"  Schedule saved → {out}")

    def plot_all(self) -> None:
        """Generate all visualisations."""
        if self.result is None:
            raise RuntimeError("Run solve() first.")

        od = self.output_dir
        print("\nGenerating visualisations …")

        plot_pit_by_period(self.df, self.result,
                           str(od / "01_pit_by_period.html"))
        plot_pit_by_destination(self.df, self.result,
                                str(od / "02_pit_by_destination.html"))
        plot_production_dashboard(self.result,
                                  str(od / "03_production_dashboard.html"))
        plot_pit_by_value(self.df, self.result,
                          str(od / "04_pit_by_value.html"))
        plot_cross_section(self.df, self.result, iz_level=9,
                           filename=str(od / "05_cross_section_iz9.html"))
        plot_pit_surface(self.df, self.result,
                         str(od / "06_pit_surface.html"))

        print("\n══ All done ══════════════════════════════════════════════")
        print(f"HTML files written to: {self.output_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Marvin Mine PCPSP solver + 3-D visualisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",   default="Marvin.csv",
                        help="Block model CSV  (default: Marvin.csv)")
    parser.add_argument("--pcpsp", default="marvin.pcpsp",
                        help="MineLib .pcpsp file  (default: marvin.pcpsp)")
    parser.add_argument("--prec",  default="Marvin_Prece.xlsx",
                        help="Precedence xlsx  (default: Marvin_Prece.xlsx)")
    parser.add_argument("--out",   default="marvin_output",
                        help="Output directory  (default: marvin_output)")
    args = parser.parse_args()

    model = MarvinPCPSP(
        csv_path=args.csv,
        pcpsp_path=args.pcpsp,
        prec_path=args.prec,
        output_dir=args.out,
    )
    model.solve()
    model.save_schedule()
    model.plot_all()


if __name__ == "__main__":
    main()

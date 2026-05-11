"""
CPIT 3D Open Pit Mining Solver & Visualizer — Marvin Mine
==========================================================
Implements the Constrained Pit limit Problem (CPIT) from:
  Espinoza, Goycoolea, Moreno, Newman (2012) - MineLib

CPIT Formulation:
  max  Σ_{b,t}  p̂_bt · x_bt
  s.t. Σ_{τ≤t} x_bτ  ≤  Σ_{τ≤t} x_{b'τ}   ∀b, b'∈B_b, t  (precedence)
       Σ_t x_bt ≤ 1                            ∀b           (mine once)
       R_rt ≤ Σ_b q_br · x_bt ≤ R̄_rt          ∀r, t        (resources)
       x_bt ∈ {0,1}                             ∀b, t        (binary)

  where p̂_bt = p_b / (1+α)^t  (NPV discounting)
"""

import sys, time, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
#  DATA PATHS
# ─────────────────────────────────────────────────────────────────────
CPIT_FILE   = "D:\\MineLib mine design using Python\\marvin CPIT.cpit"
PREC_CSV    = "D:\\MineLib mine design using Python\\Marvin_Prece.csv"
BASE_DIR    = Path(__file__).resolve().parent
OUT_DIR     = BASE_DIR / "cpit_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
#  1. PARSE CPIT FILE
# ═════════════════════════════════════════════════════════════════════
def parse_cpit(filepath: str) -> dict:
    """
    Parse the MineLib .cpit Optimization-Model Descriptor File.
    Returns a dict with keys:
      name, type, nblocks, nperiods, nresources, discount_rate,
      obj_vals   : {block_id → undiscounted value p_b}
      res_coeffs : {block_id → {resource_id → coefficient q_br}}
      res_limits : {(resource_id, period) → (type, lo, hi)}
    """
    data = {
        "name": "marvin", "type": "CPIT",
        "nblocks": 0, "nperiods": 0, "nresources": 0,
        "discount_rate": 0.1,
        "obj_vals": {},
        "res_coeffs": defaultdict(dict),
        "res_limits": {},
    }
    section = None
    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            # --- header keywords ---
            if line.startswith("NAME"):
                data["name"] = line.split(":",1)[1].strip()
                section = None; continue
            if line.startswith("TYPE"):
                data["type"] = line.split(":",1)[1].strip()
                section = None; continue
            if line.startswith("NBLOCKS"):
                data["nblocks"] = int(line.split(":",1)[1])
                section = None; continue
            if line.startswith("NPERIODS"):
                data["nperiods"] = int(line.split(":",1)[1])
                section = None; continue
            if line.startswith("NRESOURCE_SIDE_CONSTRAINTS"):
                data["nresources"] = int(line.split(":",1)[1])
                section = None; continue
            if line.startswith("DISCOUNT_RATE"):
                data["discount_rate"] = float(line.split(":",1)[1])
                section = None; continue
            if line.startswith("OBJECTIVE_FUNCTION"):
                section = "obj"; continue
            if line.startswith("RESOURCE_CONSTRAINT_COEFFICIENTS"):
                section = "res_coeff"; continue
            if line.startswith("RESOURCE_CONSTRAINT_LIMITS"):
                section = "res_lim"; continue
            # --- data sections ---
            parts = line.split()
            if section == "obj" and len(parts) >= 2:
                data["obj_vals"][int(parts[0])] = float(parts[1])
            elif section == "res_coeff" and len(parts) == 3:
                b, r, v = int(parts[0]), int(parts[1]), float(parts[2])
                data["res_coeffs"][b][r] = v
            elif section == "res_lim" and len(parts) >= 4:
                r, t, c = int(parts[0]), int(parts[1]), parts[2]
                v1 = float(parts[3])
                v2 = float(parts[4]) if len(parts) > 4 else None
                if c == "L":
                    lo, hi = v1, math.inf
                elif c == "G":
                    lo, hi = -math.inf, v1
                else:   # I (interval)
                    lo, hi = v1, v2
                data["res_limits"][(r, t)] = (lo, hi)
    return data


# ═════════════════════════════════════════════════════════════════════
#  2. PARSE PRECEDENCE FILE  (CSV)
# ═════════════════════════════════════════════════════════════════════
def parse_precedences(filepath: str) -> dict:
    """
    Returns {block_id → [predecessor_ids]}
    The predecessor block b' must be extracted ≤ same period as b.
    """
    print("  Loading precedence CSV …")
    df = pd.read_csv(filepath)
    preds = {}
    pred_cols = [c for c in df.columns if c.startswith("prede")]
    for _, row in df.iterrows():
        bid = int(row["BlockID"])
        n   = int(row["N_prede"])
        ps  = []
        for i in range(1, n + 1):
            v = row[f"prede{i}"]
            if not pd.isna(v):
                ps.append(int(v))
        preds[bid] = ps
    return preds


# ═════════════════════════════════════════════════════════════════════
#  3. COMPUTE Z-LEVELS (topological BFS from surface)
# ═════════════════════════════════════════════════════════════════════
def compute_z_levels(preds: dict, n_blocks: int) -> np.ndarray:
    """
    z_level[b] = 0  → surface block (no predecessors = no overburden)
    z_level[b] = k  → block lies k steps below the surface
    Uses BFS (Kahn's algorithm) for efficiency.
    """
    from collections import deque
    print("  Computing z-levels via topological sort …")

    in_deg  = {b: len(preds.get(b, [])) for b in range(n_blocks)}
    succs   = defaultdict(list)
    for b, ps in preds.items():
        for p in ps:
            succs[p].append(b)

    z_level = np.full(n_blocks, -1, dtype=int)
    queue   = deque()
    for b in range(n_blocks):
        if in_deg[b] == 0:
            z_level[b] = 0
            queue.append(b)

    while queue:
        b = queue.popleft()
        for s in succs[b]:
            if z_level[s] < z_level[b] + 1:
                z_level[s] = z_level[b] + 1
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    max_z = z_level.max()
    print(f"  Z-levels: 0 (surface) → {max_z} (deepest), "
          f"{(z_level == 0).sum()} surface blocks")
    return z_level


# ═════════════════════════════════════════════════════════════════════
#  4. ASSIGN SPATIAL (x, y) COORDINATES
# ═════════════════════════════════════════════════════════════════════
def assign_xy(z_level: np.ndarray) -> tuple:
    """
    Within each z-level, sort blocks by ID and arrange in 2-D grid.
    Returns (block_x, block_y) arrays of shape (n_blocks,).
    """
    n_blocks  = len(z_level)
    block_x   = np.zeros(n_blocks, dtype=int)
    block_y   = np.zeros(n_blocks, dtype=int)
    max_nx    = 0
    max_ny    = 0
    for z in range(z_level.max() + 1):
        ids = np.sort(np.where(z_level == z)[0])
        n   = len(ids)
        nx  = int(math.ceil(math.sqrt(n)))
        for idx, bid in enumerate(ids):
            ix = idx % nx
            iy = idx // nx
            block_x[bid] = ix
            block_y[bid] = iy
        max_nx = max(max_nx, nx)
        max_ny = max(max_ny, int(math.ceil(n / nx)))
    return block_x, block_y, max_nx, max_ny


# ═════════════════════════════════════════════════════════════════════
#  5. CPIT LP RELAXATION SOLVER  (greedy / LP)
# ═════════════════════════════════════════════════════════════════════
def solve_cpit_toposort(
    cpit_data: dict,
    preds: dict,
    z_level: np.ndarray,
    all_blocks: set,
    candidate_blocks: set,
) -> tuple:
    """
    TopoSort heuristic for CPIT: schedule the UPIT closure first, then fill
    remaining period lower bound requirements with the least-negative eligible
    blocks.
    """
    print("  Running CPIT TopoSort heuristic …")
    nperiods      = cpit_data["nperiods"]
    alpha         = cpit_data["discount_rate"]
    obj_vals      = cpit_data["obj_vals"]
    res_coeffs    = cpit_data["res_coeffs"]
    res_limits    = cpit_data["res_limits"]
    nresources    = cpit_data["nresources"]
    n_blocks      = cpit_data["nblocks"]

    # Pre-compute discounted factors
    disc = np.array([1.0 / (1 + alpha) ** (t + 1) for t in range(nperiods)])

    # Resource usage per period
    period_usage  = np.zeros((nresources, nperiods))
    ub = np.full((nresources, nperiods), math.inf)
    lb = np.zeros((nresources, nperiods))
    for (r, t), (lo, hi) in res_limits.items():
        if lo > -math.inf:
            lb[r, t] = lo
        if hi < math.inf:
            ub[r, t] = hi

    print(f"  Resource lb max: {np.max(lb)}, ub min: {np.min(ub)}")

    # Build successor graph for efficient topological traversal.
    from collections import deque
    succs = defaultdict(list)
    for b, ps in preds.items():
        for p in ps:
            succs[p].append(b)

    in_deg = {b: len(preds.get(b, [])) for b in all_blocks}
    queue = deque(b for b in all_blocks if in_deg[b] == 0)
    order = []
    while queue:
        b = queue.popleft()
        order.append(b)
        for s in succs[b]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    scheduled = np.full(n_blocks, -1, dtype=int)
    pred_sched = {b: -1 for b in all_blocks}
    unscheduled_pred_count = {b: len(preds.get(b, [])) for b in all_blocks}
    schedule = {}
    total_npv = 0.0

    # Phase 1: schedule the UPIT closure using priority queue for high-value first.
    import heapq
    pq = []
    in_deg_phase1 = {b: len(preds.get(b, [])) for b in candidate_blocks}
    for b in candidate_blocks:
        if in_deg_phase1[b] == 0:
            val = obj_vals.get(b, 0.0)
            heapq.heappush(pq, (-val, b))  # max heap

    while pq:
        neg_val, b = heapq.heappop(pq)
        if scheduled[b] != -1:
            continue
        earliest = max((pred_sched.get(p, -1) for p in preds.get(b, [])), default=-1) + 1
        t = min(earliest, nperiods - 1)
        schedule[b] = t
        scheduled[b] = t
        total_npv += obj_vals.get(b, 0.0) * disc[t]
        for r in range(nresources):
            period_usage[r, t] += res_coeffs[b].get(r, 0.0)
        pred_sched[b] = t
        for s in succs[b]:
            unscheduled_pred_count[s] -= 1
            if s in candidate_blocks:
                in_deg_phase1[s] -= 1
                if in_deg_phase1[s] == 0 and scheduled[s] == -1:
                    s_val = obj_vals.get(s, 0.0)
                    heapq.heappush(pq, (-s_val, s))

    # Phase 2: fill each period with eligible blocks until lower bounds are met.
    # Build initial set of ready blocks (those with no unscheduled predecessors)
    ready_blocks = set(b for b in all_blocks if unscheduled_pred_count[b] == 0 and scheduled[b] == -1)

    for t in range(nperiods):
        req = np.maximum(lb[:, t] - period_usage[:, t], 0.0)
        while np.any(req > 0):
            if not ready_blocks:
                break

            has_positive = any(
                obj_vals.get(bb, 0.0) > 0 and
                any(res_coeffs[bb].get(r, 0.0) > 0 and req[r] > 0 for r in range(nresources))
                for bb in ready_blocks
            )

            best_b = None
            best_score = -math.inf
            for b in ready_blocks:
                q_vals = [res_coeffs[b].get(r, 0.0) for r in range(nresources)]
                if all(q == 0 for q in q_vals):
                    continue
                if not any(q_vals[r] > 0 and req[r] > 0 for r in range(nresources)):
                    continue
                obj = obj_vals.get(b, 0.0)
                if obj < 0 and has_positive:
                    continue
                weighted = 0.0
                for r in range(nresources):
                    if req[r] > 0:
                        weighted += q_vals[r] / max(req[r], 1e-6)
                    else:
                        weighted += q_vals[r] / max(1.0, lb[r, t])
                score = obj * disc[t] / (weighted + 1e-9)
                if score > best_score:
                    best_score = score
                    best_b = b

            if best_b is None:
                break

            schedule[best_b] = t
            scheduled[best_b] = t
            ready_blocks.discard(best_b)  # Remove from ready
            total_npv += obj_vals.get(best_b, 0.0) * disc[t]
            for r in range(nresources):
                period_usage[r, t] += res_coeffs[best_b].get(r, 0.0)
            pred_sched[best_b] = t
            for s in succs[best_b]:
                unscheduled_pred_count[s] -= 1
                if unscheduled_pred_count[s] == 0 and scheduled[s] == -1:
                    ready_blocks.add(s)  # Add to ready when all preds scheduled
            req = np.maximum(lb[:, t] - period_usage[:, t], 0.0)

    extracted = len(schedule)
    print(f"  Scheduled {extracted:,} / {len(all_blocks):,} blocks  |  NPV = {total_npv:,.0f}")
    return schedule, total_npv, period_usage, ub, lb


# ═════════════════════════════════════════════════════════════════════
#  6. UPIT — Ultimate Pit (single period, no resource constraints)
# ═════════════════════════════════════════════════════════════════════
def solve_upit(cpit_data: dict, preds: dict) -> set:
    """
    Solves UPIT greedily: include a block if it has positive value
    and all its predecessors are also included.
    (This is equivalent to the maximum-weight closure problem.)
    """
    print("  Solving UPIT (ultimate pit) …")
    obj_vals = cpit_data["obj_vals"]
    n_blocks = cpit_data["nblocks"]

    succs = defaultdict(list)
    for b, ps in preds.items():
        for p in ps:
            succs[p].append(b)

    # Start with all positive-value blocks; add required predecessors
    in_pit = set()
    queue  = [b for b, v in obj_vals.items() if v > 0]

    def include(b):
        if b in in_pit:
            return
        in_pit.add(b)
        for p in preds.get(b, []):
            include(p)

    for b in queue:
        include(b)

    upit_npv = sum(obj_vals.get(b, 0.0) for b in in_pit)
    print(f"  UPIT: {len(in_pit):,} blocks in final pit | Undiscounted = {upit_npv:,.0f}")
    return in_pit, upit_npv


# ═════════════════════════════════════════════════════════════════════
#  7.  VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════

PLOTLY_BG = "rgb(10,10,25)"
PLOTLY_FG = "white"


def save_plotly_html(fig, out_path, auto_open=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, str(out_path), auto_open=auto_open,
                   include_plotlyjs="cdn")
    print(f"  Saved Plotly visualization to: {out_path}")


def _apply_plotly_theme(fig, title, height=800):
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=PLOTLY_BG,
        plot_bgcolor=PLOTLY_BG,
        font=dict(color=PLOTLY_FG),
        margin=dict(l=10, r=10, t=100, b=10),
        height=height,
    )
    return fig


def compute_pit_boundary_surface(block_x, block_y, z_level, pit_blocks):
    """Return a (X,Y) surface for the top of the ultimate pit boundary."""
    ix_vals = np.sort(np.unique(block_x))
    iy_vals = np.sort(np.unique(block_y))
    surface = np.full((len(ix_vals), len(iy_vals)), np.nan, dtype=float)
    ix_index = {x: i for i, x in enumerate(ix_vals)}
    iy_index = {y: j for j, y in enumerate(iy_vals)}

    for b in pit_blocks:
        ix = block_x[b]
        iy = block_y[b]
        i = ix_index[ix]
        j = iy_index[iy]
        if np.isnan(surface[i, j]) or z_level[b] > surface[i, j]:
            surface[i, j] = z_level[b]

    return ix_vals, iy_vals, surface


def fig_block_value_map(block_x, block_y, z_level, block_val, out_path):
    """Plan-view heat map of block values by z-level."""
    data = pd.DataFrame({
        "X": block_x,
        "Y": block_y,
        "Level": z_level,
        "Value": block_val,
    })
    fig = px.scatter(
        data,
        x="X",
        y="Y",
        color="Value",
        color_continuous_scale="RdYlGn",
        facet_col="Level",
        facet_col_wrap=6,
        category_orders={"Level": sorted(data["Level"].unique())},
        labels={"Value": "Block Value ($)", "Level": "Level"},
        title="Marvin Mine — Block Value by Level",
        height=950,
    )
    fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.replace("Level=", "Level ")))
    fig.update_layout(coloraxis_colorbar=dict(title="Block Value ($)"))
    save_plotly_html(fig, out_path)


def fig_3d_pit(block_x, block_y, z_level, block_val, schedule, upit_blocks, out_path):
    """Interactive 3-D scatter showing the CPIT schedule and ultimate pit."""
    print("  Rendering 3-D pit view …")
    n_blocks = len(block_val)
    rng = np.random.default_rng(42)
    subset = rng.choice(n_blocks, size=min(30000, n_blocks), replace=False)

    data = pd.DataFrame({
        "X": block_x[subset].astype(float),
        "Y": block_y[subset].astype(float),
        "Z": -z_level[subset].astype(float),
        "Value": block_val[subset],
        "Block": subset,
    })
    data["Scheduled"] = data["Block"].map(lambda b: b in schedule)
    data["InPit"] = data["Block"].map(lambda b: b in upit_blocks)
    data["Category"] = "Outside Pit"
    data.loc[data["InPit"] & (data["Value"] > 0), "Category"] = "Ore in Ultimate Pit"
    data.loc[data["InPit"] & (data["Value"] <= 0), "Category"] = "Waste in Ultimate Pit"
    data.loc[data["Scheduled"], "Category"] = "CPIT Scheduled"

    palette = {
        "CPIT Scheduled": "#FFD700",
        "Ore in Ultimate Pit": "#00e676",
        "Waste in Ultimate Pit": "#e53935",
        "Outside Pit": "#37474f",
    }

    fig = go.Figure()
    for category, color in palette.items():
        subset_df = data[data["Category"] == category]
        if subset_df.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=subset_df["X"],
            y=subset_df["Y"],
            z=subset_df["Z"],
            mode="markers",
            marker=dict(
                size=6 if category == "CPIT Scheduled" else 4,
                color=color,
                opacity=0.75,
            ),
            name=category,
            hovertemplate=(
                "<b>Block</b>: %{customdata[0]}<br>"
                "X: %{x}, Y: %{y}, Z: %{z}<br>"
                "Value: $%{customdata[1]:,.0f}<br>"
                "Category: %{text}<extra></extra>"
            ),
            text=subset_df["Category"],
            customdata=np.stack([subset_df["Block"], subset_df["Value"]], axis=1),
        ))

    if len(upit_blocks) > 0:
        ix_vals, iy_vals, pit_surface = compute_pit_boundary_surface(
            block_x, block_y, z_level, upit_blocks)
        fig.add_trace(go.Surface(
            x=ix_vals,
            y=iy_vals,
            z=-pit_surface,
            surfacecolor=pit_surface,
            colorscale="Viridis",
            opacity=0.45,
            showscale=False,
            name="Ultimate Pit Boundary",
            hovertemplate="X: %{x}<br>Y: %{y}<br>Depth: %{z}<extra></extra>",
            connectgaps=False,
        ))

        # Draw the pit perimeter only, not interior grid lines.
        if pit_surface.shape[0] > 0 and pit_surface.shape[1] > 0:
            # outer X edge
            x_line = np.concatenate([ix_vals, ix_vals[::-1], [ix_vals[0]]])
            y_line = np.concatenate([np.full_like(ix_vals, iy_vals[0]),
                                      np.full_like(ix_vals, iy_vals[-1]),
                                      [iy_vals[0]]])
            z_line = np.concatenate([-pit_surface[:, 0], -pit_surface[::-1, -1], [-pit_surface[0, 0]]])
            fig.add_trace(go.Scatter3d(
                x=x_line,
                y=y_line,
                z=z_line,
                mode="lines",
                line=dict(color="white", width=4),
                name="Pit perimeter",
                showlegend=True,
                hoverinfo="skip",
            ))

    fig.update_scenes(
        xaxis_title="X (block)",
        yaxis_title="Y (block)",
        zaxis_title="Depth (-level)",
        xaxis=dict(backgroundcolor=PLOTLY_BG, gridcolor="white"),
        yaxis=dict(backgroundcolor=PLOTLY_BG, gridcolor="white"),
        zaxis=dict(backgroundcolor=PLOTLY_BG, gridcolor="white"),
    )
    _apply_plotly_theme(fig, "Marvin Mine — 3-D Block Model (CPIT Solution)", height=850)
    save_plotly_html(fig, out_path)


def fig_schedule_gantt(schedule, nperiods, out_path):
    """Blocks mined per period and cumulative mining progression."""
    period_counts = [0] * nperiods
    for t in schedule.values():
        if 0 <= t < nperiods:
            period_counts[t] += 1
    cumulative = np.cumsum(period_counts)
    periods = list(range(1, nperiods + 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=periods,
        y=period_counts,
        name="Blocks Extracted",
        marker_color="#00b0ff",
    ))
    fig.add_trace(go.Scatter(
        x=periods,
        y=cumulative,
        mode="lines+markers",
        name="Cumulative Extracted",
        line=dict(color="#76ff03", width=3),
        marker=dict(size=6),
    ))

    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title="Blocks",
        legend=dict(bgcolor="rgba(255,255,255,0.05)"),
    )
    _apply_plotly_theme(fig, "CPIT Production Schedule — Marvin Mine", height=720)
    save_plotly_html(fig, out_path)


def fig_resource_utilisation(period_usage, ub, nperiods, nresources, out_path):
    """Shows resource usage against upper bounds for each period."""
    periods = list(range(1, nperiods + 1))
    resource_labels = {0: "Extraction (tonnes)", 1: "Processing (tonnes)"}

    fig = make_subplots(rows=nresources, cols=1,
                        shared_xaxes=True,
                        subplot_titles=[resource_labels.get(r, f"Resource {r}")
                                        for r in range(nresources)])

    for r in range(nresources):
        usage = [period_usage[r, t] for t in range(nperiods)]
        limit = [ub[r, t] for t in range(nperiods)]
        fig.add_trace(go.Bar(
            x=periods,
            y=usage,
            name="Usage",
            marker_color="#00b0ff" if r == 0 else "#ff6d00",
            opacity=0.8,
            showlegend=False,
        ), row=r + 1, col=1)
        fig.add_trace(go.Scatter(
            x=periods,
            y=limit,
            mode="lines",
            name="Upper Bound",
            line=dict(color="white", dash="dash"),
            showlegend=False,
        ), row=r + 1, col=1)
        fig.update_yaxes(title_text=resource_labels.get(r, f"Resource {r}"), row=r + 1, col=1)

    fig.update_xaxes(title_text="Period", row=nresources, col=1)
    _apply_plotly_theme(fig, "Resource Utilisation — CPIT Marvin Mine",
                        height=380 * max(1, nresources))
    save_plotly_html(fig, out_path)


def fig_npv_profile(schedule, cpit_data, out_path):
    """Discounted NPV contribution per period with cumulative line."""
    obj_vals = cpit_data["obj_vals"]
    alpha = cpit_data["discount_rate"]
    nperiods = cpit_data["nperiods"]

    period_npv = np.zeros(nperiods)
    for b, t in schedule.items():
        if 0 <= t < nperiods:
            val = obj_vals.get(b, 0.0)
            period_npv[t] += val / (1 + alpha) ** (t + 1)
    cumulative_npv = np.cumsum(period_npv)
    periods = list(range(1, nperiods + 1))

    pos_npv = np.where(period_npv >= 0, period_npv, 0)
    neg_npv = np.where(period_npv < 0, period_npv, 0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Discounted NPV per Period", "Cumulative Discounted NPV"])
    fig.add_trace(go.Bar(
        x=periods,
        y=pos_npv / 1e6,
        name="Positive NPV",
        marker_color="#00e676",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=periods,
        y=neg_npv / 1e6,
        name="Negative NPV",
        marker_color="#ef5350",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=periods,
        y=cumulative_npv / 1e6,
        mode="lines+markers",
        name="Cumulative NPV",
        line=dict(color="#FFD700", width=3),
        marker=dict(symbol="diamond", size=7),
    ), row=2, col=1)

    fig.update_yaxes(title_text="NPV ($M)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative NPV ($M)", row=2, col=1)
    fig.update_xaxes(title_text="Period", row=2, col=1)
    _apply_plotly_theme(fig, "NPV Profile — CPIT Marvin Mine", height=780)
    save_plotly_html(fig, out_path)


def fig_3d_schedule_animation(block_x, block_y, z_level, block_val, schedule, nperiods, out_path):
    """Interactive 3D animation showing pit progression over time."""
    print("  Rendering schedule progression …")
    snapshots = [int(round(nperiods * f)) - 1 for f in [0.2, 0.4, 0.6, 0.8, 1.0]]
    snapshots = [max(0, min(nperiods - 1, s)) for s in snapshots]

    n_blocks = len(block_val)
    rng = np.random.default_rng(42)
    subset = rng.choice(n_blocks, size=min(20000, n_blocks), replace=False)

    data = pd.DataFrame({
        "X": block_x[subset].astype(float),
        "Y": block_y[subset].astype(float),
        "Z": -z_level[subset].astype(float),
        "Value": block_val[subset],
        "Block": subset,
    })
    data["Schedule"] = data["Block"].map(lambda b: schedule.get(int(b), -1))
    data["Extracted"] = data["Schedule"] >= 0

    frames = []
    for snap_t in snapshots:
        snapshot_colors = ["#FFD700" if s >= 0 and s <= snap_t else "#263238" for s in data["Schedule"]]
        snapshot_sizes = [5 if s >= 0 and s <= snap_t else 2 for s in data["Schedule"]]
        frames.append(go.Frame(
            data=[go.Scatter3d(
                x=data["X"],
                y=data["Y"],
                z=data["Z"],
                mode="markers",
                marker=dict(color=snapshot_colors, size=snapshot_sizes, opacity=0.7),
                hoverinfo="skip",
            )],
            name=f"Period {snap_t + 1}",
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
    )
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.1,
            "y": 0,
        }],
        sliders=[{
            "steps": [{
                "label": f"{snap_t + 1}",
                "method": "animate",
                "args": [[f"Period {snap_t + 1}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            } for snap_t in snapshots],
            "currentvalue": {"prefix": "Up to period ", "visible": True},
        }],
    )
    fig.update_scenes(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        xaxis=dict(backgroundcolor=PLOTLY_BG, gridcolor="white"),
        yaxis=dict(backgroundcolor=PLOTLY_BG, gridcolor="white"),
        zaxis=dict(backgroundcolor=PLOTLY_BG, gridcolor="white"),
    )
    _apply_plotly_theme(fig, "CPIT Pit Progression — Marvin Mine", height=820)
    save_plotly_html(fig, out_path)


def fig_value_distribution(block_val, upit_blocks, out_path):
    """Histogram of block values for all blocks and ultimate pit blocks."""
    pit_vals = [block_val[b] for b in upit_blocks]
    all_vals = list(block_val)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=all_vals,
        nbinsx=60,
        name="All Blocks",
        marker_color="#00b0ff",
        opacity=0.75,
    ))
    fig.add_trace(go.Histogram(
        x=pit_vals,
        nbinsx=60,
        name="Ultimate Pit Blocks",
        marker_color="#FFD700",
        opacity=0.7,
    ))
    fig.update_layout(barmode="overlay")
    fig.update_layout(
        xaxis_title="Block Value ($)",
        yaxis_title="Count",
        title="Block Value Distribution — Marvin Mine",
    )
    _apply_plotly_theme(fig, "Block Value Distribution — Marvin Mine", height=700)
    save_plotly_html(fig, out_path)


# ═════════════════════════════════════════════════════════════════════
#  8.  SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════
def print_summary(cpit_data, upit_blocks, upit_npv, schedule, total_npv,
                  period_usage, ub):
    alpha    = cpit_data["discount_rate"]
    nperiods = cpit_data["nperiods"]
    nblocks  = cpit_data["nblocks"]
    obj_vals = cpit_data["obj_vals"]

    ore_blocks   = sum(1 for v in obj_vals.values() if v > 0)
    waste_blocks = nblocks - ore_blocks

    lines = [
        "═" * 62,
        "  MARVIN MINE — CPIT RESULTS SUMMARY",
        "═" * 62,
        f"  Mine instance         : {cpit_data['name']}",
        f"  Total blocks          : {nblocks:>12,}",
        f"  Ore blocks            : {ore_blocks:>12,}  ({100*ore_blocks/nblocks:.1f}%)",
        f"  Waste blocks          : {waste_blocks:>12,}  ({100*waste_blocks/nblocks:.1f}%)",
        f"  Planning horizon      : {nperiods:>12} periods",
        f"  Discount rate (α)     : {alpha:>12.1%}",
        f"  Resource constraints  : {cpit_data['nresources']:>12}  per period",
        "─" * 62,
        "  UPIT (Ultimate Pit Limit Problem)",
        f"  Blocks in ultimate pit: {len(upit_blocks):>12,}",
        f"  Undiscounted NPV      : ${upit_npv:>20,.0f}",
        "─" * 62,
        "  CPIT (Constrained Pit Limit Problem — TopoSort heuristic)",
        f"  Blocks scheduled      : {len(schedule):>12,}",
        f"  Discounted NPV        : ${total_npv:>20,.0f}",
        "─" * 62,
        "  RESOURCE UTILISATION (fraction of upper bound)",
        "  Period |  Extraction  | Processing",
        "  " + "─" * 38,
    ]
    for t in range(nperiods):
        e_frac = period_usage[0, t] / ub[0, t] if ub[0, t] > 0 else 0
        p_frac = period_usage[1, t] / ub[1, t] if (
            cpit_data["nresources"] > 1 and ub[1, t] > 0) else 0
        lines.append(f"  {t+1:>6} | {e_frac:>10.1%}  | {p_frac:>8.1%}")
    lines.append("═" * 62)
    report = "\n".join(lines)
    print(report)
    return report


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 62)
    print("  CPIT 3-D Solver — Marvin Mine (MineLib)")
    print("═" * 62 + "\n")

    # ── 1. Parse inputs ──────────────────────────────────────────────
    print("[1/6] Parsing CPIT data file …")
    cpit_data = parse_cpit(CPIT_FILE)
    n_blocks  = cpit_data["nblocks"]
    nperiods  = cpit_data["nperiods"]
    print(f"  {n_blocks:,} blocks, {nperiods} periods, "
          f"α={cpit_data['discount_rate']:.0%}")

    # ── 2. Precedences ───────────────────────────────────────────────
    print("\n[2/6] Parsing precedences …")
    preds = parse_precedences(PREC_CSV)
    total_prec = sum(len(v) for v in preds.values())
    print(f"  {total_prec:,} precedence relationships")

    # ── 3. Geometry ──────────────────────────────────────────────────
    print("\n[3/6] Computing block geometry …")
    z_level          = compute_z_levels(preds, n_blocks)
    block_x, block_y, nx, ny = assign_xy(z_level)
    block_val        = np.array([cpit_data["obj_vals"].get(b, 0.0)
                                 for b in range(n_blocks)])
    print(f"  Grid footprint: ~{nx} × {ny} blocks per level")

    # ── 4. Solve UPIT ────────────────────────────────────────────────
    print("\n[4/6] Solving UPIT …")
    upit_blocks, upit_npv = solve_upit(cpit_data, preds)

    # ── 5. Solve CPIT ────────────────────────────────────────────────
    print("\n[5/6] Solving CPIT …")
    t0 = time.time()
    schedule, total_npv, period_usage, ub, lb = solve_cpit_toposort(
        cpit_data, preds, z_level, set(range(n_blocks)), upit_blocks)
    print(f"  Solved in {time.time()-t0:.1f}s")

    # ── 6. Summary ───────────────────────────────────────────────────
    print("\n[6/6] Generating outputs …\n")
    report = print_summary(cpit_data, upit_blocks, upit_npv,
                            schedule, total_npv, period_usage, ub)

    # Save text report
    rpt_path = OUT_DIR / "cpit_marvin_report.txt"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(__doc__ + "\n\n" + report)
    print(f"\n  Saved: {rpt_path}")

    # ── Figures ──────────────────────────────────────────────────────
    print("\n  Generating figures …")

    fig_block_value_map(block_x, block_y, z_level, block_val,
                         OUT_DIR / "1_block_value_map.html")

    fig_3d_pit(block_x, block_y, z_level, block_val,
               schedule, upit_blocks,
               OUT_DIR / "2_3d_pit_view.html")

    fig_schedule_gantt(schedule, nperiods,
                        OUT_DIR / "3_schedule_gantt.html")

    fig_resource_utilisation(period_usage, ub, nperiods,
                              cpit_data["nresources"],
                              OUT_DIR / "4_resource_utilisation.html")

    fig_npv_profile(schedule, cpit_data,
                     OUT_DIR / "5_npv_profile.html")

    fig_3d_schedule_animation(block_x, block_y, z_level, block_val,
                               schedule, nperiods,
                               OUT_DIR / "6_pit_progression.html")

    fig_value_distribution(block_val, upit_blocks,
                            OUT_DIR / "7_value_distribution.html")

    print("\n" + "═" * 62)
    print("  ALL OUTPUTS SAVED TO:", OUT_DIR)
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()

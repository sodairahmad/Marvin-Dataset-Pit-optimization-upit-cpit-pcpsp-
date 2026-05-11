"""
UPIT (Ultimate Pit with Incremental Traversal) Optimizer
=========================================================
Implements the Lerchs-Grossmann (LG) / maximum-weight closure algorithm
on a 3-D block model with inter-block predecessor (slope) constraints.

Data files:
  Marvin.csv          – block coordinates (IX, IY, IZ) + economic values
  Marvin_Prece.xlsx   – predecessor (slope constraint) graph for each block

The economic value used is the "process" revenue (positive = ore, negative = waste).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict, deque
import time

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  UPIT OPTIMIZER  –  Marvin Block Model")
print("=" * 60)

print("\n[1/5] Loading block model …")
df = pd.read_csv("D:\MineLib mine design using Python\Marvin.csv")
df.columns = df.columns.str.strip()

# Use "economic value process" as block economic value (BEV)
df["BEV"] = df["economic value process"].astype(float)
df["NetValue"] = df["BEV"]
df["BlockType"] = np.where(df["BEV"] > 0, "Ore", "Waste")

# Build a unique 0-based block index that aligns with the predecessor file
#   The predecessor file uses BlockID = 0, 1, … N-1 (row order)
#   Re-index the CSV in the same order it was written (IX outer, IZ inner)
df = df.sort_values(["IX", "IY", "IZ"]).reset_index(drop=True)
df["BlockID"] = df.index

n_blocks = len(df)
print(f"   Blocks loaded : {n_blocks:,}")
print(f"   Grid          : IX {df.IX.min()}–{df.IX.max()} "
      f"| IY {df.IY.min()}–{df.IY.max()} | IZ {df.IZ.min()}–{df.IZ.max()}")
print(f"   BEV range     : [{df.BEV.min():,.0f}, {df.BEV.max():,.0f}]")
print(f"   Ore blocks (BEV > 0) : {(df.BEV > 0).sum():,}")

print("\n[2/5] Loading predecessor (slope-constraint) graph …")
df_pre = pd.read_excel("D:\MineLib mine design using Python\Marvin_Prece.xlsx")
df_pre.columns = df_pre.columns.str.strip()

# Build adjacency: pred_cols = prede1 … prede17
pred_cols = [c for c in df_pre.columns if c.startswith("prede") and c != "prede"]

# predecessors[block_id] = list of block_ids that must be removed BEFORE block_id
predecessors = defaultdict(list)
for _, row in df_pre.iterrows():
    bid = int(row["BlockID"])
    for col in pred_cols:
        v = row[col]
        if not np.isnan(v):
            predecessors[bid].append(int(v))

total_arcs = sum(len(v) for v in predecessors.values())
print(f"   Total slope arcs : {total_arcs:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MAXIMUM-WEIGHT CLOSURE  (Pseudo-flow / Lerchs-Grossmann style)
#     Implemented as a graph-theoretic max-closure via min-cut.
#     We use the classic construction:
#       • source S → block_i  with capacity  BEV_i   (if BEV_i > 0)
#       • block_i  → sink T   with capacity -BEV_i   (if BEV_i < 0)
#       • block_i  → pred_j   with capacity  ∞        (slope constraint)
#     Optimal pit = blocks reachable from S in the residual graph after max-flow.
#     We use a fast BFS-based push-relabel (FIFO) approximation suited
#     for large sparse graphs typical of block models.
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/5] Solving maximum-weight closure (LG algorithm) …")
t0 = time.time()

INF = 1e15   # large capacity for slope arcs

# Node numbering:   0 = source,  1..n = blocks,  n+1 = sink
S = 0
T = n_blocks + 1
N = n_blocks + 2          # total nodes

# Adjacency list (forward edges only stored; residual managed by back-pointers)
# Each edge: [to, cap, rev_index]
graph = [[] for _ in range(N)]

def add_edge(u, v, cap):
    graph[u].append([v, cap, len(graph[v])])
    graph[v].append([u, 0,  len(graph[u]) - 1])

# Source / sink arcs
total_positive_bev = 0.0
for bid in range(n_blocks):
    bev = df.at[bid, "BEV"]
    if bev > 0:
        add_edge(S, bid + 1, bev)
        total_positive_bev += bev
    elif bev < 0:
        add_edge(bid + 1, T, -bev)

# Slope arcs: block_i must wait for ALL its predecessors
for bid, preds in predecessors.items():
    for pid in preds:
        add_edge(bid + 1, pid + 1, INF)   # bid depends on pid → mine pid first

# ── BFS-level labelling (Dinic's BFS phase) ──────────────────────────────────
def bfs_level(s, t, level):
    level[:] = -1
    level[s] = 0
    q = deque([s])
    while q:
        u = q.popleft()
        for v, cap, _ in graph[u]:
            if cap > 0 and level[v] < 0:
                level[v] = level[u] + 1
                q.append(v)
    return level[t] >= 0

def dfs_flow(u, t, pushed, level, iter_):
    if u == t:
        return pushed
    while iter_[u] < len(graph[u]):
        v, cap, rev = graph[u][iter_[u]]
        if cap > 0 and level[v] == level[u] + 1:
            d = dfs_flow(v, t, min(pushed, cap), level, iter_)
            if d > 0:
                graph[u][iter_[u]][1] -= d
                graph[v][rev][1]       += d
                return d
        iter_[u] += 1
    return 0

# Dinic's max-flow
level = np.full(N, -1, dtype=np.int64)
max_flow = 0.0
while bfs_level(S, T, level):
    iter_ = [0] * N
    while True:
        f = dfs_flow(S, T, INF, level, iter_)
        if f == 0:
            break
        max_flow += f

# ── Find min-cut reachable set (= optimal pit) ───────────────────────────────
# Blocks reachable from S in the residual graph → inside pit
visited = [False] * N
q = deque([S])
visited[S] = True
while q:
    u = q.popleft()
    for v, cap, _ in graph[u]:
        if cap > 0 and not visited[v]:
            visited[v] = True
            q.append(v)

in_pit = np.array([visited[bid + 1] for bid in range(n_blocks)], dtype=bool)
optimal_value = total_positive_bev - max_flow

elapsed = time.time() - t0
print(f"   Solved in {elapsed:.1f}s")
print(f"   Blocks in optimal pit : {in_pit.sum():,} / {n_blocks:,}")
print(f"   Optimal pit value     : {optimal_value:,.0f}")

df["in_pit"] = in_pit


class UPITOptimizer:
    def __init__(self, df):
        self.df = df.copy()
        self.n_blocks = len(self.df)
        self.ix_vals = sorted(self.df.IX.unique())
        self.iy_vals = sorted(self.df.IY.unique())
        self.grid_ix, self.grid_iy = np.meshgrid(self.ix_vals, self.iy_vals, indexing="ij")
        self.IZ_surf = None

    @property
    def pit_df(self):
        return self.df[self.df["in_pit"]]

    @property
    def outside_df(self):
        return self.df[~self.df["in_pit"]]

    def compute_boundary_surface(self):
        pit_df = self.pit_df
        surface = pit_df.groupby(["IX", "IY"])["IZ"].max().reset_index()
        surface.columns = ["IX", "IY", "IZ_surface"]

        self.IZ_surf = np.full(self.grid_ix.shape, np.nan)
        surf_lookup = surface.set_index(["IX", "IY"])["IZ_surface"]
        for i, ix in enumerate(self.ix_vals):
            for j, iy in enumerate(self.iy_vals):
                key = (ix, iy)
                if key in surf_lookup.index:
                    self.IZ_surf[i, j] = surf_lookup[key]

    @staticmethod
    def save_plotly_html(fig, filename):
        pio.write_html(fig, filename, auto_open=True, include_plotlyjs="cdn")
        print(f"  Saved Plotly visualization to: {filename}")

    @staticmethod
    def get_hover_text(df, status_label):
        return [
            f"BlockID: {bid}<br>IX: {ix}<br>IY: {iy}<br>IZ: {iz}<br>BEV: {bev:,.0f}<br>Status: {status_label}"
            for bid, ix, iy, iz, bev in zip(
                df["BlockID"], df["IX"], df["IY"], df["IZ"], df["BEV"]
            )
        ]

    def plot_pit_blocks(self, filename="upit_pit_blocks_plot.html"):
        pit_df = self.pit_df
        if pit_df.empty:
            print("No blocks are currently assigned to the pit; skipping pit block plot.")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=pit_df["IX"],
            y=pit_df["IY"],
            z=pit_df["IZ"],
            mode="markers",
            marker=dict(
                size=4,
                color=pit_df["NetValue"],
                colorscale="Turbo",
                opacity=0.8,
                colorbar=dict(title="Net Value ($)"),
            ),
            text=pit_df["BlockType"],
            customdata=pit_df[["CU", "AU", "density"]].values,
            hovertemplate=
                "<b>Block Details</b><br>" +
                "IX: %{x}, IY: %{y}, IZ: %{z}<br>" +
                "CU: %{customdata[0]:.3f}% | AU: %{customdata[1]:.3f}g/t<br>" +
                "Density: %{customdata[2]:.3f} t/m³<br>" +
                "Type: %{text}<br>" +
                "Net Value: $%{marker.color:,.0f}<extra></extra>",
            name="Pit Blocks"
        ))

        if self.IZ_surf is not None:
            pit_surface = np.nan_to_num(self.IZ_surf, nan=np.nanmin(self.IZ_surf) - 1)
            fig.add_trace(go.Surface(
                x=self.ix_vals,
                y=self.iy_vals,
                z=pit_surface.T,
                surfacecolor=pit_surface.T,
                colorscale="Viridis",
                showscale=False,
                opacity=0.35,
                name="Pit boundary surface"
            ))

        fig.update_layout(
            title=f"Marvin UPIT - Total Profit: ${pit_df['NetValue'].sum():,.0f}",
            scene=dict(
                xaxis_title="IX (East)",
                yaxis_title="IY (North)",
                zaxis_title="IZ (Depth)",
                aspectmode="data",
                xaxis=dict(backgroundcolor="rgb(10,10,25)", gridcolor="white", showbackground=True),
                yaxis=dict(backgroundcolor="rgb(10,10,25)", gridcolor="white", showbackground=True),
                zaxis=dict(backgroundcolor="rgb(10,10,25)", gridcolor="white", showbackground=True),
            ),
            hovermode="closest",
            paper_bgcolor="rgb(10,10,25)",
            plot_bgcolor="rgb(10,10,25)",
            margin=dict(l=0, r=0, t=100, b=0),
            height=800,
        )

        self.save_plotly_html(fig, filename)

    def plot_all_blocks(self, filename="upit_all_blocks_plot.html"):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=self.outside_df["IX"],
            y=self.outside_df["IY"],
            z=self.outside_df["IZ"],
            mode="markers",
            marker=dict(
                size=3,
                color="green",
                opacity=0.35,
            ),
            text=self.outside_df["BlockType"],
            customdata=self.outside_df[["CU", "AU", "density", "NetValue"]].values,
            hovertemplate=
                "<b>Block Details</b><br>" +
                "IX: %{x}, IY: %{y}, IZ: %{z}<br>" +
                "CU: %{customdata[0]:.3f}% | AU: %{customdata[1]:.3f}g/t<br>" +
                "Density: %{customdata[2]:.3f} t/m³<br>" +
                "Type: %{text}<br>" +
                "Net Value: $%{customdata[3]:.0f}<extra></extra>",
            hoverlabel=dict(bgcolor="white", font_size=10, font_family="Arial"),
            name="Outside pit"
        ))
        fig.add_trace(go.Scatter3d(
            x=self.pit_df["IX"],
            y=self.pit_df["IY"],
            z=self.pit_df["IZ"],
            mode="markers",
            marker=dict(
                size=4,
                color="red",
                opacity=0.9,
            ),
            text=self.pit_df["BlockType"],
            customdata=self.pit_df[["CU", "AU", "density", "NetValue"]].values,
            hovertemplate=
                "<b>Block Details</b><br>" +
                "IX: %{x}, IY: %{y}, IZ: %{z}<br>" +
                "CU: %{customdata[0]:.3f}% | AU: %{customdata[1]:.3f}g/t<br>" +
                "Density: %{customdata[2]:.3f} t/m³<br>" +
                "Type: %{text}<br>" +
                "Net Value: $%{customdata[3]:.0f}<extra></extra>",
            hoverlabel=dict(bgcolor="white", font_size=10, font_family="Arial"),
            name="Inside pit"
        ))

        if self.IZ_surf is not None:
            pit_surface = np.nan_to_num(self.IZ_surf, nan=np.nanmin(self.IZ_surf) - 1)
            fig.add_trace(go.Surface(
                x=self.ix_vals,
                y=self.iy_vals,
                z=pit_surface.T,
                surfacecolor=pit_surface.T,
                colorscale="Greys",
                showscale=False,
                opacity=0.3,
                name="Pit boundary surface"
            ))

        fig.update_layout(
            title="All Blocks Colored by Ultimate Pit Limit",
            scene=dict(
                xaxis_title="IX (East)",
                yaxis_title="IY (North)",
                zaxis_title="IZ (Depth)",
                aspectmode="data",
                xaxis=dict(backgroundcolor="rgb(10,10,25)", gridcolor="white", showbackground=True),
                yaxis=dict(backgroundcolor="rgb(10,10,25)", gridcolor="white", showbackground=True),
                zaxis=dict(backgroundcolor="rgb(10,10,25)", gridcolor="white", showbackground=True),
                camera=dict(eye=dict(x=1.45, y=1.45, z=0.75)),
                dragmode="orbit",
            ),
            hovermode="closest",
            legend=dict(itemsizing="constant", bgcolor="rgba(10,10,25,0.7)", font=dict(color="white")),
            paper_bgcolor="rgb(10,10,25)",
            plot_bgcolor="rgb(10,10,25)",
            margin=dict(l=0, r=0, t=80, b=0),
            height=800,
        )

        self.save_plotly_html(fig, filename)

    
# ─────────────────────────────────────────────────────────────────────────────
# 3.  PIT BOUNDARY SURFACE
#     For each (IX, IY) column find the highest IZ that is in-pit.
# ─────────────────────────────────────────────────────────────────────────────

optimizer = UPITOptimizer(df)
optimizer.compute_boundary_surface()

print("\n[4/5] Saving Plotly visualizations to local HTML files …")
optimizer.plot_pit_blocks("upit_pit_blocks_plot.html")
optimizer.plot_all_blocks("upit_all_blocks_plot.html")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  Total blocks             : {n_blocks:>10,}")
print(f"  Blocks in optimal pit    : {in_pit.sum():>10,}  ({100*in_pit.sum()/n_blocks:.1f}%)")
print(f"  Ore blocks in pit        : {(optimizer.pit_df.BEV > 0).sum():>10,}")
print(f"  Waste blocks in pit      : {(optimizer.pit_df.BEV <= 0).sum():>10,}")
print(f"  Max flow (waste cost)    : {max_flow/1e6:>10.2f} M")
print(f"  Gross ore value          : {total_positive_bev/1e6:>10.2f} M")
print(f"  Optimal pit value (NPV)  : {optimal_value/1e6:>10.2f} M")
print(f"  Pit depth (benches)      : {optimizer.pit_df.IZ.min():>6} – {optimizer.pit_df.IZ.max()}")
print(f"  Pit footprint (IX)       : {optimizer.pit_df.IX.min():>6} – {optimizer.pit_df.IX.max()}")
print(f"  Pit footprint (IY)       : {optimizer.pit_df.IY.min():>6} – {optimizer.pit_df.IY.max()}")
print("=" * 60)
print("\n  Plotly HTML visualizations saved: upit_pit_solid_plot.html, upit_pit_blocks_plot.html, upit_all_blocks_plot.html")
print("  Open these files in a browser or a Plotly-compatible viewer to inspect the interactive 3D models.\n")

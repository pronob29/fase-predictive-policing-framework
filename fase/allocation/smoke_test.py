"""
Phase 4 smoke test.

1. Loads real meta.pkl (for pct_minority → group labels)
2. Loads real X_test / y_test tensors → runs FASEModel forward pass for risk mu
3. Solves constrained LP for 50 test timesteps
4. Compares constrained vs unconstrained (proportional) allocation
5. Asserts DIR is within epsilon for every solved timestep
"""

import pickle, pathlib, sys
import numpy as np
import torch

ROOT   = pathlib.Path(__file__).resolve().parents[2]
TENSOR = ROOT / "data/tensors/baltimore"

print("── Loading tensors and metadata ───────────────────────────────────────")
with open(TENSOR / "meta.pkl", "rb") as f:
    meta = pickle.load(f)
with open(TENSOR / "graph.pkl", "rb") as f:
    graph = pickle.load(f)

X_test = torch.load(TENSOR / "X_test.pt", weights_only=True)   # (N, T, F)
y_test = torch.load(TENSOR / "y_test.pt", weights_only=True)   # (N, T)
N, T_full, F = X_test.shape
print(f"  X_test: {tuple(X_test.shape)}")

# ── Derive group labels from ACS pct_minority ─────────────────────────────
acs_df         = meta["acs"]
node_ids       = meta["node_ids"]
pct_minority   = acs_df.set_index("ZCTA5").reindex(node_ids)["pct_minority"].fillna(0).values
MINORITY_THRESHOLD = 0.5
group = (pct_minority >= MINORITY_THRESHOLD).astype(int)   # (N,)

print(f"  Nodes: {N}  |  minority (≥{MINORITY_THRESHOLD}): {group.sum()}  "
      f"majority: {(1-group).sum()}")
print(f"  pct_minority range: [{pct_minority.min():.3f}, {pct_minority.max():.3f}]")

# ── Get predicted risk from FASEModel ────────────────────────────────────────
print("\n── Running FASEModel forward pass ─────────────────────────────────────")
from fase.models import FASEModel

T_window = 50   # use 50 consecutive test timesteps for speed

x = X_test[:, :T_window, :].unsqueeze(0)   # (1, N, T_window, F)
y = y_test[:, :T_window].unsqueeze(0)      # (1, N, T_window)

A_combined = torch.tensor(graph["A_combined"], dtype=torch.float32)
A_geo      = torch.tensor(graph["A_geo"],      dtype=torch.float32)

model = FASEModel(
    num_nodes          = N,
    in_channels        = F,
    residual_channels  = 32,
    skip_channels      = 32,
    stgnn_out_channels = 16,
    num_layers         = 4,
    dropout            = 0.0,
)
model.eval()
with torch.no_grad():
    mu, theta, pi = model(x, A_combined, A_geo, y)

# mu: (1, N, T_window)  — use as risk proxy
mu_np = mu.squeeze(0).T.numpy()   # (T_window, N)
print(f"  mu shape: {mu_np.shape}  range: [{mu_np.min():.4f}, {mu_np.max():.4f}]")

# ── Instantiate allocator ─────────────────────────────────────────────────────
print("\n── Solving constrained LP for each timestep ───────────────────────────")
from fase.allocation import PatrolAllocator, compare_allocations, batch_metrics

BUDGET  = 60.0
EPSILON = 0.05

allocator = PatrolAllocator(
    num_nodes     = N,
    group         = group,
    patrol_budget = BUDGET,
    epsilon       = EPSILON,
)

P_constrained  = allocator.solve_batch(mu_np)                     # (T, N)
P_proportional = np.stack([allocator._proportional(mu_np[t])
                            for t in range(T_window)])            # (T, N)

print(f"  Solved {T_window} timesteps")

# ── Compute metrics ──────────────────────────────────────────────────────────
print("\n── Allocation metrics ──────────────────────────────────────────────────")
from fase.allocation.metrics import dir_score

m_con = batch_metrics(P_constrained,  mu_np, group, BUDGET)
m_pro = batch_metrics(P_proportional, mu_np, group, BUDGET)

header = f"{'Metric':<18} {'Constrained':>22} {'Proportional':>22}"
print(header)
print("─" * len(header))
for k in m_con:
    c = m_con[k]
    p = m_pro[k]
    print(f"  {k:<16} {c['mean']:>10.4f} ± {c['std']:.4f}   "
          f"{p['mean']:>10.4f} ± {p['std']:.4f}")

# ── Assert fairness constraint holds ────────────────────────────────────────
print("\n── Checking fairness constraint ────────────────────────────────────────")
violations = 0
for t in range(T_window):
    d = dir_score(P_constrained[t], group)
    if abs(d - 1.0) > EPSILON + 1e-4:   # small tolerance for solver inaccuracy
        violations += 1

print(f"  DIR violations (|DIR-1| > {EPSILON}): {violations} / {T_window}")

if violations == 0:
    print("\n✓  Phase 4 smoke test PASSED")
else:
    print(f"\n✗  {violations} DIR violations — check solver accuracy")
    sys.exit(1)

"""
Phase 5 smoke test — deployment feedback simulator.

Runs 3 cycles (instead of 6) with 3 retrain epochs each for speed.
Verifies:
  - Each cycle completes without error
  - history list has correct length and keys
  - All metrics are finite
  - DIR stays within epsilon (LP constraint is not broken by retraining)
"""

import pickle, pathlib, sys
import numpy as np
import torch

ROOT   = pathlib.Path(__file__).resolve().parents[2]
TENSOR = ROOT / "data/tensors/baltimore"

# ── Load artefacts ────────────────────────────────────────────────────────────
print("── Loading artefacts ──────────────────────────────────────────────────")
with open(TENSOR / "meta.pkl", "rb") as f:
    meta = pickle.load(f)
with open(TENSOR / "graph.pkl", "rb") as f:
    graph = pickle.load(f)

X_test = torch.load(TENSOR / "X_test.pt", weights_only=True)   # (N, T, F)
y_test = torch.load(TENSOR / "y_test.pt", weights_only=True)   # (N, T)
N, T_full, F = X_test.shape

# Use a short window for speed: first 128 timesteps
T_USE = 128
X = X_test[:, :T_USE, :]
y = y_test[:, :T_USE]
print(f"  X: {tuple(X.shape)}  y: {tuple(y.shape)}")

# Group labels
node_ids   = meta["node_ids"]
acs_df     = meta["acs"]
pct_min    = acs_df.set_index("ZCTA5").reindex(node_ids)["pct_minority"].fillna(0).values
group      = (pct_min >= 0.5).astype(int)
print(f"  minority={group.sum()}  majority={(1-group).sum()}")

# Graph tensors
A_combined = torch.tensor(graph["A_combined"], dtype=torch.float32)
A_geo      = torch.tensor(graph["A_geo"],      dtype=torch.float32)

# ── Build model & allocator ───────────────────────────────────────────────────
print("\n── Building model and allocator ───────────────────────────────────────")
from fase.models import FASEModel
from fase.allocation import PatrolAllocator
from fase.simulation import DeploymentSimulator

model = FASEModel(
    num_nodes          = N,
    in_channels        = F,
    residual_channels  = 32,
    skip_channels      = 32,
    stgnn_out_channels = 16,
    num_layers         = 4,
    dropout            = 0.0,
)

allocator = PatrolAllocator(
    num_nodes     = N,
    group         = group,
    patrol_budget = 60.0,
    epsilon       = 0.05,
)

# ── Build simulator ───────────────────────────────────────────────────────────
sim = DeploymentSimulator(
    model               = model,
    allocator           = allocator,
    X                   = X,
    y_true              = y,
    A_combined          = A_combined,
    A_binary            = A_geo,
    group               = group,
    patrol_budget       = 60.0,
    detection_base_prob = 0.30,
    detection_max_prob  = 0.90,
    retrain_epochs      = 3,       # small for speed
    window_size         = 32,
    seed                = 42,
)

# ── Run 3 cycles ──────────────────────────────────────────────────────────────
print("\n── Running 3 deployment cycles ────────────────────────────────────────")
history = sim.run(num_cycles=3)

# ── Validate ──────────────────────────────────────────────────────────────────
print("\n── Validation ─────────────────────────────────────────────────────────")
EPSILON   = 0.05
required_keys = {
    "cycle", "zinb_loss_mean", "zinb_loss_final",
    "dir_mean", "dir_std", "gini_mean", "coverage_mean",
    "mean_min_patrol", "mean_maj_patrol",
    "obs_dir_mean", "true_dir_mean",
    "detection_ratio_min", "detection_ratio_maj",
}

assert len(history) == 3, f"Expected 3 cycles, got {len(history)}"

for rec in history:
    missing = required_keys - set(rec.keys())
    assert not missing, f"Cycle {rec['cycle']} missing keys: {missing}"

    for k, v in rec.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"Cycle {rec['cycle']} key '{k}' = {v} is not finite"

    dir_ok = abs(rec["dir_mean"] - 1.0) <= EPSILON + 0.01  # 0.01 tolerance for solver
    assert dir_ok, (
        f"Cycle {rec['cycle']} mean DIR={rec['dir_mean']:.4f} "
        f"violates |DIR-1|≤{EPSILON}"
    )

print("  All 3 cycles complete, all metrics finite, DIR constraint held.")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── Per-cycle summary ───────────────────────────────────────────────────")
print(f"{'Cycle':>5}  {'ZINB loss':>10}  {'DIR':>7}  {'Gini':>7}  "
      f"{'Coverage':>9}  {'obs_DIR':>8}  {'det_min':>8}  {'det_maj':>8}")
print("─" * 78)
for r in history:
    print(
        f"  {r['cycle']:>3}  {r['zinb_loss_final']:>10.4f}  "
        f"{r['dir_mean']:>7.4f}  {r['gini_mean']:>7.4f}  "
        f"{r['coverage_mean']:>9.4f}  {r['obs_dir_mean']:>8.4f}  "
        f"{r['detection_ratio_min']:>8.4f}  {r['detection_ratio_maj']:>8.4f}"
    )

print("\n✓  Phase 5 smoke test PASSED")

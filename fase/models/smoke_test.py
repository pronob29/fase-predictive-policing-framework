"""
Quick end-to-end smoke test for Phase 3 models.
Loads real tensors + graph, runs one forward pass, checks shapes and loss.
"""

import pickle, sys, pathlib
import torch
from fase.models import FASEModel
from fase.models.losses import zinb_loss

ROOT   = pathlib.Path(__file__).resolve().parents[2]
TENSOR = ROOT / "data/tensors/baltimore"

print("── Loading tensors and graph ──────────────────────────────────────────")
with open(TENSOR / "meta.pkl", "rb") as f:
    meta = pickle.load(f)
with open(TENSOR / "graph.pkl", "rb") as f:
    graph = pickle.load(f)

X_train = torch.load(TENSOR / "X_train.pt", weights_only=True)  # (N, T, F)
y_train = torch.load(TENSOR / "y_train.pt", weights_only=True)  # (N, T)

N, T_full, F = X_train.shape
print(f"  X_train: {tuple(X_train.shape)}   y_train: {tuple(y_train.shape)}")

# ── Use a small window to keep the smoke test fast ──────────────────────────
T   = 32                   # 32 time steps
B   = 2                    # fake batch size: slice two non-overlapping windows

x = torch.stack([
    X_train[:, :T, :],     # window 0
    X_train[:, T:2*T, :],  # window 1
])                         # (B, N, T, F)

y = torch.stack([
    y_train[:, :T],
    y_train[:, T:2*T],
])                         # (B, N, T)

A_combined = torch.tensor(graph["A_combined"], dtype=torch.float32)  # (N,N)
A_geo      = torch.tensor(graph["A_geo"],      dtype=torch.float32)

print(f"  x: {tuple(x.shape)}   y: {tuple(y.shape)}")
print(f"  A_combined: {tuple(A_combined.shape)}")

# ── Build model ─────────────────────────────────────────────────────────────
print("\n── Building FASEModel ─────────────────────────────────────────────────")
model = FASEModel(
    num_nodes          = N,
    in_channels        = F,
    residual_channels  = 32,   # smaller for speed
    skip_channels      = 32,
    stgnn_out_channels = 16,
    num_layers         = 4,
    kernel_size        = 2,
    dropout            = 0.0,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# ── Forward pass ─────────────────────────────────────────────────────────────
print("\n── Forward pass ───────────────────────────────────────────────────────")
model.eval()
with torch.no_grad():
    mu, theta, pi = model(x, A_combined, A_geo, y)

print(f"  mu    : {tuple(mu.shape)}  min={mu.min():.4f}  max={mu.max():.4f}")
print(f"  theta : {tuple(theta.shape)}  min={theta.min():.4f}  max={theta.max():.4f}")
print(f"  pi    : {tuple(pi.shape)}  min={pi.min():.4f}  max={pi.max():.4f}")

assert mu.shape    == (B, N, T), f"mu shape mismatch: {mu.shape}"
assert theta.shape == (B, N, T), f"theta shape mismatch: {theta.shape}"
assert pi.shape    == (B, N, T), f"pi shape mismatch: {pi.shape}"
assert (mu    > 0).all(),        "mu must be positive"
assert (theta > 0).all(),        "theta must be positive"
assert ((pi > 0) & (pi < 1)).all(), "pi must be in (0,1)"

# ── ZINB loss ────────────────────────────────────────────────────────────────
print("\n── ZINB loss ──────────────────────────────────────────────────────────")
loss = zinb_loss(y, mu, theta, pi)
print(f"  zinb_loss = {loss.item():.6f}")
assert torch.isfinite(loss), "Loss is not finite!"

# ── Backward pass ────────────────────────────────────────────────────────────
print("\n── Backward pass ──────────────────────────────────────────────────────")
model.train()
mu, theta, pi = model(x, A_combined, A_geo, y)
loss = zinb_loss(y, mu, theta, pi)
loss.backward()

grad_norms = {
    name: p.grad.norm().item()
    for name, p in model.named_parameters()
    if p.grad is not None
}
assert len(grad_norms) > 0, "No gradients computed!"
print(f"  Gradients on {len(grad_norms)} parameter tensors — all finite: "
      f"{all(torch.isfinite(torch.tensor(v)) for v in grad_norms.values())}")

print("\n✓  Phase 3 smoke test PASSED")

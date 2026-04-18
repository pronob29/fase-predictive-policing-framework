"""
fase/models/hawkes.py
=====================
Multivariate Hawkes Process excitation layer — GPU-vectorized.

Intensity model
---------------
    λ_v(t) = μ_v(t)  +  A_v(t)

    A_v(t) = Σ_{k=0}^{H-1}  exp(-β·k) · z_v(t-1-k)

where z_v(t) = Σ_u α_uv · y_u(t)  (neighbourhood influence)
and   H = history_window (default 168 = 7 days at 1-hour bins).

Equivalently, A satisfies the O(1)-step recurrence:
    A_v(t) = exp(-β) · A_v(t-1)  +  z_v(t-1)

GPU-efficient implementation
-----------------------------
The original O(T) Python loop (T ≈ 26,280) launches tens of thousands of
tiny CUDA kernels, destroying GPU throughput.  We replace it with:

  1.  einsum   — z = α^T ⊙ y_hist         shape (B, N, T)  one CUDA call
  2.  F.pad    — causal shift z → z_shift  shape (B, N, T)  one CUDA call
  3.  F.conv1d — causal exponential decay  shape (B, N, T)  one CUDA call

Total: 3 CUDA kernel launches regardless of T.
All three ops are fully differentiable so gradients flow back to α and β.

Kernel derivation
-----------------
We want A[t] = Σ_{k=0}^{H-1} ker[k] · z_shift[t-k]
             (standard causal linear convolution)

F.conv1d computes cross-correlation:
  output[t] = Σ_{j=0}^{H-1} weight[j] · input[t+j]

With (H-1) zeros left-padded onto input and weight = ker.flip(0):
  output[t] = Σ_{j=0}^{H-1} ker[H-1-j] · z_shift_padded[t+j]
            = Σ_{k=0}^{H-1} ker[k] · z_shift[t-k]   ✓  (substitution k=H-1-j)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HawkesExcitation(nn.Module):
    """
    Learnable Hawkes excitation A_v(t) via causal exponential-decay convolution.

    Parameters
    ----------
    num_nodes      : N — number of spatial nodes
    alpha_init     : initial per-edge excitation strength (all edges equal)
    beta_init      : initial global decay rate
    history_window : H — max lag in time steps (default 168 = 7 days @ 1h bins)
    """

    def __init__(
        self,
        num_nodes: int,
        alpha_init: float = 0.5,
        beta_init: float = 1.0,
        history_window: int = 168,
    ) -> None:
        super().__init__()
        self.N = num_nodes
        self.H = history_window

        self.log_alpha = nn.Parameter(
            torch.full((num_nodes, num_nodes), _inv_softplus(alpha_init))
        )
        self.log_beta = nn.Parameter(
            torch.tensor(_inv_softplus(beta_init))
        )

    def forward(
        self,
        y_hist: torch.Tensor,   # (B, N, T)
        A_bin:  torch.Tensor,   # (N, N) binary adjacency mask
    ) -> torch.Tensor:
        """
        Returns
        -------
        excitation : (B, N, T) — A_v(t) for every (batch, node, timestep)
        """
        B, N, T = y_hist.shape
        device  = y_hist.device
        dtype   = y_hist.dtype

        alpha = F.softplus(self.log_alpha) * A_bin   # (N, N), masked
        beta  = F.softplus(self.log_beta)            # scalar

        # ── Step 1: neighbourhood influence ──────────────────────────────────
        # z[b, v, t] = Σ_u alpha[u, v] * y[b, u, t]
        z = torch.einsum("uv,but->bvt", alpha, y_hist)  # (B, N, T)

        # ── Step 2: causal shift — A(t) depends on z(t-1), z(t-2), … ────────
        # z_shift[:, v, 0]  = 0
        # z_shift[:, v, t]  = z[:, v, t-1]  for t ≥ 1
        z_shift = F.pad(z, (1, 0))[..., :T]             # (B, N, T)

        # ── Step 3: exponential-decay causal convolution ──────────────────────
        H   = min(self.H, T)
        k_  = torch.arange(H, device=device, dtype=dtype)
        ker = torch.exp(-beta * k_)                      # (H,) ∝ softplus(log_beta)
        # flip for cross-correlation → convolution (see module docstring)
        ker_w = ker.flip(0).view(1, 1, H)               # (1, 1, H)

        z_bn     = z_shift.reshape(B * N, 1, T)
        z_padded = F.pad(z_bn, (H - 1, 0))              # left-pad for causality
        A_bn     = F.conv1d(z_padded, ker_w)             # (B*N, 1, T)

        return A_bn.reshape(B, N, T)


def _inv_softplus(x: float, eps: float = 1e-6) -> float:
    """softplus⁻¹(x) = log(exp(x) - 1)  used for parameter initialisation."""
    x = max(x, eps)
    return math.log(math.expm1(x)) if x < 20 else x

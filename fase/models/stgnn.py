"""
fase/models/stgnn.py
====================
Residual-aware Spatiotemporal Graph Neural Network (STGNN).

Architecture — simplified Graph WaveNet with explicit graph diffusion
---------------------------------------------------------------------
Input  X  :  (B, N, T, F)
Output     :  (B, N, T, out_channels)   base-risk embedding  μ_v(t)

Each WaveNetBlock applies, in order:
  1. Dilated causal temporal convolution  (left-padded, no look-ahead)
  2. Gated activation  (tanh · sigmoid split)
  3. Graph diffusion convolution  (A_combined @ h)
  4. Dropout
  5. Skip projection  →  accumulated across blocks
  6. Residual add

The temporal receptive field with default settings
(num_layers=4, kernel_size=2, dilations=[1,2,4,8]) is 16 hours.
Double num_layers or add a second cycle (dilations=[1,2,4,8,1,2,4,8])
to extend to 32 h, etc.

All graph operations use plain torch.einsum on a dense (N, N) matrix.
N = 25 ZCTAs, so dense matmul is faster than sparse for this problem size.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building block: dilated causal 1-D convolution
# ─────────────────────────────────────────────────────────────────────────────

class CausalDilatedConv(nn.Module):
    """
    1-D convolution with automatic left-only padding to enforce causality.

    output[t] depends only on input[t], input[t-1], …  (no future leakage).

    Parameters
    ----------
    in_channels, out_channels : channel dimensions
    kernel_size               : filter width (default 2 gives compact support)
    dilation                  : exponentially grown across blocks (1, 2, 4, 8, …)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation          # left-pad only
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B·N, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


# ─────────────────────────────────────────────────────────────────────────────
# Building block: one WaveNet residual block
# ─────────────────────────────────────────────────────────────────────────────

class WaveNetBlock(nn.Module):
    """
    One residual block: temporal TCN → gated activation → graph diffusion.

    Canonical tensor format throughout: (B, N, T, C).

    Parameters
    ----------
    residual_channels : C — channels carried through residual stream
    skip_channels     : channels contributed to the skip accumulator
    kernel_size       : temporal filter width
    dilation          : dilation rate for this block
    dropout           : applied after gated activation, before graph conv
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        C = residual_channels
        # Single conv that outputs 2C — split into filter and gate halves
        self.tcn      = CausalDilatedConv(C, 2 * C, kernel_size, dilation)
        self.skip_lin = nn.Linear(C, skip_channels)
        self.res_lin  = nn.Linear(C, C)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # (B, N, T, C)
        A: torch.Tensor,           # (N, N) adjacency — applied after gate
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        h    : (B, N, T, C)         residual output for the next block
        skip : (B, N, T, skip_ch)   accumulated skip contribution
        """
        B, N, T, C = x.shape
        residual = x

        # ── 1. Dilated causal temporal convolution ────────────────────────────
        # Reshape to (B·N, C, T) for Conv1d, then restore
        h = x.reshape(B * N, T, C).permute(0, 2, 1)   # (B·N, C, T)
        h = self.tcn(h)                                 # (B·N, 2C, T)
        h_f, h_g = h.chunk(2, dim=1)                   # each (B·N, C, T)
        h = torch.tanh(h_f) * torch.sigmoid(h_g)       # (B·N, C, T)  gated
        h = h.permute(0, 2, 1).reshape(B, N, T, C)     # (B, N, T, C)
        h = self.drop(h)

        # ── 2. Graph diffusion convolution ────────────────────────────────────
        # h_out[b,n,t,c] = Σ_m  A[n,m] · h[b,m,t,c]
        h = torch.einsum("nm,bmtc->bntc", A, h)        # (B, N, T, C)

        # ── 3. Skip and residual projections ─────────────────────────────────
        skip = self.skip_lin(h)                         # (B, N, T, skip_ch)
        h    = self.res_lin(h) + residual               # (B, N, T, C)

        return h, skip


# ─────────────────────────────────────────────────────────────────────────────
# Full STGNN
# ─────────────────────────────────────────────────────────────────────────────

class STGNN(nn.Module):
    """
    Stacked WaveNet blocks with a learned output projection.

    Parameters
    ----------
    in_channels       : F = 13 (input feature dimension)
    residual_channels : channel width inside each block (default 64)
    skip_channels     : skip accumulator width (default 64)
    out_channels      : embedding dimension returned to FASEModel (default 32)
    num_layers        : number of WaveNet blocks
    kernel_size       : temporal conv kernel width
    dropout           : per-block dropout rate
    dilations         : explicit dilation schedule; if None, uses [1,2,4,...,2^(L-1)]

    Receptive field (default 4 layers, kernel=2):
        RF = 1 + (2-1)·(1+2+4+8) = 16 time steps
    """

    def __init__(
        self,
        in_channels: int = 13,
        residual_channels: int = 64,
        skip_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 2,
        dropout: float = 0.2,
        dilations: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        if dilations is None:
            dilations = [2 ** i for i in range(num_layers)]
        if len(dilations) != num_layers:
            raise ValueError(
                f"len(dilations)={len(dilations)} must equal num_layers={num_layers}"
            )

        self.input_proj = nn.Linear(in_channels, residual_channels)

        self.blocks = nn.ModuleList([
            WaveNetBlock(
                residual_channels, skip_channels,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            for d in dilations
        ])

        # Two-layer output MLP on the accumulated skip signal
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(skip_channels, skip_channels),
            nn.ReLU(),
            nn.Linear(skip_channels, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,   # (B, N, T, F)
        A: torch.Tensor,   # (N, N)  combined adjacency
    ) -> torch.Tensor:
        """
        Returns
        -------
        embedding : (B, N, T, out_channels)
        """
        h         = self.input_proj(x)      # (B, N, T, residual_ch)
        skip_acc  = None

        for block in self.blocks:
            h, skip = block(h, A)
            skip_acc = skip if skip_acc is None else skip_acc + skip

        return self.output_proj(skip_acc)   # (B, N, T, out_channels)

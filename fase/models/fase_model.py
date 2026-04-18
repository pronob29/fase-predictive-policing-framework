"""
fase/models/fase_model.py
=========================
FASEModel — top-level module combining all Phase 3 sub-models.

Data flow
---------
                                ┌──────────────┐
  x (B,N,T,F)  ─────────────►  │    STGNN     │ ──► μ_embed (B,N,T,D)
  A_combined (N,N)              └──────────────┘
                                                         ↓  (D channels)
  y_hist (B,N,T)  ──────────►  ┌──────────────┐ ──► A_excite (B,N,T,1)
  A_binary (N,N)               │    Hawkes    │      broadcast → (B,N,T,D)
                                └──────────────┘
                                                    combined = μ_embed + A_excite
                                                         ↓
                                ┌──────────────────────────────────┐
                                │  ZINB head  (3 × Linear)         │
                                │  combined → (mu, theta, pi)      │
                                └──────────────────────────────────┘
                                    each shape: (B, N, T)

The ZINB head maps the D-dim embedding to three scalars per (b,n,t):
  mu    → softplus (> 0)
  theta → softplus (> 0)
  pi    → sigmoid  (∈ (0,1))

Training
--------
from fase.models import FASEModel
from fase.models.losses import zinb_loss

model = FASEModel.from_config(cfg)
mu, theta, pi = model(x, A_combined, A_binary, y_hist)
loss = zinb_loss(y_target, mu, theta, pi)
loss.backward()
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgnn import STGNN
from .hawkes import HawkesExcitation


class ZINBHead(nn.Module):
    """
    Maps a D-dimensional embedding to ZINB parameters (mu, theta, pi).

    Two-layer MLP shared across all (b, n, t).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head    = nn.Linear(hidden_dim, 1)
        self.theta_head = nn.Linear(hidden_dim, 1)
        self.pi_head    = nn.Linear(hidden_dim, 1)

    def forward(
        self, h: torch.Tensor  # (B, N, T, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns mu, theta, pi each of shape (B, N, T)."""
        feat  = self.net(h)                                 # (B, N, T, hidden)
        mu    = F.softplus(self.mu_head(feat)).squeeze(-1)  # (B, N, T)
        theta = F.softplus(self.theta_head(feat)).squeeze(-1)
        pi    = torch.sigmoid(self.pi_head(feat)).squeeze(-1)
        return mu, theta, pi


class FASEModel(nn.Module):
    """
    Full FASE dual-engine model.

    Parameters
    ----------
    num_nodes         : N (25 for Baltimore)
    in_channels       : F (13 engineered features)
    residual_channels : STGNN hidden width (default 64)
    skip_channels     : STGNN skip accumulator width (default 64)
    stgnn_out_channels: D — embedding dim passed to ZINB head (default 32)
    num_layers        : STGNN depth (default 4)
    kernel_size       : temporal conv kernel (default 2)
    dropout           : per-block dropout (default 0.2)
    dilations         : explicit dilation list; None → [1,2,4,...,2^(L-1)]
    hawkes_alpha_init : initial Hawkes excitation strength (default 0.5)
    hawkes_beta_init  : initial Hawkes decay rate (default 1.0)
    zinb_hidden       : hidden dim inside ZINB head (default 32)
    """

    def __init__(
        self,
        num_nodes: int = 25,
        in_channels: int = 13,
        residual_channels: int = 64,
        skip_channels: int = 64,
        stgnn_out_channels: int = 32,
        num_layers: int = 4,
        kernel_size: int = 2,
        dropout: float = 0.2,
        dilations: Optional[list] = None,
        hawkes_alpha_init: float = 0.5,
        hawkes_beta_init: float = 1.0,
        hawkes_history_window: int = 168,
        zinb_hidden: int = 32,
    ) -> None:
        super().__init__()

        self.stgnn = STGNN(
            in_channels=in_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            out_channels=stgnn_out_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            dilations=dilations,
        )

        self.hawkes = HawkesExcitation(
            num_nodes=num_nodes,
            alpha_init=hawkes_alpha_init,
            beta_init=hawkes_beta_init,
            history_window=hawkes_history_window,
        )

        # Project scalar Hawkes excitation to D dims before adding to embedding
        self.hawkes_proj = nn.Linear(1, stgnn_out_channels)

        self.zinb_head = ZINBHead(
            in_dim=stgnn_out_channels,
            hidden_dim=zinb_hidden,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,         # (B, N, T, F)
        A_combined: torch.Tensor, # (N, N)  weighted, row-stochastic
        A_binary: torch.Tensor,   # (N, N)  binary mask for Hawkes
        y_hist: torch.Tensor,     # (B, N, T)  observed counts (lagged or full)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mu, theta, pi : each (B, N, T)  — ZINB distribution parameters
        """
        # 1. Base-risk embedding from STGNN
        mu_embed = self.stgnn(x, A_combined)    # (B, N, T, D)

        # 2. Hawkes excitation (scalar per node/time)
        A_exc = self.hawkes(y_hist, A_binary)   # (B, N, T)

        # 3. Project excitation scalar → D-dim and add to embedding
        A_exc_proj = self.hawkes_proj(
            A_exc.unsqueeze(-1)                 # (B, N, T, 1)
        )                                       # (B, N, T, D)
        combined = mu_embed + A_exc_proj        # (B, N, T, D)

        # 4. ZINB head
        return self.zinb_head(combined)         # mu, theta, pi each (B, N, T)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "FASEModel":
        """
        Instantiate from the phase3 section of fase_config.yaml (already
        loaded as a plain dict, e.g. via yaml.safe_load).

        Expected keys (all optional — defaults mirror __init__):
            stgnn.in_channels, stgnn.hidden_channels, stgnn.out_channels,
            stgnn.num_layers, stgnn.kernel_size, stgnn.dropout, stgnn.dilations
            hawkes.alpha_init, hawkes.beta_init
            num_nodes  (top-level or inferred as 25)
        """
        p3  = cfg.get("phase3", cfg)
        st  = p3.get("stgnn", {})
        hw  = p3.get("hawkes", {})

        return cls(
            num_nodes          = cfg.get("num_nodes", 25),
            in_channels        = st.get("in_channels", 13),
            residual_channels  = st.get("hidden_channels", 64),
            skip_channels      = st.get("hidden_channels", 64),
            stgnn_out_channels = st.get("out_channels", 32),
            num_layers         = st.get("num_layers", 4),
            kernel_size        = st.get("kernel_size", 2),
            dropout            = st.get("dropout", 0.2),
            dilations          = st.get("dilations", None),
            hawkes_alpha_init      = hw.get("alpha_init", 0.5),
            hawkes_beta_init       = hw.get("beta_init", 1.0),
            hawkes_history_window  = hw.get("history_window", 168),
        )

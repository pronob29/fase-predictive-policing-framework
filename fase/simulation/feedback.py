"""
fase/simulation/feedback.py
===========================
Deployment Feedback Simulator (Phase 5).

Feedback loop — one cycle k:
─────────────────────────────
  1. INFERENCE   model(X, A, y_hist) → μ(t) for all t in deploy window
  2. ALLOCATION  PatrolAllocator.solve_batch(μ) → P  shape (T, N)
  3. DETECTION   det_prob_v(t) = base + (max−base) · P[t,v] / B
                 y_obs[v,t]  = y_true[v,t] · det_prob_v(t)   (expected value)
  4. RETRAIN     model is fine-tuned on (X, y_obs) for retrain_epochs
                 using sliding windows of size window_size
  5. RECORD      cycle metrics: ZINB loss, DIR, Gini, coverage

Detection model
───────────────
Using the *expected* detection rate (deterministic) rather than Bernoulli
sampling keeps the loop reproducible and avoids stochastic noise masking
the fairness signal.

    det_prob(p) = p_base + (p_max − p_base) · clip(p / B, 0, 1)

Feedback bias mechanism
───────────────────────
More patrol → higher detection prob → larger y_obs in patrolled ZCTAs →
model learns to predict higher risk there → reinforced allocation → …

The simulator tracks both the *allocation fairness* (DIR of P) and the
*training-data bias* (DIR of y_obs) across cycles so researchers can
quantify feedback amplification.

Usage
─────
    sim = DeploymentSimulator.from_config(cfg, model, allocator, tensors, graph)
    history = sim.run(num_cycles=6)
    # history: list of dicts, one per cycle
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from fase.models.losses import zinb_loss
from fase.allocation import PatrolAllocator
from fase.allocation.metrics import dir_score, gini_coefficient, coverage_score


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detection_prob(
    patrol: np.ndarray,          # (T, N)
    budget: float,
    base_prob: float,
    max_prob: float,
) -> np.ndarray:
    """Linear detection probability model, shape (T, N), values in [base, max]."""
    p_norm = np.clip(patrol / budget, 0.0, 1.0)
    return base_prob + (max_prob - base_prob) * p_norm


def _simulate_observed(
    y_true: np.ndarray,          # (T, N)
    det_prob: np.ndarray,        # (T, N)
) -> np.ndarray:
    """Expected observed counts = true counts × detection probability."""
    return y_true * det_prob


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class DeploymentSimulator:
    """
    K-cycle deployment feedback simulator.

    Parameters
    ----------
    model         : FASEModel  (will be deep-copied so caller's model is untouched)
    allocator     : PatrolAllocator
    X             : (N, T, F) feature tensor for the deployment period
    y_true        : (N, T)    true crime counts
    A_combined    : (N, N)    weighted row-stochastic adjacency
    A_binary      : (N, N)    binary adjacency (for Hawkes)
    group         : (N,)      binary group labels (1 = minority ZCTA)
    patrol_budget : B
    detection_base_prob : p_base
    detection_max_prob  : p_max
    retrain_epochs : epochs per cycle
    lr, weight_decay, grad_clip : Adam hyperparameters
    window_size   : sliding window length for training mini-batches
    seed          : RNG seed for reproducibility
    device        : torch device string
    """

    def __init__(
        self,
        model: nn.Module,
        allocator: PatrolAllocator,
        X: torch.Tensor,
        y_true: torch.Tensor,
        A_combined: torch.Tensor,
        A_binary: torch.Tensor,
        group: np.ndarray,
        patrol_budget: float = 60.0,
        detection_base_prob: float = 0.30,
        detection_max_prob: float = 0.90,
        retrain_epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
        window_size: int = 64,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model      = copy.deepcopy(model).to(device)
        self.allocator  = allocator
        self.X          = X.to(device)
        self.y_true     = y_true                             # keep on CPU for numpy ops
        self.A_combined = A_combined.to(device)
        self.A_binary   = A_binary.to(device)
        self.group      = np.asarray(group, dtype=int)
        self.B          = patrol_budget
        self.base_prob  = detection_base_prob
        self.max_prob   = detection_max_prob
        self.n_epochs   = retrain_epochs
        self.lr         = lr
        self.wd         = weight_decay
        self.clip       = grad_clip
        self.W          = window_size
        self.device     = device

        # y_true as numpy (T, N) for detection simulation
        self._y_true_np = y_true.T.numpy() if isinstance(y_true, torch.Tensor) \
                          else y_true.T

        self.N = X.shape[0]
        self.T = X.shape[1]

    # ── Inference ─────────────────────────────────────────────────────────────

    def _predict(self) -> np.ndarray:
        """Full-sequence inference. Returns mu (T, N)."""
        self.model.eval()
        x_b = self.X.unsqueeze(0)                       # (1, N, T, F)
        y_b = self.y_true.to(self.device).unsqueeze(0)  # (1, N, T)
        with torch.no_grad():
            mu, _, _ = self.model(x_b, self.A_combined, self.A_binary, y_b)
        return mu.squeeze(0).T.cpu().numpy()             # (T, N)

    # ── Training ──────────────────────────────────────────────────────────────

    def _make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

    def _train_epoch(
        self,
        y_obs_t: torch.Tensor,     # (N, T) observed counts on device
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model.train()
        T     = self.X.shape[1]
        # Non-overlapping windows, shuffled order
        starts = list(range(0, T - self.W + 1, self.W))
        np.random.shuffle(starts)

        total, count = 0.0, 0
        for s in starts:
            e   = s + self.W
            x_w = self.X[:, s:e, :].unsqueeze(0)      # (1, N, W, F)
            y_w = y_obs_t[:, s:e].unsqueeze(0)        # (1, N, W)

            optimizer.zero_grad()
            mu, theta, pi = self.model(x_w, self.A_combined, self.A_binary, y_w)
            loss = zinb_loss(y_w, mu, theta, pi)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            optimizer.step()

            total += loss.item()
            count += 1

        return total / max(count, 1)

    # ── Metrics helpers ───────────────────────────────────────────────────────

    def _allocation_metrics(
        self,
        P: np.ndarray,             # (T, N)
        mu_np: np.ndarray,         # (T, N)
    ) -> Dict[str, float]:
        dirs  = [dir_score(P[t], self.group) for t in range(self.T)]
        ginis = [gini_coefficient(P[t])      for t in range(self.T)]
        covs  = [coverage_score(P[t], mu_np[t], self.B) for t in range(self.T)]
        min_idx = np.where(self.group == 1)[0]
        maj_idx = np.where(self.group == 0)[0]
        return {
            "dir_mean":       float(np.mean(dirs)),
            "dir_std":        float(np.std(dirs)),
            "gini_mean":      float(np.mean(ginis)),
            "coverage_mean":  float(np.mean(covs)),
            "mean_min_patrol":float(P[:, min_idx].mean()),
            "mean_maj_patrol":float(P[:, maj_idx].mean()),
        }

    def _data_bias_metrics(
        self,
        y_obs: np.ndarray,         # (T, N)
        y_true: np.ndarray,        # (T, N)
    ) -> Dict[str, float]:
        """DIR of observed vs true counts — quantifies feedback-induced data bias."""
        min_idx = np.where(self.group == 1)[0]
        maj_idx = np.where(self.group == 0)[0]

        obs_dir_vals  = [dir_score(y_obs[t],  self.group) for t in range(self.T)]
        true_dir_vals = [dir_score(y_true[t], self.group) for t in range(self.T)]

        detection_ratio_min = y_obs[:, min_idx].sum() / (y_true[:, min_idx].sum() + 1e-9)
        detection_ratio_maj = y_obs[:, maj_idx].sum() / (y_true[:, maj_idx].sum() + 1e-9)

        return {
            "obs_dir_mean":          float(np.mean(obs_dir_vals)),
            "true_dir_mean":         float(np.mean(true_dir_vals)),
            "detection_ratio_min":   float(detection_ratio_min),
            "detection_ratio_maj":   float(detection_ratio_maj),
        }

    # ── Single cycle ──────────────────────────────────────────────────────────

    def run_cycle(self, cycle: int) -> Dict:
        """Execute one deployment cycle. Returns a metrics dict."""
        print(f"\n  [Cycle {cycle}] Step 1/4: Inference …", end="", flush=True)
        mu_np = self._predict()                              # (T, N)

        print(f" Step 2/4: Allocation …", end="", flush=True)
        P = self.allocator.solve_batch(mu_np)                # (T, N)

        print(f" Step 3/4: Detection simulation …", end="", flush=True)
        det_prob = _detection_prob(P, self.B, self.base_prob, self.max_prob)
        y_obs_np = _simulate_observed(self._y_true_np, det_prob)  # (T, N)
        y_obs_t  = torch.tensor(y_obs_np.T, dtype=torch.float32, device=self.device)

        print(f" Step 4/4: Retraining ({self.n_epochs} epochs) …", flush=True)
        optimizer  = self._make_optimizer()
        epoch_losses = []
        for _ in range(self.n_epochs):
            epoch_losses.append(self._train_epoch(y_obs_t, optimizer))

        # Collect metrics
        alloc_m = self._allocation_metrics(P, mu_np)
        bias_m  = self._data_bias_metrics(y_obs_np, self._y_true_np)

        return {
            "cycle":          cycle,
            "zinb_loss_mean": float(np.mean(epoch_losses)),
            "zinb_loss_final":float(epoch_losses[-1]),
            **alloc_m,
            **bias_m,
        }

    # ── Full simulation ───────────────────────────────────────────────────────

    def run(self, num_cycles: int = 6) -> List[Dict]:
        """
        Run the full K-cycle feedback loop.

        Returns
        -------
        history : list of num_cycles dicts, one per cycle
        """
        history = []
        for k in range(1, num_cycles + 1):
            metrics = self.run_cycle(k)
            history.append(metrics)
            self._print_cycle_summary(metrics)
        return history

    def _print_cycle_summary(self, m: Dict) -> None:
        print(
            f"    ↳ loss={m['zinb_loss_final']:.4f}  "
            f"DIR={m['dir_mean']:.4f}±{m['dir_std']:.4f}  "
            f"Gini={m['gini_mean']:.4f}  "
            f"coverage={m['coverage_mean']:.4f}  "
            f"obs_DIR={m['obs_dir_mean']:.4f}  "
            f"det_min/maj={m['detection_ratio_min']:.3f}/{m['detection_ratio_maj']:.3f}"
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        model: nn.Module,
        allocator: PatrolAllocator,
        X: torch.Tensor,
        y_true: torch.Tensor,
        A_combined: torch.Tensor,
        A_binary: torch.Tensor,
        group: np.ndarray,
    ) -> "DeploymentSimulator":
        """
        Build from the phase3 + phase5 sections of fase_config.yaml.

        cfg should be the full top-level config dict.
        """
        p5 = cfg.get("phase5", {})
        p3 = cfg.get("phase3", {})
        tr = p3.get("training", {})

        return cls(
            model               = model,
            allocator           = allocator,
            X                   = X,
            y_true              = y_true,
            A_combined          = A_combined,
            A_binary            = A_binary,
            group               = group,
            patrol_budget       = cfg.get("phase4", {}).get("patrol_budget", 60.0),
            detection_base_prob = p5.get("detection_base_prob", 0.30),
            detection_max_prob  = p5.get("detection_max_prob",  0.90),
            retrain_epochs      = p5.get("retrain_epochs", 50),
            lr                  = tr.get("learning_rate", 1e-3),
            weight_decay        = tr.get("weight_decay",  1e-4),
            grad_clip           = tr.get("grad_clip", 5.0),
            seed                = tr.get("seed", 42),
        )

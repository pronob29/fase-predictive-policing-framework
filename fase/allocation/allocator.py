"""
fase/allocation/allocator.py
============================
Fairness-Constrained Patrol Allocation (Phase 4).

Optimization problem (one LP per timestep)
-------------------------------------------
Given:
  mu_v   — predicted crime risk at ZCTA v  (from FASEModel)
  g_v    — group label: 1 if pct_minority ≥ minority_threshold, else 0
  B      — total patrol budget (units, continuous)
  ε      — max allowable Demographic Impact Ratio (DIR) deviation

Solve:
  maximize    Σ_v  mu_v · p_v          (risk-weighted coverage)
  subject to
    Σ_v p_v  ≤  B                      (budget)
    p_v      ≥  0                      (non-negativity)
    p_v      ≤  B                      (per-node cap = whole budget)
    |DIR − 1| ≤  ε                     (fairness)

DIR definition used here (ratio of mean patrol intensities):
    DIR = mean_{v: g_v=1}(p_v)  /  mean_{v: g_v=0}(p_v)

Linearised fairness constraints (both are linear in p):
    mean_min  ≤  (1 + ε) · mean_maj
    mean_min  ≥  (1 − ε) · mean_maj

This is a bounded LP, solved via CVXPY (default solver: CLARABEL or ECOS).

Batch mode
----------
`solve_batch(mu)` loops over a (T, N) risk matrix and returns a (T, N)
allocation matrix.  Each timestep is an independent LP; they share the same
constraint structure so we build the problem once and re-parametrise mu.

Fallback
--------
If the constrained LP is infeasible (epsilon too tight), the solver silently
falls back to the unconstrained proportional allocation:
    p_v = B · mu_v / Σ_v mu_v
"""

from __future__ import annotations

import warnings
from typing import Optional

import cvxpy as cp
import numpy as np


class PatrolAllocator:
    """
    Solves the fairness-constrained patrol allocation LP.

    Parameters
    ----------
    num_nodes         : N — number of ZCTA nodes
    group             : (N,) int array — 1 = minority ZCTA, 0 = majority
    patrol_budget     : B — total patrol units per timestep
    epsilon           : ε — max |DIR - 1| allowed
    per_node_cap      : optional hard cap per ZCTA (default = patrol_budget)
    solver            : CVXPY solver string; None = CVXPY default
    """

    def __init__(
        self,
        num_nodes: int,
        group: np.ndarray,
        patrol_budget: float = 60.0,
        epsilon: float = 0.05,
        per_node_cap: Optional[float] = None,
        solver: Optional[str] = None,
    ) -> None:
        self.N      = num_nodes
        self.group  = np.asarray(group, dtype=int)
        self.B      = float(patrol_budget)
        self.eps    = float(epsilon)
        self.cap    = float(per_node_cap) if per_node_cap is not None else self.B
        self.solver = solver

        self._min_idx = np.where(self.group == 1)[0]
        self._maj_idx = np.where(self.group == 0)[0]
        self._n_min   = len(self._min_idx)
        self._n_maj   = len(self._maj_idx)

        # Build parametric CVXPY problem once; reuse across timesteps
        self._p   = cp.Variable(self.N, nonneg=True)
        self._mu_param = cp.Parameter(self.N, nonneg=False)  # risk vector
        self._problem  = self._build_problem()

    # ── LP construction ───────────────────────────────────────────────────────

    def _build_problem(self) -> cp.Problem:
        p, mu = self._p, self._mu_param

        constraints = [
            cp.sum(p) <= self.B,
            p <= self.cap,
        ]

        # Fairness only makes sense when both groups exist
        if self._n_min > 0 and self._n_maj > 0:
            mean_min = cp.sum(p[self._min_idx]) / self._n_min
            mean_maj = cp.sum(p[self._maj_idx]) / self._n_maj
            constraints += [
                mean_min <= (1.0 + self.eps) * mean_maj,
                mean_min >= (1.0 - self.eps) * mean_maj,
            ]

        objective = cp.Maximize(mu @ p)
        return cp.Problem(objective, constraints)

    # ── Single-timestep solve ─────────────────────────────────────────────────

    def solve(self, mu: np.ndarray) -> np.ndarray:
        """
        Allocate patrols for one timestep.

        Parameters
        ----------
        mu : (N,) float array — predicted risk scores (must be non-negative)

        Returns
        -------
        p  : (N,) float array — patrol allocation per ZCTA
        """
        mu = np.asarray(mu, dtype=float)
        mu = np.clip(mu, 0, None)          # ensure non-negative

        self._mu_param.value = mu

        try:
            kwargs = dict(warm_start=True)
            if self.solver:
                kwargs["solver"] = self.solver
            self._problem.solve(**kwargs)
        except cp.SolverError:
            pass

        if self._p.value is not None and self._problem.status in (
            cp.OPTIMAL, cp.OPTIMAL_INACCURATE
        ):
            return np.clip(self._p.value, 0, None)

        # Fallback: proportional allocation (unconstrained)
        warnings.warn(
            f"LP status='{self._problem.status}' — falling back to "
            "proportional allocation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return self._proportional(mu)

    def _proportional(self, mu: np.ndarray) -> np.ndarray:
        """Risk-proportional allocation (no fairness constraint)."""
        total = mu.sum()
        if total <= 0:
            return np.full(self.N, self.B / self.N)
        return (mu / total) * self.B

    # ── Batch solve ───────────────────────────────────────────────────────────

    def solve_batch(self, mu: np.ndarray) -> np.ndarray:
        """
        Allocate patrols for every timestep in a (T, N) risk matrix.

        Parameters
        ----------
        mu : (T, N) float array

        Returns
        -------
        P  : (T, N) float array — allocation per (timestep, ZCTA)
        """
        mu = np.asarray(mu, dtype=float)
        T  = mu.shape[0]
        P  = np.zeros((T, self.N), dtype=float)
        for t in range(T):
            P[t] = self.solve(mu[t])
        return P

    # ── Class method: build from config dict ─────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        group: np.ndarray,
    ) -> "PatrolAllocator":
        """
        Instantiate from the phase4 section of fase_config.yaml.

        Parameters
        ----------
        cfg   : full config dict (top-level or just the phase4 sub-dict)
        group : (N,) binary group array (derived from ACS pct_minority)
        """
        p4 = cfg.get("phase4", cfg)
        return cls(
            num_nodes     = len(group),
            group         = group,
            patrol_budget = p4.get("patrol_budget", 60),
            epsilon       = p4.get("epsilon", 0.05),
        )

"""
fase/allocation/metrics.py
==========================
Fairness and efficiency metrics for patrol allocation evaluation.

Metrics
-------
dir_score          : Demographic Impact Ratio (minority vs majority mean patrol)
gini_coefficient   : Gini inequality index of the allocation vector
coverage_score     : risk-weighted coverage = Σ mu_v · p_v / (B · max(mu))
allocation_metrics : convenience wrapper returning all metrics as a dict
compare_allocations: side-by-side dict for constrained vs unconstrained
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def dir_score(
    p: np.ndarray,
    group: np.ndarray,
    smoothing: float = 1e-6,
) -> float:
    """
    Demographic Impact Ratio.

    DIR = mean_patrol(minority ZCTAs) / mean_patrol(majority ZCTAs)

    DIR = 1   → perfectly equal mean allocation
    DIR > 1   → minority ZCTAs receive more patrol (over-policed)
    DIR < 1   → minority ZCTAs receive less patrol (under-policed)

    Parameters
    ----------
    p         : (N,) allocation vector
    group     : (N,) binary array (1 = minority)
    smoothing : additive smoothing to prevent division by zero

    Returns
    -------
    float
    """
    p     = np.asarray(p, dtype=float)
    group = np.asarray(group, dtype=int)

    min_idx = np.where(group == 1)[0]
    maj_idx = np.where(group == 0)[0]

    mean_min = p[min_idx].mean() if len(min_idx) > 0 else 0.0
    mean_maj = p[maj_idx].mean() if len(maj_idx) > 0 else 0.0

    return (mean_min + smoothing) / (mean_maj + smoothing)


def gini_coefficient(p: np.ndarray) -> float:
    """
    Gini coefficient of a non-negative allocation vector.

    Returns 0 (perfect equality) to 1 (all resources to one node).
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0, None)
    if p.sum() == 0:
        return 0.0
    p_sorted = np.sort(p)
    N        = len(p_sorted)
    idx      = np.arange(1, N + 1)
    return float((2 * (idx * p_sorted).sum() - (N + 1) * p_sorted.sum()) /
                 (N * p_sorted.sum()))


def coverage_score(
    p: np.ndarray,
    mu: np.ndarray,
    budget: float,
) -> float:
    """
    Risk-weighted coverage: fraction of maximum achievable coverage attained.

      coverage = (Σ mu_v · p_v) / (B · max(mu_v))

    Returns a value in [0, 1].  A perfectly risk-proportional allocation of
    the full budget at the highest-risk node gives coverage = 1.
    """
    p   = np.asarray(p,  dtype=float)
    mu  = np.asarray(mu, dtype=float)
    denom = budget * mu.max()
    if denom <= 0:
        return 0.0
    return float(np.dot(mu, p) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def allocation_metrics(
    p: np.ndarray,
    mu: np.ndarray,
    group: np.ndarray,
    budget: float,
    smoothing: float = 1e-6,
) -> dict:
    """
    Return all metrics for a single allocation vector.

    Returns
    -------
    dict with keys: dir, gini, coverage, total_patrol,
                    mean_minority, mean_majority
    """
    p     = np.asarray(p, dtype=float)
    group = np.asarray(group, dtype=int)

    min_idx = np.where(group == 1)[0]
    maj_idx = np.where(group == 0)[0]

    return {
        "dir":           dir_score(p, group, smoothing),
        "gini":          gini_coefficient(p),
        "coverage":      coverage_score(p, mu, budget),
        "total_patrol":  float(p.sum()),
        "mean_minority": float(p[min_idx].mean()) if len(min_idx) > 0 else 0.0,
        "mean_majority": float(p[maj_idx].mean()) if len(maj_idx) > 0 else 0.0,
    }


def compare_allocations(
    p_constrained: np.ndarray,
    p_baseline: np.ndarray,
    mu: np.ndarray,
    group: np.ndarray,
    budget: float,
    smoothing: float = 1e-6,
) -> dict:
    """
    Side-by-side comparison of constrained vs baseline (proportional) allocation.

    Returns
    -------
    dict with keys 'constrained' and 'baseline', each containing the
    metrics dict from allocation_metrics().  Also includes 'delta' showing
    the difference for each metric.
    """
    m_c = allocation_metrics(p_constrained, mu, group, budget, smoothing)
    m_b = allocation_metrics(p_baseline,    mu, group, budget, smoothing)

    delta = {k: m_c[k] - m_b[k] for k in m_c}

    return {
        "constrained": m_c,
        "baseline":    m_b,
        "delta":       delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_metrics(
    P: np.ndarray,
    Mu: np.ndarray,
    group: np.ndarray,
    budget: float,
    smoothing: float = 1e-6,
) -> dict:
    """
    Compute mean metrics over a batch of (T, N) allocation + risk matrices.

    Returns
    -------
    dict with mean and std of each metric across T timesteps.
    """
    T = P.shape[0]
    rows = [allocation_metrics(P[t], Mu[t], group, budget, smoothing)
            for t in range(T)]

    keys = rows[0].keys()
    return {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "std":  float(np.std( [r[k] for r in rows])),
        }
        for k in keys
    }

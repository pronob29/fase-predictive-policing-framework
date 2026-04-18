"""
fase/models/losses.py
=====================
Zero-Inflated Negative Binomial (ZINB) negative log-likelihood.

Crime incident counts are highly zero-inflated (≈83% zeros at 1-hour / ZCTA
granularity) and overdispersed, so standard Poisson or MSE losses badly
underfit.  ZINB handles both properties with three parameters:

    mu    (float, > 0)  — NB mean (= expected count)
    theta (float, > 0)  — NB dispersion; larger ⟹ less overdispersion
                          (theta → ∞  recovers Poisson)
    pi    (float, 0–1)  — probability of a structural zero (non-event area)

NB parametrisation used here  (mean / dispersion):
    P(Y = k) = Γ(k+θ) / [Γ(θ) k!]  ·  (θ/(μ+θ))^θ  ·  (μ/(μ+θ))^k

ZINB log-likelihood:
    y = 0:   log[ π + (1−π) · (θ/(μ+θ))^θ ]
    y > 0:   log(1−π) + lgamma(y+θ) − lgamma(θ) − lgamma(y+1)
             + θ·log(θ/(μ+θ)) + y·log(μ/(μ+θ))
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def zinb_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Per-element ZINB negative log-likelihood.

    Parameters
    ----------
    y     : observed counts, shape (*)  — float32 or int cast to float
    mu    : NB mean,         shape (*)  — strictly positive
    theta : NB dispersion,   shape (*)  — strictly positive
    pi    : zero-gate,       shape (*)  — in (0, 1)
    eps   : numerical floor applied to log arguments

    Returns
    -------
    nll : tensor of same shape as y, dtype float32
    """
    y     = y.float()
    mu    = mu.float().clamp(min=eps)
    theta = theta.float().clamp(min=eps)
    pi    = pi.float().clamp(min=eps, max=1.0 - eps)

    # ── Shared NB log-probabilities ───────────────────────────────────────────
    # log( θ / (μ+θ) )  and  log( μ / (μ+θ) )
    log_theta_over_sum = torch.log(theta) - torch.log(theta + mu)   # log(1-p)
    log_mu_over_sum    = torch.log(mu)    - torch.log(theta + mu)   # log(p)

    # log P_NB(y=0) = θ · log(θ/(μ+θ))
    log_nb_zero = theta * log_theta_over_sum

    # log P_NB(y=k) for k ≥ 0
    log_nb_y = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * log_theta_over_sum
        + y     * log_mu_over_sum
    )

    # ── ZINB log-probabilities ────────────────────────────────────────────────
    log_pi     = torch.log(pi)
    log_1_pi   = torch.log(1.0 - pi)

    # y = 0:  log( π + (1−π)·P_NB(0) )
    #   stable form via logaddexp:  log(e^a + e^b) where
    #     a = log π,   b = log(1−π) + log P_NB(0)
    log_p0 = torch.logaddexp(log_pi, log_1_pi + log_nb_zero)

    # y > 0:  log(1−π) + log P_NB(y)
    log_py = log_1_pi + log_nb_y

    # Select branch and negate (we minimise NLL)
    nll = -torch.where(y == 0, log_p0, log_py)
    return nll


def zinb_loss(
    y: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Scalar ZINB loss (mean or sum of per-element NLL).

    Parameters
    ----------
    reduction : 'mean' (default) | 'sum' | 'none'
    """
    nll = zinb_nll(y, mu, theta, pi, eps)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll

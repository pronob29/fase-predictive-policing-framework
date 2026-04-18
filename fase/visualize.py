"""
fase/visualize.py
=================
Generate all 7 paper-quality figures from training and simulation results.

Usage
-----
    python -m fase.visualize
    python -m fase.visualize --results-dir results/fase --out-dir results/fase/figures
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")          # headless — no display needed on HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "lines.linewidth":   2.0,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

BLUE   = "#2E86AB"
ORANGE = "#E84855"
GREEN  = "#3BB273"
PURPLE = "#7B2D8B"
GREY   = "#888888"


def save(fig: plt.Figure, out_dir: pathlib.Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  saved → {out_dir}/{name}.{{pdf,png}}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 · Training & Validation ZINB Loss Curve
# ─────────────────────────────────────────────────────────────────────────────
def fig1_loss_curve(log: pd.DataFrame, out_dir: pathlib.Path) -> None:
    best_epoch = log.loc[log["val_loss"].idxmin()]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log["epoch"], log["train_loss"], color=BLUE,   label="Train loss")
    ax.plot(log["epoch"], log["val_loss"],   color=ORANGE, label="Val loss")
    ax.axvline(best_epoch["epoch"], color=GREEN, linestyle=":", linewidth=1.5,
               label=f"Best checkpoint (epoch {int(best_epoch['epoch'])})")
    ax.scatter([best_epoch["epoch"]], [best_epoch["val_loss"]],
               color=GREEN, zorder=5, s=60)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("ZINB NLL Loss")
    ax.set_title("Fig 1 · Training & Validation ZINB Loss")
    ax.legend(framealpha=0.9)
    ax.grid(True)
    ax.set_xlim(1, log["epoch"].max())

    # Annotation
    ax.annotate(
        f"Best val = {best_epoch['val_loss']:.4f}",
        xy=(best_epoch["epoch"], best_epoch["val_loss"]),
        xytext=(best_epoch["epoch"] + 5, best_epoch["val_loss"] + 0.004),
        fontsize=9, color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
    )
    fig.tight_layout()
    save(fig, out_dir, "fig1_loss_curve")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 · Generalisation Gap (val − train)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_loss_gap(log: pd.DataFrame, out_dir: pathlib.Path) -> None:
    gap = log["val_loss"] - log["train_loss"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(log["epoch"], gap, alpha=0.25, color=PURPLE)
    ax.plot(log["epoch"], gap, color=PURPLE, label="Val − Train loss")
    ax.axhline(0, color=GREY, linewidth=1.0, linestyle="-")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Generalisation Gap (Val − Train)")
    ax.set_title("Fig 2 · Generalisation Gap (Overfitting Monitor)")
    ax.legend(framealpha=0.9)
    ax.grid(True)
    ax.set_xlim(1, log["epoch"].max())
    fig.tight_layout()
    save(fig, out_dir, "fig2_loss_gap")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 · DIR per Cycle with ε-band
# ─────────────────────────────────────────────────────────────────────────────
def fig3_dir_cycles(sim: pd.DataFrame, out_dir: pathlib.Path, epsilon: float = 0.05) -> None:
    cycles = sim["cycle"]

    fig, ax = plt.subplots(figsize=(7, 4))
    # Fairness band
    ax.axhspan(1 - epsilon, 1 + epsilon, alpha=0.12, color=GREEN,
               label=f"Fair zone |DIR−1| ≤ {epsilon}")
    ax.axhline(1.0, color=GREEN, linestyle="-", linewidth=1.2, alpha=0.6)

    ax.errorbar(
        cycles, sim["dir_mean"], yerr=sim["dir_std"],
        fmt="o-", color=BLUE, capsize=5, capthick=1.5,
        label="DIR mean ± std",
    )
    ax.set_xlabel("Deployment Cycle")
    ax.set_ylabel("Demographic Impact Ratio (DIR)")
    ax.set_title("Fig 3 · Allocation Fairness (DIR) across Feedback Cycles")
    ax.set_xticks(cycles)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y")

    # Value labels
    for _, row in sim.iterrows():
        ax.annotate(f"{row['dir_mean']:.3f}",
                    xy=(row["cycle"], row["dir_mean"] + row["dir_std"] + 0.003),
                    ha="center", fontsize=8.5, color=BLUE)
    fig.tight_layout()
    save(fig, out_dir, "fig3_dir_cycles")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 · Mean Patrol — Minority vs Majority per Cycle
# ─────────────────────────────────────────────────────────────────────────────
def fig4_patrol_bars(sim: pd.DataFrame, out_dir: pathlib.Path) -> None:
    cycles = np.array(sim["cycle"])
    w = 0.35
    x = np.arange(len(cycles))

    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, sim["mean_min_patrol"], w, color=ORANGE, alpha=0.85,
                label="Minority ZCTAs")
    b2 = ax.bar(x + w/2, sim["mean_maj_patrol"], w, color=BLUE,   alpha=0.85,
                label="Majority ZCTAs")

    ax.set_xlabel("Deployment Cycle")
    ax.set_ylabel("Mean Patrol Units per ZCTA")
    ax.set_title("Fig 4 · Patrol Allocation: Minority vs Majority ZCTAs")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cycle {c}" for c in cycles])
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y")

    # Value labels on bars
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save(fig, out_dir, "fig4_patrol_bars")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 · Detection Rate Disparity per Cycle
# ─────────────────────────────────────────────────────────────────────────────
def fig5_detection_disparity(sim: pd.DataFrame, out_dir: pathlib.Path) -> None:
    cycles = np.array(sim["cycle"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cycles, sim["detection_ratio_min"], "o-", color=ORANGE,
            label="Minority detection rate")
    ax.plot(cycles, sim["detection_ratio_maj"], "s-", color=BLUE,
            label="Majority detection rate")

    ax.fill_between(cycles,
                    sim["detection_ratio_min"], sim["detection_ratio_maj"],
                    alpha=0.12, color=GREY, label="Disparity gap")

    ax.set_xlabel("Deployment Cycle")
    ax.set_ylabel("Detection Rate  (observed / true)")
    ax.set_title("Fig 5 · Patrol-Induced Detection Rate Disparity")
    ax.set_xticks(cycles)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # Disparity annotation on last cycle
    last = sim.iloc[-1]
    gap  = last["detection_ratio_maj"] - last["detection_ratio_min"]
    ax.annotate(
        f"Gap = {gap:.3f}",
        xy=(last["cycle"], (last["detection_ratio_min"] + last["detection_ratio_maj"]) / 2),
        xytext=(last["cycle"] - 0.5, (last["detection_ratio_min"] + last["detection_ratio_maj"]) / 2),
        fontsize=9, color=GREY,
        arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0),
    )
    fig.tight_layout()
    save(fig, out_dir, "fig5_detection_disparity")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 · Coverage & Gini per Cycle (dual axis)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_coverage_gini(sim: pd.DataFrame, out_dir: pathlib.Path) -> None:
    cycles = sim["cycle"]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    l1, = ax1.plot(cycles, sim["coverage_mean"], "o-", color=GREEN,
                   label="Risk-weighted coverage")
    l2, = ax2.plot(cycles, sim["gini_mean"],     "s--", color=PURPLE,
                   label="Gini inequality index")

    ax1.set_xlabel("Deployment Cycle")
    ax1.set_ylabel("Risk-Weighted Coverage", color=GREEN)
    ax2.set_ylabel("Gini Index", color=PURPLE)
    ax1.set_title("Fig 6 · Coverage vs Inequality across Feedback Cycles")
    ax1.set_xticks(cycles)
    ax1.tick_params(axis="y", labelcolor=GREEN)
    ax2.tick_params(axis="y", labelcolor=PURPLE)
    ax1.grid(True, axis="y", alpha=0.3)

    ax1.set_ylim(0.85, 1.00)
    ax2.set_ylim(0.90, 0.95)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, framealpha=0.9, loc="lower left")
    fig.tight_layout()
    save(fig, out_dir, "fig6_coverage_gini")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 · ZINB Retraining Loss per Cycle
# ─────────────────────────────────────────────────────────────────────────────
def fig7_retrain_loss(sim: pd.DataFrame, out_dir: pathlib.Path) -> None:
    cycles = sim["cycle"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cycles, sim["zinb_loss_mean"],  "o-", color=BLUE,
            label="Mean loss (all retrain epochs)")
    ax.plot(cycles, sim["zinb_loss_final"], "s--", color=ORANGE,
            label="Final epoch loss")
    ax.fill_between(cycles,
                    sim["zinb_loss_mean"] - (sim["zinb_loss_mean"] - sim["zinb_loss_final"]).abs(),
                    sim["zinb_loss_mean"] + (sim["zinb_loss_mean"] - sim["zinb_loss_final"]).abs(),
                    alpha=0.10, color=BLUE)

    ax.set_xlabel("Deployment Cycle")
    ax.set_ylabel("ZINB NLL Loss")
    ax.set_title("Fig 7 · Retraining Loss Stability across Feedback Cycles")
    ax.set_xticks(cycles)
    ax.legend(framealpha=0.9)
    ax.grid(True, axis="y")

    # Annotate mean ± range
    for _, row in sim.iterrows():
        ax.annotate(f"{row['zinb_loss_final']:.4f}",
                    xy=(row["cycle"], row["zinb_loss_final"] - 0.0008),
                    ha="center", fontsize=8, color=ORANGE)
    fig.tight_layout()
    save(fig, out_dir, "fig7_retrain_loss")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FASE result figures")
    parser.add_argument("--results-dir", default="results/fase")
    parser.add_argument("--out-dir",     default="results/fase/figures")
    args = parser.parse_args()

    res_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)

    # Load
    log_path = res_dir / "training_log.csv"
    sim_path = res_dir / "simulation_history.csv"

    if not log_path.exists():
        raise FileNotFoundError(f"Not found: {log_path}")
    if not sim_path.exists():
        raise FileNotFoundError(f"Not found: {sim_path}")

    log = pd.read_csv(log_path)
    sim = pd.read_csv(sim_path)

    print(f"Training log : {len(log)} epochs")
    print(f"Simulation   : {len(sim)} cycles")
    print(f"Output dir   : {out_dir}")
    print()

    fig1_loss_curve(log, out_dir)
    fig2_loss_gap(log, out_dir)
    fig3_dir_cycles(sim, out_dir)
    fig4_patrol_bars(sim, out_dir)
    fig5_detection_disparity(sim, out_dir)
    fig6_coverage_gini(sim, out_dir)
    fig7_retrain_loss(sim, out_dir)

    print(f"\nAll 7 figures saved to {out_dir}/")
    print("Each figure has .pdf (300 dpi, paper) and .png (150 dpi, slides) versions.")


if __name__ == "__main__":
    main()

"""
fase/simulate.py
================
Production Phase 5 deployment feedback simulator.

Loads a trained FASEModel checkpoint, runs K deployment cycles on the test
set, and saves per-cycle metrics to CSV.

Usage
-----
    python -m fase.simulate                         # uses best_model.pt
    python -m fase.simulate --checkpoint checkpoints/fase/best_model.pt
    python -m fase.simulate --cycles 6 --retrain-epochs 50
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import pickle

import numpy as np
import torch
import yaml

from fase.models import FASEModel
from fase.allocation import PatrolAllocator
from fase.simulation import DeploymentSimulator


# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU  : {prop.name}  |  VRAM {prop.total_memory / 1e9:.1f} GB")
        return torch.device("cuda")
    print("WARNING: CUDA not available — running on CPU")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="FASE Phase 5 deployment simulator")
    parser.add_argument("--config",          default="configs/fase_config.yaml")
    parser.add_argument("--checkpoint",      default="checkpoints/fase/best_model.pt")
    parser.add_argument("--results-dir",     default="results/fase")
    parser.add_argument("--cycles",          type=int,   default=None, help="override config")
    parser.add_argument("--retrain-epochs",  type=int,   default=None, help="override config")
    parser.add_argument("--window-size",     type=int,   default=64)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    # ── Load artefacts ─────────────────────────────────────────────────────────
    tensor_dir = pathlib.Path("data/tensors/baltimore")
    print("\nLoading tensors and graph …")

    with open(tensor_dir / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(tensor_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)

    X_test = torch.load(tensor_dir / "X_test.pt", weights_only=True)   # (N, T, F)
    y_test = torch.load(tensor_dir / "y_test.pt", weights_only=True)   # (N, T)
    N, T, F = X_test.shape
    print(f"  X_test : (N={N}, T={T}, F={F})")

    # Group labels from ACS
    node_ids   = meta["node_ids"]
    pct_min    = (
        meta["acs"]
        .set_index("ZCTA5")
        .reindex(node_ids)["pct_minority"]
        .fillna(0)
        .values
    )
    threshold = cfg["phase4"]["minority_threshold"]
    group     = (pct_min >= threshold).astype(int)
    print(f"  Minority ZCTAs (≥{threshold} pct_minority): {group.sum()} / {N}")

    # Graph tensors on device
    A_combined = torch.tensor(graph["A_combined"], dtype=torch.float32, device=device)
    A_binary   = torch.tensor(graph["A_geo"],      dtype=torch.float32, device=device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run 'python -m fase.train' first."
        )

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FASEModel.from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded from epoch {ckpt['epoch']}  |  val_loss={ckpt['val_loss']:.4f}")

    # ── Build allocator ───────────────────────────────────────────────────────
    p4 = cfg["phase4"]
    allocator = PatrolAllocator(
        num_nodes     = N,
        group         = group,
        patrol_budget = p4["patrol_budget"],
        epsilon       = p4["epsilon"],
    )

    # ── Simulation hyperparameters ─────────────────────────────────────────────
    p5      = cfg["phase5"]
    p3_tr   = cfg["phase3"]["training"]
    cycles  = args.cycles         or p5.get("deploy_cycles",    6)
    epochs  = args.retrain_epochs or p5.get("retrain_epochs",  50)

    print(
        f"\nSimulation : {cycles} cycles  |  {epochs} retrain epochs/cycle"
        f"  |  window={args.window_size}"
    )

    # ── Run simulator ─────────────────────────────────────────────────────────
    sim = DeploymentSimulator(
        model               = model,
        allocator           = allocator,
        X                   = X_test.to(device),
        y_true              = y_test,
        A_combined          = A_combined,
        A_binary            = A_binary,
        group               = group,
        patrol_budget       = p4["patrol_budget"],
        detection_base_prob = p5["detection_base_prob"],
        detection_max_prob  = p5["detection_max_prob"],
        retrain_epochs      = epochs,
        lr                  = p3_tr.get("learning_rate", 1e-3),
        weight_decay        = p3_tr.get("weight_decay",  1e-4),
        grad_clip           = p3_tr.get("grad_clip",     5.0),
        window_size         = args.window_size,
        seed                = p3_tr.get("seed", 42),
        device              = str(device),
    )

    history = sim.run(num_cycles=cycles)

    # ── Save results ──────────────────────────────────────────────────────────
    res_dir = pathlib.Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    out_csv = res_dir / "simulation_history.csv"

    fieldnames = list(history[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"\nSimulation history saved → {out_csv}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Final summary ───────────────────────────────────────────────────────")
    print(f"{'Cycle':>5}  {'ZINB':>8}  {'DIR':>7}  {'Gini':>7}  "
          f"{'Cover':>7}  {'obs_DIR':>8}  {'det_min':>8}  {'det_maj':>8}")
    print("─" * 70)
    for r in history:
        print(
            f"  {r['cycle']:>3}  {r['zinb_loss_final']:>8.4f}  "
            f"{r['dir_mean']:>7.4f}  {r['gini_mean']:>7.4f}  "
            f"{r['coverage_mean']:>7.4f}  {r['obs_dir_mean']:>8.4f}  "
            f"{r['detection_ratio_min']:>8.4f}  {r['detection_ratio_maj']:>8.4f}"
        )


if __name__ == "__main__":
    main()

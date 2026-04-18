"""
fase/train.py
=============
Production GPU training for the FASE dual-engine model (Phase 3).

Usage
-----
    python -m fase.train                          # uses configs/fase_config.yaml
    python -m fase.train --config path/to/cfg.yaml --epochs 200
    python -m fase.train --checkpoint-dir checkpoints/run2 --window-size 64

Outputs
-------
    checkpoints/<run>/best_model.pt    — checkpoint with lowest val ZINB loss
    checkpoints/<run>/final_model.pt   — weights after last epoch
    results/fase/training_log.csv      — epoch, train_loss, val_loss, lr
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import yaml

from fase.models import FASEModel
from fase.models.losses import zinb_loss


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """Slice (N, T, F) and (N, T) tensors into non-overlapping time windows."""

    def __init__(
        self,
        X: torch.Tensor,     # (N, T, F)
        y: torch.Tensor,     # (N, T)
        window_size: int,
    ) -> None:
        N, T, F = X.shape
        self.xs, self.ys = [], []
        for s in range(0, T - window_size + 1, window_size):
            self.xs.append(X[:, s : s + window_size, :])
            self.ys.append(y[:, s : s + window_size])

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int):
        return self.xs[idx], self.ys[idx]


def collate_windows(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)  # (B, N, W, F), (B, N, W)


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    A_combined: torch.Tensor,
    A_binary: torch.Tensor,
    optimizer: optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> float:
    model.train()
    total, count = 0.0, 0
    for x_b, y_b in loader:
        x_b = x_b.to(device, non_blocking=True)   # (B, N, W, F)
        y_b = y_b.to(device, non_blocking=True)   # (B, N, W)
        optimizer.zero_grad()
        mu, theta, pi = model(x_b, A_combined, A_binary, y_b)
        loss = zinb_loss(y_b, mu, theta, pi)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    A_combined: torch.Tensor,
    A_binary: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for x_b, y_b in loader:
        x_b = x_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)
        mu, theta, pi = model(x_b, A_combined, A_binary, y_b)
        loss = zinb_loss(y_b, mu, theta, pi)
        total += loss.item()
        count += 1
    return total / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev  = torch.device("cuda")
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU  : {prop.name}  |  VRAM {prop.total_memory / 1e9:.1f} GB")
        print(f"CUDA : {torch.version.cuda}  |  cuDNN {torch.backends.cudnn.version()}")
    else:
        dev = torch.device("cpu")
        print("WARNING: CUDA not available — running on CPU (performance will be poor)")
    return dev


def main() -> None:
    parser = argparse.ArgumentParser(description="FASE Phase 3 GPU training")
    parser.add_argument("--config",         default="configs/fase_config.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/fase")
    parser.add_argument("--results-dir",    default="results/fase")
    parser.add_argument("--epochs",         type=int,   default=None,  help="override config")
    parser.add_argument("--window-size",    type=int,   default=None,  help="time steps per window")
    parser.add_argument("--batch-size",     type=int,   default=16,    help="windows per gradient step")
    parser.add_argument("--lr",             type=float, default=None,  help="override config lr")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    torch.backends.cudnn.benchmark = True       # auto-tune convolutions

    # ── Load data ─────────────────────────────────────────────────────────────
    tensor_dir = pathlib.Path("data/tensors/baltimore")
    print("\nLoading tensors …")
    X_train = torch.load(tensor_dir / "X_train.pt", weights_only=True)   # (N, T, F)
    y_train = torch.load(tensor_dir / "y_train.pt", weights_only=True)
    X_val   = torch.load(tensor_dir / "X_val.pt",   weights_only=True)
    y_val   = torch.load(tensor_dir / "y_val.pt",   weights_only=True)
    X_test  = torch.load(tensor_dir / "X_test.pt",  weights_only=True)
    y_test  = torch.load(tensor_dir / "y_test.pt",  weights_only=True)

    with open(tensor_dir / "graph.pkl", "rb") as f:
        graph = pickle.load(f)

    A_combined = torch.tensor(graph["A_combined"], dtype=torch.float32, device=device)
    A_binary   = torch.tensor(graph["A_geo"],      dtype=torch.float32, device=device)

    N, T_train, F = X_train.shape
    print(f"  Train : (N={N}, T={T_train}, F={F})"
          f"  Val : T={X_val.shape[1]}  Test : T={X_test.shape[1]}")

    # ── Hyperparameters from config ───────────────────────────────────────────
    p3  = cfg["phase3"]
    tr  = p3["training"]
    W   = args.window_size  or tr.get("batch_size",     32)
    ep  = args.epochs       or tr.get("epochs",         100)
    lr  = args.lr           or tr.get("learning_rate",  1e-3)
    wd  =                      tr.get("weight_decay",   1e-4)
    gc  =                      tr.get("grad_clip",      5.0)
    seed=                      tr.get("seed",           42)

    torch.manual_seed(seed)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = WindowDataset(X_train, y_train, W)
    val_ds   = WindowDataset(X_val,   y_val,   W)
    test_ds  = WindowDataset(X_test,  y_test,  W)

    loader_kw = dict(
        collate_fn  = collate_windows,
        num_workers = 2,
        pin_memory  = (device.type == "cuda"),
        persistent_workers = True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, **loader_kw)

    print(f"  Windows — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}"
          f"  (W={W}, batch={args.batch_size})")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FASEModel.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nFASEModel  {n_params:,} parameters  →  {device}\n")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-6
    )

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    res_dir  = pathlib.Path(args.results_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val   = float("inf")
    log_rows   = []
    header_fmt = f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}  {'s/ep':>7}  {'Status'}"
    print(header_fmt)
    print("─" * len(header_fmt))

    for epoch in range(1, ep + 1):
        t0         = time.perf_counter()
        train_loss = train_epoch(model, train_loader, A_combined, A_binary, optimizer, gc, device)
        val_loss   = evaluate(model, val_loader, A_combined, A_binary, device)
        scheduler.step(val_loss)
        elapsed    = time.perf_counter() - t0
        cur_lr     = optimizer.param_groups[0]["lr"]

        status = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch":           epoch,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss":        val_loss,
                    "cfg":             cfg,
                },
                ckpt_dir / "best_model.pt",
            )
            status = "✓ best"

        log_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": cur_lr}
        )

        if epoch % 10 == 0 or epoch == 1 or status:
            print(f"{epoch:6d}  {train_loss:10.4f}  {val_loss:10.4f}"
                  f"  {cur_lr:10.2e}  {elapsed:7.2f}  {status}")

    # ── Final checkpoint and test evaluation ──────────────────────────────────
    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    test_loss = evaluate(model, test_loader, A_combined, A_binary, device)
    print(f"\nTest ZINB loss : {test_loss:.4f}")
    print(f"Best val loss  : {best_val:.4f}   →  {ckpt_dir}/best_model.pt")

    with open(res_dir / "training_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Training log   : {res_dir}/training_log.csv")


if __name__ == "__main__":
    main()

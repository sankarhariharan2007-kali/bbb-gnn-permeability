"""
Priority 2: train the edge-aware ablation grid -- gine, gat_edge, sage_edge --
across both datasets and all three seeds, using the identical protocol as
src/train.py (same split, pos_weight, early stopping, Youden's-J threshold).

Usage:
    python -m src.train_edge --model gat_edge --dataset bbbp --seed 0
    python -m src.train_edge --run-all                     # all 18 runs

Every model here consumes data.edge_attr, which src/featurize.py already
computes but the base four models (models.py) never touch -- this is
`ABLATION_MODELS` in models_gine.py generalized from one operator to all
three that can use edge features. Results land in
results/edge_ablation/edge_ablation_runs.csv, in the same schema as the
existing gine_ablation_results.csv, so `gine` rows here can be checked
against (and, once re-run under this script, replace) the earlier ones.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .datasets import ROOT, load_dataset
from .evaluate import best_threshold, compute_metrics
from .models_edge import EDGE_MODELS, build_edge_model, count_parameters
from .split import check_split, scaffold_split

RESULTS_DIR = ROOT / "results" / "edge_ablation"
SEEDS = [0, 1, 2]
DATASET_NAMES = ["bbbp", "b3db"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        probs.append(torch.sigmoid(model(batch)).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train_edge_one(
    model_name: str,
    dataset_name: str,
    seed: int = 0,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 64,
    hidden: int = 128,
    num_layers: int = 3,
    dropout: float = 0.3,
    heads: int = 4,
    device: str = "cpu",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = torch.device(device)

    graphs = load_dataset(dataset_name, verbose=False)
    smiles = [g.smiles for g in graphs]
    labels = [int(g.y.item()) for g in graphs]

    train_idx, val_idx, test_idx = scaffold_split(smiles, seed=seed, verbose=False)
    check_split(smiles, train_idx, val_idx, test_idx, labels)

    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]
    test_set = [graphs[i] for i in test_idx]

    drop_last = len(train_set) % batch_size == 1
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    model = build_edge_model(model_name, hidden=hidden, num_layers=num_layers,
                              dropout=dropout, heads=heads).to(dev)

    n_pos = sum(labels[i] for i in train_idx)
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float, device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    best_val_auc, best_state, best_epoch, since_improve = -1.0, None, -1, 0
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(train_loader.dataset)

        val_prob, val_y = predict(model, val_loader, dev)
        from sklearn.metrics import roc_auc_score
        val_auc = float(roc_auc_score(val_y, val_prob))
        scheduler.step(val_auc)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc,
                        "lr": optimizer.param_groups[0]["lr"]})

        if val_auc > best_val_auc:
            best_val_auc, best_epoch, since_improve = val_auc, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            since_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"    epoch {epoch:>3}  loss={train_loss:.4f}  "
                  f"val_auc={val_auc:.4f}  best={best_val_auc:.4f}@{best_epoch}")

        if since_improve >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} "
                      f"(no val improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)

    val_prob, val_y = predict(model, val_loader, dev)
    test_prob, test_y = predict(model, test_loader, dev)

    thr = best_threshold(val_y, val_prob)
    val_metrics = compute_metrics(val_y, val_prob, thr)
    test_metrics = compute_metrics(test_y, test_prob, thr)

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "n_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(time.time() - start, 1),
        "val": val_metrics,
        "test": test_metrics,
    }

    if out_dir is None:
        out_dir = RESULTS_DIR / dataset_name / model_name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save(best_state, out_dir / "model.pt")
    np.savez(out_dir / "test_preds.npz", prob=test_prob, y=test_y, idx=np.array(test_idx))
    np.savez(out_dir / "val_preds.npz", prob=val_prob, y=val_y, idx=np.array(val_idx))
    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_auc", "lr"])
        writer.writeheader()
        writer.writerows(history)

    if verbose:
        print(f"    test ROC-AUC={test_metrics['roc_auc']:.4f}  "
              f"PR-AUC={test_metrics['pr_auc']:.4f}  "
              f"bal-acc={test_metrics['balanced_accuracy']:.4f}  "
              f"MCC={test_metrics['mcc']:.4f}  ({result['train_seconds']}s)")

    return result


def collect() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*/*/seed*/metrics.json")):
        r = json.loads(path.read_text())
        rows.append({
            "dataset": r["dataset"], "model": r["model"], "seed": r["seed"],
            "n_parameters": r["n_parameters"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "train_seconds": r["train_seconds"],
            "test_roc_auc": r["test"]["roc_auc"], "test_pr_auc": r["test"]["pr_auc"],
            "test_balanced_accuracy": r["test"]["balanced_accuracy"],
            "test_mcc": r["test"]["mcc"],
        })
    return pd.DataFrame(rows)


def run_all(models=EDGE_MODELS, datasets=DATASET_NAMES, seeds=SEEDS,
            epochs=200, patience=30, device="cpu") -> pd.DataFrame:
    for name in datasets:
        load_dataset(name, verbose=False)

    total = len(datasets) * len(models) * len(seeds)
    done, failed = 0, []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                done += 1
                print(f"\n[{done}/{total}] {dataset}/{model}/seed{seed}")
                try:
                    train_edge_one(model_name=model, dataset_name=dataset, seed=seed,
                                    epochs=epochs, patience=patience, device=device,
                                    verbose=True)
                except Exception:
                    print(f"    FAILED:\n{traceback.format_exc()}")
                    failed.append((dataset, model, seed))

    df = collect()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "edge_ablation_runs.csv", index=False)
    metrics = ["test_roc_auc", "test_pr_auc", "test_balanced_accuracy", "test_mcc"]
    df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"]).round(4).to_csv(
        RESULTS_DIR / "edge_ablation_summary.csv"
    )
    print(f"\nwrote {RESULTS_DIR / 'edge_ablation_runs.csv'} ({len(df)} runs)")
    if failed:
        print(f"FAILED runs: {failed}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--model", choices=EDGE_MODELS)
    p.add_argument("--dataset", choices=DATASET_NAMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.run_all:
        run_all(epochs=args.epochs, patience=args.patience, device=args.device)
        return

    if not args.model or not args.dataset:
        p.error("either --run-all, or both --model and --dataset")

    print(f"[{args.dataset}/{args.model}/seed{args.seed}]")
    train_edge_one(model_name=args.model, dataset_name=args.dataset, seed=args.seed,
                    epochs=args.epochs, patience=args.patience, device=args.device)


if __name__ == "__main__":
    main()

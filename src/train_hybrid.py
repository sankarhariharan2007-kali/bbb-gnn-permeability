"""
Priority 1: train one (model, dataset, seed) hybrid GNN+descriptor run, and
run the full 4 models x 2 datasets x 3 seeds grid.

Usage:
    python -m src.train_hybrid --model gcn --dataset bbbp --seed 0   # one run
    python -m src.train_hybrid --run-all                             # 24 runs

Mirrors src/train.py exactly (same split, same pos_weight, same early
stopping, same Youden's-J threshold-on-val) so a hybrid-vs-base row-by-row
comparison isolates the effect of the descriptor concatenation and nothing
else. The only additions: descriptors are computed once per molecule,
standardized using TRAIN-FOLD statistics only (never val/test, so scaling
itself introduces no leakage), and attached to each graph as
`data.descriptors` before the loaders are built.
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
from .hybrid_features import compute_descriptor_matrix, standardize_with_train_stats
from .models import MODELS
from .models_hybrid import build_hybrid_model, count_parameters
from .featurize import EDGE_DIM, FEATURE_ID, NODE_DIM
from .split import (SPLIT_ID, SPLIT_ID_MURCKO_ONLY, check_split,
                    scaffold_split)

RESULTS_DIR = ROOT / "results" / "hybrid"
BATCH_RESULTS_DIR = ROOT / "results" / "hybrid_batch"
SEEDS = list(range(10))
DATASET_NAMES = ["bbbp", "b3db"]


def results_dir(norm: str) -> Path:
    """Where an arm's runs live. The two hybrid arms differ ONLY in the norm.

    `hybrid_batch` holds the repo's best in-distribution numbers (limitations
    5.0e) and for a long time no committed command could regenerate it -- it was
    produced ad hoc, while `eval_external_holdout` read it. Keeping the two
    trees apart is deliberate: the BatchNorm arm must never be loaded as a
    frozen reward, because `C(m)` would then depend on the GRPO group it was
    scored in.
    """
    return BATCH_RESULTS_DIR if norm == "batch" else RESULTS_DIR


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


def train_hybrid_one(
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
    norm: str = "graph",
    cluster_acyclic: bool = True,
    device: str = "cpu",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = torch.device(device)

    graphs = load_dataset(dataset_name, verbose=False)
    smiles = [g.smiles for g in graphs]
    labels = [int(g.y.item()) for g in graphs]

    train_idx, val_idx, test_idx = scaffold_split(
        smiles, seed=seed, verbose=False, cluster_acyclic=cluster_acyclic)
    check_split(smiles, train_idx, val_idx, test_idx, labels,
                cluster_acyclic=cluster_acyclic)

    # Descriptors computed once for the whole dataset, then standardized using
    # TRAIN-fold statistics only -- val/test never influence the scaling.
    raw_desc = compute_descriptor_matrix(smiles)
    desc_std, mu, sigma = standardize_with_train_stats(raw_desc, train_idx)
    for i, g in enumerate(graphs):
        g.descriptors = torch.tensor(desc_std[i], dtype=torch.float).unsqueeze(0)

    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]
    test_set = [graphs[i] for i in test_idx]

    drop_last = len(train_set) % batch_size == 1
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    model = build_hybrid_model(model_name, hidden=hidden, num_layers=num_layers,
                                dropout=dropout, heads=heads,
                               norm=norm).to(dev)

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
        # The run contract -- see train.py. Validated and replayed by
        # featurize.run_contract when a checkpoint is reloaded as a reward.
        "feature_id": FEATURE_ID,
        "node_dim": NODE_DIM,
        "edge_dim": EDGE_DIM,
        "hidden": hidden,
        "num_layers": num_layers,
        "dropout": dropout,
        "heads": heads,
        "norm": norm,
        "split": SPLIT_ID if cluster_acyclic else SPLIT_ID_MURCKO_ONLY,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_seconds": round(time.time() - start, 1),
        "val": val_metrics,
        "test": test_metrics,
    }

    if out_dir is None:
        out_dir = results_dir(norm) / dataset_name / model_name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save(best_state, out_dir / "model.pt")
    np.savez(out_dir / "descriptor_scaling.npz", mu=mu, sigma=sigma)
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


def collect(root: Path | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted((root or RESULTS_DIR).glob("*/*/seed*/metrics.json")):
        r = json.loads(path.read_text())
        rows.append({
            "dataset": r["dataset"], "model": r["model"], "seed": r["seed"],
            "n_parameters": r["n_parameters"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "train_seconds": r["train_seconds"],
            "val_roc_auc": r["val"]["roc_auc"],
            **{f"test_{k}": v for k, v in r["test"].items() if k != "confusion_matrix"},
        })
    return pd.DataFrame(rows)


def run_all(models=MODELS, datasets=DATASET_NAMES, seeds=SEEDS,
            epochs=200, patience=30, device="cpu", norm="graph",
            force=False) -> pd.DataFrame:
    for name in datasets:
        load_dataset(name, verbose=False)

    root = results_dir(norm)
    arm = root.name
    total = len(datasets) * len(models) * len(seeds)
    done, failed = 0, []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                done += 1
                out_dir = root / dataset / model / f"seed{seed}"
                if (out_dir / "metrics.json").exists() and not force:
                    print(f"[{done}/{total}] {arm} {dataset}/{model}/seed{seed} "
                          f"-- already done, skipping")
                    continue
                print(f"\n[{done}/{total}] {arm} {dataset}/{model}/seed{seed}")
                try:
                    train_hybrid_one(model_name=model, dataset_name=dataset, seed=seed,
                                      epochs=epochs, patience=patience, device=device,
                                      norm=norm, verbose=True)
                except Exception:
                    print(f"    FAILED:\n{traceback.format_exc()}")
                    failed.append((dataset, model, seed))

    df = collect(root)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / f"{arm}_runs.csv", index=False)
    metrics = ["test_roc_auc", "test_pr_auc", "test_balanced_accuracy", "test_f1", "test_mcc"]
    df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"]).round(4).to_csv(
        root / f"{arm}_summary.csv"
    )
    print(f"\nwrote {root / f'{arm}_runs.csv'} ({len(df)} runs)")
    if failed:
        print(f"FAILED runs: {failed}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-all", action="store_true", help="run the full grid")
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--dataset", choices=DATASET_NAMES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--norm", default="graph", choices=["layer", "graph", "batch"],
                   help="graph (default) is batch-independent and is what a "
                        "frozen reward must use. batch scores higher and writes "
                        "to results/hybrid_batch (limitations 5.0e)")
    p.add_argument("--force", action="store_true",
                   help="retrain runs that already have metrics.json")
    p.add_argument("--threads", type=int, default=None,
                   help="torch.set_num_threads; authoritative where "
                        "OMP_NUM_THREADS is not (CLAUDE.md). Pinned because "
                        "OpenMP changes float reduction order.")
    args = p.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)


    if args.run_all:
        run_all(epochs=args.epochs, patience=args.patience, device=args.device,
                norm=args.norm, force=args.force)
        return

    if not args.model or not args.dataset:
        p.error("either --run-all, or both --model and --dataset")

    print(f"[{results_dir(args.norm).name} {args.dataset}/{args.model}/seed{args.seed}]")
    train_hybrid_one(model_name=args.model, dataset_name=args.dataset, seed=args.seed,
                      epochs=args.epochs, patience=args.patience, device=args.device,
                      norm=args.norm)


if __name__ == "__main__":
    main()

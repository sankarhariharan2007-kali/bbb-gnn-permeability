"""Train one hierarchical motif model (arm 1a / the BRICS family).

Mirrors src/train.py exactly -- same optimizer, schedule, patience, pos_weight,
threshold selection and contract fields -- so a difference against the base
family is attributable to the two-level architecture and not to the loop.

Graphs are decorated with their fragment assignment at load time via
models_brics.to_motif_data; featurize.py is untouched, so FEATURE_ID does not
move and no committed checkpoint is invalidated.

Usage:
    python -m src.train_brics --model gin --dataset bbbp --seed 0
    python -m src.train_brics --run-all
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .datasets import ROOT, load_dataset
from .featurize import EDGE_DIM, FEATURE_ID, NODE_DIM
from .evaluate import best_threshold, compute_metrics
from .models import NORMS, count_parameters
from .models_brics import BRICS_MODELS, build_brics_model, to_motif_data
from .split import SPLIT_ID, SPLIT_ID_MURCKO_ONLY, check_split, scaffold_split

RESULTS_DIR = ROOT / "results" / "brics_motif"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, true labels) for a whole loader."""
    model.eval()
    probs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        probs.append(torch.sigmoid(model(batch)).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


def train_one(
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
    motif_layers: int = 2,
    intra_only: bool = False,
    fuse: bool = True,
    device: str = "cpu",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    set_seed(seed)
    dev = torch.device(device)

    raw = load_dataset(dataset_name, verbose=False)
    graphs = [to_motif_data(g) for g in raw]
    kept = [i for i, g in enumerate(graphs) if g is not None]
    if len(kept) != len(raw) and verbose:
        print(f"    dropped {len(raw) - len(kept)} molecules with no motif view")
    graphs = [graphs[i] for i in kept]
    smiles = [g.smiles for g in graphs]
    labels = [int(g.y.item()) for g in graphs]

    # The split is seeded, so each seed sees a different scaffold partition and
    # the reported spread captures split variance as well as init variance.
    train_idx, val_idx, test_idx = scaffold_split(
        smiles, seed=seed, verbose=False, cluster_acyclic=cluster_acyclic)
    check_split(smiles, train_idx, val_idx, test_idx, labels,
                cluster_acyclic=cluster_acyclic)

    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]
    test_set = [graphs[i] for i in test_idx]

    # BatchNorm needs >1 sample per batch; drop a trailing singleton batch.
    # Harmless under LayerNorm, kept so every norm trains on identical batches.
    drop_last = len(train_set) % batch_size == 1
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=drop_last)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    # atom_layers == num_layers so the atom pathway is EXACTLY the flat
    # model's depth. The motif graph is then strictly additional, and the head
    # can learn to ignore it -- which is what makes this a test of whether
    # fragment structure adds anything rather than whether it can replace
    # atom-level detail.
    model = build_brics_model(model_name, hidden=hidden,
                              atom_layers=num_layers,
                              motif_layers=motif_layers,
                              dropout=dropout, heads=heads, norm=norm,
                              intra_only=intra_only, fuse=fuse).to(dev)

    # pos_weight from the training fold only. Both datasets skew positive, so
    # this mainly helps accuracy/F1; ROC-AUC is threshold-free.
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

    model.load_state_dict(best_state)  # restore best-on-val before testing

    val_prob, val_y = predict(model, val_loader, dev)
    test_prob, test_y = predict(model, test_loader, dev)

    # Threshold picked on validation, then applied unchanged to test.
    thr = best_threshold(val_y, val_prob)
    val_metrics = compute_metrics(val_y, val_prob, thr)
    test_metrics = compute_metrics(test_y, test_prob, thr)

    result = {
        "model": model_name,
        "family": "brics_motif",
        "motif_layers": motif_layers,
        "intra_only": intra_only,
        "fuse": fuse,
        "dataset": dataset_name,
        "seed": seed,
        "n_parameters": count_parameters(model),
        # The run contract. Checkpoints are bare state_dicts, so without these
        # nothing on disk says which featurizer, architecture or partition
        # produced them -- and `build_model`'s defaults silently became the
        # contract. `featurize.run_contract` validates and replays them.
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
        out_dir = RESULTS_DIR / dataset_name / model_name / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    torch.save(best_state, out_dir / "model.pt")
    # Molecule indices are saved so predictions from different models can be
    # aligned for the ensemble; val predictions let a stacker fit on a clean fold.
    np.savez(out_dir / "test_preds.npz", prob=test_prob, y=test_y,
             idx=np.array(test_idx))
    np.savez(out_dir / "val_preds.npz", prob=val_prob, y=val_y,
             idx=np.array(val_idx))
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=BRICS_MODELS, required=True)
    p.add_argument("--dataset", choices=["bbbp", "b3db"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--motif-layers", type=int, default=2)
    p.add_argument("--intra-only", action="store_true",
                   help="arm 1a: level 1 sees only intra-fragment bonds")
    p.add_argument("--no-fuse", action="store_true",
                   help="drop the atom-level readout channel")
    p.add_argument("--norm", choices=NORMS, default="graph",
                   help="layer (default) is batch-independent; batch reproduces "
                        "the pre-contract runs")
    p.add_argument("--murcko-only-split", action="store_true",
                   help="disable Tanimoto clustering of acyclic molecules "
                        "(reproduces the pre-contract split)")
    p.add_argument("--device", default="cpu",
                   help="cpu (default) is fastest for these small graphs")
    p.add_argument("--threads", type=int, default=None,
                   help="torch.set_num_threads; authoritative where "
                        "OMP_NUM_THREADS is not (CLAUDE.md). Pinned because "
                        "OpenMP changes float reduction order.")
    args = p.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)


    print(f"[{args.dataset}/{args.model}/seed{args.seed}]")
    train_one(
        model_name=args.model, dataset_name=args.dataset, seed=args.seed,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
        weight_decay=args.weight_decay, batch_size=args.batch_size,
        hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout,
        heads=args.heads, norm=args.norm,
        motif_layers=args.motif_layers,
        intra_only=args.intra_only, fuse=not args.no_fuse,
        cluster_acyclic=not args.murcko_only_split,
        device=args.device,
    )


if __name__ == "__main__":
    main()

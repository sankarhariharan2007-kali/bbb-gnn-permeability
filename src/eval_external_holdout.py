"""
Evaluate every already-trained checkpoint (base GNN, hybrid GNN+descriptor,
edge-feature ablation, D-MPNN) against the leak-free external holdout sets
(Adenot / Wang), which src/external_holdout.py builds but which no training
or eval script in this repo currently touches.

This closes that gap: it answers "do these models generalize to molecules
outside BBBP/B3DB entirely?" using the *same* checkpoints already sitting in
results/, no retraining involved.

WHERE TO PUT THIS FILE
-----------------------
Drop this file in the project as `src/eval_external_holdout.py` (next to
train.py, models.py, etc.) so the relative imports below resolve. Then run
it as a module from the project root, exactly like the other scripts:

    python -m src.eval_external_holdout

Requirements: same environment as the rest of the repo (requirements.txt) --
torch, torch_geometric, rdkit, scikit-learn, pandas, numpy.

WHAT IT DOES
------------
1. Builds (or reuses) data/external_holdout/{adenot,wang}_clean.csv via
   src/external_holdout.py -- these are already guaranteed scaffold-disjoint
   from both BBBP and B3DB, so no model here has seen them in any form.
2. Converts each holdout SMILES to a PyG graph (skips anything RDKit can't
   parse, same as datasets.py does for the training sets).
3. Walks every checkpoint under results/{, hybrid/, edge_ablation/, dmpnn/}
   and, for each, loads its saved state_dict into the matching architecture.
4. For hybrid checkpoints, recomputes the 8 descriptors for the holdout
   molecules and standardizes them using that SPECIFIC seed's saved
   train-fold mean/std (descriptor_scaling.npz) -- never refit on the
   holdout, to keep this an honest generalization test.
5. Scores every checkpoint against both holdout sets: ROC-AUC / PR-AUC
   (threshold-free) plus balanced accuracy / F1 / MCC at the checkpoint's
   OWN validation-selected threshold (from its metrics.json) -- never a
   threshold fit on the holdout itself.
6. Writes per-run rows and a (family, dataset, model, holdout) summary
   (mean +/- std over seeds) to results/external_holdout_eval/.

Small-n caveat: adenot_clean.csv and wang_clean.csv are ~55-60 molecules
each after leak removal. Point estimates here will be noisy -- this script
also reports the raw confusion matrix per run so single-molecule swings are
visible rather than hidden behind a single number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .datasets import ROOT
from .evaluate import compute_metrics
from .external_holdout import EXTERNAL_SOURCES, build_clean_holdout
from .featurize import smiles_to_graph
from .hybrid_features import compute_descriptors
from .models import MODELS, build_model
from .models_dmpnn import build_dmpnn_model
from .models_edge import EDGE_MODELS, build_edge_model
from .models_hybrid import build_hybrid_model

RESULTS_DIR = ROOT / "results"
HOLDOUT_DIR = ROOT / "data" / "external_holdout"
OUT_DIR = RESULTS_DIR / "external_holdout_eval"
HOLDOUT_SOURCES = list(EXTERNAL_SOURCES)  # ["adenot", "wang"]


# --------------------------------------------------------------------------
# Holdout loading (build once if not already present on disk)
# --------------------------------------------------------------------------

def ensure_holdout_files(training_pool: list[str]) -> dict[str, Path]:
    paths = {}
    for source in HOLDOUT_SOURCES:
        path = HOLDOUT_DIR / f"{source}_clean.csv"
        if not path.exists():
            print(f"[{source}] clean holdout not found on disk, building it now...")
            clean_df, report = build_clean_holdout(source, training_pool)
            HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
            clean_df.to_csv(path, index=False)
            print(f"  wrote {path} ({report['rows_out']} rows)")
        paths[source] = path
    return paths


def load_holdout_graphs(path: Path) -> tuple[list, int]:
    """Return (graphs, n_smiles_dropped_as_unparseable)."""
    df = pd.read_csv(path)
    graphs, dropped = [], 0
    for smi, label in zip(df["smiles"], df["label"]):
        g = smiles_to_graph(smi, label=float(label))
        if g is None:
            dropped += 1
            continue
        g.smiles = smi
        graphs.append(g)
    return graphs, dropped


# --------------------------------------------------------------------------
# Checkpoint discovery
# --------------------------------------------------------------------------
# Each entry: (family, dataset, model_name, seed_dir_path)

def discover_checkpoints() -> list[dict]:
    found = []

    # base: results/<dataset>/<model>/seed<N>/model.pt
    for dataset in ("bbbp", "b3db"):
        for model_name in MODELS:
            for seed_dir in sorted((RESULTS_DIR / dataset / model_name).glob("seed*")):
                if (seed_dir / "model.pt").exists():
                    found.append(dict(family="base", dataset=dataset,
                                       model=model_name, seed_dir=seed_dir))

    # hybrid: results/hybrid/<dataset>/<model>/seed<N>/model.pt
    for dataset in ("bbbp", "b3db"):
        for model_name in MODELS:
            for seed_dir in sorted((RESULTS_DIR / "hybrid" / dataset / model_name).glob("seed*")):
                if (seed_dir / "model.pt").exists():
                    found.append(dict(family="hybrid", dataset=dataset,
                                       model=model_name, seed_dir=seed_dir))

    # edge_ablation: results/edge_ablation/<dataset>/<model>/seed<N>/model.pt
    for dataset in ("bbbp", "b3db"):
        for model_name in EDGE_MODELS:
            for seed_dir in sorted((RESULTS_DIR / "edge_ablation" / dataset / model_name).glob("seed*")):
                if (seed_dir / "model.pt").exists():
                    found.append(dict(family="edge_ablation", dataset=dataset,
                                       model=model_name, seed_dir=seed_dir))

    # dmpnn: results/dmpnn/<dataset>/seed<N>/model.pt  (no per-model subdir)
    for dataset in ("bbbp", "b3db"):
        for seed_dir in sorted((RESULTS_DIR / "dmpnn" / dataset).glob("seed*")):
            if (seed_dir / "model.pt").exists():
                found.append(dict(family="dmpnn", dataset=dataset,
                                   model="dmpnn", seed_dir=seed_dir))

    return found


def build_model_for(family: str, model_name: str) -> torch.nn.Module:
    if family == "base":
        return build_model(model_name)
    if family == "hybrid":
        return build_hybrid_model(model_name)
    if family == "edge_ablation":
        return build_edge_model(model_name)
    if family == "dmpnn":
        return build_dmpnn_model()
    raise ValueError(f"unknown family: {family!r}")


def load_val_threshold(seed_dir: Path) -> float:
    """Use the SAME threshold the checkpoint was scored with on its own
    in-distribution test set (chosen on val, per evaluate.best_threshold).
    Fitting a new threshold on the holdout would leak/cheat."""
    metrics = json.loads((seed_dir / "metrics.json").read_text())
    return float(metrics["test"]["threshold"])


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        batch = batch.to(device)
        probs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probs)


def attach_hybrid_descriptors(graphs: list, mu: np.ndarray, sigma: np.ndarray) -> None:
    """In-place: attach data.descriptors, standardized with a CHECKPOINT'S
    OWN saved train-fold mu/sigma (never stats from the holdout itself)."""
    for g in graphs:
        raw = np.array(compute_descriptors(g.smiles), dtype=np.float64)
        std = (raw - mu) / sigma
        g.descriptors = torch.tensor(std, dtype=torch.float).unsqueeze(0)


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-pool", nargs="+", default=["bbbp", "b3db"],
                     help="datasets to treat as 'already seen' when building "
                          "the clean holdout (passed to external_holdout.py)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    holdout_paths = ensure_holdout_files(args.training_pool)

    holdout_graphs: dict[str, list] = {}
    for source, path in holdout_paths.items():
        graphs, dropped = load_holdout_graphs(path)
        n_pos = sum(int(g.y.item()) for g in graphs)
        print(f"[{source}] loaded {len(graphs)} graphs "
              f"({n_pos} pos / {len(graphs) - n_pos} neg), "
              f"{dropped} unparseable SMILES dropped")
        holdout_graphs[source] = graphs

    checkpoints = discover_checkpoints()
    print(f"\nfound {len(checkpoints)} checkpoints to evaluate\n")

    rows = []
    for ckpt in checkpoints:
        family, dataset, model_name, seed_dir = (
            ckpt["family"], ckpt["dataset"], ckpt["model"], ckpt["seed_dir"]
        )
        seed = int(seed_dir.name.replace("seed", ""))

        model = build_model_for(family, model_name).to(device)
        state = torch.load(seed_dir / "model.pt", map_location=device)
        model.load_state_dict(state)

        threshold = load_val_threshold(seed_dir)

        mu = sigma = None
        if family == "hybrid":
            scaling = np.load(seed_dir / "descriptor_scaling.npz")
            mu, sigma = scaling["mu"], scaling["sigma"]

        for source, graphs in holdout_graphs.items():
            if family == "hybrid":
                attach_hybrid_descriptors(graphs, mu, sigma)

            loader = DataLoader(graphs, batch_size=256)
            probs = predict(model, loader, device)
            y_true = np.array([int(g.y.item()) for g in graphs])

            m = compute_metrics(y_true, probs, threshold)
            rows.append({
                "family": family, "train_dataset": dataset, "model": model_name,
                "seed": seed, "holdout": source, "threshold_used": threshold,
                "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                "balanced_accuracy": m["balanced_accuracy"], "f1": m["f1"],
                "mcc": m["mcc"], "n": m["n"], "n_positive": m["n_positive"],
                "tn": m["confusion_matrix"]["tn"], "fp": m["confusion_matrix"]["fp"],
                "fn": m["confusion_matrix"]["fn"], "tp": m["confusion_matrix"]["tp"],
            })
            print(f"  [{family}/{dataset}/{model_name}/seed{seed}] "
                  f"vs {source}: ROC-AUC={m['roc_auc']:.3f}  MCC={m['mcc']:.3f}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "external_holdout_runs.csv", index=False)

    metric_cols = ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "mcc"]
    summary = (
        df.groupby(["family", "train_dataset", "model", "holdout"])[metric_cols]
        .agg(["mean", "std"]).round(4)
    )
    summary.to_csv(OUT_DIR / "external_holdout_summary.csv")

    print(f"\nwrote {len(df)} rows to {OUT_DIR / 'external_holdout_runs.csv'}")
    print(f"wrote summary to {OUT_DIR / 'external_holdout_summary.csv'}")


if __name__ == "__main__":
    main()

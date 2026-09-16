"""Run every (model, dataset, seed) combination and collect results.

Usage:
    python -m src.run_all                 # all 24 runs
    python -m src.run_all --datasets bbbp # subset
"""

from __future__ import annotations

import argparse
import json
import traceback

import numpy as np
import pandas as pd

from .datasets import ROOT, load_dataset
from .models import MODELS
from .train import RESULTS_DIR, train_one

SEEDS = [0, 1, 2]
DATASET_NAMES = ["bbbp", "b3db"]


def collect() -> pd.DataFrame:
    """Gather every metrics.json under results/ into a tidy frame."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("*/*/seed*/metrics.json")):
        r = json.loads(path.read_text())
        rows.append({
            "dataset": r["dataset"], "model": r["model"], "seed": r["seed"],
            "n_parameters": r["n_parameters"], "best_epoch": r["best_epoch"],
            "epochs_run": r["epochs_run"], "train_seconds": r["train_seconds"],
            "val_roc_auc": r["val"]["roc_auc"],
            **{f"test_{k}": v for k, v in r["test"].items()
               if k != "confusion_matrix"},
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across seeds for each (dataset, model)."""
    metrics = ["test_roc_auc", "test_pr_auc", "test_balanced_accuracy",
               "test_f1", "test_mcc"]
    agg = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"])
    return agg.round(4)


def print_summary(df: pd.DataFrame) -> None:
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        print(f"\n{'=' * 72}\n{dataset.upper()}  (mean +/- std over {sub['seed'].nunique()} seeds)\n{'=' * 72}")
        print(f"{'model':<8} {'ROC-AUC':<18} {'PR-AUC':<18} {'bal-acc':<18} {'MCC':<16}")
        for model in MODELS:
            m = sub[sub["model"] == model]
            if m.empty:
                continue
            cells = "".join(
                f"{m[c].mean():.4f} +/- {m[c].std():.4f}  "
                for c in ["test_roc_auc", "test_pr_auc",
                          "test_balanced_accuracy", "test_mcc"]
            )
            print(f"{model:<8} {cells}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=DATASET_NAMES, choices=DATASET_NAMES)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    for name in args.datasets:  # warm the caches once, not per run
        load_dataset(name, verbose=False)

    total = len(args.datasets) * len(args.models) * len(args.seeds)
    done, failed = 0, []
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                done += 1
                print(f"\n[{done}/{total}] {dataset}/{model}/seed{seed}")
                try:
                    train_one(model_name=model, dataset_name=dataset, seed=seed,
                              epochs=args.epochs, patience=args.patience,
                              device=args.device, verbose=True)
                except Exception:
                    print(f"    FAILED:\n{traceback.format_exc()}")
                    failed.append((dataset, model, seed))

    df = collect()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "all_runs.csv", index=False)
    summarize(df).to_csv(RESULTS_DIR / "summary.csv")
    print_summary(df)
    print(f"\nwrote {RESULTS_DIR / 'all_runs.csv'} ({len(df)} runs)")
    if failed:
        print(f"FAILED runs: {failed}")


if __name__ == "__main__":
    main()

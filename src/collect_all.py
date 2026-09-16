"""
Collect every results table produced across the base run + priorities 0-3
into one master comparison, tabularized and ranked.

Usage:
    python -m src.collect_all
    python -m src.collect_all --dataset bbbp   # filter to one dataset

Reads whatever is present and skips whatever isn't (so you can run this
after any subset of the training scripts, not just after everything):

    results/all_runs.csv                          <- src/run_all.py (gcn/sage/gin/gat)
    results/descriptor_baseline_v2_runs.csv        <- descriptor_baseline_v2.py (LightGBM)
    results/edge_ablation/edge_ablation_runs.csv   <- src/train_edge.py (gine/gat_edge/sage_edge)
    results/hybrid/hybrid_runs.csv                 <- src/train_hybrid.py (gcn/sage/gin/gat + descriptors)
    results/dmpnn/dmpnn_runs.csv                   <- src/train_dmpnn.py

    gine_ablation_results.csv (repo root, if you haven't re-run gine under
    train_edge.py yet) is picked up too, but a row for the same
    (dataset, model=gine) in edge_ablation_runs.csv takes priority since
    it's the more recent, identically-protocoled run.

Writes results/master_comparison.csv (one row per (dataset, source, model),
mean +/- std across seeds) and prints it as a ranked table per dataset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["test_roc_auc", "test_pr_auc", "test_balanced_accuracy", "test_f1", "test_mcc"]

# Each entry: (label shown in the table, path, kind)
#   kind "gnn"        -> already has dataset/model/seed + the METRICS columns
#   kind "descriptor" -> descriptor_baseline_v2 schema (roc_auc/pr_auc/... without test_ prefix,
#                        and a `feature_set` column instead of `model`)
SOURCES = [
    ("base_gnn", ROOT / "results" / "all_runs.csv", "gnn"),
    ("edge_ablation", ROOT / "results" / "edge_ablation" / "edge_ablation_runs.csv", "gnn"),
    ("gine_ablation_legacy", ROOT / "gine_ablation_results.csv", "gnn"),
    ("hybrid", ROOT / "results" / "hybrid" / "hybrid_runs.csv", "gnn"),
    ("dmpnn", ROOT / "results" / "dmpnn" / "dmpnn_runs.csv", "gnn"),
    ("descriptor_gbm", ROOT / "results" / "descriptor_baseline_v2_runs.csv", "descriptor"),
]


def _load_gnn(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep_cols = ["dataset", "model", "seed"] + [c for c in METRICS if c in df.columns]
    df = df[keep_cols].copy()
    df["source"] = source_label
    return df


def _load_descriptor(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "dataset": df["dataset"],
        "model": "lightgbm_" + df["feature_set"],
        "seed": df["seed"],
        "test_roc_auc": df["roc_auc"],
        "test_pr_auc": df["pr_auc"],
        "test_balanced_accuracy": df["balanced_accuracy"],
        "test_f1": df["f1"],
        "test_mcc": df["mcc"],
    })
    out["source"] = source_label
    return out


def load_all() -> pd.DataFrame:
    frames = []
    seen_gine = False
    for label, path, kind in SOURCES:
        if not path.exists():
            continue
        if label == "gine_ablation_legacy" and seen_gine:
            continue  # edge_ablation's gine rows (if present) supersede this
        df = _load_gnn(path, label) if kind == "gnn" else _load_descriptor(path, label)
        if label == "edge_ablation" and "gine" in df["model"].unique():
            seen_gine = True
        frames.append(df)
    if not frames:
        raise SystemExit(
            "No results files found yet. Run at least one training script first "
            "(see the docstring in this file for expected paths)."
        )
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    present_metrics = [m for m in METRICS if m in df.columns]
    agg = (
        df.groupby(["dataset", "source", "model"])[present_metrics]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    n_seeds = df.groupby(["dataset", "source", "model"])["seed"].nunique()
    agg["n_seeds"] = n_seeds
    return agg.reset_index()


def print_table(summary: pd.DataFrame) -> None:
    for dataset in sorted(summary["dataset"].unique()):
        sub = summary[summary["dataset"] == dataset].sort_values(
            "test_roc_auc_mean", ascending=False
        )
        print(f"\n{'=' * 100}\n{dataset.upper()}  (ranked by test ROC-AUC, mean +/- std)\n{'=' * 100}")
        header = f"{'source':<16} {'model':<20} {'n':>2}  {'ROC-AUC':<16} {'PR-AUC':<16} {'bal-acc':<16} {'F1':<16} {'MCC':<16}"
        print(header)
        for _, r in sub.iterrows():
            def cell(m):
                mean, std = r.get(f"{m}_mean"), r.get(f"{m}_std")
                if pd.isna(mean):
                    return f"{'--':<16}"
                std = 0.0 if pd.isna(std) else std
                return f"{mean:.4f}+/-{std:.4f}  "
            print(f"{r['source']:<16} {r['model']:<20} {int(r['n_seeds']):>2}  "
                  f"{cell('test_roc_auc')}{cell('test_pr_auc')}{cell('test_balanced_accuracy')}"
                  f"{cell('test_f1')}{cell('test_mcc')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["bbbp", "b3db"], default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "master_comparison.csv"))
    args = ap.parse_args()

    df = load_all()
    if args.dataset:
        df = df[df["dataset"] == args.dataset]

    summary = summarize(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print_table(summary)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

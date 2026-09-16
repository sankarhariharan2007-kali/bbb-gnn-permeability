"""Combine the saved per-model predictions into ensembles.

Because the scaffold split is seeded, each seed defines a *different* test fold.
Models are therefore combined **within a seed** -- where all four share an
identical split and their saved `idx` arrays match exactly -- and the resulting
scores are averaged across seeds. Averaging predictions across seeds would be
meaningless, since those predictions describe different molecules.

Usage:
    python -m src.ensemble
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from .evaluate import best_threshold, compute_metrics
from .models import MODELS
from .train import RESULTS_DIR


def _load(dataset: str, model: str, seed: int, fold: str) -> dict | None:
    path = RESULTS_DIR / dataset / model / f"seed{seed}" / f"{fold}_preds.npz"
    if not path.exists():
        return None
    d = np.load(path)
    return {"prob": d["prob"], "y": d["y"], "idx": d["idx"]}


def gather(dataset: str, seed: int, models: list[str]) -> dict | None:
    """Stack the four models' val/test predictions for one seed."""
    val, test = {}, {}
    for m in models:
        v, t = _load(dataset, m, seed, "val"), _load(dataset, m, seed, "test")
        if v is None or t is None:
            return None
        val[m], test[m] = v, t

    ref = models[0]
    for m in models[1:]:
        # All four models must have seen the identical split for this seed.
        assert np.array_equal(test[m]["idx"], test[ref]["idx"]), \
            f"{dataset}/seed{seed}: {m} and {ref} have different test folds"
        assert np.array_equal(val[m]["idx"], val[ref]["idx"]), \
            f"{dataset}/seed{seed}: {m} and {ref} have different val folds"

    return {
        "val_X": np.column_stack([val[m]["prob"] for m in models]),
        "val_y": val[ref]["y"],
        "test_X": np.column_stack([test[m]["prob"] for m in models]),
        "test_y": test[ref]["y"],
    }


def ensemble_seed(data: dict) -> dict[str, np.ndarray]:
    """Produce test scores for each combination strategy."""
    val_X, val_y, test_X = data["val_X"], data["val_y"], data["test_X"]

    scores = {"soft_vote": test_X.mean(axis=1)}

    # Rank averaging is scale-free: it only needs each model's ordering, so a
    # poorly calibrated but well-ranked model is not penalised.
    scores["rank_avg"] = np.column_stack(
        [rankdata(test_X[:, j]) / len(test_X) for j in range(test_X.shape[1])]
    ).mean(axis=1)

    # Stacking learns per-model weights on the validation fold, which the base
    # models were early-stopped on but never trained on.
    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(val_X, val_y)
    scores["stacking"] = stacker.predict_proba(test_X)[:, 1]
    scores["_stack_weights"] = stacker.coef_[0]

    return scores


def run(models: list[str] | None = None, seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    models = models or MODELS
    rows, weights = [], []

    for dataset in ("bbbp", "b3db"):
        for seed in seeds:
            data = gather(dataset, seed, models)
            if data is None:
                continue
            scores = ensemble_seed(data)
            weights.append({"dataset": dataset, "seed": seed,
                            **dict(zip(models, scores.pop("_stack_weights").round(3)))})

            # Individual models, for a like-for-like baseline on this same fold.
            for j, m in enumerate(models):
                thr = best_threshold(data["val_y"], data["val_X"][:, j])
                metrics = compute_metrics(data["test_y"], data["test_X"][:, j], thr)
                rows.append({"dataset": dataset, "seed": seed, "method": m,
                             "kind": "single", **metrics})

            for name, test_prob in scores.items():
                # Threshold from the same strategy applied to the val fold.
                if name == "soft_vote":
                    val_prob = data["val_X"].mean(axis=1)
                elif name == "rank_avg":
                    val_prob = np.column_stack(
                        [rankdata(data["val_X"][:, j]) / len(data["val_X"])
                         for j in range(data["val_X"].shape[1])]).mean(axis=1)
                else:
                    val_prob = None
                thr = best_threshold(data["val_y"], val_prob) if val_prob is not None else 0.5
                metrics = compute_metrics(data["test_y"], test_prob, thr)
                rows.append({"dataset": dataset, "seed": seed, "method": name,
                             "kind": "ensemble", **metrics})

    df = pd.DataFrame(rows).drop(columns=["confusion_matrix"])
    df.to_csv(RESULTS_DIR / "ensemble_runs.csv", index=False)
    pd.DataFrame(weights).to_csv(RESULTS_DIR / "stack_weights.csv", index=False)
    return df


def print_report(df: pd.DataFrame) -> None:
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        print(f"\n{'=' * 66}\n{dataset.upper()}  test ROC-AUC, mean +/- std over seeds\n{'=' * 66}")
        agg = (sub.groupby(["kind", "method"])["roc_auc"]
                  .agg(["mean", "std"]).sort_values("mean", ascending=False))
        for (kind, method), row in agg.iterrows():
            marker = "  <-- ensemble" if kind == "ensemble" else ""
            print(f"  {method:<12} {row['mean']:.4f} +/- {row['std']:.4f}{marker}")


if __name__ == "__main__":
    df = run()
    if df.empty:
        print("No predictions found. Run `python -m src.run_all` first.")
    else:
        print_report(df)
        print(f"\nwrote {RESULTS_DIR / 'ensemble_runs.csv'}")

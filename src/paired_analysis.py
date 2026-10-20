"""Paired-seed comparison of model architectures.

The seed in this project controls the *split* (see src/split.py), and every
model is trained on the same seed-s split. So the architecture comparison is a
paired design, and reporting it with marginal (across-seed) standard
deviations overstates the noise: most of that spread is split difficulty,
which is common to all models on that seed and cancels in a per-seed
difference.

Concretely on BBBP, base-GNN marginal SDs run up to 0.045 while the gaps
between operators are ~0.013 -- which reads as hopeless. The paired SD for
gat-gin is 0.009, so the same gap is a ~1.4 sd effect. The comparison was
always better powered than the marginal numbers suggested.

This script verifies the pairing assumption before relying on it, rather than
inferring it from the code: within each (dataset, seed) every model must have
been scored on a test set of the same size AND the same positive count. A
source that fails that check is reported and excluded rather than silently
pooled.

    python -m src.paired_analysis                  # base GNNs, both datasets
    python -m src.paired_analysis --all-sources    # include hybrid/edge
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# t critical values, two-sided 95%, n-1 df. Spelled out because the whole
# point of this script is that n is small.
_TCRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
          8: 2.365, 9: 2.306, 10: 2.262}


def _tcrit(n: int) -> float:
    return _TCRIT.get(n, 1.96)


SOURCES = {
    "base_gnn": RESULTS / "all_runs.csv",
    "hybrid": RESULTS / "hybrid" / "hybrid_runs.csv",
    "edge_ablation": RESULTS / "edge_ablation" / "edge_ablation_runs.csv",
}


def load(all_sources: bool = False) -> pd.DataFrame:
    frames = []
    for name, path in SOURCES.items():
        if not all_sources and name != "base_gnn":
            continue
        if not path.exists():
            print(f"  skip {name}: {path} not found")
            continue
        df = pd.read_csv(path)
        df["source"] = name
        # Prefix non-base models so gat (base) and gat (hybrid) stay distinct.
        if name != "base_gnn":
            df["model"] = name.split("_")[0] + ":" + df["model"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def check_pairing(df: pd.DataFrame) -> pd.DataFrame:
    """Which (dataset, seed) cells have a genuinely shared test set.

    Same test_n alone is not enough -- two different splits can easily have
    the same size. Requiring the positive count to match as well makes an
    accidental collision much less likely.
    """
    need = [c for c in ("test_n", "test_n_positive") if c in df]
    g = df.groupby(["dataset", "seed"]).agg(
        n_models=("model", "nunique"),
        **{f"uniq_{c}": (c, "nunique") for c in need},
    )
    g["paired"] = np.logical_and.reduce([g[f"uniq_{c}"] == 1 for c in need])
    return g.reset_index()


def paired_table(df: pd.DataFrame, dataset: str, metric: str) -> pd.DataFrame:
    sub = df[df.dataset == dataset].pivot_table(
        index="seed", columns="model", values=metric
    ).dropna(axis=1)
    rows = []
    for a, b in itertools.combinations(sorted(sub.columns), 2):
        diff = (sub[a] - sub[b]).dropna()
        n = len(diff)
        if n < 2:
            continue
        md, sd = diff.mean(), diff.std(ddof=1)
        se = sd / np.sqrt(n)
        indep_sd = np.sqrt(sub[a].var(ddof=1) + sub[b].var(ddof=1))
        lo, hi = md - _tcrit(n) * se, md + _tcrit(n) * se
        # Seeds needed for a 95% CI clear of zero at the observed paired sd,
        # using the large-n critical value -- an optimistic floor, not a
        # promise, since t is larger at the n it would actually be run at,
        # and since sd itself is estimated from n-1 df (2, at n=3) and is
        # therefore unstable. Floored at 2: a difference cannot have a
        # confidence interval at n=1.
        n_need = max(2.0, np.ceil((1.96 * sd / abs(md)) ** 2)) if md else np.nan
        rows.append({
            "pair": f"{a} - {b}", "n": n, "mean_diff": md,
            "paired_sd": sd, "indep_sd": indep_sd,
            "sd_ratio": sd / indep_sd if indep_sd else np.nan,
            "ci95_lo": lo, "ci95_hi": hi,
            "sig": "yes" if lo * hi > 0 else "no",
            "dz": md / sd if sd else np.nan,
            "n_for_sig": n_need,
            # Statistical separation is not practical relevance. A pair can be
            # highly consistent across seeds and still differ by less than
            # anyone would act on, so the magnitude is flagged separately.
            "negligible": abs(md) < 0.005,
        })
    return pd.DataFrame(rows).sort_values("dz", key=abs, ascending=False)


def main(all_sources: bool, metric: str) -> None:
    df = load(all_sources)

    print("=" * 78)
    print("PAIRING CHECK -- do all models share a test set within (dataset, seed)?")
    print("=" * 78)
    chk = check_pairing(df)
    print(chk.to_string(index=False))
    bad = chk[~chk.paired]
    if len(bad):
        print(f"\n  !! {len(bad)} cell(s) are NOT paired -- excluded from the "
              f"analysis below, since a per-seed difference there would be "
              f"comparing scores on different molecules:")
        print(bad.to_string(index=False))
        df = df.merge(chk[chk.paired][["dataset", "seed"]], on=["dataset", "seed"])
    else:
        print("\n  all cells paired: same test_n and same test_n_positive across models.")

    for ds in sorted(df.dataset.unique()):
        sub = df[df.dataset == ds]
        marg = sub.groupby("model")[metric].agg(["mean", "std", "count"])
        print()
        print("=" * 78)
        print(f"{ds.upper()}  --  {metric}")
        print("=" * 78)
        print("\nmarginal (across-seed, what the summary tables report):")
        print(marg.round(4).to_string())

        t = paired_table(sub, ds, metric)
        if t.empty:
            print("\n  not enough paired seeds for a comparison")
            continue
        print("\npaired (per-seed differences):")
        print(t.round(4).to_string(index=False))
        print(f"\n  median paired_sd / indep_sd = {t.sd_ratio.median():.2f}"
              f"   (<1 means split variance cancelled)")
        n_sig = (t.sig == "yes").sum()
        n_sig_real = ((t.sig == "yes") & ~t.negligible).sum()
        print(f"  {n_sig}/{len(t)} pairs separated at 95% with n={t.n.max()} seeds"
              f"  ({n_sig_real} of them by more than 0.005 AUC)")
        best = t.iloc[0]
        print(f"  largest effect: {best.pair}  {best.mean_diff:+.4f} "
              f"(dz={best.dz:+.2f}), ~{best.n_for_sig:.0f} seeds to clear zero")
        if t.n.max() < 5:
            print(f"  NOTE: paired_sd has {t.n.max() - 1} df here, so dz and "
                  f"n_for_sig are themselves noisy -- read them as ranking "
                  f"candidates for the 10-seed run, not as results.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-sources", action="store_true",
                    help="include hybrid/edge_ablation, not just base GNNs")
    ap.add_argument("--metric", default="test_roc_auc")
    args = ap.parse_args()
    main(args.all_sources, args.metric)

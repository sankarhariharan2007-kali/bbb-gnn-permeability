"""Score the molecules the screen THREW AWAY. Was the reward pushing toward them?

THE GAP THIS CLOSES. `postfilter.add_scores` scores only molecules with
`passes == True`, so 45,701 of the 72,528 pooled molecules across the three
arms carry no classifier score at all -- everything rejected for size,
structural defect, or a targeted/catalogue alert. Every claim in this project
about `C` is therefore a claim about the survivor subset.

THE QUESTION THAT LEAVES OPEN, which matters. If the rejected molecules score
HIGHER than the survivors, the reward was actively driving the policy toward
exactly the liabilities the screen removes, and the screen is load-bearing
rather than cosmetic. If they score the same or lower, the screen is removing
molecules the reward was indifferent to, and the two mechanisms are
independent. Either answer is worth having and neither is currently known.

Scored with the frozen reward ensemble, the same one that trained arm A and the
gate arm, so the numbers are directly comparable to the committed `c_score`
column.

    python -m src.score_rejected --arm cgate
    python -m src.score_rejected --all
    python -m src.score_rejected --self-check
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from .datasets import ROOT
from .reward import load_frozen_classifier, sanitizable

ARMS = {"a": "brics10", "b": "brics_hybrid", "cgate": "brics_cgate"}
OUT = ROOT / "results" / "pubchem" / "rejected_scores.csv"


def reason(r) -> str:
    """One label per molecule, in the screen's own precedence order.

    A molecule can trip several filters at once; `postfilter` rejects on any,
    so attributing it to the first in this order keeps the groups disjoint and
    matches how the screen counts them.
    """
    if bool(r.undersized):
        return "undersized"
    if isinstance(r.structural, str) and r.structural.strip():
        return "structural defect"
    if r.n_targeted > 0:
        return "targeted alert"
    if r.n_catalogue > 0:
        return "catalogue alert"
    if bool(r.in_training_flat):
        return "in training data"
    return "survivor"


def score_all(smiles: list[str], classify, batch: int = 256) -> np.ndarray:
    """NaN where RDKit will not sanitize, never 0.0 -- a zero would be averaged
    in as "predicted impermeable" and drag every group mean down."""
    out = np.full(len(smiles), np.nan)
    for i in range(0, len(smiles), batch):
        idx = list(range(i, min(i + batch, len(smiles))))
        mols = [(j, sanitizable(smiles[j])) for j in idx]
        keep = [(j, m) for j, m in mols if m is not None]
        if keep:
            out[[j for j, _ in keep]] = classify([m for _, m in keep])
    return out


def run(arm: str, classify) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results" / "candidates" / ARMS[arm] / "screened.csv")
    d["reason"] = [reason(r) for r in d.itertuples()]
    print(f"  {arm}: scoring all {len(d)} pooled molecules")
    d["c_all"] = score_all(list(d.smiles), classify)
    d["arm"] = arm

    g = (d.groupby("reason")
          .agg(n=("c_all", "size"), scored=("c_all", "count"),
               c_mean=("c_all", "mean"), c_sd=("c_all", "std"),
               heavy=("heavy_atoms", "mean"))
          .sort_values("c_mean", ascending=False))
    surv = d[d.reason == "survivor"].c_all
    print(f"\n  {arm}  (survivor mean C = {surv.mean():.4f})")
    print(f"    {'group':20} {'n':>7} {'mean C':>8} {'sd':>7} {'vs surv':>9} {'heavy':>7}")
    for r in g.itertuples():
        delta = r.c_mean - surv.mean()
        print(f"    {r.Index:20} {int(r.n):>7} {r.c_mean:>8.4f} {r.c_sd:>7.4f} "
              f"{delta:>+9.4f} {r.heavy:>7.1f}")
    return d


def _self_check() -> None:
    # Precedence must be disjoint and ordered as the screen counts: a molecule
    # that is both undersized and alert-bearing is attributed to undersized.
    probe = pd.DataFrame([
        dict(undersized=True, structural="", n_targeted=3, n_catalogue=1, in_training_flat=False),
        dict(undersized=False, structural="valence", n_targeted=0, n_catalogue=0, in_training_flat=False),
        dict(undersized=False, structural="", n_targeted=1, n_catalogue=1, in_training_flat=False),
        dict(undersized=False, structural="", n_targeted=0, n_catalogue=2, in_training_flat=False),
        dict(undersized=False, structural="", n_targeted=0, n_catalogue=0, in_training_flat=True),
        dict(undersized=False, structural="", n_targeted=0, n_catalogue=0, in_training_flat=False),
    ])
    got = [reason(r) for r in probe.itertuples()]
    assert got == ["undersized", "structural defect", "targeted alert",
                   "catalogue alert", "in training data", "survivor"], got
    # An unsanitizable SMILES must be NaN, not 0.0.
    v = score_all(["not_a_molecule", "CCO"], lambda m: np.full(len(m), 0.5))
    assert np.isnan(v[0]) and v[1] == 0.5, v
    print("precedence disjoint and ordered; unscorable rows are NaN")
    print("score_rejected self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="cgate")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()

    torch.set_num_threads(6)
    classify = load_frozen_classifier()
    print("reward ensemble: gin/gat/gine, GraphNorm, seed 0\n")
    arms = sorted(ARMS) if args.all else [args.arm]
    frames = [run(a, classify) for a in arms]
    df = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["arm", "smiles", "reason", "c_all", "heavy_atoms", "passes"]].to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

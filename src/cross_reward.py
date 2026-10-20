"""Score both generator arms with both rewards. The 2x2 that detects hacking.

THE QUESTION. One arm was trained against the gin/gat/gine GraphNorm ensemble,
the other against hybrid+BatchNorm. Each arm's score under its OWN reward is
near-tautological -- it says the policy hit the target it was given. What
separates "found real permeable chemistry" from "found this reward's blind
spots" is the off-diagonal: how does each arm score under the reward it never
saw?

  own >> other, both arms   each reward is being exploited on its own terms;
                            the agreement of either with reality is unproven
  own ~ other,  both arms   the signal transfers between materially different
                            classifiers, so it is not an artefact of either

Real BBBP molecules are scored by both rewards in the same run, because an
absolute probability is not interpretable on its own -- "generated sits where
real drugs sit, on a reward it never saw" is.

A NOTE ON WHAT THIS CANNOT SHOW. Both rewards are trained on BBBP and both
rate carbon tetrachloride at the top (0.987 and 1.000). Transfer between them
rules out reward-specific exploitation, not the objective: they agree that
solvents cross membranes because solvents cross membranes.

    python -m src.cross_reward
    python -m src.cross_reward --self-check
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from .datasets import ROOT, load_dataset
from .reward import load_frozen_classifier, load_frozen_hybrid, sanitizable

RDLogger.DisableLog("rdApp.*")

# Three ARMS, two CLASSIFIERS. The gate arm uses the same classifier as arm A;
# what differs is whether C is a reward term or a hurdle, so its in-loop column
# is the ensemble one -- but it was never paid to climb that column, which is
# the point of including it here.
ARMS = {
    "A: ensemble as term": ROOT / "results" / "candidates" / "brics10" / "screened.csv",
    "B: hybrid_batch as term": ROOT / "results" / "candidates" / "brics_hybrid" / "screened.csv",
    "GATE: ensemble as hurdle": ROOT / "results" / "candidates" / "brics_cgate" / "screened.csv",
}
OUT = ROOT / "results" / "pubchem" / "cross_reward.csv"
TRAINED_ON = {"A: ensemble as term": "ensemble",
              "B: hybrid_batch as term": "hybrid_batch",
              # gated, never rewarded -- marked separately so the * in the
              # table does not claim it was an optimization target
              "GATE: ensemble as hurdle": "ensemble"}
GATED = {"GATE: ensemble as hurdle"}


def _score(smiles: list[str], classify, batch: int = 256) -> np.ndarray:
    """NaN where RDKit will not sanitize, so absences never read as zeros."""
    out = np.full(len(smiles), np.nan)
    for i in range(0, len(smiles), batch):
        chunk = list(range(i, min(i + batch, len(smiles))))
        mols = [(j, sanitizable(smiles[j])) for j in chunk]
        keep = [(j, m) for j, m in mols if m is not None]
        if keep:
            out[[j for j, _ in keep]] = classify([m for _, m in keep])
    return out


def survivors(path) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d[d.passes.fillna(False).astype(bool)
             & ~d.in_training_flat.fillna(False).astype(bool)].copy()


def _self_check() -> None:
    a, b = load_frozen_classifier(), load_frozen_hybrid()
    # Both rewards must separate a CNS drug from a BBB- molecule, or neither
    # off-diagonal number means anything.
    pos, neg = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1", "NC(=O)c1ccc(N)cc1S(=O)(=O)O"
    for name, c in (("ensemble", a), ("hybrid_batch", b)):
        sp, sn = _score([pos], c)[0], _score([neg], c)[0]
        assert sp > sn, f"{name}: haloperidol {sp:.3f} <= sulfanilamide-acid {sn:.3f}"
        print(f"  {name:13} haloperidol {sp:.3f} > {sn:.3f}")
    # An unsanitizable SMILES must come back NaN, not 0.0 -- a zero would be
    # averaged in as "predicted impermeable".
    assert np.isnan(_score(["not_a_molecule"], a)[0])
    print("cross_reward self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()

    rewards = {"ensemble": load_frozen_classifier(),
               "hybrid_batch": load_frozen_hybrid()}
    print("rewards loaded: ensemble (gin/gat/gine, GraphNorm), "
          "hybrid_batch (4 archs, BatchNorm)\n")

    frames = []
    for arm, path in ARMS.items():
        if not path.exists():
            print(f"skip {arm}: {path} missing")
            continue
        s = survivors(path)
        print(f"{arm}: scoring {len(s)} survivors with both rewards")
        for rname, rfn in rewards.items():
            s[f"R_{rname}"] = _score(list(s.smiles), rfn)
        s["arm"] = arm
        frames.append(s)
    if not frames:
        raise SystemExit("no arms found")
    allsurv = pd.concat(frames, ignore_index=True)

    g = load_dataset("bbbp", verbose=False)
    rs = [x.smiles for x in g]
    ry = np.array([int(x.y.item()) for x in g])
    print(f"\nscoring {len(rs)} real BBBP molecules with both rewards")
    real = {k: _score(rs, v) for k, v in rewards.items()}

    print(f"\n{'=' * 74}\nTHE 2x2: each arm under each reward\n{'=' * 74}")
    print(f"\n{'population':<30} {'R=ensemble':>12} {'R=hybrid_batch':>16}")
    print("-" * 74)
    for arm in ARMS:
        sub = allsurv[allsurv.arm == arm]
        if not len(sub):
            continue
        e, h = sub.R_ensemble.mean(), sub.R_hybrid_batch.mean()
        own = TRAINED_ON[arm]
        gated = arm in GATED
        mark = lambda k: (" g" if gated else " *") if k == own else "  "
        print(f"{arm:<30} {e:>10.4f}{mark('ensemble')} {h:>14.4f}{mark('hybrid_batch')}")
    print("-" * 74)
    print(f"{'real BBB+ (control)':<30} {real['ensemble'][ry == 1].mean():>10.4f}   "
          f"{real['hybrid_batch'][ry == 1].mean():>14.4f}")
    print(f"{'real BBB- (control)':<30} {real['ensemble'][ry == 0].mean():>10.4f}   "
          f"{real['hybrid_batch'][ry == 0].mean():>14.4f}")
    print("\n  * = optimized against (in-loop)   "
          "g = gated on, never rewarded")

    print(f"\n{'=' * 74}\nOFF-DIAGONAL: does the signal transfer?\n{'=' * 74}")
    for arm in ARMS:
        sub = allsurv[allsurv.arm == arm]
        if not len(sub):
            continue
        own = f"R_{TRAINED_ON[arm]}"
        other = "R_hybrid_batch" if TRAINED_ON[arm] == "ensemble" else "R_ensemble"
        both = sub.dropna(subset=[own, other])
        drop = both[own].mean() - both[other].mean()
        print(f"\n{arm}")
        print(f"  own reward   {both[own].mean():.4f}")
        print(f"  other reward {both[other].mean():.4f}   drop {drop:+.4f}")
        print(f"  spearman     {both[own].corr(both[other], method='spearman'):+.4f}")
        top = both.nlargest(250, own)
        print(f"  top 250 by own reward: own {top[own].mean():.4f}, "
              f"other {top[other].mean():.4f}  (gap {top[own].mean() - top[other].mean():+.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    allsurv.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("\n  Both rewards are BBBP-trained and both rate carbon tetrachloride")
    print("  at the top. Transfer between them rules out reward-specific")
    print("  exploitation, not the objective.")


if __name__ == "__main__":
    main()

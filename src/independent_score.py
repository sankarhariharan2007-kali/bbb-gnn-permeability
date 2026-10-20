"""Score the generated molecules with a classifier the generator never saw.

WHY THIS IS THE TEST THAT MATTERS. `screened.csv`'s `c_score` comes from
`reward.load_frozen_classifier` -- the gin/gat/gine GraphNorm ensemble the
generator was *optimized against*. A high `c_score` therefore says the policy
succeeded at its own objective, which is very close to tautological and cannot
distinguish "found permeable chemistry" from "found this ensemble's blind
spots". Reward hacking is invisible to the reward.

So score the same molecules with the REPORTED classifier instead:
hybrid + BatchNorm (`results/hybrid_batch/`), which differs on every axis that
matters here. It has 8 RDKit descriptors the reward model does not see, a
different normalization, and is averaged over all ten split seeds so it is not
tied to the single fold the reward ensemble used. Nothing in GRPO ever
consulted it.

HOW TO READ THE RESULT. Real BBB+ molecules are scored the same way, in the
same run, as the calibration the comparison needs -- an absolute probability
means little, but "generated sits where real drugs sit" is interpretable. Two
outcomes and what each would mean:

  agreement    generated molecules score like real drugs on an out-of-loop
               judge -> the permeability signal is real, not ensemble-specific
  divergence   high c_score, low independent score -> the generator found
               this reward ensemble's blind spots, i.e. reward hacking that
               the in-loop number could never have revealed

BatchNorm is safe HERE and nowhere else in this pipeline. In `.eval()` it uses
running statistics, so a molecule's score does not depend on its batch. That
is why it may judge but must never be the reward (see documentation/
full_pipeline.md S4).

    python -m src.independent_score              # survivors + real control
    python -m src.independent_score --all        # the full 19219 pool
    python -m src.independent_score --self-check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .datasets import ROOT, load_dataset
from .featurize import run_contract, smiles_to_graph
from .hybrid_features import compute_descriptors
from .models_hybrid import build_hybrid_model

SCREENED = ROOT / "results" / "candidates" / "brics10" / "screened.csv"
HYBRID_BATCH = ROOT / "results" / "hybrid_batch"
OUT = ROOT / "results" / "pubchem" / "independent_score.csv"

# gcn is the best hybrid_batch cell on BBBP (0.9214 over 10 seeds). All ten
# seeds are averaged: each is a different scaffold split, so the ensemble is
# not tied to the one fold the reward ensemble was built from.
MODEL = "gcn"
SEEDS = list(range(10))


def _members(dataset: str = "bbbp"):
    """(model, mu, sigma) per seed, each rebuilt as its run recorded it."""
    out = []
    for seed in SEEDS:
        d = HYBRID_BATCH / dataset / MODEL / f"seed{seed}"
        if not (d / "model.pt").exists():
            continue
        arch = run_contract(d)              # raises on a feature_id mismatch
        norm = json.loads((d / "metrics.json").read_text())["norm"]
        assert norm == "batch", f"{d} is norm={norm}, expected the BatchNorm arm"
        net = build_hybrid_model(MODEL, **arch)
        net.load_state_dict(torch.load(d / "model.pt", map_location="cpu"))
        net.eval()                          # running stats, not batch stats
        net.requires_grad_(False)
        sc = np.load(d / "descriptor_scaling.npz")
        out.append((net, sc["mu"], sc["sigma"]))
    if not out:
        raise SystemExit(f"no hybrid_batch/{dataset}/{MODEL} checkpoints found")
    return out


@torch.no_grad()
def score(smiles: list[str], members, batch_size: int = 256) -> np.ndarray:
    """Mean probability over the members. NaN for unparseable SMILES."""
    graphs, keep = [], []
    for i, s in enumerate(smiles):
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
            keep.append(i)
    out = np.full(len(smiles), np.nan)
    if not graphs:
        return out
    # Descriptors are standardized with each CHECKPOINT'S OWN train-fold
    # mu/sigma, never with statistics refit on the molecules being scored --
    # that would leak the generated distribution into its own judge.
    raw = np.array([compute_descriptors(g.smiles) for g in graphs], dtype=np.float64)
    per_member = []
    for net, mu, sigma in members:
        std = (raw - mu) / sigma
        for g, row in zip(graphs, std):
            g.descriptors = torch.tensor(row, dtype=torch.float).unsqueeze(0)
        probs = []
        for batch in DataLoader(graphs, batch_size=batch_size):
            probs.append(torch.sigmoid(net(batch)).numpy())
        per_member.append(np.concatenate(probs))
    out[keep] = np.mean(per_member, axis=0)
    return out


def real_control(dataset: str = "bbbp") -> tuple[list[str], np.ndarray]:
    g = load_dataset(dataset, verbose=False)
    return [x.smiles for x in g], np.array([int(x.y.item()) for x in g])


def _band(v: np.ndarray) -> str:
    v = v[~np.isnan(v)]
    return (f"mean {v.mean():.3f}  sd {v.std(ddof=1):.3f}  "
            f"median {np.median(v):.3f}  >0.5 {np.mean(v > 0.5):.1%}")


def _self_check() -> None:
    members = _members()
    assert len(members) == 10, f"{len(members)} members, expected 10"

    # The judge must be batch-independent in eval(). If it is not, a molecule's
    # score depends on what it was scored alongside and every comparison below
    # is meaningless.
    probe = ["CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
             "ClC(Cl)(Cl)Cl", "O=C(O)c1ccccc1O"]
    alone = np.array([score([p], members)[0] for p in probe])
    crowd = score(probe, members)
    drift = np.abs(alone - crowd).max()
    assert drift < 1e-5, f"batch dependence {drift:.2e} -- eval() not in force"

    # It must separate a known CNS drug from a solvent, or it cannot speak to
    # the question the generator's output raises.
    caff, ccl4 = score(["Cn1cnc2c1c(=O)n(C)c(=O)n2C"], members)[0], alone[2]
    print(f"  batch-independence drift {drift:.2e}")
    print(f"  caffeine {caff:.3f} vs carbon tetrachloride {ccl4:.3f}")
    print("independent_score self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="score the full pool, not just the survivors")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()

    members = _members()
    print(f"judge: hybrid_batch/{MODEL}, {len(members)} seeds, BatchNorm, "
          f"eval() -> batch-independent\n")

    scr = pd.read_csv(SCREENED)
    sel = scr if args.all else scr[scr.passes.fillna(False).astype(bool)
                                   & ~scr.in_training_flat.fillna(False).astype(bool)]
    sel = sel.copy()
    print(f"scoring {len(sel)} generated molecules")
    sel["independent"] = score(list(sel.smiles), members)

    rs, ry = real_control()
    print(f"scoring {len(rs)} real BBBP molecules as the control")
    real = score(rs, members)

    print(f"\n{'=' * 72}\nOUT-OF-LOOP CLASSIFIER vs THE REWARD IT WAS TRAINED ON\n{'=' * 72}")
    print(f"\n  generated, independent judge : {_band(sel.independent.to_numpy())}")
    print(f"  real BBB+ , independent judge : {_band(real[ry == 1])}")
    print(f"  real BBB- , independent judge : {_band(real[ry == 0])}")
    if "c_score" in sel:
        print(f"\n  generated, the REWARD ensemble: {_band(sel.c_score.to_numpy())}")
        both = sel.dropna(subset=["c_score", "independent"])
        if len(both) > 2:
            pear = both.c_score.corr(both.independent)
            spear = both.c_score.corr(both.independent, method="spearman")
            gap = (both.c_score - both.independent).mean()
            print(f"\n  agreement over {len(both)} molecules: "
                  f"pearson {pear:+.3f}, spearman {spear:+.3f}")
            print(f"  mean(reward - independent) = {gap:+.3f}  "
                  f"{'<- reward is OPTIMISTIC' if gap > 0.05 else ''}")
            top = both.nlargest(100, "c_score")
            print(f"  top 100 by reward: reward {top.c_score.mean():.3f} vs "
                  f"independent {top.independent.mean():.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("\n  The independent judge never influenced training, so where it")
    print("  agrees the permeability signal is not an artefact of the reward")
    print("  ensemble. It is still a BBBP-trained model: it answers 'does this")
    print("  cross a membrane', not 'is this a drug'.")


if __name__ == "__main__":
    main()

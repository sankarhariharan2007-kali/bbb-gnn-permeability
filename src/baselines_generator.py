"""Does GRPO beat its own warm start, and does either beat random assembly?

THE QUESTION A REVIEWER ASKS FIRST, and which this project had never answered.
Every generator result here is reported against other arms of itself. If the
trained policy does not beat (a) the MLE warm start it began from and (b) a
randomly initialized policy over the same action space, then the reinforcement
learning contributes nothing and no comparison between arms means anything.

The two baselines are deliberately the cheap, damaging ones rather than a
published pipeline. A REINVENT or MOSES/GuacaMol comparison would differ in
action space AND training corpus -- this generator trains on 1,965 BBBP
molecules, not ZINC or ChEMBL -- so a difference would be uninterpretable, and
it answers a question about distribution modelling that is not the question
here. Random assembly over the SAME vocabulary is the honest floor: it is the
AddCarbon-style trivial baseline for this action space.

MATCHED SAMPLE SIZE IS NOT OPTIONAL. Scaffold fraction reads 0.758 at n=199 and
0.595 at n=982 on the same real molecules, a swing larger than any effect here,
so every arm is sampled as the same number of groups of exactly 64 and the
diversity statistic is computed per group and averaged. Comparing a pooled
figure against a per-group one would manufacture an effect.

    python -m src.baselines_generator              # 8 groups of 64 per arm
    python -m src.baselines_generator --groups 16
    python -m src.baselines_generator --self-check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

from .datasets import ROOT, load_dataset
from .diversity import scaffold_count
from .reward import load_frozen_classifier, sanitizable

RDLogger.DisableLog("rdApp.*")

WARM_START = ROOT / "results" / "generator" / "bbbp_brics_pretrained.pt"
TRAINED = ROOT / "results" / "brics_cgate"      # the recommended arm
TRAINED_A = ROOT / "results" / "brics_kl1"      # the original arm
GROUP = 64
OUT = ROOT / "results" / "pubchem" / "generator_baselines.csv"


def _untrained():
    """A randomly initialized policy over the identical action space.

    Built from the same vocabulary as the warm start so the action space is
    held fixed and only the weights differ -- otherwise this would compare two
    different problems.
    """
    from .brics_generator import BricsGenerator
    ck = torch.load(WARM_START, map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    # Same vocabulary and same architecture dims as the warm start; only the
    # weights are random. Anything else would change the action space.
    return BricsGenerator(ck["vocab"], emb=ck["emb"], hidden=ck["hidden"],
                          layers=ck["layers"], z_dim=ck["z_dim"],
                          type_emb=ck["type_emb"])


def _sample(gen, groups: int, seed: int) -> list[list[str]]:
    """`groups` lists of exactly GROUP raw samples each."""
    out = []
    for g in range(groups):
        torch.manual_seed(seed + 7919 * g)
        out.append(gen.sample(GROUP, device="cpu")["smiles"])
    return out


def _metrics(groups: list[list[str]], classify) -> dict:
    """Per-group statistics, averaged. Validity is the share of GROUP, not of
    whatever survived, so an arm cannot look good by emitting fewer molecules.
    """
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED

    from .hybrid_features import CNSMPO

    per = []
    for smis in groups:
        mols = [sanitizable(s) for s in smis if s]
        mols = [m for m in mols if m is not None]
        n = len(mols)
        row = {"valid": n / GROUP}
        if n:
            row["c"] = float(np.mean(classify(mols)))
            row["qed"] = float(np.mean([QED.qed(m) for m in mols]))
            row["mpo"] = float(np.mean([
                CNSMPO.score(Descriptors.MolWt(m), Crippen.MolLogP(m),
                             Descriptors.TPSA(m), Lipinski.NumHDonors(m))[0] / 5.0
                for m in mols]))
            row["scaffold"] = scaffold_count(mols)[1]
            row["uniq"] = len({Chem.MolToSmiles(m) for m in mols}) / n
            row["heavy"] = float(np.mean([m.GetNumHeavyAtoms() for m in mols]))
        per.append(row)
    df = pd.DataFrame(per)
    return {k: (df[k].mean(), df[k].std(ddof=1)) for k in df.columns}


def _real_control(groups: int, classify, seed: int = 0) -> dict:
    """Real BBBP molecules drawn the same way: groups of exactly GROUP.

    The reference the generated arms are read against. Without it an absolute
    scaffold fraction is uninterpretable, because the statistic depends on
    sample size rather than only on the policy.
    """
    g = load_dataset("bbbp", verbose=False)
    pool = [x.smiles for x in g]
    rng = np.random.default_rng(seed)
    return _metrics([list(rng.choice(pool, GROUP, replace=False))
                     for _ in range(groups)], classify)


def _self_check() -> None:
    gen = _untrained()
    s = _sample(gen, 1, seed=0)[0]
    assert len(s) == GROUP, f"{len(s)} samples, expected {GROUP}"
    # An untrained policy must still produce mostly sanitizable output -- that
    # is the point of assembly being valid by construction, and if it did not
    # the baseline would be measuring the engine rather than the policy.
    ok = sum(1 for x in s if x and sanitizable(x) is not None)
    assert ok > GROUP * 0.5, f"only {ok}/{GROUP} sanitize from an untrained policy"
    # Sampling must be seed-deterministic, or none of these means are stable.
    assert _sample(gen, 1, seed=0)[0] == s, "sampling is not reproducible"
    # Validity is a share of GROUP, so an arm emitting nothing scores 0 rather
    # than being excluded.
    assert _metrics([[""] * GROUP], lambda m: np.zeros(len(m)))["valid"][0] == 0.0
    print(f"untrained policy sanitizes {ok}/{GROUP}; sampling reproducible")
    print("baselines_generator self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, default=8,
                    help=f"groups of {GROUP} per arm")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()

    from .brics_generator import load_generator

    classify = load_frozen_classifier()
    torch.set_num_threads(6)
    arms: dict[str, list[list[str]]] = {}

    print(f"sampling {args.groups} groups of {GROUP} per arm\n")
    print("  random policy (untrained, same action space)")
    arms["random policy"] = _sample(_untrained(), args.groups, seed=11)

    print("  MLE warm start (no GRPO)")
    arms["MLE warm start"] = _sample(
        load_generator(WARM_START, "cpu"), args.groups, seed=22)

    for label, root in (("GRPO (arm A)", TRAINED_A), ("GRPO (gate arm)", TRAINED)):
        ck = root / "pseed3" / "bbbp_grpo.pt"
        if not ck.exists():
            ck = next((p for p in sorted(root.glob("*/bbbp_grpo.pt"))), None)
        if ck is None:
            print(f"  {label}: no checkpoint on disk, skipping "
                  f"(GRPO checkpoints are deliberately uncommitted)")
            continue
        print(f"  {label} <- {ck.parent.name}")
        arms[label] = _sample(load_generator(ck, "cpu"), args.groups, seed=33)

    rows = []
    for label, groups in arms.items():
        m = _metrics(groups, classify)
        rows.append({"arm": label, **{k: v[0] for k, v in m.items()}})
    ctl = _real_control(args.groups, classify)
    rows.append({"arm": "real BBBP (control)", **{k: v[0] for k, v in ctl.items()}})

    df = pd.DataFrame(rows)
    cols = ["arm", "valid", "c", "qed", "mpo", "scaffold", "uniq", "heavy"]
    df = df[[c for c in cols if c in df.columns]]
    print(f"\n{'=' * 78}\nGENERATOR BASELINES "
          f"({args.groups} groups of {GROUP}, matched)\n{'=' * 78}")
    print(df.round(4).to_string(index=False))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("\n  scaffold is PER GROUP at n=64 and averaged, never pooled: the")
    print("  statistic depends on sample size, so only matched draws compare.")
    print("  If a GRPO row does not beat both baselines, the RL earns nothing")
    print("  and no comparison between reward arms is meaningful.")


if __name__ == "__main__":
    main()

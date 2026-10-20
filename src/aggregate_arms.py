"""Does a Pareto reduction beat the linear one? linear vs nsga2 vs composite.

The reward R = w_D*D + w_C*C + w_Q*QED + w_S*SA is a Minkowski-weighted
linear scalarization, so Theorem 1 of Das & Dennis (1997) applies: no choice
of weights reaches a Pareto-optimal candidate inside a concave fold of the
frontier. Measured on this project's own sampled groups, a linear reduction
reaches **0.48** of the per-group Pareto front over 2000 random weightings,
and the four objectives are nearly uncorrelated (mean pairwise +0.009, D vs C
= -0.258) -- the regime where a rank-based reduction has the most to find.
`src/pareto.py` has the derivation and a runnable concave-fold demo.

All three arms share the four seeds and the per-seed warm starts, so the
comparison is paired on both. That matters more than it sounds: warm-start
variance is ~2x GRPO-seed variance here, and pairing on only the seed is what
made the VAE comparison read as a clean regression when it was noise.

A metrics table cannot tell "better" from "Goodharting differently", so the
report ends with the two off-instrument checks:

  * D_real -- the discriminator's score on real BBBP+ drugs is 0.94 while it
    gives the generator 0.80 in every configuration tried so far. Closing
    that is the realism question.
  * the INSTRUMENT GAP, C(BBBP) - C(B3DB): the training ensemble minus a
    judge it never saw. Real BBB+ drugs sit at **+0.037** under the current
    ensembles (n=1560; recomputed 2026-10-11). The figure here previously read
    -0.029, from the retired pre-contract ensemble -- and since the diagnostic
    measures DISTANCE from the real-drug gap, a reference of the wrong sign
    inverts every verdict drawn from it. Recompute it whenever the ensemble
    changes. A gap further from the reference means the arm found more of the
    selector's blind spots, not better molecules -- so an arm may "win" on
    c_mean and still be the worse arm.

  * D_real is NOT a symmetric judge. The discriminator loaded below comes from
    results/gan/seed0, a pre-contract SELFIES run at kl_coef=0.02, so it was
    trained to separate real molecules from a SELFIES generator specifically.
    Scoring a different generator family with it is out of domain: a high
    D_real may mean only that the family lacks the artifacts this D learned.
    For a cross-family realism claim, train one discriminator on real versus a
    mixture of both families instead.

    python -m src.aggregate_arms
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

from .datasets import ROOT, _load_raw
from .kl_sweep import summarize_run

RDLogger.DisableLog("rdApp.*")

SEEDS = (1, 2, 3, 4)
ARMS = {
    "linear": ROOT / "results" / "factorial_ws" / "mle_withC",
    "nsga2": ROOT / "results" / "pareto_arm" / "nsga2",
    "composite": ROOT / "results" / "pareto_arm" / "composite",
}
METRICS = ("scaffold_frac", "tanimoto_dist", "unique_smiles_frac", "valid_frac",
           "c_mean", "kl", "tpsa_spread_ratio", "logp_spread_ratio",
           "mw_spread_ratio")
LAST_N = 20


def pool(d: Path, last_n: int = LAST_N) -> list[str]:
    seen: set[str] = set()
    for f in sorted((d / "samples").glob("step*.json"))[-last_n:]:
        seen.update(s for s in json.loads(f.read_text()) if s)
    return [Chem.MolToSmiles(m)
            for m in (Chem.MolFromSmiles(s) for s in sorted(seen))
            if m is not None]


def table() -> None:
    runs = {a: [summarize_run(d / f"seed{s}") for s in SEEDS]
            for a, d in ARMS.items()}
    base = runs["linear"]
    others = [a for a in ARMS if a != "linear"]
    head = "".join(f"{a:>12}{'up':>5}" for a in others)
    print(f"{'metric':<20}{'linear':>9}" + head)
    print("-" * (29 + 17 * len(others)))
    for m in METRICS:
        b = np.array([r[m] for r in base])
        line = f"{m:<20}{b.mean():>9.3f}"
        for a in others:
            x = np.array([r[m] for r in runs[a]])
            d = x - b
            line += f"{x.mean():>12.3f}{f'{int((d > 0).sum())}/{len(SEEDS)}':>5}"
        print(line)
    print("\n(per-arm column is the arm's mean; 'up' is seeds improved vs linear)")


def offinstrument() -> None:
    from .reward import load_frozen_classifier
    from .train_gan import Discriminator

    bbbp = load_frozen_classifier(dataset="bbbp")
    b3db = load_frozen_classifier(dataset="b3db", models=("gin", "gat", "gine"))
    # norm="batch" replays the architecture this checkpoint was TRAINED with
    # rather than imposing today's default -- the same principle
    # featurize.run_contract applies to classifiers. Without it the load fails
    # on running_mean against GraphNorm's mean_scale, and this whole function
    # has been unrunnable since the norm default changed.
    disc = Discriminator(norm="batch")
    disc.load_state_dict(torch.load(ROOT / "results/gan/seed0/bbbp_disc.pt",
                                    map_location="cpu"))
    disc.eval()

    real = [s for s in _load_raw("bbbp").query("label==1")["smiles"].dropna()
            if Chem.MolFromSmiles(s)]
    rm = [Chem.MolFromSmiles(s) for s in real]

    print(f"\n{'arm':<12}{'D_real':>9}{'C(BBBP)':>10}{'C(B3DB)':>10}{'gap':>9}"
          f"{'gap sd':>9}")
    print("-" * 59)
    for a, d in ARMS.items():
        ds, cb, c3, gaps = [], [], [], []
        for s in SEEDS:
            smis = pool(d / f"seed{s}")
            mols = [Chem.MolFromSmiles(x) for x in smis]
            x1, x2 = bbbp(mols).mean(), b3db(mols).mean()
            ds.append(disc.score(smis).mean()); cb.append(x1); c3.append(x2)
            gaps.append(x1 - x2)
        print(f"{a:<12}{np.mean(ds):>9.3f}{np.mean(cb):>10.3f}"
              f"{np.mean(c3):>10.3f}{np.mean(gaps):>+9.3f}{np.std(gaps):>9.3f}")
    rb, r3 = bbbp(rm).mean(), b3db(rm).mean()
    print(f"{'REAL drugs':<12}{disc.score(real).mean():>9.3f}{rb:>10.3f}"
          f"{r3:>10.3f}{rb - r3:>+9.3f}{'':>9}")
    print("\nD_real: higher is more real-looking. gap = selector minus unseen")
    print("judge; real drugs sit at +0.037, so a gap FURTHER FROM THAT is worse even when")
    print("C(BBBP) is higher -- that is the instrument being gamed, not progress.")


def main() -> int:
    missing = [d / f"seed{s}" for d in ARMS.values() for s in SEEDS
               if not (d / f"seed{s}" / "config.json").exists()]
    if missing:
        print("missing runs:\n  " + "\n  ".join(str(p) for p in missing))
        return 1
    # Guard the premise: --aggregate was one of the three flags silently
    # dropped by the duplicated assemble() call, so an arm whose config does
    # not record the mode it was supposed to run is a linear run wearing a
    # different directory name.
    for a, d in ARMS.items():
        want = "linear" if a == "linear" else a
        for s in SEEDS:
            got = json.loads((d / f"seed{s}" / "config.json").read_text())["aggregate"]
            assert got == want, f"{a}/seed{s} recorded aggregate={got!r}"
    table()
    offinstrument()
    return 0


if __name__ == "__main__":
    sys.exit(main())

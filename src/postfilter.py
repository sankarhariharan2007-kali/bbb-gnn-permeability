"""Post-generation filtering: pool a run's samples, screen them, rank survivors.

Path 2. The active loop is left as the `gate` arm -- size gate only -- and
structural screening happens HERE, after training, where the generator
cannot optimize against it.

That separation is the whole point. A reactive-group penalty inside the
reward was measured to reduce the patterns it was trained against (0.339 ->
0.203) while leaving independent catalogues unmoved (BRENK 0.438 -> 0.406,
within seed noise) -- the generator learned the checklist, not the
chemistry. A filter the policy never sees cannot be gamed, because no
gradient, and no reward, ever reaches it.

The screen is the union of the targeted SMARTS set and BRENK, PAINS A/B/C
and NIH. Using everything is safe here for the same reason it was unsafe in
the loop: nothing is being trained, so no instrument is being consumed.

    python -m src.postfilter results/guarded/gate results/arm_study/gate_s1 \
        --from-step 120 --out results/candidates

TWO CAUTIONS, both measured rather than asserted, and both printed by the
script so they cannot be lost between here and a slide:

  * Passing this screen is NOT evidence a molecule is safe. 39% of the real
    BBB+ drugs in BBBP fail the same union. The filter removes recognized
    liabilities; it says nothing about the ones no catalogue encodes.

  * Ranking by C is only meaningful if C still discriminates. In the
    collapsed baseline the within-group sd of C fell to 0.006 with every
    molecule above 0.95, and MC dropout put the model's own uncertainty
    well above that -- a ranking there is noise. The script reports the
    spread of C among survivors and says so when it is too tight to rank on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, FilterCatalog, QED
from rdkit.Chem.FilterCatalog import FilterCatalogParams as P

from .alerts import alert_names, count_alerts
from .datasets import ROOT
from .diversity import summarize

RDLogger.DisableLog("rdApp.*")

CATALOGUES = ("BRENK", "PAINS_A", "PAINS_B", "PAINS_C", "NIH")


def _union_catalog() -> FilterCatalog.FilterCatalog:
    p = P()
    for name in CATALOGUES:
        p.AddCatalog(getattr(P.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(p)


def pool_samples(run_dirs: list[Path], from_step: int = 120) -> list[str]:
    """Canonical, deduplicated SMILES from the per-step dumps.

    Late steps only: early groups are close to the warm start and say
    little about what the trained policy produces. Deduplication is on the
    CANONICAL form, so two spellings of one molecule count once -- without
    it a collapsed run would look productive.
    """
    seen: dict[str, None] = {}
    for d in run_dirs:
        sdir = Path(d) / "samples"
        if not sdir.exists():
            print(f"  no samples/ in {d}")
            continue
        for f in sorted(sdir.glob("step*.json")):
            if int(f.stem[4:]) < from_step:
                continue
            for s in json.load(open(f)):
                m = Chem.MolFromSmiles(s) if s else None
                if m is not None:
                    seen.setdefault(Chem.MolToSmiles(m), None)
    return list(seen)


def structural_problems(mol: Chem.Mol) -> list[str]:
    """Defects RDKit sanitization accepts but a chemist would not.

    Added because the candidate list was wrong without it. 7.7% of
    survivors, and 16.4% of those with >=25 heavy atoms, carried radical or
    carbene centres -- [C] and [CH] parse, sanitize, pass the size gate and
    trip no substructure catalogue, so they reached the ranked output. They
    are also enriched exactly where nearest-neighbour similarity is lowest,
    so ranking by novelty surfaced them first: three of the twelve most
    distant large molecules were radicals.

    This is a validity check, not a liability filter, which is why it is
    separate from the alert catalogues.
    """
    bad = []
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        bad.append("radical")
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        # Net-neutral zwitterions are fine; a lone uncompensated charge on a
        # generated structure is usually a decoding artifact.
        if Chem.GetFormalCharge(mol) != 0:
            bad.append("net_charge")
    # A generated SMILES can request a ring the geometry cannot close;
    # embedding failure is the cheapest available proxy for that.
    if mol.GetRingInfo().NumRings() and not mol.GetRingInfo().AtomRings():
        bad.append("ring_perception_failed")
    return bad


def screen(smiles: list[str], min_heavy_atoms: int = 10) -> pd.DataFrame:
    """One row per molecule with every rejection reason, not just the first.

    Rows are kept rather than dropped so the discarded set stays inspectable
    -- knowing WHAT was thrown away is how the next round of patterns gets
    chosen, and a script that silently returns survivors hides that.
    """
    cat = _union_catalog()
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        cat_hits = [e.GetDescription() for e in cat.GetMatches(m)]
        targeted = alert_names(m)
        structural = structural_problems(m)
        hv = m.GetNumHeavyAtoms()
        rows.append({
            "smiles": s, "heavy_atoms": hv, "mw": Descriptors.MolWt(m),
            "qed": QED.qed(m),
            "n_targeted": len(targeted), "n_catalogue": len(cat_hits),
            "alerts": "|".join(sorted(set(targeted + cat_hits))),
            "structural": "|".join(structural),
            "undersized": hv < min_heavy_atoms,
            "passes": (hv >= min_heavy_atoms and not cat_hits
                       and not targeted and not structural),
        })
    return pd.DataFrame(rows)


def add_scores(df: pd.DataFrame, dataset: str = "bbbp",
               device: str = "cpu") -> pd.DataFrame:
    """Frozen-classifier score for the survivors, in batches."""
    from .reward import load_frozen_classifier, sanitizable

    classify = load_frozen_classifier(dataset=dataset, device=device)
    out = np.full(len(df), np.nan)
    idx = np.where(df["passes"].to_numpy())[0]
    for i in range(0, len(idx), 256):
        chunk = idx[i:i + 256]
        mols = [sanitizable(df.smiles.iloc[j]) for j in chunk]
        keep = [(j, m) for j, m in zip(chunk, mols) if m is not None]
        if keep:
            out[[j for j, _ in keep]] = classify([m for _, m in keep])
    df = df.copy()
    df["c_score"] = out
    return df


def novelty(df: pd.DataFrame, dataset: str = "bbbp") -> pd.DataFrame:
    """Flag survivors that are literally training molecules.

    A generator that rediscovers its training set is not generating. This
    is cheap to check and easy to forget.
    """
    from .datasets import DATASETS

    cfg = DATASETS[dataset]
    raw = pd.read_csv(cfg["path"], sep=cfg["sep"])[cfg["smiles_col"]].dropna()
    known, known_flat = set(), set()
    for s in raw:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        known.add(Chem.MolToSmiles(m))
        # Constitution only. An exact-SMILES test is STEREO-SENSITIVE, and the
        # BRICS generator emits flat molecules (brics_assembly.corpus strips
        # stereo, because reassemble flips the anchor's parity), so a training
        # molecule with defined centres canonicalizes differently and never
        # matches. Measured on a 5888-molecule BRICS pool: the exact test found
        # 1009 training molecules, the flat test 1486 -- 477 missed, and 171 of
        # the 1891 "survivors" were training molecules wearing no stereo.
        #
        # Excluding on the flat form is the defensible default rather than a
        # convenience: the featurizer has no chirality features, so C(m) is
        # identical across stereoisomers, and a candidate differing from a
        # training molecule only at a centre the scorer cannot see is not a new
        # candidate. `in_training_set` is kept so the old number stays visible.
        flat = Chem.MolFromSmiles(Chem.MolToSmiles(m))
        if flat is not None:
            Chem.RemoveStereochemistry(flat)
            known_flat.add(Chem.MolToSmiles(flat))
    df = df.copy()
    df["in_training_set"] = df.smiles.isin(known)
    df["in_training_flat"] = df.smiles.isin(known_flat) | df["in_training_set"]
    return df


#: Permeability as a requirement for the shortlist, at the same threshold the
#: reward gate uses: the 5th percentile of real BBB+ molecules under the
#: ensemble classifier. Calibrated on the positive class, never on generated
#: output.
RANK_MIN_C = 0.615
RANK_BY = ("c_score", "mpo", "mpo_gated")


def add_mpo(df: pd.DataFrame) -> pd.DataFrame:
    """CNS-MPO (Wager, 5 of 6 terms) per molecule, reusing hybrid_features.

    Normalized to [0, 1] by dividing by 5, because that implementation omits
    the basic-pKa term rather than fabricating it -- so its maximum is 5, not 6.
    """
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    from .hybrid_features import CNSMPO

    vals = []
    for s in df.smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            vals.append(np.nan)
            continue
        composite, _ = CNSMPO.score(Descriptors.MolWt(m), Crippen.MolLogP(m),
                                    Descriptors.TPSA(m), Lipinski.NumHDonors(m))
        vals.append(composite / 5.0)
    df = df.copy()
    df["cns_mpo"] = vals
    return df


def rank(survivors: pd.DataFrame, by: str, top: int) -> pd.DataFrame:
    """Order the shortlist. WHY NOT ABSOLUTE C, WHICH WAS THE DEFAULT.

    Ranking by absolute C selects the lipophilic extreme of the survivor set.
    Measured: the top 25 by C run logP 4.89 and TPSA 14.4 against the
    population's 3.04 and 49.9, and against real BBB+ drugs' 2.67 and 54.1.
    The population is drug-like; the 25 rows anyone actually reads are not.

    That is the permeability objective's bias -- the same gradient that once
    found carbon tetrachloride -- acting inside the size-gated space. And it is
    a property of the RANKING, not of the generator: the C-gate arm shows the
    same drift (+2.02 logP) despite never optimizing C, so no reward change
    fixes it.

    `mpo_gated` is the honest default for a shortlist: require permeability at
    the real-BBB+ 5th percentile, then order by CNS-MPO -- permeability as a
    hurdle, drug-likeness as the objective. `c_score` is kept so the original
    candidate files reproduce exactly. `mpo` drops the permeability
    requirement and is a diagnostic only.
    """
    if by == "c_score":
        return survivors.sort_values("c_score", ascending=False).head(top)
    # CNS-MPO SATURATES AND MUST BE TIE-BROKEN. 950 of the 9722 survivors that
    # clear the C hurdle score a perfect 1.000, so "top 25 by MPO" would be an
    # arbitrary 25 of 950 -- the same defect as ranking a saturated C, which is
    # what this function exists to fix. Sorting by (MPO, C) breaks the tie on
    # permeability *within* ideal drug-space, which is the question being asked.
    if by == "mpo":
        return survivors.sort_values(["cns_mpo", "c_score"],
                                     ascending=False).head(top)
    if by == "mpo_gated":
        ok = survivors[survivors.c_score >= RANK_MIN_C]
        if not len(ok):
            print(f"  !! no survivor reaches C >= {RANK_MIN_C}; using mpo alone")
            ok = survivors
        return ok.sort_values(["cns_mpo", "c_score"], ascending=False).head(top)
    raise ValueError(f"unknown rank_by {by!r}")


def report(df: pd.DataFrame, top: int = 25,
           rank_by: str = "c_score") -> pd.DataFrame:
    total = len(df)
    survivors = df[df["passes"] & ~df["in_training_flat"]].copy()
    print("=" * 74)
    print("POST-GENERATION SCREEN")
    print("=" * 74)
    print(f"  pooled, deduplicated : {total}")
    print(f"  undersized           : {int(df.undersized.sum())}")
    print(f"  structural defect    : {int((df.structural != '').sum())}")
    print(f"  targeted alert       : {int((df.n_targeted > 0).sum())}")
    print(f"  catalogue alert      : {int((df.n_catalogue > 0).sum())}")
    print(f"  already in training  : {int(df.in_training_flat.sum())}"
          f"  (exact-SMILES only: {int(df.in_training_set.sum())})")
    print(f"  SURVIVORS            : {len(survivors)} ({len(survivors)/max(total,1):.0%})")

    pre = summarize(df.smiles.tolist())
    post = summarize(survivors.smiles.tolist()) if len(survivors) else None
    if post:
        print(f"\n  scaffold fraction    : {pre['scaffold_frac']:.3f} -> "
              f"{post['scaffold_frac']:.3f}")
        print(f"  tanimoto distance    : {pre['tanimoto_dist']:.3f} -> "
              f"{post['tanimoto_dist']:.3f}")

    if len(survivors) and survivors.c_score.notna().any():
        c = survivors.c_score.dropna()
        print(f"\n  C among survivors    : mean {c.mean():.3f}  sd {c.std():.3f}  "
              f"{(c > 0.95).mean():.0%} above 0.95")
        # The saturation test. MC dropout put the classifier's own
        # uncertainty near 0.95 in logit space; once the spread between
        # molecules falls far below that, ranking on C orders noise.
        if c.std() < 0.02:
            print("  !! C is saturated here (sd < 0.02). Ranking by it orders")
            print("     noise rather than permeability -- rank on something else,")
            print("     or treat the survivor set as unordered.")

    print("\n  Passing this screen is NOT evidence of safety: 39% of the real")
    print("  BBB+ drugs in BBBP fail the same union. It removes recognized")
    print("  liabilities and says nothing about the ones no catalogue encodes.")
    print("  Every survivor is a hypothesis for evaluation, not for synthesis.")

    survivors = add_mpo(survivors)
    ranked = rank(survivors, rank_by, top)
    if len(survivors) and survivors.cns_mpo.notna().any():
        print(f"\n  ranked by {rank_by}. CNS-MPO (Wager 5/6, normalized):")
        print(f"    all survivors            : {survivors.cns_mpo.mean():.3f}")
        print(f"    the shortlist ({len(ranked)})        : {ranked.cns_mpo.mean():.3f}")
        print(f"    real BBB+ drugs reference: 0.753")
    return ranked


def main(run_dirs: list[Path], from_step: int, out: Path, dataset: str,
         device: str, min_heavy_atoms: int, top: int,
         rank_by: str = "c_score") -> None:
    smiles = pool_samples(run_dirs, from_step)
    print(f"pooled {len(smiles)} distinct molecules from {len(run_dirs)} run(s) "
          f"at step >= {from_step}\n")
    if not smiles:
        return
    df = novelty(add_scores(screen(smiles, min_heavy_atoms), dataset, device), dataset)
    ranked = report(df, top, rank_by)

    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "screened.csv", index=False)
    ranked.to_csv(out / "candidates.csv", index=False)
    print(f"\ntop {len(ranked)} by C:")
    print(ranked[["smiles", "heavy_atoms", "qed", "c_score"]]
          .head(10).round(3).to_string(index=False))
    print(f"\nwrote {out}/screened.csv (all, with reasons) and candidates.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--from-step", type=int, default=120)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "candidates")
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-heavy-atoms", type=int, default=10)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--rank-by", default="c_score", choices=RANK_BY,
                    help="how to order the shortlist. c_score is the original "
                         "and selects the lipophilic extreme; mpo_gated "
                         "requires C >= 0.615 then orders by CNS-MPO")
    args = ap.parse_args()
    main(args.run_dirs, args.from_step, args.out, args.dataset, args.device,
         args.min_heavy_atoms, args.top, args.rank_by)


def sample_batches(checkpoints: list[Path], n_batches: int, group_size: int,
                   out_dir: Path, seed: int, device: str = "cpu") -> int:
    """Draw fresh groups from trained checkpoints at FIXED theta.

    Sampling, not further training. More steps was measured to be actively
    harmful -- scaffold diversity collapses 0.80 -> 0.07 by step 200 and the
    classifier's signal-to-noise crosses 1 between steps 100 and 150 -- so
    the candidate pool is widened by drawing more from a good policy rather
    than by optimizing a good policy into a worse one.

    The seed is an explicit integer per checkpoint. The first bulk run used
    `hash(path) % 2**31`, and Python randomizes string hashing per process
    unless PYTHONHASHSEED is set, so those 450 batches are not reproducible
    and had to be committed as data.
    """
    import torch

    from .brics_generator import load_generator

    out_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    for ci, ck in enumerate(checkpoints):
        torch.manual_seed(seed + 1000 * ci)
        gen = load_generator(Path(ck) / "bbbp_grpo.pt", device)
        for _ in range(n_batches):
            smi = gen.sample(group_size, device=device)["smiles"]
            (out_dir / f"step{k:04d}.json").write_text(
                json.dumps([x for x in smi if x]))
            k += 1
    return k

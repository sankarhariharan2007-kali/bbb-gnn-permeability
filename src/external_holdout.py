"""
Leak-safe construction of the Adenot/Wang "different molecule types" holdout.

The problem this fixes
-----------------------
B3DB is a known aggregator of exactly the Adenot (2004) and Wang et al. source
sets, and BBBP shares many of the same public compounds. If Adenot/Wang is
used as a held-out "unseen molecule type" generalization test while BBBP
and/or B3DB is the training pool, most of that "held-out" set is not new
chemistry to the model -- it is the same molecule, by canonical SMILES,
counted a second time. Measured overlap (canonical-SMILES exact match):

    held-out candidate   overlaps B3DB      overlaps BBBP
    Adenot (1,649 mols)  1,475   (89%)      1,132  (69%)
    Wang   (1,587 mols)  1,410   (89%)        774  (49%)

Label agreement on the overlap is ~98-99%, so this isn't a labeling
disagreement that could be used as a tiebreaker -- it is the same molecule,
same label, appearing in both "train" and "test".

What's already clean and is NOT touched by this module
--------------------------------------------------------
- check_split() (src/split.py) already enforces disjoint folds and zero
  scaffold overlap *within* one dataset's own train/val/test split.
- Threshold selection is fit on val only (src/evaluate.py).
- Dedup within a single dataset happens pre-split on canonical SMILES
  (src/datasets.py).
- BBBP and B3DB are deliberately never cross-trained/tested.
None of that guards the *external* holdout against the training pool, which
is the gap this module closes.

The fix
-------
1. Canonicalize every SMILES in the external holdout source and in the
   training pool, exactly the way src/datasets.py does (Chem.MolToSmiles,
   default canonical form -- same function, so the comparison is apples to
   apples with what actually got trained on).
2. Any held-out molecule whose canonical SMILES already appears in the
   training pool is a direct duplicate -- drop it.
3. Dropping individual molecules can still leave the rest of their Murcko
   scaffold group in the holdout, which is the same scaffold-leak vector
   check_split() already guards against inside a single dataset. So: for any
   Murcko scaffold group in the holdout that has at least one direct-overlap
   member, drop the WHOLE group, not just the overlapping molecule(s).
4. Report every count so the drop is auditable, in the same style as
   src/datasets.py's cleaning report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent

EXTERNAL_SOURCES = {
    "adenot": dict(path=ROOT / "Adenot_final.csv", smiles_col="smiles", label_col="BBB"),
    "wang": dict(path=ROOT / "Wang_final.csv", smiles_col="smiles", label_col="BBB"),
}

# Training-pool datasets, same loader config as src/datasets.py::DATASETS.
TRAINING_SOURCES = {
    "bbbp": dict(path=ROOT / "BBBP.csv", sep=",", smiles_col="smiles", label_col="p_np"),
    "b3db": dict(path=ROOT / "B3DB_classification.tsv", sep="\t",
                 smiles_col="SMILES", label_col="BBB+/BBB-"),
}


def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(mol)


def murcko_scaffold(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _canonical_pool(dataset_names: list[str]) -> set[str]:
    """Canonical SMILES actually usable for training, across the given datasets."""
    pool: set[str] = set()
    for name in dataset_names:
        cfg = TRAINING_SOURCES[name]
        df = pd.read_csv(cfg["path"], sep=cfg.get("sep", ","))
        for smi in df[cfg["smiles_col"]].dropna():
            c = canonical_smiles(smi)
            if c is not None:
                pool.add(c)
    return pool


def load_external(name: str) -> pd.DataFrame:
    cfg = EXTERNAL_SOURCES[name]
    df = pd.read_csv(cfg["path"])
    df = df[[cfg["smiles_col"], cfg["label_col"]]].rename(
        columns={cfg["smiles_col"]: "smiles", cfg["label_col"]: "label"}
    )
    return df.dropna(subset=["smiles", "label"]).reset_index(drop=True)


def build_clean_holdout(source_name: str, train_dataset_names: list[str],
                         verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). clean_df has columns [smiles, label], where
    `smiles` is canonical and guaranteed scaffold-disjoint from the given
    training pool.
    """
    df = load_external(source_name)
    report = {
        "source": source_name,
        "training_pool": train_dataset_names,
        "rows_in": len(df),
    }

    train_pool = _canonical_pool(train_dataset_names)

    canon, labels, invalid = [], [], 0
    for smi, label in zip(df["smiles"], df["label"]):
        c = canonical_smiles(smi)
        if c is None:
            invalid += 1
            continue
        canon.append(c)
        labels.append(int(label))
    report["invalid_smiles_dropped"] = invalid

    # Direct duplicate: exact canonical-SMILES match against the training pool.
    direct_overlap = [c in train_pool for c in canon]
    report["direct_overlap_molecules"] = sum(direct_overlap)

    # Scaffold groups within the holdout itself.
    scaffolds = [murcko_scaffold(c) for c in canon]
    tainted_scaffolds = {s for s, overlap in zip(scaffolds, direct_overlap) if overlap}

    keep_smiles, keep_labels = [], []
    group_dropped_extra = 0  # molecules dropped only because their group is tainted
    for c, lbl, scaf, overlap in zip(canon, labels, scaffolds, direct_overlap):
        if scaf in tainted_scaffolds:
            if not overlap:
                group_dropped_extra += 1
            continue
        keep_smiles.append(c)
        keep_labels.append(lbl)

    report["scaffold_groups_dropped"] = len(tainted_scaffolds)
    report["additional_molecules_dropped_via_group"] = group_dropped_extra
    report["rows_out"] = len(keep_smiles)
    n_pos = sum(keep_labels)
    report["positives"], report["negatives"] = n_pos, len(keep_labels) - n_pos
    report["pct_dropped_total"] = round(
        1 - report["rows_out"] / max(report["rows_in"] - invalid, 1), 4
    )

    if verbose:
        print(f"[{source_name}] external holdout vs. training pool {train_dataset_names}")
        print(f"  rows in                          : {report['rows_in']}")
        print(f"  invalid SMILES                   : -{invalid}")
        print(f"  direct canonical-SMILES overlap  : {report['direct_overlap_molecules']}")
        print(f"  scaffold groups tainted           : {len(tainted_scaffolds)} "
              f"(+{group_dropped_extra} extra molecules pulled in with their group)")
        print(f"  rows out (leak-free)              : {report['rows_out']} "
              f"({n_pos} pos / {len(keep_labels) - n_pos} neg)")
        print(f"  total dropped                     : {report['pct_dropped_total']:.1%}\n")

    clean_df = pd.DataFrame({"smiles": keep_smiles, "label": keep_labels})
    return clean_df, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=list(EXTERNAL_SOURCES),
                     choices=list(EXTERNAL_SOURCES))
    ap.add_argument("--training-pool", nargs="+", default=["bbbp", "b3db"],
                     choices=list(TRAINING_SOURCES))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "external_holdout"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for source in args.sources:
        clean_df, report = build_clean_holdout(source, args.training_pool)
        clean_df.to_csv(out_dir / f"{source}_clean.csv", index=False)
        reports.append(report)
    (out_dir / "leak_report.json").write_text(json.dumps(reports, indent=2))
    print(f"wrote clean holdout sets + leak_report.json to {out_dir}")


if __name__ == "__main__":
    main()

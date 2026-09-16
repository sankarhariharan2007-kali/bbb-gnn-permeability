"""Load, clean, and cache the two BBB permeability datasets.

BBBP and B3DB overlap substantially (B3DB aggregates BBBP as one of its
sources), so the two are always trained and evaluated independently. Training
on one and testing on the other would be a leaked evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .featurize import smiles_to_graph

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

DATASETS = {
    "bbbp": {
        "path": ROOT / "BBBP.csv",
        "sep": ",",
        "smiles_col": "smiles",
        "label_col": "p_np",
        "label_map": None,  # already 0/1
    },
    "b3db": {
        "path": ROOT / "B3DB_classification.tsv",
        "sep": "\t",
        "smiles_col": "SMILES",
        "label_col": "BBB+/BBB-",
        "label_map": {"BBB+": 1, "BBB-": 0},
    },
}


def _load_raw(name: str) -> pd.DataFrame:
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["path"], sep=cfg["sep"])
    df = df[[cfg["smiles_col"], cfg["label_col"]]].rename(
        columns={cfg["smiles_col"]: "smiles", cfg["label_col"]: "label"}
    )
    if cfg["label_map"] is not None:
        df["label"] = df["label"].map(cfg["label_map"])
    return df


def build_dataset(name: str, verbose: bool = True) -> tuple[list, dict]:
    """Featurize one dataset, returning PyG graphs and a cleaning report."""
    df = _load_raw(name)
    report = {"dataset": name, "rows_in": len(df)}

    # Drop rows with a missing label before doing any chemistry work.
    n_before = len(df)
    df = df.dropna(subset=["smiles", "label"])
    report["missing_dropped"] = n_before - len(df)

    graphs, canon, labels, invalid = [], [], [], 0
    for smi, label in zip(df["smiles"], df["label"]):
        g = smiles_to_graph(smi, label=float(label))
        if g is None:
            invalid += 1
            continue
        graphs.append(g)
        canon.append(g.smiles)
        labels.append(int(label))
    report["invalid_smiles_dropped"] = invalid

    # Deduplicate on the canonical form, not the raw string: the same molecule
    # is often written several ways across these sources.
    seen: dict[str, list[int]] = {}
    for i, smi in enumerate(canon):
        seen.setdefault(smi, []).append(i)

    keep, dup_merged, conflicts = [], 0, 0
    for smi, idxs in seen.items():
        if len(idxs) == 1:
            keep.append(idxs[0])
            continue
        if len({labels[i] for i in idxs}) > 1:
            # Contradictory labels for one molecule: drop every copy rather
            # than guess which source is right.
            conflicts += len(idxs)
        else:
            keep.append(idxs[0])
            dup_merged += len(idxs) - 1
    keep.sort()

    report["duplicates_merged"] = dup_merged
    report["label_conflicts_dropped"] = conflicts
    graphs = [graphs[i] for i in keep]
    report["rows_out"] = len(graphs)
    n_pos = sum(int(g.y.item()) for g in graphs)
    report["positives"] = n_pos
    report["negatives"] = len(graphs) - n_pos

    if verbose:
        print(f"[{name}] cleaning report")
        print(f"  rows in                 : {report['rows_in']}")
        print(f"  missing smiles/label    : -{report['missing_dropped']}")
        print(f"  invalid SMILES (RDKit)  : -{report['invalid_smiles_dropped']}")
        print(f"  duplicates merged       : -{report['duplicates_merged']}")
        print(f"  label conflicts dropped : -{report['label_conflicts_dropped']}")
        print(f"  rows out                : {report['rows_out']} "
              f"({n_pos} pos / {len(graphs) - n_pos} neg, "
              f"{n_pos / len(graphs):.1%} positive)")
        accounted = (report["rows_in"] - report["missing_dropped"]
                     - report["invalid_smiles_dropped"] - report["duplicates_merged"]
                     - report["label_conflicts_dropped"])
        assert accounted == report["rows_out"], "cleaning report does not balance"

    return graphs, report


def cache_path(name: str) -> Path:
    return PROCESSED_DIR / f"{name}.pt"


def load_dataset(name: str, rebuild: bool = False, verbose: bool = True) -> list:
    """Load featurized graphs, building and caching them on first use."""
    path = cache_path(name)
    if path.exists() and not rebuild:
        return torch.load(path, weights_only=False)

    graphs, report = build_dataset(name, verbose=verbose)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, path)
    (PROCESSED_DIR / f"{name}_report.json").write_text(json.dumps(report, indent=2))
    return graphs


if __name__ == "__main__":
    from .split import check_split, scaffold_split

    for name in DATASETS:
        graphs = load_dataset(name, rebuild=True)
        smiles = [g.smiles for g in graphs]
        labels = [int(g.y.item()) for g in graphs]
        for seed in (0, 1, 2):
            train, val, test = scaffold_split(smiles, seed=seed, verbose=(seed == 0))
            check_split(smiles, train, val, test, labels)
            balance = "  ".join(
                f"{fold}={sum(labels[i] for i in idxs) / len(idxs):.1%}"
                for fold, idxs in (("train", train), ("val", val), ("test", test))
            )
            print(f"  seed {seed} positives : {balance}")
        print("  split integrity : OK (disjoint, complete, no scaffold leak, "
              "both classes in every fold)\n")

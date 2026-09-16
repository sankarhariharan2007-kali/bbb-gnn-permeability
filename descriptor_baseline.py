"""
Descriptor-based (non-graph) baseline for BBB permeability classification.

Standalone by design: it reads BBBP.csv / B3DB_classification.tsv directly and
does not import anything from src/, because the src/ package referenced in
README.md was not present in the uploaded archive (only data/, results/,
notebooks/, and README.md were included). Drop this into src/baseline.py and
swap the cleaning/split calls for the project's own src.datasets / src.split
once available, to guarantee identical folds to the GNN runs.

Answers one question: does a classical descriptor + gradient-boosted-tree
baseline get anywhere near the GNN numbers in results/summary.csv, or does
the graph structure buy something descriptors alone can't?

Two feature sets are compared per dataset:
  raw       -- MW, HBA, HBD, LogP, TPSA, RotatableBonds (Lipinski-style)
  raw+mpo   -- raw plus a CNS-MPO-*proxy* composite and a TPSA x LogP
               interaction term (see CNSMPOProxy docstring for exactly what
               is approximated vs. the canonical 6-parameter Wager score)

Usage:
    python descriptor_baseline.py --dataset bbbp
    python descriptor_baseline.py --dataset b3db
    python descriptor_baseline.py --dataset both
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
)

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install lightgbm") from e

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

SEEDS = (0, 1, 2)
SPLIT_FRACS = (0.8, 0.1, 0.1)  # train, val, test


# --------------------------------------------------------------------------- #
# Loading + cleaning (mirrors the cleaning numbers reported in README.md so
# results are comparable to the GNN runs; re-derived here rather than reused
# because src/datasets.py wasn't in the archive)
# --------------------------------------------------------------------------- #

def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def load_bbbp(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"p_np": "label", "smiles": "smiles"})
    return _clean(df[["smiles", "label"]], name="bbbp")


def load_b3db(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.rename(columns={"SMILES": "smiles"})
    df["label"] = (df["BBB+/BBB-"].str.strip() == "BBB+").astype(int)
    return _clean(df[["smiles", "label"]], name="b3db")


def _clean(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rows_in = len(df)
    df = df.dropna(subset=["smiles", "label"]).copy()
    df["canon"] = df["smiles"].apply(canonical_smiles)
    invalid = df["canon"].isna().sum()
    df = df.dropna(subset=["canon"])

    # duplicates: same canonical SMILES, keep if labels agree, drop if they conflict
    grouped = df.groupby("canon")["label"].nunique()
    conflicting = grouped[grouped > 1].index
    conflicts = len(conflicting)
    df = df[~df["canon"].isin(conflicting)]
    dupes_merged = df.duplicated("canon").sum()
    df = df.drop_duplicates("canon").reset_index(drop=True)

    print(
        f"[{name}] rows_in={rows_in} invalid_smiles={invalid} "
        f"label_conflicts_dropped={conflicts} dupes_merged={dupes_merged} "
        f"rows_out={len(df)} positive_rate={df['label'].mean():.3f}"
    )
    out = df[["canon", "label"]].rename(columns={"canon": "smiles"})
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #

def compute_descriptors(smiles: pd.Series) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append(
            {
                "mw": Descriptors.MolWt(mol),
                "hba": Lipinski.NumHAcceptors(mol),
                "hbd": Lipinski.NumHDonors(mol),
                "logp": Crippen.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "rot_bonds": Descriptors.NumRotatableBonds(mol),
            }
        )
    return pd.DataFrame(rows)


class CNSMPOProxy:
    """
    Approximation of Wager et al.'s 6-parameter CNS-MPO desirability score.

    NOT the canonical score -- two of the six parameters aren't computable
    from SMILES alone with what's installed here, and are substituted /
    dropped rather than faked:
      - ClogD (pH 7.4): approximated by reusing the ClogP desirability curve.
        LogD requires an ionization-state-aware calculation; plain LogP is
        used as a stand-in, which will overstate desirability for strongly
        ionizable compounds. Flagged, not corrected.
      - pKa of the most basic center: omitted entirely (needs a pKa predictor
        that isn't part of this environment). The composite here sums 5
        terms, not 6.
    Treat this as "CNS-MPO-proxy", not CNS-MPO, when reporting results.
    """

    @staticmethod
    def _trapezoid(x, low0, low1, high1, high0):
        # 0 below low0, ramps to 1 over [low0, low1], flat 1 over [low1, high1],
        # ramps down to 0 over [high1, high0], 0 above high0
        if x <= low0 or x >= high0:
            return 0.0
        if x < low1:
            return (x - low0) / (low1 - low0)
        if x <= high1:
            return 1.0
        return 1.0 - (x - high1) / (high0 - high1)

    @staticmethod
    def _decreasing_ramp(x, good, bad):
        if x <= good:
            return 1.0
        if x >= bad:
            return 0.0
        return 1.0 - (x - good) / (bad - good)

    @classmethod
    def score(cls, mw, logp, tpsa, hbd):
        d_logp = cls._decreasing_ramp(logp, good=3.0, bad=5.0)
        d_logd_proxy = d_logp  # substitution, see class docstring
        d_mw = cls._decreasing_ramp(mw, good=360.0, bad=500.0)
        d_tpsa = cls._trapezoid(tpsa, 20.0, 40.0, 90.0, 120.0)
        d_hbd = cls._decreasing_ramp(hbd, good=1.0, bad=4.0)
        composite = d_logp + d_logd_proxy + d_mw + d_tpsa + d_hbd  # max 5, not 6
        interaction = d_tpsa * d_logp  # the literal "q x z"-style term
        return composite, interaction, d_tpsa, d_logp


def add_mpo_features(desc: pd.DataFrame) -> pd.DataFrame:
    out = desc.copy()
    composites, interactions, d_tpsas, d_logps = [], [], [], []
    for _, row in desc.iterrows():
        c, i, dt, dl = CNSMPOProxy.score(row.mw, row.logp, row.tpsa, row.hbd)
        composites.append(c)
        interactions.append(i)
        d_tpsas.append(dt)
        d_logps.append(dl)
    out["mpo_proxy_composite"] = composites
    out["tpsa_x_logp_desirability"] = interactions
    out["d_tpsa"] = d_tpsas
    out["d_logp"] = d_logps
    return out


# --------------------------------------------------------------------------- #
# Balanced scaffold split (shuffled scaffold groups, not size-sorted --
# matches the "balanced=True" protocol described in README.md)
# --------------------------------------------------------------------------- #

def scaffold_for(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return smi


def balanced_scaffold_split(df: pd.DataFrame, seed: int, fracs=SPLIT_FRACS):
    rng = np.random.RandomState(seed)
    scaffolds: dict[str, list[int]] = {}
    for idx, smi in zip(df.index, df["smiles"]):
        scaffolds.setdefault(scaffold_for(smi), []).append(idx)

    groups = list(scaffolds.values())
    rng.shuffle(groups)  # shuffled, NOT sorted by size -- avoids the
    # size-sorted degeneracy documented in README.md

    n = len(df)
    targets = {k: f * n for k, f in zip(("train", "val", "test"), fracs)}
    buckets = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for group in groups:
        # assign each whole scaffold group to whichever split is furthest
        # below its target share -- keeps train/val/test balanced without
        # ever sorting groups by size
        deficit = {k: targets[k] - counts[k] for k in counts}
        dest = max(deficit, key=deficit.get)
        buckets[dest].extend(group)
        counts[dest] += len(group)

    train_idx, val_idx, test_idx = buckets["train"], buckets["val"], buckets["test"]
    assert not (set(train_idx) & set(val_idx) & set(test_idx))
    return train_idx, val_idx, test_idx


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

def best_threshold(y_val, p_val) -> float:
    thresholds = np.linspace(0.05, 0.95, 37)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1 = f1_score(y_val, (p_val >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def run_seed(df: pd.DataFrame, feature_cols: list[str], seed: int) -> dict:
    train_idx, val_idx, test_idx = balanced_scaffold_split(df, seed)
    train, val, test = df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        train[feature_cols],
        train["label"],
        eval_set=[(val[feature_cols], val["label"])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    p_val = model.predict_proba(val[feature_cols])[:, 1]
    thr = best_threshold(val["label"].values, p_val)

    p_test = model.predict_proba(test[feature_cols])[:, 1]
    y_test = test["label"].values
    y_pred = (p_test >= thr).astype(int)

    return {
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "test_roc_auc": roc_auc_score(y_test, p_test),
        "test_pr_auc": average_precision_score(y_test, p_test),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_mcc": matthews_corrcoef(y_test, y_pred),
        "test_threshold": thr,
    }


def run_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    desc = compute_descriptors(df["smiles"])
    desc = add_mpo_features(desc)
    full = pd.concat([df.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)

    raw_cols = ["mw", "hba", "hbd", "logp", "tpsa", "rot_bonds"]
    mpo_cols = raw_cols + ["mpo_proxy_composite", "tpsa_x_logp_desirability"]

    records = []
    for feature_set_name, cols in (("raw", raw_cols), ("raw+mpo_proxy", mpo_cols)):
        for seed in SEEDS:
            r = run_seed(full, cols, seed)
            r["dataset"] = dataset_name
            r["feature_set"] = feature_set_name
            records.append(r)
    return pd.DataFrame(records)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["test_roc_auc", "test_pr_auc", "test_balanced_accuracy", "test_f1", "test_mcc"]
    return runs.groupby(["dataset", "feature_set"])[metrics].agg(["mean", "std"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bbbp", "b3db", "both"], default="both")
    ap.add_argument("--bbbp-path", default="BBBP.csv")
    ap.add_argument("--b3db-path", default="B3DB_classification.tsv")
    ap.add_argument("--out", default="results/descriptor_baseline_runs.csv")
    args = ap.parse_args()

    all_runs = []
    if args.dataset in ("bbbp", "both"):
        df = load_bbbp(args.bbbp_path)
        all_runs.append(run_dataset(df, "bbbp"))
    if args.dataset in ("b3db", "both"):
        df = load_b3db(args.b3db_path)
        all_runs.append(run_dataset(df, "b3db"))

    runs = pd.concat(all_runs, ignore_index=True)
    runs.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}\n")
    print(summarize(runs).round(4))


if __name__ == "__main__":
    main()

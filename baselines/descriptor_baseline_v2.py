"""
Descriptor-based (non-graph) baseline, sharing identical scaffold folds with
the GNN runs in results/summary.csv.

Unlike the first draft (written before src/ was available), this:
  - reproduces src/datasets.py's cleaning algorithm exactly (verified to match
    data/processed/{bbbp,b3db}_report.json row-for-row, including the
    label_conflicts_dropped count, which counts every row in a conflicting
    group -- not the number of conflicting groups)
  - imports the project's actual src/split.py for scaffold_split/check_split,
    so seed 0/1/2 here are the SAME train/val/test molecule sets the GNNs saw
  - does NOT import src/datasets.py, src/featurize.py, src/models.py, or
    src/train.py, because those pull in torch + torch_geometric, which could
    not be installed in this environment (disk quota). Nothing here needed
    them: cleaning only needs RDKit, and scaffold splitting only needs SMILES
    strings, which src/split.py already doesn't depend on torch for either.

This means the ROC-AUC/F1/MCC numbers below are directly comparable, row for
row, to results/summary.csv -- not just "close" as in the first draft.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import split as project_split  # the project's real src/split.py, no torch needed

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError as e:
    raise SystemExit("pip install lightgbm") from e

SEEDS = tuple(range(10))
ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Cleaning -- line-for-line equivalent of src/datasets.py::build_dataset,
# minus the torch Data object construction (only validity + canonical SMILES
# are needed for a descriptor baseline)
# --------------------------------------------------------------------------- #

def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(mol)


DATASET_CFG = {
    "bbbp": dict(path=ROOT / "data" / "raw" / "BBBP.csv", sep=",", smiles_col="smiles",
                 label_col="p_np", label_map=None),
    "b3db": dict(path=ROOT / "data" / "raw" / "B3DB_classification.tsv", sep="\t", smiles_col="SMILES",
                 label_col="BBB+/BBB-", label_map={"BBB+": 1, "BBB-": 0}),
}


def build_dataset(name: str) -> tuple[list[str], list[int], dict]:
    cfg = DATASET_CFG[name]
    df = pd.read_csv(cfg["path"], sep=cfg["sep"])
    df = df[[cfg["smiles_col"], cfg["label_col"]]].rename(
        columns={cfg["smiles_col"]: "smiles", cfg["label_col"]: "label"}
    )
    if cfg["label_map"] is not None:
        df["label"] = df["label"].map(cfg["label_map"])

    report = {"dataset": name, "rows_in": len(df)}
    n_before = len(df)
    df = df.dropna(subset=["smiles", "label"])
    report["missing_dropped"] = n_before - len(df)

    canon, labels, invalid = [], [], 0
    for smi, label in zip(df["smiles"], df["label"]):
        c = canonical_smiles(smi)
        if c is None:
            invalid += 1
            continue
        canon.append(c)
        labels.append(int(label))
    report["invalid_smiles_dropped"] = invalid

    seen: dict[str, list[int]] = {}
    for i, smi in enumerate(canon):
        seen.setdefault(smi, []).append(i)

    keep, dup_merged, conflicts = [], 0, 0
    for smi, idxs in seen.items():
        if len(idxs) == 1:
            keep.append(idxs[0])
            continue
        if len({labels[i] for i in idxs}) > 1:
            conflicts += len(idxs)  # matches src/datasets.py exactly
        else:
            keep.append(idxs[0])
            dup_merged += len(idxs) - 1
    keep.sort()

    report["duplicates_merged"] = dup_merged
    report["label_conflicts_dropped"] = conflicts
    report["rows_out"] = len(keep)
    n_pos = sum(labels[i] for i in keep)
    report["positives"], report["negatives"] = n_pos, len(keep) - n_pos

    smiles_out = [canon[i] for i in keep]
    labels_out = [labels[i] for i in keep]
    print(f"[{name}] rows_out={report['rows_out']} "
          f"({n_pos} pos / {len(keep) - n_pos} neg, {n_pos/len(keep):.1%} positive)")
    return smiles_out, labels_out, report


# --------------------------------------------------------------------------- #
# Descriptors + CNS-MPO-proxy (same as v1; see class docstring for exactly
# which of the 6 canonical Wager et al. parameters are approximated/omitted)
# --------------------------------------------------------------------------- #

def compute_descriptors(smiles: list[str]) -> pd.DataFrame:
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        rows.append({
            "mw": Descriptors.MolWt(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "hbd": Lipinski.NumHDonors(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rot_bonds": Descriptors.NumRotatableBonds(mol),
        })
    return pd.DataFrame(rows)


class CNSMPO:
    """
    Exact piecewise-linear desirability functions for all six canonical
    CNS-MPO parameters (Wager et al.), as specified by the user:

        f(CLogP): 1.0 <=3, linear 1.0-0.5(x-3) on (3,5), 0.0 >=5
        f(CLogD): 1.0 <=2, linear 1.0-0.5(x-2) on (2,4), 0.0 >=4
        f(MW):    1.0 <=360, linear 1.0-(x-360)/140 on (360,500), 0.0 >=500
        f(TPSA):  x/40 on [0,40], 1.0 on (40,90], linear 1.0-(x-90)/30 on
                  (90,120), 0.0 >=120   <-- note: ramps UP from 0 at TPSA=0,
                  not from 20 as my first two drafts had it; fixed here.
        f(HBD):   1.0 at 0 or 1, 0.5 at 2, 0.0 at >=3 (discrete steps, not a
                  continuous ramp -- my earlier drafts used a continuous ramp
                  and were wrong at HBD=2 (gave 0.67, not 0.5) and HBD=3
                  (gave 0.33, not 0.0). Fixed here.
        f(pKa):   1.0 <=8, linear 1.0-0.5(x-8) on (8,10), 0.0 >=10

    What's still a proxy, not the real thing, and why:
      - CLogD (pH 7.4): still substituted with CLogP. Real LogD needs an
        ionization-state-aware calculation (e.g. via a pKa predictor), which
        isn't installed here. f_clogd() is implemented correctly and ready to
        take a real CLogD value the moment one is available -- it's just fed
        the CLogP value in the meantime, which will overstate desirability
        for compounds that are substantially ionized at pH 7.4.
      - Most basic pKa: f_pka() is implemented but not called -- there's no
        pKa predictor in this environment, and fabricating a value would be
        worse than omitting the term. The composite below sums 5 of 6 terms
        (max 5.0, not 6.0) for that reason; report it as such, not as "CNS
        MPO score" without qualification.
    """

    @staticmethod
    def f_clogp(x: float) -> float:
        if x <= 3:
            return 1.0
        if x < 5:
            return 1.0 - 0.5 * (x - 3)
        return 0.0

    @staticmethod
    def f_clogd(x: float) -> float:
        if x <= 2:
            return 1.0
        if x < 4:
            return 1.0 - 0.5 * (x - 2)
        return 0.0

    @staticmethod
    def f_mw(x: float) -> float:
        if x <= 360:
            return 1.0
        if x < 500:
            return 1.0 - (x - 360) / 140
        return 0.0

    @staticmethod
    def f_tpsa(x: float) -> float:
        if x <= 40:
            return x / 40
        if x <= 90:
            return 1.0
        if x < 120:
            return 1.0 - (x - 90) / 30
        return 0.0

    @staticmethod
    def f_hbd(n: float) -> float:
        if n <= 1:
            return 1.0
        if n == 2:
            return 0.5
        return 0.0

    @staticmethod
    def f_pka(x: float) -> float:
        if x <= 8:
            return 1.0
        if x < 10:
            return 1.0 - 0.5 * (x - 8)
        return 0.0

    @classmethod
    def score(cls, mw, logp, tpsa, hbd):
        d_clogp = cls.f_clogp(logp)
        d_clogd = cls.f_clogd(logp)  # proxy: real CLogD not available, see docstring
        d_mw = cls.f_mw(mw)
        d_tpsa = cls.f_tpsa(tpsa)
        d_hbd = cls.f_hbd(hbd)
        # pKa term omitted entirely -- 5 of 6 terms, max possible = 5.0
        composite = d_clogp + d_clogd + d_mw + d_tpsa + d_hbd
        interaction = d_tpsa * d_clogp  # TPSA x LogP, the "q x z"-style term
        return composite, interaction


def add_mpo_features(desc: pd.DataFrame) -> pd.DataFrame:
    out = desc.copy()
    comps, inters = [], []
    for _, r in desc.iterrows():
        c, i = CNSMPO.score(r.mw, r.logp, r.tpsa, r.hbd)
        comps.append(c)
        inters.append(i)
    out["mpo_composite_5of6"] = comps
    out["tpsa_x_logp_desirability"] = inters
    return out


# --------------------------------------------------------------------------- #
# Train / eval -- uses the project's own best_threshold logic (Youden's J,
# from src/evaluate.py) rather than an F1 sweep, to match the GNN protocol
# exactly rather than approximate it
# --------------------------------------------------------------------------- #

def best_threshold_youden(y_true, y_prob) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def compute_metrics(y_true, y_prob, threshold) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def run_seed(smiles, labels, features_df, feature_cols, seed) -> dict:
    train_idx, val_idx, test_idx = project_split.scaffold_split(
        smiles, seed=seed, verbose=False
    )
    project_split.check_split(smiles, train_idx, val_idx, test_idx, labels)

    X = features_df[feature_cols].values
    y = np.array(labels)

    Xtr, ytr = X[train_idx], y[train_idx]
    Xval, yval = X[val_idx], y[val_idx]
    Xtest, ytest = X[test_idx], y[test_idx]

    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    p_val = model.predict_proba(Xval)[:, 1]
    thr = best_threshold_youden(yval, p_val)
    p_test = model.predict_proba(Xtest)[:, 1]
    metrics = compute_metrics(ytest, p_test, thr)
    metrics.update(seed=seed, n_train=len(train_idx), n_val=len(val_idx),
                    n_test=len(test_idx), threshold=thr)
    return metrics


def run_dataset(name: str) -> pd.DataFrame:
    smiles, labels, _ = build_dataset(name)
    desc = compute_descriptors(smiles)
    desc = add_mpo_features(desc)

    raw_cols = ["mw", "hba", "hbd", "logp", "tpsa", "rot_bonds"]
    mpo_cols = raw_cols + ["mpo_composite_5of6", "tpsa_x_logp_desirability"]

    rows = []
    for feature_set_name, cols in (("raw", raw_cols), ("raw+mpo_5of6", mpo_cols)):
        for seed in SEEDS:
            r = run_seed(smiles, labels, desc, cols, seed)
            r["dataset"] = name
            r["feature_set"] = feature_set_name
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    all_runs = pd.concat([run_dataset("bbbp"), run_dataset("b3db")], ignore_index=True)
    out = ROOT / "results" / "descriptor_baseline_v2_runs.csv"
    all_runs.to_csv(out, index=False)

    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "mcc"]
    summary = all_runs.groupby(["dataset", "feature_set"])[metrics].agg(["mean", "std"]).round(4)
    print(f"\nwrote {out}\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()

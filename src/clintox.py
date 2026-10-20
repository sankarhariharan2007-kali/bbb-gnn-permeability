"""ClinTox FDA_APPROVED as an "is this a drug" term. Candidate for 3.5.

THE GAP THIS IS MEANT TO FILL. BBBP labels a TRANSPORT property, so optimizing
it finds anything that crosses a membrane -- which is why the generator once
converged on carbon tetrachloride, freon-12 and tetrafluoromethane. No amount
of tuning the permeability objective fixes that, because the objective is not
wrong, it is answering a different question. 3.5 asks for a second labelled
signal that says "is this a drug".

WHY TOX21 FAILED AT THIS, AND WHY IT IS THE TEST TO BEAT. The existing
`src/tox.py` was built for the same purpose and must not be switched on: its
twelve nuclear-receptor and stress-response assays fire on drug-sized aromatic
scaffolds, so it scores CCl4 at 0.251 and caffeine at 0.302 -- it *rewards the
solvent and penalizes the drug* (limitations 3.3). Any replacement has to clear
that bar before it is wired into anything.

THE STRUCTURAL WORRY WITH CLINTOX, STATED UP FRONT. Its negatives are drugs
that FAILED clinical trials, not non-drugs. The label separates "approved" from
"failed candidate", both of which are drug-like. It contains no halogenated
solvents at all -- carbon tetrachloride, chloroform, freon-12 and
tetrafluoromethane are all absent -- so the model has never seen the class it
would be asked to reject and must extrapolate. It is also 93.7% positive
(1390/94). Whether it extrapolates in the right direction is an empirical
question, which is what `_decisive_test` answers.

    python -m src.clintox            # train + the decisive test
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from .datasets import ROOT
from .featurize import FEATURE_ID, smiles_to_graph
from .models import build_model, count_parameters
from .split import check_split, scaffold_split

RDLogger.DisableLog("rdApp.*")

RAW = ROOT / "data" / "external_untrusted" / "clintox.csv"
OUT = ROOT / "results" / "clintox"


def load(task: str = "FDA_APPROVED"):
    df = pd.read_csv(RAW)
    graphs = []
    for s, y in zip(df["smiles"], df[task]):
        g = smiles_to_graph(s, label=float(y)) if isinstance(s, str) else None
        if g is not None:
            graphs.append(g)
    return graphs


def train_one(seed: int = 0, epochs: int = 150, patience: int = 30,
              task: str = "FDA_APPROVED", verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    graphs = load(task)
    smiles = [g.smiles for g in graphs]
    labels = [int(g.y.item()) for g in graphs]
    tr, va, te = scaffold_split(smiles, seed=seed, verbose=False)
    # 94 negatives over the whole set, so a fold can come out single-class.
    # check_split's two-class assertion is the right guard; report rather than
    # silently proceed on a fold where ROC-AUC is undefined.
    try:
        check_split(smiles, tr, va, te, labels)
    except AssertionError as e:
        if verbose:
            print(f"    seed {seed}: {e}")
        return {}

    dev = torch.device("cpu")
    loaders = [DataLoader([graphs[i] for i in idx], batch_size=bs, shuffle=sh)
               for idx, bs, sh in ((tr, 64, True), (va, 256, False), (te, 256, False))]
    model = build_model("gcn").to(dev)
    n_pos = sum(labels[i] for i in tr)
    pw = torch.tensor([(len(tr) - n_pos) / max(n_pos, 1)], dtype=torch.float)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best, best_state, since = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in loaders[0]:
            opt.zero_grad()
            crit(model(b), b.y).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = np.concatenate([torch.sigmoid(model(b)).numpy() for b in loaders[1]])
            y = np.concatenate([b.y.numpy() for b in loaders[1]])
        auc = roc_auc_score(y, p) if len(set(y)) == 2 else 0.5
        if auc > best:
            best, best_state, since = auc, deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(model(b)).numpy() for b in loaders[2]])
        y = np.concatenate([b.y.numpy() for b in loaders[2]])
    test_auc = roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")

    OUT.mkdir(parents=True, exist_ok=True)
    d = OUT / f"{task}_seed{seed}"
    d.mkdir(exist_ok=True)
    torch.save(best_state, d / "model.pt")
    (d / "metrics.json").write_text(json.dumps({
        "task": task, "seed": seed, "val_auc": best, "test_auc": test_auc,
        "n": len(graphs), "n_pos": int(sum(labels)),
        "feature_id": FEATURE_ID, "n_parameters": count_parameters(model),
    }, indent=2))
    if verbose:
        print(f"    seed {seed}: val {best:.3f}  test {test_auc:.3f}  "
              f"(n={len(graphs)}, {sum(labels)} positive)")
    return {"model": model, "test_auc": test_auc}


def _decisive_test(model) -> bool:
    """Does it rank a solvent BELOW a drug? Tox21 failed exactly here.

    Returns True only if every solvent scores below every reference drug. A
    term that cannot do this cannot be wired into the reward, however good its
    held-out AUC, because the molecules it must reject are the ones it has
    never seen.
    """
    SOLVENTS = {"carbon tetrachloride": "ClC(Cl)(Cl)Cl", "chloroform": "ClC(Cl)Cl",
                "freon-12": "FC(F)(Cl)Cl", "tetrafluoromethane": "FC(F)(F)F"}
    DRUGS = {"caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
             "diazepam": "CN1c2ccc(Cl)cc2C(c2ccccc2)=NCC1=O",
             "haloperidol": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"}

    @torch.no_grad()
    def score(smi):
        g = smiles_to_graph(smi)
        if g is None:
            return float("nan")
        from torch_geometric.data import Batch
        return float(torch.sigmoid(model(Batch.from_data_list([g]))).item())

    sv = {k: score(v) for k, v in SOLVENTS.items()}
    dr = {k: score(v) for k, v in DRUGS.items()}
    print("\n  THE DECISIVE TEST -- P(FDA_APPROVED), higher should mean more drug-like\n")
    for k, v in sorted(sv.items(), key=lambda x: -x[1]):
        print(f"    SOLVENT  {k:22} {v:.3f}")
    for k, v in sorted(dr.items(), key=lambda x: -x[1]):
        print(f"    DRUG     {k:22} {v:.3f}")
    ok = max(sv.values()) < min(dr.values())
    print(f"\n    worst drug {min(dr.values()):.3f} vs best solvent {max(sv.values()):.3f}"
          f"  ->  {'SEPARATES' if ok else 'FAILS -- same mode as Tox21 (3.3)'}")
    return ok


if __name__ == "__main__":
    print(f"ClinTox FDA_APPROVED, scaffold split, feature_id={FEATURE_ID}\n")
    res = [train_one(seed=s) for s in range(3)]
    aucs = [r["test_auc"] for r in res if r and not np.isnan(r.get("test_auc", np.nan))]
    if aucs:
        print(f"\n  test AUC over {len(aucs)} usable seeds: "
              f"{np.mean(aucs):.3f} +/- {np.std(aucs, ddof=1) if len(aucs) > 1 else 0:.3f}")
    usable = [r for r in res if r]
    if usable:
        _decisive_test(usable[0]["model"])

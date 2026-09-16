"""Bemis-Murcko scaffold splitting.

Test molecules get core structures never seen during training, which is the
honest measure of generalization to new chemistry. A random split on these
datasets reports ~0.90 ROC-AUC largely by memorizing close analogs.

Two strategies are implemented:

`balanced=True` (default, Chemprop-style) shuffles scaffold groups under a
per-seed RNG, routing groups too large for val/test into train first.

`balanced=False` is the classic DeepChem `ScaffoldSplitter`: groups sorted by
size descending, filled train-first. **It is degenerate on BBBP.csv.** That file
is ordered so its entire back half is class 1; because val/test receive only the
smallest (singleton) scaffold groups and ties are broken by file position, both
folds come out 100% positive and ROC-AUC becomes undefined. Kept only so that
artifact can be reproduced and discussed.
"""

from __future__ import annotations

import random
from collections import defaultdict

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES. Acyclic molecules yield ''."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_groups(smiles_list: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        groups[murcko_scaffold(smi)].append(idx)
    return dict(groups)


def scaffold_split(
    smiles_list: list[str],
    frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    balanced: bool = True,
    verbose: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices by scaffold group. Groups are never broken across folds."""
    groups = scaffold_groups(smiles_list)
    n = len(smiles_list)
    n_train, n_val = frac[0] * n, frac[1] * n

    if balanced:
        # Groups big enough to distort a small fold go to train; the rest are
        # shuffled, so fold membership is independent of file order.
        big, small = [], []
        for members in groups.values():
            if len(members) > n_val / 2:
                big.append(members)
            else:
                small.append(members)
        rng = random.Random(seed)
        rng.shuffle(big)
        rng.shuffle(small)
        ordered = big + small
    else:
        ordered = sorted(groups.values(), key=lambda m: (-len(m), m[0]))

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for members in ordered:
        if len(train) + len(members) <= n_train:
            train += members
        elif len(val) + len(members) <= n_val:
            val += members
        else:
            test += members

    if verbose:
        acyclic = len(groups.get("", []))
        print(f"  scaffold groups : {len(groups)} distinct "
              f"({acyclic} acyclic molecules share the empty scaffold)")
        print(f"  split (seed={seed}) : train={len(train)} ({len(train)/n:.1%})  "
              f"val={len(val)} ({len(val)/n:.1%})  "
              f"test={len(test)} ({len(test)/n:.1%})")

    return sorted(train), sorted(val), sorted(test)


def check_split(
    smiles_list: list[str],
    train: list[int],
    val: list[int],
    test: list[int],
    labels: list[int] | None = None,
) -> None:
    """Assert the split is disjoint, complete, scaffold-clean, and two-class.

    A silent scaffold leak inflates every downstream number and a single-class
    fold makes ROC-AUC undefined, so this runs on every training run rather
    than only in tests.
    """
    n = len(smiles_list)
    assert not (set(train) & set(val)), "train/val overlap"
    assert not (set(train) & set(test)), "train/test overlap"
    assert not (set(val) & set(test)), "val/test overlap"
    assert len(train) + len(val) + len(test) == n, "split does not cover all molecules"
    assert set(train) | set(val) | set(test) == set(range(n)), "index coverage gap"

    scaf = [murcko_scaffold(s) for s in smiles_list]
    s_train = {scaf[i] for i in train}
    s_val = {scaf[i] for i in val}
    s_test = {scaf[i] for i in test}
    assert not (s_train & s_test), "SCAFFOLD LEAK: train/test share scaffolds"
    assert not (s_train & s_val), "SCAFFOLD LEAK: train/val share scaffolds"
    assert not (s_val & s_test), "SCAFFOLD LEAK: val/test share scaffolds"

    if labels is not None:
        for fold, idxs in (("train", train), ("val", val), ("test", test)):
            present = {labels[i] for i in idxs}
            assert len(present) == 2, (
                f"{fold} fold is single-class ({present}); ROC-AUC is undefined. "
                "This is the failure mode of the size-sorted split on BBBP."
            )

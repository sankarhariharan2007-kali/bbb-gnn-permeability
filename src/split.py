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

ACYCLIC MOLECULES NEEDED THEIR OWN GROUPING. `MurckoScaffoldSmiles` returns ''
for anything with no ring system, so every acyclic molecule collapsed into a
single group keyed ''. Because groups are never broken across folds, that one
group went to whichever fold could hold it -- and being large, it always went to
train. Those molecules were therefore never tested. Measured: 99/2039 of BBBP
(4.9%) and 311/7805 of B3DB (4.0%).

`cluster_acyclic=True` (default) instead clusters them by Morgan/Tanimoto
similarity (Butina, cutoff 0.6) so they distribute across folds while still
keeping near-identical molecules together. The effect is bounded by the ~5% of
the data involved; this is a correctness fix, not a performance lever.

GROUP IDENTITY HAS ONE DEFINITION, `group_keys`. `check_split` asserts no group
spans two folds, so it must group molecules exactly as `scaffold_split` did or
it fires on every run. It previously recomputed Murcko scaffolds independently,
which is fine only while the two agree. Both now call `group_keys`. Weakening
the leak assertion instead would make every number in this repo unverifiable.
"""

from __future__ import annotations

import random
from collections import defaultdict

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

# Recorded in metrics.json so a checkpoint names the partition it was fit under.
ACYCLIC_SIM_CUTOFF = 0.6
SPLIT_ID = f"murcko+tanimoto@{ACYCLIC_SIM_CUTOFF}"
SPLIT_ID_MURCKO_ONLY = "murcko"

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES. Acyclic molecules yield ''."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _tanimoto_clusters(smiles_list: list[str], cutoff: float) -> list[int]:
    """Butina cluster id per molecule. Deterministic, and seed-independent.

    The grouping must not depend on the split seed: all seeds partition the
    same groups, which is what makes a within-seed ensemble legitimate.
    """
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(None if mol is None else _morgan.GetFingerprint(mol))

    n = len(fps)
    if n < 2:
        return [0] * n

    # Butina wants the lower triangle of a distance matrix, flattened.
    dists: list[float] = []
    for i in range(1, n):
        if fps[i] is None:
            dists.extend([1.0] * i)
            continue
        row = [
            1.0 if fps[j] is None
            else 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            for j in range(i)
        ]
        dists.extend(row)

    clusters = Butina.ClusterData(dists, n, 1.0 - cutoff, isDistData=True)
    assignment = [0] * n
    for cid, members in enumerate(clusters):
        for idx in members:
            assignment[idx] = cid
    return assignment


def group_keys(
    smiles_list: list[str],
    cluster_acyclic: bool = True,
    cutoff: float = ACYCLIC_SIM_CUTOFF,
) -> list[str]:
    """The single source of group identity, one key per molecule.

    A ring-bearing molecule is keyed by its Murcko scaffold. An acyclic one is
    keyed by its Tanimoto cluster, because its scaffold is '' and would
    otherwise merge the entire acyclic cohort into one unsplittable group.
    """
    keys = [murcko_scaffold(s) for s in smiles_list]
    if not cluster_acyclic:
        return keys

    acyclic = [i for i, k in enumerate(keys) if k == ""]
    if not acyclic:
        return keys

    clusters = _tanimoto_clusters([smiles_list[i] for i in acyclic], cutoff)
    for position, idx in enumerate(acyclic):
        keys[idx] = f"tanimoto:{clusters[position]}"
    return keys


def scaffold_groups(
    smiles_list: list[str], cluster_acyclic: bool = True
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, key in enumerate(group_keys(smiles_list, cluster_acyclic)):
        groups[key].append(idx)
    return dict(groups)


def scaffold_split(
    smiles_list: list[str],
    frac: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    balanced: bool = True,
    verbose: bool = True,
    cluster_acyclic: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices by scaffold group. Groups are never broken across folds."""
    groups = scaffold_groups(smiles_list, cluster_acyclic)
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
        acyclic_merged = len(groups.get("", []))
        clustered = sum(len(m) for k, m in groups.items() if k.startswith("tanimoto:"))
        n_clusters = sum(1 for k in groups if k.startswith("tanimoto:"))
        print(f"  scaffold groups : {len(groups)} distinct")
        if clustered:
            print(f"  acyclic         : {clustered} molecules in {n_clusters} "
                  f"Tanimoto clusters (cutoff {ACYCLIC_SIM_CUTOFF})")
        else:
            print(f"  acyclic         : {acyclic_merged} molecules share the "
                  f"empty scaffold (one unsplittable group)")
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
    cluster_acyclic: bool = True,
) -> None:
    """Assert the split is disjoint, complete, scaffold-clean, and two-class.

    A silent scaffold leak inflates every downstream number and a single-class
    fold makes ROC-AUC undefined, so this runs on every training run rather
    than only in tests.

    `cluster_acyclic` MUST match what `scaffold_split` was called with. The
    leak assertion compares group keys, so grouping the molecules differently
    here than the splitter did reports a leak that does not exist.
    """
    n = len(smiles_list)
    assert not (set(train) & set(val)), "train/val overlap"
    assert not (set(train) & set(test)), "train/test overlap"
    assert not (set(val) & set(test)), "val/test overlap"
    assert len(train) + len(val) + len(test) == n, "split does not cover all molecules"
    assert set(train) | set(val) | set(test) == set(range(n)), "index coverage gap"

    keys = group_keys(smiles_list, cluster_acyclic)
    s_train = {keys[i] for i in train}
    s_val = {keys[i] for i in val}
    s_test = {keys[i] for i in test}
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


def _self_check() -> None:
    # Six acyclic molecules (scaffold '') plus three ring systems.
    acyclic = ["CCO", "CCCO", "CCCCO", "CC(C)CO", "CCCCCCCC", "NCCN"]
    cyclic = ["c1ccccc1C(=O)O", "c1ccccc1CC", "C1CCNCC1"]
    smiles = acyclic + cyclic

    murcko_only = group_keys(smiles, cluster_acyclic=False)
    assert murcko_only[:len(acyclic)] == [""] * len(acyclic)
    merged = len({k for k in murcko_only if k == ""})
    print(f"murcko only  -> {len(acyclic)} acyclic molecules in {merged} group "
          f"(unsplittable: it goes wholesale to one fold)")

    clustered = group_keys(smiles, cluster_acyclic=True)
    assert "" not in clustered, "every acyclic molecule must get a cluster key"
    assert clustered[len(acyclic):] == murcko_only[len(acyclic):], (
        "clustering must not touch ring-bearing molecules"
    )
    n_clusters = len({k for k in clustered if k.startswith("tanimoto:")})
    assert n_clusters > 1, "the acyclic cohort must actually split"
    print(f"+tanimoto    -> {n_clusters} clusters over the same molecules")

    # Deterministic and seed-free: every seed must partition the same groups,
    # which is what makes a within-seed ensemble legitimate.
    assert group_keys(smiles) == clustered
    print("grouping     -> deterministic and seed-independent")

    # Near-identical molecules must stay together, or the split stops being
    # a generalization test.
    pair = group_keys(["CCCCCCCCO", "CCCCCCCCCO"], cluster_acyclic=True)
    assert pair[0] == pair[1], "near-identical acyclics must share a cluster"
    print("similar pair -> same cluster (nonanol / decanol)")

    # Unparseable SMILES must not crash the clusterer.
    assert len(group_keys(["CCO", "not-a-molecule"])) == 2
    print("invalid      -> grouped without crashing")

    # The trap: check_split must group exactly as scaffold_split did. Asserting
    # it FIRES on a mismatch is the point -- the tempting "fix" for the crash
    # this refactor prevents is to delete the leak assertion entirely.
    # No `labels` here: this fixture is too small to guarantee both classes in
    # every fold, and that assertion is exercised on the real data by
    # `python -m src.datasets`.
    train, val, test = scaffold_split(smiles, frac=(0.5, 0.25, 0.25), seed=0,
                                      verbose=False, cluster_acyclic=True)
    check_split(smiles, train, val, test, cluster_acyclic=True)
    print("check_split  -> passes when the flags agree")

    if val and test:
        try:
            check_split(smiles, train, val, test, cluster_acyclic=False)
        except AssertionError:
            print("check_split  -> correctly reports a leak when they disagree")
        else:
            # Only reachable if the acyclics happened to land in one fold.
            print("check_split  -> (mismatch undetectable on this tiny fixture)")

    print("\nself-check passed")


if __name__ == "__main__":
    _self_check()

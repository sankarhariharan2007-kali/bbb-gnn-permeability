"""
Descriptor features for the hybrid GNN+descriptor model (priority 1).

Six Lipinski-style descriptors plus the CNS-MPO-proxy composite and its
TPSA x LogP interaction term -- the same feature set as the descriptor
baseline (descriptor_baseline_v2.py), reusing its corrected CNSMPO
piecewise-linear desirability functions verbatim so the "does the descriptor
signal help" story is comparable end to end: same descriptors, whether they
are the whole model (the LightGBM baseline) or bolted onto a graph embedding
(this module).

8 features, in fixed order: mw, hba, hbd, logp, tpsa, rot_bonds,
mpo_composite_5of6, tpsa_x_logp_desirability.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski

DESCRIPTOR_NAMES = [
    "mw", "hba", "hbd", "logp", "tpsa", "rot_bonds",
    "mpo_composite_5of6", "tpsa_x_logp_desirability",
]
DESCRIPTOR_DIM = len(DESCRIPTOR_NAMES)


class CNSMPO:
    """Exact piecewise-linear desirability functions (Wager et al.), 5 of 6
    terms -- CLogD is proxied by CLogP (no pKa-aware LogD calculator here) and
    the basic-pKa term is omitted rather than fabricated. Copied verbatim from
    descriptor_baseline_v2.py so both baselines use the identical composite.
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

    @classmethod
    def score(cls, mw, logp, tpsa, hbd):
        d_clogp = cls.f_clogp(logp)
        d_clogd = cls.f_clogd(logp)  # proxy, see class docstring
        d_mw = cls.f_mw(mw)
        d_tpsa = cls.f_tpsa(tpsa)
        d_hbd = cls.f_hbd(hbd)
        composite = d_clogp + d_clogd + d_mw + d_tpsa + d_hbd  # max 5, not 6
        interaction = d_tpsa * d_clogp
        return composite, interaction


def compute_descriptors(smiles: str) -> list[float]:
    """Raw (unstandardized) descriptor vector for one canonical SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    composite, interaction = CNSMPO.score(mw, logp, tpsa, hbd)
    return [mw, hba, hbd, logp, tpsa, rot_bonds, composite, interaction]


def compute_descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    return np.array([compute_descriptors(s) for s in smiles_list], dtype=np.float64)


def standardize_with_train_stats(
    raw: np.ndarray, train_idx: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score every column using TRAIN-fold statistics only, so scaling
    itself can't leak val/test distribution into the model. Returns
    (standardized_full_matrix, mean, std).
    """
    mu = raw[train_idx].mean(axis=0)
    sigma = raw[train_idx].std(axis=0)
    sigma[sigma == 0] = 1.0
    return (raw - mu) / sigma, mu, sigma

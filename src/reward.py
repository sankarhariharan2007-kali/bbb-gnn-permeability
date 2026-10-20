"""Reward assembly for GRPO: frozen permeability classifier + QED + SA + D.

Implements Eq. 5 (reward), Eq. 6 (annealed weights) and Eq. 7 (validity-gated
switch) from the continuation draft. The discriminator term is passed IN rather
than owned here -- D_phi is trained by the GAN loop and is the only
non-stationary part of the reward (Section 3.1), so keeping it out of this
module makes the stationary terms independently testable.

    python -m src.reward        # self-check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Descriptors, RDConfig
from torch_geometric.data import Batch

from .featurize import run_contract, smiles_to_graph
from .pareto import reduce_objectives
from .models import build_model
from .models_edge import EDGE_MODELS, build_edge_model
from .alerts import alert_score
from .train import RESULTS_DIR

RDLogger.DisableLog("rdApp.*")

# sascorer ships with RDKit but lives in Contrib, which is not on sys.path.
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

SA_MIN, SA_MAX = 1.0, 10.0  # sascorer's documented range; lower = easier to make


def _checkpoint(root: Path, dataset: str, model: str, seed: int) -> Path:
    """Locate a run, whether it landed in the core grid or the edge ablation.

    The edge-aware operators (gine, gat_edge) were run separately and live
    under results/edge_ablation/, not results/<dataset>/. Their saved fold
    indices match the core runs seed-for-seed, so the two sets are genuinely
    ensemble-compatible -- see _self_check.
    """
    for path in (
        root / dataset / model / f"seed{seed}" / "model.pt",
        root / "edge_ablation" / dataset / model / f"seed{seed}" / "model.pt",
    ):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no checkpoint for {model}/{dataset}/seed{seed}. Train it first: "
        f"python -m src.train --model {model} --dataset {dataset} --seed {seed}"
    )


def load_frozen_classifier(
    dataset: str = "bbbp",
    models: tuple[str, ...] = ("gin", "gat", "gine"),
    seed: int = 0,
    device: str = "cpu",
    results_dir: Path | None = None,
):
    """Return C(mols) -> np.ndarray of mean P(permeable), frozen and stationary.

    eval() is still load-bearing, for a smaller reason than it used to be. The
    classifier now defaults to GraphNorm, which is batch-independent by
    construction, so C(m) no longer varies with group composition even in train
    mode. `nn.Dropout` does, so the forward pass is still stochastic without
    eval() and the "frozen C is stationary in k" premise of Section 3.1 would
    still be false. The batch-invariance assertion in _self_check stays: it is
    cheap, and it is what catches a regression back to BatchNorm.

    THE ENSEMBLE IS VERIFIED HERE, NOT IN THE SELF-CHECK. The fold-identity
    assertion used to live only in `_self_check`, hardcoded to
    ("gin","gat","gine") at bbbp seed 0 -- so every other model set, seed or
    results_dir was combined with nothing checked at all. It now runs on every
    load, which is the only version of the guarantee worth having: two machines
    pulling the same branch can otherwise assemble an ensemble out of a
    half-updated checkout and nothing complains.

    Architecture is replayed from each run's metrics.json rather than taken
    from build_model's defaults. Those defaults were the de-facto contract --
    no caller ever passed in_dim -- so changing one invalidated every committed
    checkpoint and surfaced as `size mismatch for convs.0` far from the cause.

    Checkpoints are combined within a single seed only -- each seed defines a
    different scaffold split (README), so mixing seeds mixes models that saw
    each other's validation molecules.
    """
    root = results_dir or RESULTS_DIR
    nets = []
    folds: dict[str, np.ndarray] = {}
    norms: dict[str, str] = {}
    for name in models:
        ckpt = _checkpoint(root, dataset, name, seed)
        # Raises if the run was trained under a different featurizer, and
        # returns the architecture it was trained with.
        arch = run_contract(ckpt.parent)
        norms[name] = arch.get("norm")
        folds[name] = np.load(ckpt.parent / "test_preds.npz")["idx"]
        # Edge-aware operators carry a different skeleton class, so they must
        # be rebuilt by the module that trained them or the state_dict is
        # being loaded into the wrong architecture.
        builder = build_edge_model if name in EDGE_MODELS else build_model
        net = builder(name, **arch).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()
        net.requires_grad_(False)
        nets.append(net)

    # Same fold. The seed reseeds the scaffold split, so members trained under
    # different split settings -- or pulled from a half-updated checkout --
    # have each other's held-out molecules in their training sets, and the
    # ensemble's predictions are leaked rather than merely worse.
    first = models[0]
    for name in models:
        if not np.array_equal(folds[name], folds[first]):
            # Mismatched folds are usually the SAME SIZE with different
            # contents, so report the overlap -- a size comparison reads as
            # "197 vs 197" and tells the reader nothing.
            shared = len(np.intersect1d(folds[name], folds[first]))
            raise RuntimeError(
                f"ENSEMBLE LEAK: {name} and {first} (dataset={dataset}, "
                f"seed={seed}) were trained on different folds -- "
                f"{shared}/{len(folds[first])} test molecules in common. "
                f"Members must share a split, or each has trained on the "
                f"others' held-out molecules."
            )

    # Same normalization. Not a leak, but a mixed-norm ensemble is never
    # intended -- it means a checkout caught mid-update, and the fold check
    # above cannot see it because the split is unchanged.
    if len(set(norms.values())) > 1:
        raise RuntimeError(
            f"mixed-norm ensemble: {norms}. These checkpoints come from "
            f"different runs of the project; re-pull or retrain so every "
            f"member shares a norm."
        )

    @torch.no_grad()
    def classify(mols: list[Chem.Mol]) -> np.ndarray:
        if not mols:
            return np.zeros(0, dtype=float)
        graphs = [smiles_to_graph(Chem.MolToSmiles(m)) for m in mols]
        batch = Batch.from_data_list(graphs).to(device)
        probs = [torch.sigmoid(net(batch)) for net in nets]
        return torch.stack(probs).mean(0).cpu().numpy()

    return classify


def load_frozen_hybrid(
    dataset: str = "bbbp",
    models: tuple[str, ...] = ("gcn", "sage", "gin", "gat"),
    seed: int = 0,
    device: str = "cpu",
):
    """The OTHER reward: hybrid + BatchNorm, for the matched-reward comparison.

    WHY THIS EXISTS. The default reward is gin/gat/gine under GraphNorm, with
    no descriptors. Training against a second, materially different classifier
    answers a question neither one can answer alone: does the generator's
    output depend on which judge it was optimized against? Agreement means the
    permeability signal is robust; divergence means each reward is being
    exploited on its own terms. See documentation/full_pipeline.md.

    All four architectures are used, deliberately without selection -- picking
    the three best hybrid_batch cells off a test table is the exact move the
    reward-seed note warns against.

    BATCHNORM IS THE HAZARD THIS REINTRODUCES, so it is asserted rather than
    assumed. In train mode BatchNorm1d takes its statistics from the batch, so
    C(m) would depend on the GRPO group m was sampled with and the reward would
    stop being a function of the molecule (3.97e-02 drift measured in
    limitations 5.0, against 6.0e-08 for GraphNorm). In `.eval()` it uses
    running statistics instead, and the check below proves that holds for
    these exact checkpoints before any of them is handed to a training loop.
    The default reward does not need this check, which is why GraphNorm
    remains the right default: it removes the failure mode instead of guarding
    it.
    """
    from .hybrid_features import compute_descriptors
    from .models_hybrid import build_hybrid_model

    root = RESULTS_DIR / "hybrid_batch"
    nets, scalings = [], []
    folds: dict[str, np.ndarray] = {}
    norms: dict[str, str] = {}
    for name in models:
        d = root / dataset / name / f"seed{seed}"
        ckpt = d / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt} missing. Train it with "
                f"`python -m src.train_hybrid --run-all --norm batch`."
            )
        arch = run_contract(d)
        norms[name] = arch.get("norm")
        folds[name] = np.load(d / "test_preds.npz")["idx"]
        net = build_hybrid_model(name, **arch).to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()
        net.requires_grad_(False)
        nets.append(net)
        sc = np.load(d / "descriptor_scaling.npz")
        scalings.append((sc["mu"], sc["sigma"]))

    first = models[0]
    for name in models:
        if not np.array_equal(folds[name], folds[first]):
            shared = len(np.intersect1d(folds[name], folds[first]))
            raise RuntimeError(
                f"ENSEMBLE LEAK: {name} and {first} (hybrid_batch, "
                f"dataset={dataset}, seed={seed}) were trained on different "
                f"folds -- {shared}/{len(folds[first])} test molecules shared."
            )
    if set(norms.values()) != {"batch"}:
        raise RuntimeError(
            f"load_frozen_hybrid expects the BatchNorm arm, got norms={norms}. "
            f"results/hybrid_batch must hold norm=batch runs only."
        )

    @torch.no_grad()
    def classify(mols: list[Chem.Mol]) -> np.ndarray:
        if not mols:
            return np.zeros(0, dtype=float)
        smis = [Chem.MolToSmiles(m) for m in mols]
        graphs = [smiles_to_graph(s) for s in smis]
        raw = np.array([compute_descriptors(s) for s in smis], dtype=np.float64)
        probs = []
        for net, (mu, sigma) in zip(nets, scalings):
            # Each member standardizes with ITS OWN train-fold mu/sigma. Never
            # statistics refit on the molecules being scored: that would let
            # the generated distribution move its own reward.
            std = (raw - mu) / sigma
            for g, row in zip(graphs, std):
                g.descriptors = torch.tensor(row, dtype=torch.float).unsqueeze(0)
            batch = Batch.from_data_list(graphs).to(device)
            probs.append(torch.sigmoid(net(batch)))
        return torch.stack(probs).mean(0).cpu().numpy()

    # Prove batch-independence on these checkpoints before they can be used.
    probe = [Chem.MolFromSmiles(s) for s in
             ("CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
              "ClC(Cl)(Cl)Cl", "c1ccc2c(c1)ccc1ccccc12")]
    alone = np.array([classify([m])[0] for m in probe])
    crowded = classify(probe)
    drift = float(np.abs(alone - crowded).max())
    if drift > 1e-5:
        raise RuntimeError(
            f"hybrid_batch reward is batch-DEPENDENT: C(m) drifts {drift:.2e} "
            f"between scoring a molecule alone and in a group of four. eval() "
            f"is not in force, so the reward is not a function of the molecule."
        )

    return classify


def sanitizable(smiles: str) -> Chem.Mol | None:
    """Chemical validity, i.e. RDKit sanitization -- NOT grammar validity.

    Section 1.1: a SELFIES string always parses to some molecule, so the
    generator's syntactic guarantee says nothing about whether the result
    survives sanitization. These are different failure modes; only this one
    gates the reward.

    A dummy atom also fails here. `[3*]C(C)=O` sanitizes perfectly well, so a
    BRICS fragment with an unfilled attachment point would otherwise be counted
    valid and scored -- an open valence is not a molecule, and the classifier
    has never seen one. This is the same guard as in `featurize.smiles_to_graph`
    and it is deliberately in both: this one decides `valid`, that one decides
    what reaches the network.
    """
    mol = Chem.MolFromSmiles(smiles)  # returns None iff sanitization fails
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return None
    return mol


def score_terms(smiles: list[str], classify, tox=None,
                min_heavy_atoms: int = 0, min_mw: float = 0.0,
                alert_lambda: float = 0.0, min_c: float = 0.0
                ) -> dict[str, np.ndarray]:
    """Per-molecule stationary reward terms. Invalid rows are left at zero.

    `tox` is an optional frozen toxicity scorer (src.tox.load_frozen_tox). It
    returns P(active) across the Tox21 assays, and is stored here as the
    NON-toxicity term `t = 1 - P(active)` so that, like C, QED and SA, higher
    is better and `assemble` stays a plain weighted sum with no sign
    exceptions to remember.

    Without it `t` is all ones, i.e. the term is present but inert, so a
    history logged with and without a toxicity model stays the same shape.

    `min_heavy_atoms` / `min_mw` gate out molecules too small to be drugs by
    marking them INVALID, so they take the flat penalty and the hurdle
    advantage exactly as a valence violation does. Two reasons to spend the
    gate here rather than as another weighted term:

      * a soft size reward is hackable in the other direction -- anything
        monotone in size pays the generator to append carbon chains -- while
        a gate has no gradient to climb at all;
      * a term that must drive the reward to zero has a derivative that blows
        up as it approaches zero. A hard gate sidesteps that entirely, which
        is strictly better than the same veto expressed multiplicatively.

    Both default to 0, i.e. off, so nothing that predates them changes. The
    returned `undersized` mask reports what the gate removed, separately from
    what failed sanitization, because those two are different problems and a
    group starved of valid molecules must be diagnosable.
    """
    n = len(smiles)
    valid = np.zeros(n, dtype=bool)
    c = np.zeros(n)
    qed = np.zeros(n)
    sa = np.zeros(n)
    t = np.ones(n)
    a = np.ones(n)

    undersized = np.zeros(n, dtype=bool)

    mols, keep = [], []
    for i, smi in enumerate(smiles):
        mol = sanitizable(smi)
        if mol is None:
            continue
        if (min_heavy_atoms and mol.GetNumHeavyAtoms() < min_heavy_atoms) or (
            min_mw and Descriptors.MolWt(mol) < min_mw
        ):
            undersized[i] = True
            continue
        mols.append(mol)
        keep.append(i)

    if mols:
        idx = np.array(keep)
        valid[idx] = True
        c[idx] = classify(mols)
        qed[idx] = [QED.qed(m) for m in mols]
        # Flip and rescale so higher is better, matching the other three terms.
        raw_sa = np.array([sascorer.calculateScore(m) for m in mols])
        sa[idx] = (SA_MAX - raw_sa) / (SA_MAX - SA_MIN)
        if alert_lambda:
            # Reactive-group penalty, exp(-lambda * n_distinct_alerts). Soft
            # rather than a gate because 32% of real BBB+ drugs trip the same
            # patterns -- a veto would reject a third of the answer.
            a[idx] = alert_score([Chem.MolToSmiles(m) for m in mols], alert_lambda)
        if tox is not None:
            # Scored from canonical SMILES rather than the raw input so the
            # toxicity model sees exactly the molecule the other terms did.
            t[idx] = 1.0 - np.asarray(tox([Chem.MolToSmiles(m) for m in mols]))

    # PERMEABILITY AS A GATE RATHER THAN A DIRECTION. Applied after classify()
    # because, unlike the size gates above, it needs C to exist first.
    #
    # Why a gate: `assemble` is a weighted sum, so a monotone C term pays the
    # policy to climb toward whatever maximizes C -- and what maximizes C is
    # carbon tetrachloride, which BBBP labels BBB+ correctly because it really
    # does cross the barrier (limitations 3.1). The same argument already
    # justified gating size rather than rewarding it: a gate has no gradient to
    # climb at all. Pair this with `--drop C` so permeability is a REQUIREMENT
    # and the optimization pressure sits on Q/S/D instead.
    #
    # This is nearly free: limitations 1.7 measured mean_C at 0.786 with
    # w_C = 0, against BBBP's own 0.745 -- the policy's permeability is mostly a
    # side effect of what the other terms select for, not something it chases.
    subthreshold = np.zeros(n, dtype=bool)
    if min_c:
        below = valid & (c < min_c)
        subthreshold[below] = True
        valid[below] = False

    return {"valid": valid, "c": c, "qed": qed, "sa": sa, "t": t, "a": a,
            "undersized": undersized, "subthreshold": subthreshold}


def weights_at(
    k: int,
    w0: dict[str, float],
    wf: dict[str, float] | None = None,
    schedule: str = "fixed",
    k_anneal: int = 1000,
    valid_history: list[float] | None = None,
    tau: float = 0.9,
    window: int = 10,
) -> dict[str, float]:
    """Reward weights at generator step k. Eq. 6 (linear/cosine), Eq. 7 (gated).

    Pure function: the validity-gated variant reads the caller's own history
    list rather than owning a rolling buffer, so nothing here carries state
    across steps and each schedule is checkable in isolation.
    """
    if schedule == "fixed" or wf is None:
        return dict(w0)

    if schedule == "linear":
        lam = min(k / k_anneal, 1.0)
    elif schedule == "cosine":
        lam = 0.5 * (1.0 - np.cos(np.pi * min(k / k_anneal, 1.0)))
    elif schedule == "gated":
        # Hard switch once the W-step mean validity clears tau. Before the
        # window has filled there is no evidence to switch on, so stay at w0.
        recent = (valid_history or [])[-window:]
        lam = 1.0 if len(recent) == window and float(np.mean(recent)) >= tau else 0.0
    else:
        raise ValueError(f"unknown schedule {schedule!r}")

    # A key present in w0 but not wf holds its initial value rather than
    # raising: the toxicity weight is usually meant to be constant across the
    # curriculum, and a missing endpoint should mean "do not anneal this one"
    # rather than crashing a run 80 steps in.
    return {key: w0[key] + (wf.get(key, w0[key]) - w0[key]) * lam for key in w0}


# Calibration of the logit transform, measured once on the BBBP reference set
# (n=2039, seed-0 gin/gat/gine ensemble):
#
#   raw C    mean 0.7447   sd 0.2880
#   logit C  mean 1.8591   sd 2.1581
#
# The affine map below sends logit C onto raw C's mean and sd over that
# reference. Without it the transform would also multiply the C term's
# effective weight by ~7.5x, and an A/B against raw C would be confounded by
# scale rather than isolating the change in shape. Recompute with
# `python -m src.reward --calibrate` if the ensemble or dataset changes.
LOGIT_CALIB = {"mu_raw": 0.7447, "sd_raw": 0.2880,
               "mu_logit": 1.8591, "sd_logit": 2.1581}


def transform_c(c: np.ndarray, mode: str = "raw",
                calib: dict[str, float] | None = None) -> np.ndarray:
    """Rescale the permeability term. `raw` is the identity.

    `logit` exists because the classifier's output saturates near 1 where the
    sigmoid compresses real differences into nothing: measured on the
    collapsed baseline, between-molecule sd is 0.006 in probability space but
    0.35 in logit space, and MC-dropout uncertainty is FLAT across that same
    range in logit space (0.86 on real BBBP, 0.99 on collapsed samples) while
    appearing to shrink 3.6x in probability space. The apparent certainty is
    the scale, not the model. GRPO ranks within a group, so a scale that
    compresses differences costs it exactly the signal it consumes.

    The transform is calibrated to leave the term's mean and sd unchanged on
    the reference set, so w_C keeps its meaning across modes.
    """
    if mode == "raw":
        return c
    if mode != "logit":
        raise ValueError(f"unknown c transform {mode!r}")
    k = calib or LOGIT_CALIB
    p = np.clip(c, 1e-6, 1.0 - 1e-6)
    z = (np.log(p / (1.0 - p)) - k["mu_logit"]) / k["sd_logit"]
    return k["mu_raw"] + k["sd_raw"] * z


def assemble(
    terms: dict[str, np.ndarray],
    d_scores: np.ndarray,
    w: dict[str, float],
    invalid_reward: float = 0.0,
    c_transform: str = "raw",
    aggregate: str = "linear",
    eps: float = 1e-6,
) -> np.ndarray:
    """Eq. 5. Invalid molecules collapse to a flat penalty, bypassing every term.

    `c_transform` rescales the permeability term only; see `transform_c`.
    It is applied here rather than in `score_terms` so that `terms["c"]`
    stays the raw probability for logging and diagnostics, and a history
    written under one mode remains comparable to one written under the other.
    """
    parts = {
        "D": np.asarray(d_scores, dtype=float),
        "C": transform_c(terms["c"], c_transform),
        "Q": terms["qed"],
        "S": terms["sa"],
    }
    # Non-toxicity, opt-in. Absent from `w` the term contributes nothing, so
    # existing callers and their weight dicts keep working unchanged; absent
    # from `terms` (no toxicity model loaded) it is all ones and adds a
    # constant, which a group-relative advantage subtracts away.
    if w.get("T"):
        parts["T"] = terms.get("t", np.ones_like(parts["C"]))
    # Reactive-group alerts, opt-in the same way. Inside a geometric mean the
    # 1/n exponent dilutes this heavily: at lambda 0.15 and weight 1 a
    # four-alert molecule still keeps 0.89 of its reward. Weight it to match
    # the bite intended, and see the table in the commit message.
    if w.get("A"):
        parts["A"] = terms.get("a", np.ones_like(parts["C"]))

    if aggregate == "linear":
        r = sum(w[k] * v for k, v in parts.items() if w.get(k))
    elif aggregate == "geometric":
        # Weighted geometric mean: prod(S_k ** w_k) ** (1 / sum w_k).
        #
        # The point is that a near-zero term drags the whole product down
        # instead of being outbid by a maxed-out one. Measured on the probe
        # set it closes the CCl4-vs-caffeine gap from +0.217 to +0.015, so it
        # is a real improvement and NOT a sufficient one on its own: it only
        # vetoes when some term actually approaches zero, and CCl4 is merely
        # mediocre everywhere (QED 0.470) while maxing out C. The size gate in
        # score_terms is what supplies the actual veto; this makes the
        # remaining terms harder to trade off against each other.
        #
        # eps keeps both the value and the gradient finite: d(GM)/dS_k scales
        # as GM / S_k, which diverges as S_k -> 0.
        # Structural refusal, not a value check: transform_c("logit") is
        # centred on the reference mean, so it is negative for any molecule
        # below it and unbounded above. Whether a NEGATIVE value happens to
        # appear in this particular group is luck, and a guard that only
        # fires when it does would pass in testing and raise mid-run.
        if c_transform != "raw":
            raise ValueError(
                f"aggregate='geometric' needs terms on a common [0, 1] scale, "
                f"but c_transform={c_transform!r} is unbounded and can be "
                f"negative. Use one or the other."
            )
        used = {k: v for k, v in parts.items() if w.get(k)}
        total = sum(w[k] for k in used)
        if total <= 0:
            raise ValueError("geometric aggregation needs at least one positive weight")
        acc = np.zeros_like(parts["C"], dtype=float)
        for k, v in used.items():
            # Terms must be on a common [0, 1] scale for a product to mean
            # anything. transform_c("logit") is centred on the reference mean
            # and goes negative, so the two options are incompatible and
            # silently producing NaNs here would be worse than refusing.
            if np.any(v < -eps):
                raise ValueError(
                    f"geometric aggregation needs terms in [0, 1]; {k} has "
                    f"min {float(np.min(v)):.3f}. c_transform='logit' is not "
                    f"compatible with aggregate='geometric'."
                )
            acc = acc + w[k] * np.log(np.clip(v, eps, 1.0))
        r = np.exp(acc / total)
    elif aggregate in ("nsga2", "composite"):
        # Pareto reduction. Our linear reward is a Minkowski-weighted
        # scalarization, which by Das & Dennis (1997) can only reach the
        # convex hull of the frontier -- measured on our own groups, 0.48 of
        # it. Dominance ranking does not collapse the objective vector before
        # ranking, so a concave fold is reachable. See src/pareto.py.
        #
        # Dominance and crowding are computed over the VALID subset only.
        # Invalid rows carry zeros in every term, so including them would
        # make them dominated by construction and, worse, stretch every
        # crowding range toward zero -- the diversity signal would be
        # measuring the invalid rate rather than the spread of real
        # candidates.
        used = [(k, v) for k, v in parts.items() if w.get(k)]
        if not used:
            raise ValueError(f"{aggregate} needs at least one positive weight")
        valid = terms["valid"]
        r = np.full(len(valid), float(invalid_reward))
        if valid.sum() >= 2:
            F = np.stack([v[valid] for _, v in used], axis=1)
            r[valid] = reduce_objectives(
                F, np.array([w[k] for k, _ in used]), mode=aggregate
            )
        elif valid.any():
            r[valid] = 0.0
        # Returned directly: these scores are centred near zero and go
        # negative, so `invalid_reward` is NOT a floor below them. Pair this
        # with group_advantages(valid=...), which hands invalid molecules a
        # fixed floor and never lets them into the valid subgroup's scale.
        return r
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")

    return np.where(terms["valid"], r, invalid_reward)


def _self_check() -> None:
    # Validity gate: the second string is syntactically fine and chemically not.
    smis = ["CC(=O)Oc1ccccc1C(=O)O", "C(C)(C)(C)(C)C", "CCO", "not-a-molecule"]
    assert sanitizable(smis[0]) is not None
    assert sanitizable(smis[1]) is None, "5-valent carbon must fail sanitization"
    assert sanitizable(smis[3]) is None

    # Ensemble members must share a scaffold split. The seed reseeds the split
    # (README), so averaging models from different folds would mean averaging
    # models that trained on each other's held-out molecules.
    folds = {
        name: np.load(_checkpoint(RESULTS_DIR, "bbbp", name, 0).parent / "test_preds.npz")["idx"]
        for name in ("gin", "gat", "gine")
    }
    for name, idx in folds.items():
        assert np.array_equal(idx, folds["gin"]), f"{name} trained on a different fold"

    # Frozen classifier: same molecule must score identically in any company.
    classify = load_frozen_classifier()
    alone = classify([Chem.MolFromSmiles("CCO")])
    crowded = classify([Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CCN"]])
    assert np.allclose(alone[0], crowded[0], atol=1e-6), (
        f"C(m) moved with batch composition ({alone[0]} vs {crowded[0]}): "
        "BatchNorm is in train mode and the reward is not stationary"
    )

    terms = score_terms(smis, classify)
    assert terms["valid"].tolist() == [True, False, True, False]
    assert terms["c"][1] == 0.0 and terms["qed"][3] == 0.0
    assert (0.0 <= terms["c"][terms["valid"]]).all()
    assert (terms["c"][terms["valid"]] <= 1.0).all()

    w = {"D": 1.0, "C": 1.0, "Q": 0.5, "S": 0.5}
    r = assemble(terms, np.full(4, 0.5), w)
    assert r[1] == 0.0 and r[3] == 0.0, "invalid molecules must not collect reward"
    assert r[0] > 0 and r[2] > 0

    # Schedules. Linear hits the endpoints; gated needs a full window over tau.
    w0 = {"D": 1.0, "C": 0.1, "Q": 0.5, "S": 0.5}
    wf = {"D": 0.1, "C": 1.0, "Q": 0.5, "S": 0.5}
    assert weights_at(0, w0, wf, "linear", k_anneal=100)["C"] == 0.1
    assert weights_at(100, w0, wf, "linear", k_anneal=100)["C"] == 1.0
    assert weights_at(999, w0, wf, "linear", k_anneal=100)["C"] == 1.0, "must clamp"
    assert weights_at(50, w0, wf, "cosine", k_anneal=100)["D"] == 0.55
    assert weights_at(5, w0, wf, "gated", valid_history=[1.0] * 3)["C"] == 0.1
    assert weights_at(5, w0, wf, "gated", valid_history=[1.0] * 10)["C"] == 1.0
    assert weights_at(5, w0, wf, "gated", valid_history=[0.5] * 10)["C"] == 0.1
    assert weights_at(5, w0, wf)["C"] == 0.1, "no wf -> fixed"

    # --- logit transform ------------------------------------------------
    # Identity when asked for, and order-preserving always: the transform may
    # rescale the term but must never reorder two molecules, or it is not a
    # recalibration but a different objective.
    c = np.array([0.05, 0.3, 0.5, 0.9, 0.99, 0.999])
    assert np.array_equal(transform_c(c, "raw"), c)
    t = transform_c(c, "logit")
    assert np.all(np.diff(t) > 0), f"logit transform reordered molecules: {t}"

    # Calibrated to leave the reference mean and sd alone, so w_C keeps its
    # meaning across modes and an A/B isolates shape rather than weight.
    k = LOGIT_CALIB
    at_mean = transform_c(np.array([1 / (1 + np.exp(-k["mu_logit"]))]), "logit")[0]
    assert abs(at_mean - k["mu_raw"]) < 1e-6, at_mean

    # The point of the exercise: differences that the sigmoid compresses to
    # nothing must survive. On the collapsed baseline's observed range the
    # raw spread is ~0.006 and the transform must expand it substantially.
    tight = np.array([0.959, 0.975, 0.990, 0.994])
    gain = transform_c(tight, "logit").std() / tight.std()
    assert gain > 5, f"transform recovered too little spread ({gain:.2f}x)"

    # Saturated inputs must stay finite rather than becoming inf through the
    # log; a single inf would poison the whole group's advantage.
    assert np.all(np.isfinite(transform_c(np.array([0.0, 1.0]), "logit")))

    try:
        transform_c(c, "sigmoid")
        raise AssertionError("unknown transform should have been rejected")
    except ValueError:
        pass

    # And it must reach the reward: same terms, different mode, different R.
    r_raw = assemble(terms, np.full(4, 0.5), w)
    r_log = assemble(terms, np.full(4, 0.5), w, c_transform="logit")
    assert not np.allclose(r_raw, r_log), "c_transform did not reach assemble"
    assert np.array_equal(r_raw[~terms["valid"]], r_log[~terms["valid"]]), \
        "invalid molecules must take the flat penalty under either mode"

    # --- toxicity term ----------------------------------------------------
    # Inert when no model is supplied, so nothing that predates it changes.
    assert (terms["t"] == 1.0).all(), "t must default to 1 with no tox model"
    assert np.allclose(assemble(terms, np.full(4, 0.5), w),
                       assemble(terms, np.full(4, 0.5), {**w, "T": 0.0})), \
        "T=0 must be identical to T absent"

    # With a stub scorer it must reach the reward, and in the right
    # direction: more toxic must mean less reward.
    clean = score_terms(smis, classify, tox=lambda s: np.zeros(len(s)))
    dirty = score_terms(smis, classify, tox=lambda s: np.ones(len(s)))
    assert (clean["t"][clean["valid"]] == 1.0).all()
    assert (dirty["t"][dirty["valid"]] == 0.0).all()
    wt = {**w, "T": 1.0}
    r_clean = assemble(clean, np.full(4, 0.5), wt)
    r_dirty = assemble(dirty, np.full(4, 0.5), wt)
    assert (r_clean[clean["valid"]] > r_dirty[dirty["valid"]]).all(), \
        "predicted-toxic molecules must score lower, not higher"
    # Invalid molecules keep the flat penalty; toxicity must not rescue them.
    assert (r_clean[~clean["valid"]] == 0.0).all()

    # Every other term must be untouched by adding a toxicity model.
    for key in ("c", "qed", "sa", "valid"):
        assert np.array_equal(terms[key], clean[key]), f"{key} changed"

    # A weight dict without T must still work, since every existing caller
    # has one.
    assert np.isfinite(assemble(terms, np.full(4, 0.5), {"D": 1, "C": 1,
                                                         "Q": 0.5, "S": 0.5})).all()
    # And a schedule whose wf omits T must hold T constant rather than raise.
    got = weights_at(50, {"D": 1.0, "C": 0.2, "Q": 0.5, "S": 0.5, "T": 0.8},
                     {"D": 0.3, "C": 1.5, "Q": 0.5, "S": 0.5},
                     schedule="linear", k_anneal=100)
    assert got["T"] == 0.8, got

    # --- size gate --------------------------------------------------------
    solvents = ["ClC(Cl)(Cl)Cl", "FC(F)(F)F", "FC(F)(Cl)Cl"]
    drugs = ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1"]
    both = solvents + drugs

    open_gate = score_terms(both, classify)
    assert open_gate["valid"].all(), "gate off must pass everything sanitizable"
    assert not open_gate["undersized"].any()

    gated = score_terms(both, classify, min_heavy_atoms=10)
    assert list(gated["valid"]) == [False, False, False, True, True], gated["valid"]
    assert list(gated["undersized"]) == [True, True, True, False, False]
    # Gated molecules take the flat penalty, exactly as a valence violation
    # does -- that is the whole point of routing them through `valid`.
    rg = assemble(gated, np.full(len(both), 0.5), w)
    assert (rg[:3] == 0.0).all() and (rg[3:] > 0).all(), rg
    # And the surviving molecules must be scored identically either way: the
    # gate removes candidates, it does not perturb the ones it keeps.
    assert np.allclose(gated["c"][3:], open_gate["c"][3:])

    # MW is an independent route to the same outcome, but a weaker one for
    # exactly this failure: CCl4 is MW 153.8, so a 150 cutoff lets it through
    # while the heavy-atom count catches it at 5. Halogens are heavy, and
    # weight is a poor proxy for the molecular complexity actually wanted.
    # WHAT THE C GATE DOES NOT DO: reject solvents. Carbon tetrachloride scores
    # C = 0.987, far above any threshold calibrated on real BBB+ molecules,
    # because it genuinely is among the most permeable things in BBBP. A gate on
    # C removes LOW-C molecules; it can never remove the tiny-solvent failure,
    # and it is the SIZE gate that does that (5 heavy atoms vs min 10).
    #
    # Its value is removing the gradient, not filtering: with w_C = 0 the policy
    # is no longer paid to climb toward maximal C, so it stops being pulled
    # toward the small-lipophilic corner where solvents live. The assertion
    # below pins that down so nobody later reads the gate as a solvent filter.
    solvent, drug = "ClC(Cl)(Cl)Cl", "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
    # Distinct names: `gated` belongs to the size-gate block above and is read
    # again by the summary print at the end of this function.
    c_open = score_terms([solvent, drug], classify)
    assert c_open["valid"].all(), "both should be valid with no gate"
    assert c_open["c"][0] > 0.6, f"probe assumes a high-C solvent, got {c_open['c'][0]:.3f}"
    c_gated = score_terms([solvent, drug], classify, min_c=0.615)
    assert c_gated["valid"][0] and not c_gated["subthreshold"][0], (
        "a C gate must NOT remove a high-C solvent -- if this ever fires, the "
        "threshold has been raised past the real-BBB+ distribution and the gate "
        "is now rejecting genuine CNS drugs too"
    )
    assert c_gated["valid"][1], "haloperidol must survive a gate set at the real-BBB+ p5"
    # It does remove a genuinely impermeable molecule, which is its actual job.
    low = score_terms(["NC(=O)c1ccc(N)cc1S(=O)(=O)O"], classify, min_c=0.615)
    assert low["subthreshold"][0] and not low["valid"][0], low
    # C itself is still logged for diagnostics even when the row is gated out.
    assert c_gated["c"][0] == c_open["c"][0]
    # And a gate of 0 must change nothing at all.
    assert np.array_equal(score_terms([solvent, drug], classify, min_c=0.0)["valid"],
                          c_open["valid"])

    assert not score_terms(["ClC(Cl)(Cl)Cl"], classify, min_mw=150.0)["undersized"][0]
    assert score_terms(["ClC(Cl)(Cl)Cl"], classify, min_mw=200.0)["undersized"][0]
    assert score_terms(["FC(F)(F)F"], classify, min_mw=150.0)["undersized"][0]

    # --- geometric aggregation -------------------------------------------
    gw = {"D": 1.0, "C": 1.0, "Q": 1.0, "S": 1.0}
    probe = score_terms(both, classify)
    d05 = np.full(len(both), 0.5)
    gm = assemble(probe, d05, gw, aggregate="geometric")
    assert np.all(np.isfinite(gm)) and np.all(gm >= 0)

    # Equal weights on identical terms reduce to that value.
    flat = {"valid": np.array([True]), "c": np.array([0.4]), "qed": np.array([0.4]),
            "sa": np.array([0.4]), "t": np.array([1.0]),
            "undersized": np.array([False])}
    assert abs(assemble(flat, np.array([0.4]), gw, aggregate="geometric")[0] - 0.4) < 1e-6

    # A single near-zero term must drag the product down -- the property the
    # linear form lacks. Compare a molecule good everywhere against the same
    # one with QED ~ 0.
    good = {k: v.copy() for k, v in flat.items()}
    bad = {k: v.copy() for k, v in flat.items()}
    bad["qed"] = np.array([0.0])
    g_lin = assemble(good, np.array([0.4]), gw)[0]
    b_lin = assemble(bad, np.array([0.4]), gw)[0]
    g_gm = assemble(good, np.array([0.4]), gw, aggregate="geometric")[0]
    b_gm = assemble(bad, np.array([0.4]), gw, aggregate="geometric")[0]
    assert b_gm / g_gm < 0.05, f"geometric mean failed to veto ({b_gm:.4f}/{g_gm:.4f})"
    assert b_lin / g_lin > 0.5, "linear form should barely notice, by contrast"

    # eps must keep a zero term finite rather than producing -inf / nan.
    assert np.isfinite(b_gm) and b_gm > 0

    # Logit C is centred on the reference mean and goes negative, so it
    # cannot be multiplied. Refuse rather than emit NaN.
    try:
        assemble(probe, d05, gw, aggregate="geometric", c_transform="logit")
        raise AssertionError("logit + geometric should have been rejected")
    except ValueError:
        pass
    try:
        assemble(probe, d05, gw, aggregate="cubic")
        raise AssertionError("unknown aggregate should have been rejected")
    except ValueError:
        pass

    # --- alert term -------------------------------------------------------
    assert (terms["a"] == 1.0).all(), "alerts must default to inert"
    az = "CC(NC1=CC=CC=C1)OCC(=O)N2CC2C(C)C3=CC=C(F)C=C3CC"   # aziridine, gate+GM
    al = score_terms([smis[0], az], classify, alert_lambda=0.3)
    assert al["a"][0] == 1.0, "aspirin trips no alert"
    assert al["a"][1] < 1.0, "aziridine must be penalized"
    # It must reach the reward, and only when weighted.
    d2 = np.full(2, 0.5)
    w2 = {"D": 1.0, "C": 1.0, "Q": 1.0, "S": 1.0}
    assert np.allclose(assemble(al, d2, w2), assemble(al, d2, {**w2, "A": 0.0}))
    # Compare at FIXED weights, varying only the alert score -- adding a
    # term changes the geometric normalizer (total weight), so a
    # with-versus-without comparison is not controlled and can move either
    # way regardless of the penalty.
    wa = {**w2, "A": 3.0}
    pen = assemble(al, d2, wa, aggregate="geometric")
    clean_al = {**al, "a": np.ones_like(al["a"])}
    nop = assemble(clean_al, d2, wa, aggregate="geometric")
    assert pen[1] < nop[1], "alert term did not reduce the flagged molecule"
    assert abs(pen[0] - nop[0]) < 1e-9, "unflagged molecule must be untouched"
    # And the bite must scale with lambda.
    hi = score_terms([smis[0], az], classify, alert_lambda=0.8)
    assert assemble(hi, d2, wa, aggregate="geometric")[1] < pen[1]

    print(f"C(aspirin)={terms['c'][0]:.3f}  QED={terms['qed'][0]:.3f}  "
          f"SA_norm={terms['sa'][0]:.3f}  R={r[0]:.3f}")
    print(f"C gate at 0.615: solvent C={c_open['c'][0]:.3f} "
          f"{'GATED OUT' if c_gated['subthreshold'][0] else 'KEPT'}, "
          f"haloperidol C={c_open['c'][1]:.3f} "
          f"{'kept' if c_gated['valid'][1] else 'GATED OUT'}")
    print(f"size gate: {int(gated['undersized'].sum())}/3 solvents removed, "
          f"both drugs kept")
    print(f"geometric veto: QED->0 drops reward to {b_gm / g_gm:.1%} of baseline "
          f"(linear: {b_lin / g_lin:.0%})")
    print(f"logit transform: {gain:.1f}x spread recovery on the saturated range")
    print("reward self-check passed")


def _calibrate(dataset: str = "bbbp", device: str = "cpu") -> None:
    """Recompute LOGIT_CALIB against the current ensemble and dataset."""
    import pandas as pd

    from .datasets import DATASETS

    cfg = DATASETS[dataset]
    df = pd.read_csv(cfg["path"], sep=cfg["sep"])
    smiles = df[cfg["smiles_col"]].dropna().tolist()
    classify = load_frozen_classifier(dataset=dataset, device=device)
    mols = [m for m in (sanitizable(s) for s in smiles) if m is not None]
    p = np.concatenate([classify(mols[i:i + 256]) for i in range(0, len(mols), 256)])
    q = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(q / (1 - q))
    print(f"n={len(p)}")
    print('LOGIT_CALIB = {"mu_raw": %.4f, "sd_raw": %.4f,' % (p.mean(), p.std()))
    print('               "mu_logit": %.4f, "sd_logit": %.4f}' % (lg.mean(), lg.std()))


if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        _calibrate()
    else:
        _self_check()

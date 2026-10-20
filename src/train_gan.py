"""Outer GAN+GRPO loop: sample -> reward -> advantage -> clipped update.

Closes the diagram. The three pieces that already existed (`reward.py`,
`grpo.py`, `generator.py`) are wired together here, plus D_phi, which is the
same GNN skeleton the permeability classifier uses -- real-vs-fake is the same
graph->logit problem, so it reuses `build_model` rather than introducing a
second architecture.

Two ordering constraints from Section 3 of the continuation draft are structural
here rather than advisory:

  * D_phi is scored in eval() (Section 3.1's stationarity argument applied to D:
    with BatchNorm in train mode, D(m) would depend on which other candidates
    shared the group, so the reward would stop being a function of the molecule).
  * the n_D discriminator steps run strictly AFTER the K-epoch PPO block, never
    interleaved (Section 3.4), so every advantage in a group is computed against
    one fixed phi snapshot.

    python -m src.train_gan --steps 200     # train
    python -m src.train_gan                 # self-check (tiny, no checkpoints)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch

from .datasets import ROOT, _load_raw
from .diversity import reference_stats, summarize
from .featurize import FEATURE_ID, graphs_from_smiles
from .brics_generator import BricsGenerator
from .generator import CKPT_DIR, SelfiesGenerator
from .grpo import clipped_loss, group_advantages
from .models import build_model
from .reward import (assemble, load_frozen_classifier, load_frozen_hybrid,
                     score_terms, weights_at)

# Generator families. The BRICS policy is valid by construction; the SELFIES
# one relies on training for validity. Both satisfy the same roll contract,
# so nothing below this line branches on the choice.
GENERATORS = {"selfies": SelfiesGenerator, "brics": BricsGenerator}
WARM_NAMES = {"selfies": "{dataset}_pretrained.pt",
              "brics": "{dataset}_brics_pretrained.pt"}
PRETRAIN_CMD = {"selfies": "src.generator", "brics": "src.brics_generator"}

RDLogger.DisableLog("rdApp.*")

OUT_DIR = ROOT / "results" / "gan"

W0 = {"D": 1.0, "C": 0.2, "Q": 0.5, "S": 0.5}   # validity/realism first
WF = {"D": 0.3, "C": 1.5, "Q": 0.5, "S": 0.5}   # then permeability


def _graphs(smiles: list[str]) -> Batch | None:
    """Featurize the sanitizable subset. None if nothing survives.

    Routed through featurize.graphs_from_smiles so the discriminator sees
    exactly what the classifier sees. These were separate paths: reward.py
    canonicalized first, this file featurized the generator's raw output in
    three places. They agreed only by accident, and a disagreement would
    present as a training instability rather than as a bug.
    """
    graphs = [g for g in graphs_from_smiles(smiles) if g is not None]
    return Batch.from_data_list(graphs) if graphs else None


class Discriminator(nn.Module):
    """D_phi: real-vs-fake over molecular graphs. Same skeleton as the classifier."""

    def __init__(self, conv: str = "gin", **kwargs) -> None:
        super().__init__()
        self.net = build_model(conv, **kwargs)

    def forward(self, batch) -> torch.Tensor:
        return self.net(batch)  # logit per graph

    @torch.no_grad()
    def score(self, smiles: list[str], device: str = "cpu") -> np.ndarray:
        """D(m) in [0,1] per input SMILES, 0 for anything unsanitizable.

        eval() is load-bearing for the same reason it is on the frozen
        classifier: BatchNorm in train mode makes D(m) a function of the group
        rather than of the molecule.
        """
        was_training = self.training
        self.eval()
        out = np.zeros(len(smiles))
        # One featurization pass, not two: this used to call smiles_to_graph
        # to build `keep` and then again inside _graphs.
        graphs = graphs_from_smiles(smiles)
        keep = [i for i, g in enumerate(graphs) if g is not None]
        if keep:
            batch = Batch.from_data_list([graphs[i] for i in keep]).to(device)
            out[keep] = torch.sigmoid(self(batch)).cpu().numpy()
        self.train(was_training)
        return out


def _d_step(disc, opt, real_smiles, fake_smiles, device) -> float:
    """One non-saturating GAN discriminator step. Returns BCE, or nan if skipped.

    A step is skipped rather than faked when either side is empty: early in
    training whole groups fail sanitization, and training D on reals alone
    would teach it to output 1 unconditionally.
    """
    real, fake = _graphs(real_smiles), _graphs(fake_smiles)
    if real is None or fake is None:
        return float("nan")
    disc.train()
    logits = disc(Batch.from_data_list(real.to_data_list() + fake.to_data_list()).to(device))
    target = torch.cat([torch.ones(real.num_graphs), torch.zeros(fake.num_graphs)]).to(device)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def train(
    dataset: str = "bbbp",
    steps: int = 200,
    group_size: int = 64,
    ppo_epochs: int = 4,
    n_d: int = 2,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    clip_eps: float = 0.2,
    # kl_coef=0.3 and min_heavy_atoms=10 are the defaults because they are the
    # only two interventions that have survived validation, and leaving them
    # off made the DOCUMENTED command reproduce the hack. The KL sweep's
    # three-part criterion fails at 0.02 and 0.1 and passes at 0.3 and 1.0;
    # 0.3 is the smaller passing value and keeps roughly twice the permeability
    # gain of 1.0 (c_gain 0.082 vs 0.043). At 0.02 scaffold diversity ends at
    # 0.06-0.19 against 0.78 at 0.3 -- same code, same seeds.
    #
    # Deliberately still off: the logit transform makes scaffold collapse WORSE
    # (0.741 -> 0.349, limitations 1.3 -- more gradient on a hackable objective
    # buys more hacking), the Tox21 term worse still (3.3), and the alert
    # penalty reduced only the SMARTS it was penalized on with no transfer to
    # BRENK or PAINS (the Goodhart finding, 1.2b). Geometric aggregation scored
    # worst of all four arms on every alert instrument.
    #
    # This is containment, not repair. See limitations 1.1: the failure is
    # [mitigated], never resolved, and the mechanism is unchanged.
    kl_coef: float = 0.3,
    invalid_floor: float = -2.0,
    schedule: str = "linear",
    k_anneal: int = 100,
    c_transform: str = "raw",
    w_tox: float = 0.0,
    min_heavy_atoms: int = 10,   # BBBP 5th percentile; see note on kl_coef
    min_mw: float = 0.0,
    aggregate: str = "linear",
    alert_lambda: float = 0.0,
    w_alert: float = 0.0,
    drop: tuple[str, ...] = (),
    max_len: int = 72,
    device: str = "cpu",
    out_dir: Path = OUT_DIR,
    log: bool = True,
    seed: int = 0,
    warm_start: str | None = None,
    threads: int | None = None,
    generator: str = "selfies",
    ent_coef: float = 0.0,
    # The reward ensemble's seed is a ROBUSTNESS AXIS, not a tuning knob. All
    # ten seeds carry gin+gat+gine on element-wise identical folds, so any one
    # of them is a legal ensemble, and member-mean AUC spans 0.818-0.914. A
    # seed must NOT be chosen from that table: picking the reward by a test
    # statistic selects the instrument on test-fold information, and "a seed
    # nearer the mean" is as much a selection as "the best seed". 0 is the right
    # default precisely because it was fixed before anyone looked. Vary it
    # BETWEEN runs to show a result is not an artefact of one fold; never mix
    # seeds within an ensemble, since the seed reseeds the scaffold split and
    # members from different seeds have trained on each other's held-out
    # molecules.
    reward_seed: int = 0,
    # WHICH CLASSIFIER PROVIDES C. "ensemble" is gin/gat/gine under GraphNorm,
    # no descriptors -- the default, and the only one every committed result
    # was trained against. "hybrid_batch" is the reported classifier: four
    # architectures, 8 RDKit descriptors, BatchNorm. Switching it answers a
    # question neither reward can answer alone -- does the generator's output
    # depend on which judge it was optimized against? -- and is recorded in
    # config.json so a run names the reward it chased.
    #
    # BatchNorm as a reward is only safe because load_frozen_hybrid asserts
    # batch-independence on the exact checkpoints before returning. Do not
    # route a BatchNorm model in any other way.
    reward_family: str = "ensemble",
    # Permeability as a gate instead of a reward term. 0 disables it. 0.615
    # admits 95% of real BBB+ molecules under the ensemble reward, i.e. "at
    # least as permeable as 95% of known BBB+ drugs" -- calibrated on the
    # positive class, never on generated output. Use WITH drop=("C",) or the
    # monotone term is still there to climb.
    min_c: float = 0.0,
) -> dict:
    # Which pretrained decoder seeds theta_0, and therefore pi_ref, since
    # Eq. (18) sets pi_ref := pi_theta0. Swapping it swaps the KL anchor too,
    # which is the point of comparing an MLE against a VAE warm start:
    # limitations 1.4 says no kl_coef can produce a distribution wider than
    # its own reference, so the warm start is the ceiling.
    # Pin the thread count IN PROCESS, before any tensor exists. Recording it
    # was not enough. OpenMP changes the order of float reductions and this
    # loop is chaotic, so the thread count is part of the configuration -- but
    # the env vars express it unreliably: OMP_NUM_THREADS=36 silently binds 18
    # on this machine (torch clamps the env route to physical cores) while 5
    # and 6 bind faithfully. A driver that sets the env and trusts it can run
    # two arms at different thread counts believing they match, which is how
    # the first term ablation was contaminated.
    #
    # torch.set_num_threads() is authoritative where the env var is not: it
    # accepts the value as given, including above the physical-core count, and
    # get_num_threads() reports back what was actually set. So passing
    # `threads` makes the run's reduction order a property of the command line
    # rather than of the machine's core count, which is the whole point.
    if threads is not None:
        if threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")
        torch.set_num_threads(threads)

    if generator not in GENERATORS:
        raise ValueError(f"generator must be one of {sorted(GENERATORS)}, got {generator!r}")
    Gen = GENERATORS[generator]
    warm = (Path(warm_start) if warm_start
            else CKPT_DIR / WARM_NAMES[generator].format(dataset=dataset))
    if not warm.exists():
        raise FileNotFoundError(
            f"no warm start at {warm}. GRPO cannot bootstrap from a uniform "
            f"policy -- every group would be all-invalid and carry zero "
            f"advantage. Run it first: "
            f"python -m {PRETRAIN_CMD[generator]} --pretrain --dataset {dataset}"
        )
    bad = set(drop) - set(W0)
    if bad or len(set(drop)) >= len(W0):
        raise ValueError(f"drop must be a proper subset of {sorted(W0)}, got {drop}")

    # Seed every source of randomness that differs between replicate runs: the
    # torch RNG drives both sampling and D's init, and `rng` drives D's real
    # batches. Without this a "3-seed baseline" would be three identical runs.
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = Gen.load(warm, device)
    # Reference policy for the KL term in the diagram: the pretrained generator,
    # frozen. It is what keeps GRPO from drifting off the space of molecules the
    # MLE warm start actually learned while chasing reward.
    ref = Gen.load(warm, device)
    ref.eval().requires_grad_(False)

    disc = Discriminator().to(device)
    if reward_family == "hybrid_batch":
        classify = load_frozen_hybrid(dataset=dataset, seed=reward_seed, device=device)
    elif reward_family == "ensemble":
        classify = load_frozen_classifier(dataset=dataset, seed=reward_seed, device=device)
    else:
        raise ValueError(f"unknown reward_family {reward_family!r}")
    # Frozen exactly as the permeability classifier is. Loaded only when
    # weighted, so a run without a toxicity model costs nothing and the
    # term stays inert rather than absent.
    toxic = None
    if w_tox:
        from .tox import load_frozen_tox
        toxic = load_frozen_tox(device=device)
    real_pool = [s for s in _load_raw(dataset)["smiles"].dropna().tolist()]

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d)
    rng = np.random.default_rng(seed)

    # Reference descriptor statistics, computed once on the real data and held
    # fixed for the whole run. Drift must be measured against the real
    # distribution; measuring it against a rolling window of the generator's
    # own output would normalize away the trend being looked for.
    ref_stats = reference_stats(real_pool)

    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    history, valid_history = [], []

    for k in range(steps):
        # --- 1. sample a group; pi_theta_old is snapshotted here, once -------
        roll = gen.sample(group_size, max_len=max_len, device=device)

        # --- 2. reward against a single frozen phi snapshot ------------------
        terms = score_terms(roll["smiles"], classify, tox=toxic, min_c=min_c,
                            min_heavy_atoms=min_heavy_atoms, min_mw=min_mw,
                            alert_lambda=alert_lambda)
        d_scores = disc.score(roll["smiles"], device=device)
        valid_history.append(float(terms["valid"].mean()))
        w = weights_at(k, {**W0, "T": w_tox, "A": w_alert}, WF, schedule=schedule,
                       k_anneal=k_anneal, valid_history=valid_history)
        # Ablation: zero the weight, keep computing the term. The term is still
        # scored and logged (c_mean etc.), so what the classifier thinks of the
        # generated molecules stays observable even when it no longer steers.
        w = {key: (0.0 if key in drop else val) for key, val in w.items()}
        rewards = assemble(terms, d_scores, w, c_transform=c_transform,
                           aggregate=aggregate)
        # Hurdle form: the reward from `assemble` is zero-inflated by
        # construction, so the invalid molecules must not set the scale the
        # valid ones are ranked on (see group_advantages).
        adv = group_advantages(
            torch.as_tensor(rewards, dtype=torch.float, device=device),
            valid=torch.as_tensor(terms["valid"], device=device),
            floor=invalid_floor,
        )

        # --- 3. K PPO epochs, phi held fixed throughout (Section 3.4) --------
        gen.train()
        stats = {}
        for _ in range(ppo_epochs):
            # One forward, two quantities. `log_probs` would recompute these
            # same logits, and entropy needs the whole distribution rather than
            # the chosen action's log-prob. Gathering from log_softmax here is
            # bit-identical to what `log_probs` returns in both families.
            lg = gen.logits(roll["actions"], roll["z"])
            logp_all = torch.log_softmax(lg, dim=-1)
            logp = logp_all.gather(-1, roll["actions"].unsqueeze(-1)).squeeze(-1)

            # Illegal actions sit at -inf, so p is 0 and p*logp is 0 * -inf =
            # nan. `torch.where` fixes the FORWARD value and still produces nan
            # GRADIENTS, because it backprops through both branches and
            # d/dx[exp(x)*x] at x=-inf is 0 * -inf = nan, which the zero branch
            # then multiplies to 0 * nan = nan. Measured: identical forward,
            # grad [nan,nan,nan] against [0.1175,-0.1175,0.0]. One nan step
            # poisons every parameter and the NEXT multinomial raises
            # "probability tensor contains inf, nan or element < 0".
            #
            # masked_fill's gradient is exactly zero at the filled positions, so
            # the -inf never enters the backward pass at all. This is the same
            # trap grpo.clipped_loss documents for the ratio: mask BEFORE the
            # nonlinearity, because masking the product afterwards cannot
            # recover it.
            def _entropy():
                lpa = logp_all
                return -(lpa.exp() * lpa.masked_fill(torch.isinf(lpa), 0.0)).sum(-1)

            # Build the entropy graph ONLY when the bonus is on. OpenMP changes
            # the order of float reductions and this loop is chaotic, so extra
            # autograd nodes shift allocation and drift the run: measured at
            # 6.0e-05 on `kl` over 10 steps, which is small but makes an
            # ent_coef=0 run no longer bit-comparable to the committed ones.
            # Under no_grad the diagnostic is free and identity holds.
            if ent_coef:
                ent = _entropy()
            else:
                with torch.no_grad():
                    ent = _entropy()
            with torch.no_grad():
                logp_ref = ref.log_probs(roll["actions"], roll["z"])
            loss, stats = clipped_loss(
                logp, roll["logp_old"], adv, mask=roll["mask"],
                clip_eps=clip_eps, logp_ref=logp_ref, kl_coef=kl_coef,
            )
            m = roll["mask"].float()
            ent_mean = (ent * m).sum() / m.sum().clamp(min=1.0)
            stats["entropy"] = ent_mean.item()
            # Entropy acts on PER-STEP peakedness; KL anchors to pi_ref, which
            # is a different quantity. With 5.5 decisions against SELFIES' 38.3,
            # the same per-step entropy yields far lower molecule-level entropy
            # -- diversity compounds over decisions and there are 7x fewer of
            # them. And if the warm start is itself peaked, no kl_coef recovers
            # diversity pi_ref never had. Default 0.0, so every committed run is
            # byte-identical.
            if ent_coef:
                loss = loss - ent_coef * ent_mean
            opt_g.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt_g.step()

        # --- 4. n_D discriminator steps, strictly outside the PPO window -----
        d_loss = float("nan")
        for _ in range(n_d):
            real = [real_pool[i] for i in rng.choice(len(real_pool), group_size)]
            d_loss = _d_step(disc, opt_d, real, roll["smiles"], device)

        # `summarize` measures validity with its own RDKit parse, while the
        # curriculum gate and the hurdle advantage use reward.score_terms'
        # Valid(). They should agree, but they are separate code paths and the
        # logged column must not silently become whichever one merged last --
        # so keep both, and let a divergence be visible rather than hidden.
        chem = summarize(roll["smiles"], ref_stats, group_size=group_size)
        chem["rdkit_valid_frac"] = chem.pop("valid_frac")

        row = {
            "step": k,
            "reward_mean": float(rewards.mean()),
            "valid_frac": valid_history[-1],
            "c_mean": float(terms["c"][terms["valid"]].mean()) if terms["valid"].any() else 0.0,
            "qed_mean": float(terms["qed"][terms["valid"]].mean()) if terms["valid"].any() else 0.0,
            # Logged as PREDICTED TOXICITY (1 - t), because that is the
            # quantity to watch rise; the reward carries its complement.
            "tox_mean": float(1.0 - terms["t"][terms["valid"]].mean()) if terms["valid"].any() else 0.0,
            # Group dilution: the size gate removes candidates, and a group
            # left with fewer than two valid ones carries no ranking signal
            # at all (group_advantages zeroes it). Logged separately from
            # sanitization failures so a starved group is diagnosable rather
            # than just quiet.
            "undersized_frac": float(terms["undersized"].mean()),
            # Mean distinct reactive-group alerts among valid molecules --
            # the quantity this term exists to drive down.
            "alerts_mean": float(-np.log(terms["a"][terms["valid"]]).mean()
                                 / alert_lambda) if alert_lambda and terms["valid"].any() else 0.0,
            "n_rankable": int(terms["valid"].sum()),
            # Section 3.3 diagnostic: as D sharpens, the variance of its term
            # drifts relative to the fixed-variance C term, changing their
            # effective weight on A_i even with w_D, w_C literally constant.
            "d_var": float(d_scores.var()),
            "d_loss": d_loss,
            "w_C": w["C"],
            "unique": len({s for s in roll["smiles"] if s}) / group_size,
            # Duplicates are not only a diversity metric: they shrink the set
            # group_advantages actually standardizes over. It centres and scales
            # across n_rankable rows, but repeated rows carry the same reward,
            # so the spread comes from n_eff_group distinct candidates, not from
            # group_size. At unique_smiles_frac 0.61-0.75 a group of 64 carries
            # only ~40-48 distinct ones.
            "n_eff_group": len({s for s, v in zip(roll["smiles"], terms["valid"]) if v and s}),
            "eff_group_frac": (
                len({s for s, v in zip(roll["smiles"], terms["valid"]) if v and s})
                / max(int(terms["valid"].sum()), 1)
            ),
            # Chemistry diagnostics. `unique` above is retained as the
            # contrast case: it is the metric that reported a healthy 40-step
            # run while the property spread had already collapsed, so it is
            # logged alongside the structural metrics rather than instead of
            # them. See src/diversity.py.
            **chem,
            **stats,
        }
        history.append(row)

        # Dump the raw group every step so any later analysis -- a metric not
        # thought of yet included -- can be run offline without retraining.
        if log:
            (samples_dir / f"step{k:04d}.json").write_text(
                json.dumps([s for s in roll["smiles"] if s])
            )
            # Rewrite history each step rather than once at the end: a run
            # that dies at step 140 should still yield 140 usable rows.
            with open(out_dir / "history.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)

        if log and (k % 10 == 0 or k == steps - 1):
            print(f"step {k:>4}  R={row['reward_mean']:+.3f}  valid={row['valid_frac']:.0%}  "
                  f"C={row['c_mean']:.3f}  scaf={row['scaffold_frac']:.2f}  "
                  f"tan={row['tanimoto_dist']:.3f}  "
                  f"tpsa={row['tpsa_drift_sd']:+.2f}sd/{row['tpsa_spread_ratio']:.2f}x  "
                  f"kl={row.get('kl', 0):.4f}")

    if log:
        gen.save(out_dir / f"{dataset}_grpo.pt")
        torch.save(disc.state_dict(), out_dir / f"{dataset}_disc.pt")
        # history.csv is already current -- it is rewritten every step above.
        # Pin the run's configuration and the reference statistics next to it,
        # so a later comparison between runs cannot silently compare against a
        # different baseline.
        (out_dir / "config.json").write_text(json.dumps({
            # Which classifier produced this run's reward. Without it a run
            # trained against a re-featurized classifier is indistinguishable
            # from one trained against the old one.
            "feature_id": FEATURE_ID,
            "dataset": dataset, "steps": steps, "group_size": group_size,
            "ppo_epochs": ppo_epochs, "n_d": n_d, "lr_g": lr_g, "lr_d": lr_d,
            "clip_eps": clip_eps, "kl_coef": kl_coef,
            "invalid_floor": invalid_floor, "schedule": schedule,
            "k_anneal": k_anneal, "max_len": max_len, "seed": seed,
            "c_transform": c_transform, "w_tox": w_tox,
            "min_heavy_atoms": min_heavy_atoms, "min_mw": min_mw,
            "aggregate": aggregate, "alert_lambda": alert_lambda,
            "w_alert": w_alert,
            "c_transform": c_transform, "drop": list(drop),
            "warm_start": str(warm),
            "generator": generator,
            "reward_seed": reward_seed,
            "reward_family": reward_family,
            "min_c": min_c,
            "ent_coef": ent_coef,
            # Thread count is part of the configuration, not of the machine.
            # OpenMP changes the order of float reductions, and this training
            # loop is chaotic: a run that reproduces to 2e-16 at a matching
            # thread count diverges to ~3e-03 in 20 steps at a different one,
            # which is larger than the effects the ablation measures. Without
            # this field a cross-thread comparison is indistinguishable from a
            # valid one, so --check-baseline can only report THAT a run
            # diverged, never why. torch.get_num_threads() is the value that
            # actually bound; the env vars are recorded because they are what
            # a caller sets to reproduce it.
            "threads_requested": threads,
            "torch_num_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "reference_stats": ref_stats,
        }, indent=1))
        sample = [s for s in gen.sample(200, device=device)["smiles"]
                  if s and Chem.MolFromSmiles(s) is not None]
        (out_dir / "samples.json").write_text(json.dumps(sample, indent=1))
        print(f"\nwrote {out_dir}  ({len(sample)}/200 valid samples)")

    return {"history": history, "generator": gen, "discriminator": disc}


def _self_check() -> None:
    """End-to-end on a toy generator, with a stub classifier -- no checkpoints.

    The point is the wiring and the ordering constraints, not the chemistry:
    that the reward reaches the advantage, that the advantage moves theta, and
    that phi cannot move during the PPO window.
    """
    torch.manual_seed(0)
    from .generator import build_vocab

    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1", "CCN", "C1CCCCC1", "CCOCC"]
    gen = SelfiesGenerator(build_vocab(smiles), emb=16, hidden=32, z_dim=8)
    disc = Discriminator(hidden=16, num_layers=2)

    # D must be batch-invariant, or the reward is a function of the group.
    alone = disc.score(["CCO"])
    crowded = disc.score(["CCO", "c1ccccc1", "CCN"])
    assert np.allclose(alone[0], crowded[0], atol=1e-6), \
        f"D(m) moved with group composition ({alone[0]} vs {crowded[0]})"

    # Unsanitizable and empty strings score 0 rather than crashing featurization.
    assert disc.score(["", "not-a-molecule", "CCO"]).tolist()[:2] == [0.0, 0.0]

    # An empty fake side must skip the step, not train D on reals alone.
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    before = [p.clone() for p in disc.parameters()]
    assert np.isnan(_d_step(disc, opt_d, smiles, ["", ""], "cpu"))
    assert all(torch.equal(a, b) for a, b in zip(before, disc.parameters())), \
        "D moved on a skipped step"
    assert not np.isnan(_d_step(disc, opt_d, smiles, smiles, "cpu"))

    # One full GRPO step on a stub reward: theta must move, phi must not.
    roll = gen.sample(8, max_len=10)
    terms = score_terms(roll["smiles"], lambda mols: np.full(len(mols), 0.5))
    rewards = assemble(terms, disc.score(roll["smiles"]), W0)
    assert (rewards[~terms["valid"]] == 0).all(), "invalid molecules collected reward"

    adv = group_advantages(
        torch.as_tensor(rewards, dtype=torch.float),
        valid=torch.as_tensor(terms["valid"]),
    )
    # The zero spike must not set the scale the valid molecules are ranked on.
    if terms["valid"].sum() >= 2 and not terms["valid"].all():
        v = torch.as_tensor(terms["valid"])
        assert abs(adv[v].std().item() - 1.0) < 1e-3, \
            f"invalid molecules leaked into the valid subgroup's scale ({adv[v].std():.3f})"
        assert (adv[~v] == -2.0).all(), "invalid molecules did not take the floor"
    phi_before = [p.clone() for p in disc.parameters()]
    theta_before = gen.out.weight.clone()
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-2)
    loss, stats = clipped_loss(
        gen.log_probs(roll["actions"], roll["z"]), roll["logp_old"], adv,
        mask=roll["mask"], logp_ref=roll["logp_old"], kl_coef=0.02,
    )
    opt_g.zero_grad()
    loss.backward()
    opt_g.step()

    assert all(torch.equal(a, b) for a, b in zip(phi_before, disc.parameters())), \
        "phi moved inside the PPO window -- advantages are stale by the end of it"
    if adv.abs().sum() > 0:
        assert not torch.equal(theta_before, gen.out.weight), "theta did not move"
    assert abs(stats["ratio_mean"] - 1.0) < 1e-4, \
        f"ratio != 1 on the first epoch ({stats['ratio_mean']}) -- logp_old is not pi_theta_old"

    print(f"valid={terms['valid'].mean():.0%}  R={rewards.mean():+.3f}  "
          f"adv_std={stats['adv_std']:.3f}  ratio={stats['ratio_mean']:.4f}")
    # The entropy bonus must produce FINITE gradients through a legality-masked
    # distribution. The first version did not: identical forward value, nan
    # gradients, and the failure surfaced one step later as a multinomial error
    # far from its cause. Asserted on the real masked generator, not a toy
    # tensor, because the -inf columns only exist once a mask is applied.
    from .brics_assembly import ActionTable, build_specs
    from .brics import vocab as _fvocab
    from .brics_assembly import corpus as _corpus
    _stoi = _fvocab(_corpus(["CCO", "c1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O",
                             "CN(C)CCC=C1c2ccccc2CCc2ccccc21"]), min_count=1)
    _bg = BricsGenerator(_stoi, emb=16, hidden=32, z_dim=8)
    _roll = _bg.sample(8, max_len=12)
    _lpa = torch.log_softmax(_bg.logits(_roll["actions"], _roll["z"]), dim=-1)
    assert torch.isinf(_lpa).any(), "no masked actions, so this proves nothing"
    _ent = -(_lpa.exp() * _lpa.masked_fill(torch.isinf(_lpa), 0.0)).sum(-1)
    _m = _roll["mask"].float()
    ((_ent * _m).sum() / _m.sum()).backward()
    _g = _bg.out.weight.grad
    assert _g is not None and torch.isfinite(_g).all(), "entropy bonus leaks nan gradients"
    print(f"entropy bonus: finite grads through {int(torch.isinf(_lpa).sum())} masked logits")

    print("train_gan self-check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="0 runs the self-check")
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--n-d", type=int, default=2)
    ap.add_argument("--schedule", default="linear",
                    choices=["fixed", "linear", "cosine", "gated"])
    ap.add_argument("--k-anneal", type=int, default=100)
    ap.add_argument("--kl-coef", type=float, default=0.3)
    ap.add_argument("--invalid-floor", type=float, default=-2.0,
                    help="advantage handed to unsanitizable molecules")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alert-lambda", type=float, default=0.0,
                    help="decay rate for exp(-lambda*n_alerts); 0 disables")
    ap.add_argument("--w-alert", type=float, default=0.0,
                    help="weight on the alert term (see src/alerts.py)")
    ap.add_argument("--min-heavy-atoms", type=int, default=10,
                    help="gate out molecules below this heavy-atom count")
    ap.add_argument("--min-mw", type=float, default=0.0)
    ap.add_argument("--aggregate", default="linear",
                    choices=["linear", "geometric", "nsga2", "composite"])
    ap.add_argument("--w-tox", type=float, default=0.0,
                    help="weight on the non-toxicity term (needs src.tox)")
    ap.add_argument("--c-transform", default="raw", choices=["raw", "logit"],
                    help="rescale the permeability term; see reward.transform_c")
    ap.add_argument("--drop", default="",
                    help="comma-separated reward terms to zero, from D,C,Q,S")
    ap.add_argument("--warm-start", default=None,
                    help="pretrained decoder seeding theta_0 and pi_ref")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to results/gan/seed<seed>")
    ap.add_argument("--threads", type=int, default=None,
                    help="pin torch thread count; required for any run that "
                         "will be compared against another (see findings 5)")
    ap.add_argument("--generator", default="selfies", choices=sorted(GENERATORS),
                    help="brics assembles fragments and is valid by "
                         "construction; selfies decodes tokens")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="entropy bonus on the policy. Acts on per-step "
                         "peakedness, which is the axis KL does not reach")
    ap.add_argument("--min-c", type=float, default=0.0,
                    help="gate out molecules with C below this instead of "
                         "rewarding C. Use with --drop C. 0.615 admits 95%% of "
                         "real BBB+ molecules under the ensemble reward")
    ap.add_argument("--reward-family", default="ensemble",
                    choices=["ensemble", "hybrid_batch"],
                    help="which classifier provides C. ensemble is "
                         "gin/gat/gine GraphNorm (the default, and what every "
                         "committed result used); hybrid_batch is the reported "
                         "classifier with descriptors and BatchNorm")
    ap.add_argument("--reward-seed", type=int, default=0,
                    help="scaffold-split seed of the frozen reward ensemble. A "
                         "robustness axis: vary it between runs, never within "
                         "an ensemble, and never choose it from a test table")
    args = ap.parse_args()

    if args.steps:
        out = Path(args.out_dir) if args.out_dir else OUT_DIR / f"seed{args.seed}"
        train(dataset=args.dataset, steps=args.steps, group_size=args.group_size,
              ppo_epochs=args.ppo_epochs, n_d=args.n_d, schedule=args.schedule,
              k_anneal=args.k_anneal, kl_coef=args.kl_coef,
              invalid_floor=args.invalid_floor, device=args.device,
              seed=args.seed, out_dir=out, c_transform=args.c_transform,
              w_tox=args.w_tox, min_heavy_atoms=args.min_heavy_atoms,
              min_mw=args.min_mw, aggregate=args.aggregate,
              alert_lambda=args.alert_lambda, w_alert=args.w_alert,
              drop=tuple(t for t in args.drop.split(",") if t),
              warm_start=args.warm_start, threads=args.threads,
              generator=args.generator, reward_seed=args.reward_seed,
              reward_family=args.reward_family, min_c=args.min_c,
              ent_coef=args.ent_coef)
    else:
        _self_check()

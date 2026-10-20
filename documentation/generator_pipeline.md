# Generator pipeline: what to run

The settled de-novo generation pipeline, as a runnable thing. What was *learned*
is in `brics_generator_findings.md`; this document is what to *run* and what
each number is worth. You should not need the findings doc to use this one.

**Nothing this pipeline emits is a synthesis candidate.** Every molecule is an
unvalidated hypothesis. That does not change because the pipeline is finished —
see "What the output is not".

---

## 1. The settled configuration

```bash
# 1. gate: the contract must hold before anything else
python -m src.featurize        # FEATURE_ID must print c0c3bbb8
python -m src.brics
python -m src.reward           # refuses on a feature_id or fold mismatch
python -m src.grpo

# 2. engine self-checks
python -m src.brics_assembly   # reconstruction must be 2039/2039
python -m src.brics_generator

# 3. MLE warm start (~9 min CPU, 30 epochs). NOT optional.
python -m src.brics_generator --pretrain --dataset bbbp --seed 0

# 4. GRPO at the settled point (~6 min CPU at 250 steps)
python -m src.train_gan --generator brics --steps 250 --threads 6 \
    --seed 0 --reward-seed 0 --kl-coef 1.0 \
    --warm-start results/generator/bbbp_brics_pretrained.pt \
    --out-dir results/brics_kl1/rs0_ws0

# 5. screen and rank (after the policy has settled; 250-step runs -> 150)
python -m src.postfilter results/brics_kl1/rs0_ws0 results/brics_kl1/rs0_ws1 \
    --from-step 150 --out results/candidates/brics
```

Everything else is a default. `group_size=64`, `min_heavy_atoms=10` and
`ppo_epochs=4` are unchanged from the validated SELFIES configuration.

**`--kl-coef 1.0`, not the `0.3` default.** This is the one setting that must be
passed explicitly and the single most important line in this document.
`kl_coef=0.3` is the repo default because it survived validation *for the SELFIES
generator*; it does not transfer. At 0.3 the BRICS generator loses most of its
structural diversity (scaffold 0.643, 26% of each group literal duplicates). At
1.0 it does not. The coefficient is not commensurable across the two action
spaces — the same number produces 10.7x the KL for BRICS, because the clipped
loss is a token-level masked mean and BRICS takes ~5.1 decisions over a 6051-way
softmax against SELFIES' ~38.6 over 77.

**`--threads` is configuration, not tuning.** OpenMP changes float reduction
order and this loop is chaotic, so two runs compared against each other must
pass the same value. It is recorded in each run's `config.json`.

**`--reward-seed` is a robustness axis, not a knob.** It selects which
scaffold-split fold the frozen `gin`/`gat`/`gine` ensemble comes from. All ten
seeds are legal ensembles with element-wise identical folds. Do not pick one by
looking at a test table — 0 is right because it was fixed before anyone looked.
Vary it *between* runs to show a result is not one fold's artefact; never mix
seeds within an ensemble, since the seed reseeds the split and members from
different seeds have trained on each other's held-out molecules.

## 2. What each component is worth

| component | measured | n |
|---|---|---|
| assembly reconstruction | **2039/2039 exact** | full BBBP |
| sanitization, trained policy | 0.9997 +/- 0.0001 | 3 GRPO seeds |
| sanitization, untrained policy | 0.9539 | 2560 samples |
| scaffold diversity at the settled point | **0.8643 +/- 0.0117 SEM** (sd 0.0202) | 3 GRPO seeds |
| exact-duplicate freedom | 0.9821 +/- 0.0030 SEM | 3 GRPO seeds |
| classifier term `C` | 0.8824 +/- 0.0056 SEM | 3 GRPO seeds |
| total reward | 2.1517 +/- 0.0123 SEM | 3 GRPO seeds |
| instrument gap `C(BBBP) - C(B3DB)` | **+0.0381 +/- 0.0155** vs real drugs' +0.0373; SELFIES +0.0551, Welch p = 0.003 (0.010 Bonferroni over 3 hypotheses); per-sample ranges overlap | 12 |
| screened candidate set | **1746 survivors / 5888 pooled (30%)** | 2 runs |

The three-seed figures are the headline. `scaffold 0.864` was first measured as a
single draw and reproduced at 0.864 as a three-seed mean, so it is not a
one-rollout artefact.

**The slot must be part of the action, and stereo must be stripped.** Both are
engine invariants, not tuning: choosing the lowest-index compatible dummy drops
reconstruction to 63.5%, and keeping stereochemistry drops it to 81.0% (all 387
residual failures stereo-only, because `reassemble` appends the new bond after
the dummy is removed and the anchor's parity flips). `brics_assembly`'s
self-check asserts exactness, so a regression in either is loud.

## 3. Reproducing from a clean clone

> **This section reproduces the generation half only, and it assumes a trained
> classifier.** The commands below go straight to `python -m src.reward`, which
> works solely because the three ensemble checkpoints are committed — on a clone
> with an untrained or retrained classifier it builds a reward from whatever is
> at those paths. For the whole pipeline with that seam closed, and for the
> ten-policy-seed configuration that supersedes the three-seed figures above,
> see `documentation/full_pipeline.md` or run `python -m src.pipeline --list`.

```bash
git clone <repo> && cd bbb-gnn-permeability
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # requirements.lock pins exact versions
python -m src.reward                      # the gate. If it fails, STOP.
python -m src.brics_generator --pretrain  # ~9 min
python -m src.train_gan --generator brics --steps 250 --threads 6 \
    --seed 0 --reward-seed 0 --kl-coef 1.0 \
    --warm-start results/generator/bbbp_brics_pretrained.pt \
    --out-dir results/brics_kl1/rs0_ws0
```

The classifier checkpoints the reward depends on are committed, so nothing in the
scoring half needs retraining. **Run checkpoints are deliberately not committed**
(~150 MB): they are deterministic given the committed warm start, `--seed` and
the thread count recorded in each `config.json`. `history.csv`, `config.json` and
`samples/` *are* committed, and `samples/` is what `postfilter` consumes, so the
screen reproduces without re-running GRPO.

Runs are CPU-only by design; these graphs are too small for accelerator launch
overhead to pay off. ~1.46 s/step at group 64, so 250 steps is ~6 minutes.

## 4. The candidate set

`results/candidates/brics/` holds `screened.csv` (everything, with per-molecule
rejection reasons) and `candidates.csv` (the top 25 by `C`).

```
pooled, deduplicated : 5888
undersized           :  179
structural defect    :  122
targeted alert       : 1988
catalogue alert      : 2819
already in training  : 1419   (exact-SMILES only: 1009)
SURVIVORS            : 1746 (30%)
C among survivors    : mean 0.858  sd 0.175  43% above 0.95
```

**Ranking by `C` is meaningful here, and that had to be checked.** A ranking is
noise when `C` has stopped discriminating — in the collapsed baseline its
within-group sd fell to 0.006 with everything above 0.95, below the model's own
MC-dropout uncertainty. Among these survivors the sd is **0.175**, so the
ordering carries signal.

**The ranking is one fold's opinion, and this is where reward seed actually
bites.** `C(aspirin)` spans **0.2408 (seed 8) to 0.7917 (seed 5)** across the ten
available seeds — verified on all ten, not sampled. This set was screened and
ranked at **reward seed 0**, which is recorded because "ranked survivors"
otherwise reads as fold-independent.

The scope of that concern is narrower than the raw spread suggests, and the
distinction decides where it matters:

| | measured |
|---|---|
| mean `C` per seed, one group of 64 | 0.690 - 0.796 |
| rank agreement across seeds (Spearman) | rho = 0.82 - 0.89, 45 pairs |
| after within-group z-scoring | r = 0.86 - 0.93 |
| seed agreement r, real BBBP | 0.922 +/- 0.020 (15 draws) |
| seed agreement r, generated | 0.872 +/- 0.091 (12 samples) |
| p95 \|z difference\|, real / generated | 0.873 / 1.122 |

**`group_advantages` standardizes within the group, so the absolute level is
divided out before it reaches the policy.** The seeds largely agree on *ranking*,
and post-standardization — which is what GRPO actually sees — they agree at
r ~ 0.9. Training is therefore considerably more robust to reward seed than the
0.24-0.79 spread implies, and the rs0/rs1 agreement in the 2x2 is expected
rather than lucky.

`postfilter` is the opposite case: it ranks by **absolute** `C` across a pool
with no standardization, so seed choice moves the ordering directly. That is why
the seed is recorded with the candidate set rather than merely noted.

Not zero for training either, and it is **worse on the molecules being
optimized than on real ones**. At matched n=64 over all 45 seed-pairs, the ten
seeds agree at r = 0.922 +/- 0.020 on real BBBP molecules against
r = 0.872 +/- 0.091 on generated ones (15 draws and 12 samples respectively --
single-sample figures on either side are not meaningful, see below).
The classifiers only ever saw real molecules, so the population the generator
drives toward is plausibly the population its reward is least reliable on.

**The warm-start-to-trained comparison establishes nothing, in either the mean
or the variance.** Two successive versions of this section claimed otherwise and
both are retracted; the sequence is kept because the failure is instructive.

Over 12 warm-start samples and 12 trained samples at n=64, 45 seed-pairs:

| | warm start | trained | verdict |
|---|---|---|---|
| mean r | 0.916 +/- 0.021 | 0.872 +/- 0.091 | 0.67 sd, 73% of pairings — **not established** |
| variance | sd 0.021 | sd 0.091 | **not established** (below) |

The variance claim looked strong and was not. The F-test gives F = 18.6,
p = 3e-05, but **the whole effect is one sample**: dropping `rs0_ws0` step 200
(r = 0.598) collapses it to F = 2.3, p = 0.19, while dropping any other point
leaves F ~ 20.5. With every point retained, Levene's test median-centred gives
**p = 0.19**. The F-test assumes normality and `r` is bounded on [-1, 1] and
piled up near 0.9 — exactly the shape that inflates it. Step 200 is not a
transient either: across the three runs its r is 0.598, 0.887 and 0.965, the
full range.

**Two method rules, both earned here:**

1. **Use Levene (median-centred) for a variance comparison, never the F-test**,
   on any statistic that is bounded or piled up. The F-test's p-value was wrong
   by four orders of magnitude on this data.
2. **Leave-one-out before quoting any variance ratio or any max.** Both are
   one-point statistics. A ratio that moves from 18.6 to 2.3 on removing one of
   twelve observations is describing that observation, not the populations.

The irony is the point: the variance result was offered as the explanation for
why single draws mislead, and was itself a single-draw artefact. The failure
mode survived being named, by the people who had just named it.

**This is directionally supported, NOT established, and must not be quoted with
a confidence interval that excludes 1.** Getting here took three wrong turns
worth recording, because the error is easy to repeat.

Varying *both* pools — 12 generated samples (3 runs x steps 200/220/240/249)
against 15 random BBBP draws, n=64, all 45 seed-pairs:

| statistic | real | generated | direction holds |
|---|---|---|---|
| p95 \|z difference\| | 0.873 +/- 0.124 | 1.122 +/- 0.341 | 149/180 = **83%** |
| post-standardization r | 0.922 +/- 0.020 | 0.872 +/- 0.091 | 142/180 = **79%** |

The two-sided p95 ratio is **1.31x with a 95% range of 0.63 - 2.48**, which
spans 1. One generated sample (`rs1_ws0` step 200, p95 0.597) sits *below* the
real-pool mean and reverses the direction outright.

**The generated side is the noisy half — 2.8x the sd of the real side on p95,
4.5x on r** — which is the opposite of what both halves of this project assumed.
An earlier version of this section reported 30/30 unanimity and a 1.03 - 1.89
interval; that came from resampling the real pool 30 times while holding **one**
generated sample fixed, and that sample happened to sit above the generated mean.
Fixing the noisier axis manufactures the unanimity.

The lesson generalizes past this number: **resample every axis you are not
deliberately conditioning on**, and do not identify the noisy axis by eye from
two draws — measure it. The point estimate survived all three versions (1.39x ->
1.37x -> 1.31x); only the interval moved, and the interval was the part being
quoted.

**Quote the 95th percentile with its n and its pair count; do not compare pools
on the max.** `max |z difference|` is an extreme-value statistic that grows with
both the pool size and the number of seed-pairs it is taken over — 2.9 at n=64
over 45 pairs, 4.0 at n=200 — so two measurements called "max" are not
comparable unless both state those. Worse, it is not even directionally stable
here: on one draw the real pool's max exceeded the generated pool's at 3 pairs
(2.01 vs 1.69) and the order reversed at 45 pairs (2.49 vs 2.89). The p95 is
consistent in direction at both pair counts; the max is not.

**The training-overlap check is constitution-level, and was not originally.**
`postfilter`'s exact-SMILES test is stereo-sensitive, and this generator emits
flat molecules, so training molecules with defined centres never matched: on this
pool the exact test found 1009 and the constitution test 1486, leaving 171
training molecules inside the reported survivors. `in_training_flat` now governs
exclusion and `in_training_set` keeps the old number visible. Survivors fell
1891 -> 1746 as a result.

**Novelty is weaker than the survivor count suggests, but less so than an
earlier version of this section said.** Scaffold overlap with the training
corpus:

| set | n | shares a training Murcko scaffold |
|---|---|---|
| **all survivors** | 1717 | **36.8%** |
| top 25 by `C` | 25 | 52.0% |

An earlier version quoted the 52% figure (13 of 25) as the headline. That is the
ranked display set, not the population — see the ranking caveat below. The
population figure is 36.8%, so roughly **two thirds of survivors carry a
scaffold not in training**. Constitutional novelty is screened; scaffold novelty
is reported, not enforced.

The top-ranked survivors are recognizable CNS pharmacophores — phenothiazines,
dibenzazepines, benzhydryl-piperazines — which is what recombining BRICS
fragments of CNS drugs should produce.

**The top 25 are the greasy extreme of the survivor set, and that is a property
of the ranking rather than of the generator.** Measured against real BBB+ drugs
on CNS-MPO (the Wager desirability set over logP / TPSA / MW / HBD, in
`cns_mpo_reward.py`):

| set | n | CNS-MPO | vs real |
|---|---|---|---|
| real BBB+ drugs | 600 | 0.753 +/- 0.007 | — |
| **all 1746 survivors** | 1746 | **0.752 +/- 0.005** | AUC 0.495, p = 0.87 |
| top 25 by `C` | 25 | 0.598 +/- 0.033 | AUC 0.754, p = 9e-05 |

**The survivor set as a whole is indistinguishable from real drugs on this
axis.** The top 25 are not: logP 4.44 against the survivors' 2.95, TPSA 22.1
against 47.1, MW 369 against 308, HBD 0.32 against 0.82 (all p < 1e-3). A random
25 survivors score 0.752 with a 95% interval of [0.680, 0.818], so 0.598 is
outside it.

Ranking by **absolute `C`** selects the greasy extreme. The relationship is
invisible globally — Spearman(C, CNS-MPO) = −0.032, p = 0.18 across all
survivors — because it is confined to roughly the top 1% that the ranking picks.

The same selection explains two other figures that should not be read off those
25 rows:

| | top 25 by `C` | all survivors | real BBB+ drugs |
|---|---|---|---|
| mean `C` | 0.995 | 0.851 | 0.903 |
| above 0.99 | 100% | 4.8% | 15.5% |
| shares a training scaffold | 52.0% | 36.8% | — |

**The survivor set is LESS `C`-saturated than real drugs** (0.851 vs 0.903, 4.8%
vs 15.5% above 0.99). The top 25 sit at `C` ~ 1.0 *because they are the top 25
by `C`* — quoting that as a saturation signature is circular, and an earlier
version of this document came close to doing so.

Consequence for anyone reading `candidates.csv`: **those 25 rows are not
representative of the 1746**, on greasiness, on saturation, or on scaffold
novelty. If the intended use is a property-balanced shortlist, rank on something
other than absolute `C`, or report CNS-MPO alongside it. No reward-side change is
implied — the generator's output is already real-drug-like on these four
properties and less saturated than real drugs on `C`.

## 5. What the output is not

- **Not synthesis candidates.** Every survivor is a hypothesis for evaluation.
- **Not safe.** 39% of the real BBB+ drugs in BBBP fail the same screening union,
  so passing it is not evidence of safety. The screen removes recognized
  liabilities and says nothing about liabilities no catalogue encodes.
- **Not novel scaffolds**, necessarily — see above.
- **Not validated against measured permeability.** `C` is a classifier trained on
  BBBP, which labels a *transport* property. Nothing in the label says "drug" and
  nothing says "safe"; optimizing it once found carbon tetrachloride and freon-12
  and scored them above caffeine. The size gate and `kl_coef=1.0` contain that
  failure mode; they do not remove the objective that causes it.

## 6. Known limits of the settled point

- **Three GRPO seeds, one reward seed, one warm start** at the settled point. The
  headline is a 3-seed mean; the reward-seed and warm-start axes were measured at
  `kl=0.3` (effects −0.072 and −0.081 on final scaffold, one run per cell, so
  point estimates with zero residual df).
- **`kl_coef=1.0` was selected from a 4x3 sweep** and is the knee, not an
  optimum: 3.0 buys scaffold 0.906 for `C` 0.781. One of the four paired cells
  that confirmed it is the same configuration as the sweep cell that selected it
  (it reproduced bit-identically, which is a determinism check but not
  independence).
- **Shape claims about the frontier are weaker than the operating point.** With
  the now-measured single-run sd of 0.0202 (a two-draw difference is 0.0286):
  BRICS' frontier extending past SELFIES' is 2.49 SD; BRICS@1.0 dominating
  SELFIES@1.0 is 1.82 SD and remains provisional; the apparent SELFIES
  "turn-back" is 0.56 SD and should not be claimed at all.
- **Aromaticity is not guaranteed by construction.** Type compatibility plus an
  explicit bond order guarantees connectivity and valence; joining two
  type-compatible aromatic fragments can still produce a ring system RDKit
  cannot kekulize. Untrained that is 4.6% of samples; the warm start removes it,
  which means part of what those epochs teach is toolkit avoidance rather than
  chemistry.
- **No synthesizability claim.** BRICS assembly gives a retrosynthetic
  plausibility floor by construction; nothing here tests it.

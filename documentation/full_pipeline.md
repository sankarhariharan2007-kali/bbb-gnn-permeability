# The complete pipeline

One recipe, end to end: train the classifier, freeze it as a reward, train the
generator against it, screen the output. `src/pipeline.py` runs all of it.

This supersedes the "Reproducing" sections of `documentation/pipeline.md` (§6)
and `documentation/generator_pipeline.md` (§3), which each described one half as
if it were the whole. Those documents remain the detailed evidence for their
halves; this one is the recipe and the seam between them.

```
python -m src.pipeline --list      # what would run, what would skip
python -m src.pipeline             # everything, resuming
python -m src.pipeline --only reward
python -m src.pipeline --from generator
```

---

## 1. What this produces, and what it does not

**Produces:** a BBB-permeability classifier (BBBP 0.921 in-distribution, 0.850
on novel chemistry), and a set of generated molecules screened to ~1700–5000
survivors ranked by predicted permeability.

**Does not produce synthesis candidates.** Every survivor is a hypothesis.
`C` is a classifier trained on BBBP, which labels a *transport* property —
nothing in the label says "drug" and nothing says "safe". Optimizing it once
converged on carbon tetrachloride and freon-12 and scored them above caffeine.
The size gate and `kl_coef=1.0` contain that failure mode; they do not remove
the objective that causes it. See §9.

---

## 2. Quick start

The classifier checkpoints, the MLE warm start, all ten policy-seed runs and
the screened candidate set are committed. So on a fresh clone of this branch
every training stage skips and the pipeline only re-derives the read-only
report:

```bash
pip install -r requirements.txt     # requirements.lock pins exact versions
python -m src.pipeline --list       # confirm what would skip
python -m src.pipeline              # ~5 min, trains nothing
```

Measured: 5m 4s end to end, of which the screen is 232 s. It is reproducible —
two consecutive runs leave `screened.csv` byte-identical, and the working tree
clean.

Run `--list` first. It is read-only and tells you exactly which stages have
artifacts on disk, so you find out what a run will cost before it starts.

Stages whose artifacts already exist are skipped. Resume is by artifact on
disk, not by a state file, so an interrupted run simply continues.

## 3. Full reproduction

```bash
python -m src.pipeline --force      # ~5 h, retrains the 300-run classifier grid
```

Costs on CPU (these graphs are too small for accelerator launch overhead to pay
off). The "skipping" column is measured on this branch with every artifact
present; the "training" column is what `--force` costs.

| stage | work | skipping | training |
|---|---|---|---|
| contract | 5 self-checks, cache build, split verification | 41 s | 41 s |
| classifier | 300 runs: base 80, edge 60, hybrid 80, hybrid_batch 80 | 17 s | ~4 h |
| reward | load + verify the frozen ensemble | 10 s | 10 s |
| generator | MLE warm start, then 10 GRPO runs | <1 s | 2416 s measured for 7 |
| screen | pool 19219, screen, score, rank | 239 s | 239 s |
| report | holdout eval, aggregation, paired analysis | ~25 s | ~25 s |

`contract`, `reward` and `report` always run — the first two are gates and the
third is read-only aggregation. `screen` re-runs whenever its output is absent.

The generator figure is the honest one: 2416 s trained seeds 3–9 at ~345 s each,
against the ~6 min/run the generator doc quotes. The difference is thread
contention from other work on the machine, not a configuration change.

---

## 4. The two classifier roles

**This is the seam, and it is the one thing to get right.** The classifier is
trained twice because it has two incompatible jobs.

| role | command | result tree | BBBP | B3DB | Wang |
|---|---|---|---|---|---|
| **reported** | `train_hybrid --run-all --norm batch` | `results/hybrid_batch/` | **0.921** | 0.905 | **0.850** |
| **frozen reward** | `run_all` + `train_edge --run-all` (GraphNorm default) | `results/{bbbp,b3db}/`, `results/edge_ablation/` | 0.909 | 0.900 | 0.839 |

BatchNorm scores ~0.010 higher and is **disqualified as a reward**. In train
mode its statistics come from the batch, so a molecule's logit depends on which
other molecules share its batch — the same molecule earns a different reward
depending on its GRPO group, and the reward stops being a function of the
molecule. Measured by `python -m src.models` on a 4-graph batch in train mode:
batch drifts **3.97e-02**, graph 6.0e-08, layer 3.7e-09. `eval()` guards contain
this in practice; GraphNorm removes the failure mode instead of guarding it.

**The reward ensemble is not a hybrid model.** It is base/edge `gin`/`gat`/`gine`
at seed 0 under GraphNorm — no descriptors. That is what every committed
generator result was trained against, so swapping it changes the meaning of
every number in `results/brics_kl1/`. `reward.load_frozen_classifier` verifies
three things on every load: `feature_id` per member, element-wise fold identity
across members, and norm identity. Any mismatch raises rather than scoring.

Operator choice is free — GCN / SAGE / GAT land within 0.01 of each other. GCN
is the cheapest.

---

## 5. Stage reference

### contract
`FEATURE_ID` must print `c0c3bbb8`. It is a content hash of
`atom_features`/`bond_features` over a fixed probe set, so it moves by itself if
the featurizer changes — a constant someone must remember to bump is a constant
that will be wrong exactly once. It is recorded in every `metrics.json` and
checked on every checkpoint load. `NODE_DIM` 39, `EDGE_DIM` 7.

Then the five module self-checks run exactly as their docstrings document them
(`python -m src.featurize`, `src.split`, `src.models`, `src.brics`,
`src.datasets`), including BRICS round-tripping 2039/2039 BBBP molecules and the
split-leak assertions on both datasets.

### classifier
Four arms, 300 runs, 10 seeds each. **Each seed is a different scaffold split**,
so every `±` includes split variance, not just initialization variance.

`--norm batch` routes to `results/hybrid_batch/` via `train_hybrid.results_dir`.
That arm holds the repo's best numbers and for a long time no committed command
could regenerate it — it was produced ad hoc while `eval_external_holdout` read
it. It is now reproducible.

### reward
A gate, not a build step. There is no freeze/export: the ensemble is read from
the committed checkpoints in place. If it raises, the run stops before any
generator work. **If you retrain the classifier, nothing re-points the reward —
it picks up whatever is at those paths.**

### generator
BRICS fragment assembly, valid by construction. The settled point:

```
kl_coef         1.0     steps        250     group_size  64
min_heavy_atoms 10      threads      6       reward_seed 0
warm start      results/generator/bbbp_brics_pretrained.pt
policy seeds    0-9
```

**`kl_coef=1.0` is the single most important value here.** The repo default is
0.3, validated for the SELFIES action space, and it does not transfer: the same
coefficient yields ~10.7× the KL on this action space, and at 0.3 BRICS
collapses to scaffold 0.643 with 26% literal duplicates per group. Do not
"restore the default".

**The MLE warm start is not optional.** From a uniform policy every group is
all-invalid, the group standard deviation is zero, and `group_advantages`
correctly returns no gradient — GRPO would wait for validity to appear by
chance.

**`--threads` is configuration, not tuning.** OpenMP changes the order of float
reductions, so it is part of the run identity and is recorded in each
`config.json`. Runs being compared must share it.

**Three orthogonal axes, and run directory names do not encode them.** The
committed dirs use two incompatible conventions, and `rs0_ws0`/`rs0_ws1` have
identical `seed` and `reward_seed` — they differ only in warm start. Every axis
is read from `config.json`. New runs use one convention, `pseed<n>`.

| axis | flag | levels |
|---|---|---|
| policy seed | `--seed` | 0–9 (`rs0_ws0`, `gseed1`, `gseed2`, `pseed3`…`pseed9`) |
| reward seed | `--reward-seed` | 0, 1 (`rs1_*`) |
| warm start | `--warm-start` | `bbbp_brics_pretrained.pt`, `bbbp_brics_mle_s1.pt` |

`--reward-seed` selects which scaffold-split fold the ensemble comes from. All
ten are legal ensembles with element-wise identical folds. Do not pick one by
looking at a test table — 0 is right because it was fixed before anyone looked.
Vary it *between* runs to show a result is not one fold's artefact; never mix
seeds within an ensemble, since members from different seeds have trained on
each other's held-out molecules.

### screen
`postfilter` pools `samples/` from step ≥ 150 across the ten policy-seed runs,
screens, scores and ranks. It selects run dirs by reading `config.json`, and
deliberately excludes the `rs1_*` and `*_ws1` dirs: those are different axes,
not more policy seeds.

Output goes to `results/candidates/brics10/`. The earlier two-run screen is
preserved at `results/candidates/brics/`.

**The screen is reproducible only at a pinned thread count**, and this was a
real bug in the first version of `pipeline.py`. `run_all` calls
`torch.set_num_threads` in-process, so `screen` inherited 6 threads after a full
run and the machine default (18 on this box) when invoked with `--only screen`
— and `screened.csv`'s `c_score` column then differed in 59 of 19219 rows.
Ranking was unaffected (Spearman 1.0 to 8 decimals, identical top 25), so
nothing downstream was wrong, but a frozen reward that answers differently in
two runs is precisely the hazard GraphNorm removed from the classifier. The
pipeline now pins threads once at entry, before any stage. Two pinned runs are
byte-identical; the committed artifact is at `threads=6`.

### report
Read-only. `eval_external_holdout` → `collect_all` → `paired_analysis`.
`master_comparison.csv` now carries a `norm` column, because `hybrid` and
`hybrid_batch` differ in nothing else and a table without it invites reading a
norm effect as an architecture effect.

---

## 6. What every part is worth

Two tables, two different units. They are not comparable to each other.

### Classifier — Δ ROC-AUC, pooled over 4 operators × 2 datasets × 10 seeds, paired

| component | Δ ROC-AUC | SEM | verdict |
|---|---|---|---|
| wide atom-level readout | **+0.1304** | — | the dominant factor |
| 8 RDKit descriptors | **+0.0099** | 0.0019 | significant |
| BatchNorm over GraphNorm | **+0.0074** | 0.0023 | significant |
| descriptors + BatchNorm together | **+0.0195** | 0.0024 | significant, additive |
| BRICS motif graph, intra + atom readout | +0.0064 | 0.0059 | within noise |
| virtual node | +0.0003 | 0.0023 | within noise |
| BRICS motif graph, added on top | −0.0042 | 0.0026 | within noise |
| edge (bond) features | ~0 | — | within noise |
| long-range atom connectivity | −0.0194 | — | slightly harmful |
| HRM two-timescale recurrence | **−0.0381** | 0.0059 | significant, worse |
| BRICS motif graph replacing atoms | **−0.1161** | 0.0107 | significant, much worse |

**Descriptors and BatchNorm are the only two positive effects, and they stack:**
predicted +0.0173 if independent, measured **+0.0195 ± 0.0024**. The reported
classifier is exactly those two, and that is the whole case for it.

The one sentence that explains the table: **BBB permeability behaves like a
property of bulk composition, not of long-range topology.** Everything that adds
compositional or whole-molecule information helps; everything that adds
structural reasoning is neutral or harmful. That is why LightGBM on 8
descriptors reaches 0.880, why the four operators are statistically
indistinguishable, and why both HRM and the motif model failed.

### Generator — reward, diversity and validity means

| component | measured | n |
|---|---|---|
| assembly reconstruction | **2039/2039 exact** | full BBBP |
| sanitization, trained policy | 0.9996 ± 0.0001 SEM | 10 policy seeds |
| sanitization, untrained policy | 0.9539 | 2560 samples |
| scaffold diversity at the settled point | **0.8541 ± 0.0059 SEM** (sd 0.0186) | **10 policy seeds** |
| exact-duplicate freedom | 0.9822 ± 0.0011 SEM | 10 policy seeds |
| classifier term `C` | 0.8863 ± 0.0025 SEM | 10 policy seeds |
| total reward | 2.1548 ± 0.0051 SEM | 10 policy seeds |
| instrument gap `C(BBBP) − C(B3DB)` | **+0.0381 ± 0.0155** vs real drugs' +0.0373; SELFIES +0.0551, Welch p = 0.003; per-sample ranges overlap | 12 |

### The seed bump, and what it changed

`generator_pipeline.md` reports this point at 3 policy seeds. Running the other
seven moved the headline:

| | n=3 | n=10 |
|---|---|---|
| scaffold mean | 0.8643 | **0.8541** |
| sd | 0.0202 | 0.0186 |
| SEM | 0.0117 | **0.0059** |

**The sd estimate held, so the old spread was sound — but the mean was 0.010
high.** The three committed runs happen to include `rs0_ws0` at 0.8874, the
highest of all ten. Nothing was selected; it was a three-draw sample from a
distribution with sd 0.019, and 0.8643 sat 1.7 SEM above the ten-seed mean.

The n=3 subset still returns 0.8643/0.0202 exactly under the same statistic, so
the two figures are commensurable and the move is real rather than an artefact
of recomputing it differently.

**Report 0.854 ± 0.006.** Per-seed values span 0.830 to 0.887.

**Scaffold was the only statistic that moved.** Recomputed over the same ten
runs, everything else in the table above lands on its 3-seed value:

| | n=3 | n=10 | moved? |
|---|---|---|---|
| sanitization | 0.9997 | 0.9996 | no |
| exact-duplicate freedom | 0.9821 | 0.9822 | no |
| classifier term `C` | 0.8824 | 0.8863 | +0.004, <0.5 sd |
| total reward | 2.1517 | 2.1548 | no |
| **scaffold diversity** | **0.8643** | **0.8541** | **−0.010, 0.55 sd** |

So the three-seed sample was not broadly optimistic — it was unlucky on one
axis, the one with by far the largest relative spread (sd 0.019 against
0.0003–0.016 for the rest). That is the axis where a small n costs you, and it
is also the axis the reward does not control, which is why it is the one worth
spending seeds on.

---

## 7. Numbers, and which to report

### Report ~0.85, not ~0.92

| | BBBP (in-distribution) | Wang (n=55, novel) |
|---|---|---|
| hybrid + BatchNorm | 0.921 | 0.850 |
| edge ablation | 0.893 | **0.853** |
| base | 0.895 | 0.846 |
| hybrid + GraphNorm | 0.909 | 0.839 |

Adenot (n=59) is saturated at ~0.99 for every family and discriminates nothing.
Wang is the informative set, and **everything falls to ~0.85 there.**

**In-distribution ranking does not predict generalization.** Across 11 models
the correlation between BBBP and Wang score is r = −0.45 (p = 0.16) —
suggestive rather than established, but there is certainly no evidence *for*
transfer. Plain hybrid is 1st–3rd in-distribution and 7th–9th on Wang. On the
Wang holdout the four families span 0.014 against a per-family sd of ~0.05, so
no family is separated from another there: `hybrid_batch` is chosen on
in-distribution evidence and is merely *not contradicted* out of distribution.

BatchNorm is the exception that makes the choice defensible: its gain is
**+0.0101 in-distribution and +0.0108 on Wang** — same direction, same
magnitude. It moves the hybrid family from last to second on Wang.

**Do not rank operators on this data.** At 10 seeds, 4 of 24 within-family pairs
separate at 95%, and every surviving effect is 0.004–0.02 AUC. The 3-seed claim
that `gat − gin` was a 1.4 sd effect did not replicate (limitations 4.1).

### Two generator numbers that must not be blurred

| | what it is | n |
|---|---|---|
| **scaffold diversity** | a per-run statistic at **matched n=64**, averaged over policy seeds | 10 runs |
| **candidate yield** | the size of the pooled, screened survivor set | 1 pool |

Diversity statistics are sample-size dependent — scaffold fraction reads 0.758
at n=199 and 0.595 at n=982 on the *same* BBBP molecules, a swing larger than
any effect being measured. **Pooling ten runs for screening does not license
re-quoting the diversity figure on the larger pool.** That is bias, not
variance, and more seeds do nothing for it. Comparisons are valid only at
matched molecule count.

---

## 8. The candidate set

`results/candidates/brics10/` holds `screened.csv` (everything, with
per-molecule rejection reasons) and `candidates.csv` (the top 25 by `C`).

| | 10 runs (`brics10/`) | 2 runs (`brics/`) |
|---|---|---|
| pooled, deduplicated | 19219 | 5888 |
| undersized | 475 | 179 |
| structural defect | 460 | 122 |
| targeted alert | 6597 | 1988 |
| catalogue alert | 9864 | 2819 |
| already in training | 1747 | 1419 |
| **survivors** | **6591 (34%)** | 1746 (30%) |
| `C` among survivors | 0.854 ± 0.182 | 0.858 ± 0.175 |

**3.8× the candidate set**, at an essentially unchanged survival rate and `C`
distribution — so the extra seeds buy yield rather than changing what the
generator produces. The two-run set is preserved for the figures that cite it.

**Ranking by `C` is meaningful here, and that had to be checked.** A ranking is
noise when `C` has stopped discriminating — in the collapsed baseline its
within-group sd fell to 0.006 with everything above 0.95, below the model's own
MC-dropout uncertainty. Among these survivors the sd is 0.175, so the ordering
carries signal.

**The top 25 are not representative of the population, on any axis.** This bit
three separate claims before it was caught:

| | top 25 by `C` | all survivors | real BBB+ drugs |
|---|---|---|---|
| mean `C` | 0.995 | 0.851 | 0.903 |
| above 0.99 | 100% | 4.8% | 15.5% |
| CNS-MPO | 0.598 ± 0.033 | **0.752 ± 0.005** | 0.753 ± 0.007 |
| shares a training scaffold | 52.0% | 36.8% | — |

Two readings follow. **The survivor set as a whole is indistinguishable from
real drugs on CNS-MPO** (AUC 0.495, p = 0.87) and is **less `C`-saturated than
real drugs**. The top 25 are the greasy extreme — logP 4.44 against the
survivors' 2.95, TPSA 22.1 against 47.1 — because ranking by absolute `C`
selects for it; the relationship is invisible globally (Spearman −0.032,
p = 0.18) because it is confined to roughly the top 1%. And the top 25 sitting
at `C` ≈ 1.0 *because they are the top 25 by `C`* is circular, not a saturation
signature.

If the intended use is a property-balanced shortlist, rank on something other
than absolute `C`, or report CNS-MPO alongside it.

### Permeability as a gate: a large diversity win, and no help at all on the objective

A third arm replaces the monotone `C` term with a hurdle —
`--drop C --min-c 0.615` — on the reasoning that `assemble` is a weighted sum,
so a monotone term pays the policy to *climb* toward whatever maximizes `C`.
τ = 0.615 is the 5th percentile of real BBB+ molecules under the ensemble
reward, calibrated on the positive class and never on generated output. Ten
seeds, matched to arm A otherwise.

| | A: `C` as term | GATE |
|---|---|---|
| pooled distinct molecules | 19219 | **33227** (+73%) |
| survivors | 6591 | **11329** (+72%) |
| survival rate | 34% | 34% |
| training molecules reproduced | 1747 (88.9% of corpus) | **1345 (68.4%)** |
| novel, whole pool | 72.0% | **79.4%** |
| novel survivors | 4520 | **7857** (+74%) |
| `C` among survivors | 0.854 | 0.826 |
| survivors above `C` 0.95 | 43% | 35% |

**This is the largest single improvement measured on the generation half.** The
policy explores far more, reproduces 400 fewer of its own training molecules,
and stays permeable — survivor `C` 0.826 against BBBP's own mean of 0.745,
exactly as limitations 1.7 predicted when it measured `mean_C` at 0.786 with
`w_C = 0`.

**Two numbers that look like results and are not.** `valid_frac` falls
0.979 → 0.854, but RDKit sanitization is unchanged (0.9997 → 0.9996): the drop
is molecules gated out by `C`, which is the mechanism working. And `mean_C`
*rises* in-training (0.888 → 0.906) purely because the gate truncates `C`'s low
tail — a selection effect, not an improvement. Within-group scaffold diversity
falls 0.851 → 0.820 (p = 0.19, not significant) while pooled distinct output
rises 73%; those are the two different statistics of 4.3/4.4, not a
contradiction.

**It does nothing for the objective, and cannot.** Carbon tetrachloride scores
`C = 0.987` and passes any threshold calibrated on real BBB+ drugs, because it
genuinely is among the most permeable molecules in BBBP. A gate on `C` removes
*low*-`C` molecules. The undersized fraction barely moves (2.05% → 1.87%) and
the **size** gate remains the only thing that blocks tiny solvents. Layer 3
needs a label that separates "is a drug" from "crosses a membrane" — see §9 —
and nothing derived from a permeability classifier can supply one.

### Two rewards, two arms: the generator does not depend on its judge

The pipeline has two classifiers (§4), and the generator was originally trained
against only one. A second arm trains against the other — `--reward-family
hybrid_batch`, ten policy seeds, matched on everything else (kl=1.0, 250 steps,
group 64, same warm start, reward seed 0, same seeds), so the comparison is
paired by seed and only the judge differs.

**Paired over 10 seeds, nothing meaningful separates them:**

| metric | A − B | p |
|---|---|---|
| scaffold diversity | +0.0025 | 0.73 |
| validity | −0.0001 | 0.94 |
| uniqueness | −0.0006 | 0.65 |
| Tanimoto distance | −0.0018 | **0.007** |

Only Tanimoto separates. It survives Bonferroni over four tests and is a 0.2%
relative effect, so it is real and unimportant.

**Screening and provenance are near-identical:**

| | arm A (ensemble) | arm B (hybrid_batch) |
|---|---|---|
| pooled → survivors | 19219 → 6591 (34%) | 20082 → 6848 (34%) |
| in training data | 9.1% | 8.7% |
| exist already | 18.9% | 19.1% |
| novel | 72.0% | 72.2% |
| survivors: known / novel | 31.4% / 68.6% | 31.4% / 68.6% |

**The 2×2 that detects reward hacking.** Each cell as a fraction of that
reward's own real-BBB+ control — necessary because `hybrid_batch` is globally
more conservative (real BBB+ 0.866 vs 0.908), so an arm's raw drop when moving
rewards is mostly that offset rather than anything about the arm:

| | R = ensemble | R = hybrid_batch |
|---|---|---|
| **A**: trained on ensemble | **0.941** ← in-loop | 0.929 |
| **B**: trained on hybrid_batch | 0.931 | **0.936** ← in-loop |
| real BBB+ (raw) | 0.9081 | 0.8662 |
| real BBB− (raw) | 0.2625 | 0.1673 |

**In-loop advantage is +0.010 on one reward and +0.007 on the other.** Each arm
scores under the reward it never saw almost exactly as well as the arm trained
on it, and Spearman between the two rewards is +0.71 on both arms. If either
reward were being exploited, its off-diagonal would collapse. It does not.

Run it with `python -m src.cross_reward`.

### Scored by a classifier the generator never saw

`screened.csv`'s `c_score` comes from the gin/gat/gine GraphNorm ensemble the
generator was *optimized against*, so a high value says the policy hit its own
target — close to tautological, and blind to reward hacking by construction.
The informative test scores the same molecules with the **reported** classifier
(`hybrid_batch/gcn`, 10 seeds, descriptors the reward model never sees, a
different norm, averaged over ten splits). Nothing in GRPO consulted it.
`src/independent_score.py`.

| population | independent judge | >0.5 |
|---|---|---|
| real BBB+ (control) | 0.889 ± 0.175 | 94.7% |
| **generated survivors** | **0.825 ± 0.219** | **88.9%** |
| real BBB− (control) | 0.197 ± 0.187 | 8.8% |

The judge discriminates (0.889 vs 0.197), and generated molecules sit near real
BBB+ and nowhere near BBB−. Against the reward's own 0.854 on the same set, the
mean optimism is **+0.029**.

**Where reward hacking would show up, it doesn't.** If the policy had learned
this ensemble's blind spots, the exploits would concentrate in the top-ranked
molecules, because that is what it was pushed toward. Measured over the 250
highest `c_score` survivors: mean gap **+0.008**, and **0 molecules** with a
gap above 0.3. Across all 6591 the median gap is +0.004, with 5.55% optimistic
by >0.3 against 1.32% pessimistic by >0.3 — a roughly symmetric spread of two
models disagreeing, not a systematic exploit. Agreement is pearson +0.764,
spearman +0.713.

**What this does and does not rule out.** It rules out *ensemble-specific*
hacking: the permeability signal is not an artefact of the particular reward
members. It does **not** touch the objective problem, because both classifiers
share the same label semantics — the independent judge scores carbon
tetrachloride at **1.000**, above caffeine's 0.976. Two models trained on BBBP
agree that solvents cross membranes, because they do. Independence of
architecture and fold is not independence of the question being asked.

BatchNorm is safe as a judge and unsafe as a reward: in `.eval()` it uses
running statistics, verified batch-independent to 6e-08, so a molecule's score
does not depend on its batch. See §4.

### Training data, already known, or novel

Every one of the 19219 generated molecules resolved against PubChem's full
`CID-InChI-Key` table (124.7M compounds, scanned locally in 3.2 min — see
`src/pubchem_bulk.py` for why not the web API).

| | all generated | screened survivors |
|---|---|---|
| | **n=19219** | **n=6591** |
| in its training data | 1747 — 9.1% | 0 — by construction |
| exist already, not in training | 3636 — 18.9% | 2071 — **31.4%** |
| no record found — **novel** | 13836 — **72.0% ± 0.6%** | 4520 — **68.6% ± 1.1%** |

**Memorization is low: 9.1%.** The generator is not mostly replaying its
training set. Of what it does produce, roughly a fifth is chemistry that exists
but that it never saw, and that fraction rises to 31.4% among survivors —
screening preferentially keeps real compounds, which is a point in its favour.

Matching is on the **full** InChIKey, including the protonation layer. CIDs 1
and 2 share `RDHQFKQIGNGIED-UHFFFAOYSA` and differ only in the final character,
so skeleton-only matching would conflate protonation states. Generated
molecules are flat and neutral (`-UHFFFAOYSA-N`) and match PubChem's
stereo-unspecified parent, which is why the generated lysergamide resolves to
CID 3981 and not to LSD's stereo-defined CID 5761.

The oldest survivors it rediscovered are small, very well-known compounds:
phenylpropanoic acid (CID 107), 4-hydroxybenzoic acid (135), acetanilide (904),
phenylalanine (994), phenylacetic acid (999).

**Three limits on the novel figure.** It is an **upper bound**: neither query
normalizes tautomers, so some "novel" molecules are known in another tautomeric
form. It means absent from PubChem and the training corpus, not absent from
every database. And it is not a quality signal — the *pooled* bucket contains
`C#CC1(OOOC(O)C(Cl)(Cl)Cl)CCCCC1`, a trioxide. Some of that 72% is novel
because it is bad.

**Correction.** An earlier version of this document said that trioxide passed
the screen because no catalogue encodes it. **It did not pass.** It carries
`passes=False` with 1 targeted and 7 catalogue hits, and appears in arm A's
pool only. The screen caught it eight times over, so it is evidence about the
*pooled* set and not about survivors. On survivors the stronger statement holds:
16 motifs none of the screened catalogues encode — peroxides, trioxides, azo,
diazo, isocyanate, isothiocyanate, sulfonyl halide, carbodiimide, disulfide,
thiol, P, B, Si, Se/Te — match **0 of 11,329** (`src/cluster_survivors.py`).

Validation: an independent 200-molecule random sample predicted 7.5% / 20.0% /
72.5% before the census ran, against the census's 9.1% / 18.9% / 72.0% — inside
sampling error on all three buckets.

### What the output actually looks like

Three selections from the same 6591 survivors, which is the clearest way to see
why the display set matters:

| selection | n | C | QED | logP | TPSA | MW |
|---|---|---|---|---|---|---|
| top by `C` | 6 | 0.997 | 0.72 | **4.77** | **10.7** | 386 |
| random survivors | 6 | 0.817 | 0.69 | 1.83 | 59.4 | 286 |
| `QED > 0.8` **and** `C > 0.95` | 6 | 0.982 | **0.94** | 2.83 | 47.7 | 313 |
| **whole population** | 6591 | 0.854 | 0.68 | 3.04 | 49.9 | 323 |

The top-by-`C` rows include one at logP +7.91 with QED 0.25 — ranking on
absolute `C` alone selects for greasiness. Filtering on `QED > 0.8 ∧ C > 0.95`
costs 0.015 of `C` and returns a set that is drug-like on every axis. **That,
not `candidates.csv`'s top 25, is what a shortlist should be drawn from.**

Examples from the balanced set:

```
C=0.991 QED=0.95  NC(=O)ON1CCC(c2ccccc2)(c2ccccc2)CC1        benzhydryl carbamate
C=0.990 QED=0.94  Cc1cc(CCN2CCN(c3ccccc3Cl)CC2)n[nH]1        aryl-piperazine
C=0.986 QED=0.94  CCN(CC)C(=O)C1CC2c3cccc4[nH]cc(c34)CC2N(C)C1   dihydro-lysergamide
C=0.981 QED=0.94  CC(=O)NCCN1CCC(c2noc3cc(F)ccc23)CC1        benzisoxazole-piperidine
```

**One of them is LSD.** `CCN(CC)C(=O)C1C=C2c3cccc4[nH]cc(c34)CC2N(C)C1` is
constitutionally identical to lysergic acid diethylamide (verified by flattened
canonical SMILES), and it is **not in BBBP even after flattening** — so it is
not a screening leak. BBBP does contain other ergolines (pergolide,
cabergoline, romergoline), so the generator recombined BRICS fragments of those
relatives into a compound it never saw.

Read that for exactly what it is. It is strong evidence the policy reaches real
CNS pharmacophore space rather than scoring artefacts — and it is **not** a
novelty claim (LSD is well known) and **not** a safety signal. It is the
objective working as specified: LSD crosses the blood-brain barrier extremely
effectively, which is the only question `C` asks. A controlled substance
surfacing near the top of a permeability-optimized set is §9's unresolved
objective problem, demonstrated rather than argued.

Roughly two thirds of survivors carry a scaffold not in training.
Constitutional novelty is screened; scaffold novelty is reported, not enforced.
The top-ranked survivors are recognizable CNS pharmacophores — phenothiazines,
dibenzazepines, benzhydryl-piperazines — which is what recombining BRICS
fragments of CNS drugs should produce.

---

## 9. Known limits

**Of the output**

- **Not synthesis candidates.** Every survivor is a hypothesis.
- **Not safe.** 39% of the real BBB+ drugs in BBBP fail the same screening
  union, so passing it is not evidence of safety.
- **Not novel scaffolds**, necessarily — 36.8% share a training scaffold.
- **Not validated against measured permeability.**

**Of the objective (the unresolved one)**

BBBP labels transport, not drug-likeness. Chloroform, dichloromethane,
bromoform, nitrous oxide, cyclopropane, ethanol and vinyl ether are all in BBBP
labelled BBB+, **correctly** — anaesthetics and solvents do cross the barrier.
Of the 65 molecules with ≤8 heavy atoms, 62 are positive. So a permeability
objective is not wrong, it answers a different question than "is this a drug".
Six candidate counterweights have now been tried and none holds: Tox21 scores
CCl₄ at 0.251 against caffeine's 0.302, ClinTox reaches 0.909 test AUC but rates
CCl₄ 0.920 against haloperidol 0.004, and descriptor distance, applicability
domain, uncertainty penalties and in-loop alert penalties each failed or failed
to generalize. The size gate and `kl_coef=1.0` contain the failure mode; nothing
removes it.

**Of the protocol**

- **The split clusters acyclic molecules.** `MurckoScaffoldSmiles` returns `''`
  for ring-free molecules, so they used to collapse into one group that, being
  large, always landed in train — 95/1965 BBBP and 311/7805 B3DB molecules were
  **never in a test fold**. They are now grouped by Morgan/Tanimoto (Butina,
  0.6). This costs ~0.037 ROC-AUC on BBBP and is why the headline is lower than
  this repo's earlier figures. `split.group_keys` is the single definition of
  group identity, called by both the splitter and the leak assertion.
- **Numbers are not comparable to the 0.65–0.70 BBBP literature figures**, which
  come from the size-sorted DeepChem protocol. That protocol is degenerate here:
  BBBP's back half is all class 1, so val/test come out 100% positive and
  ROC-AUC is undefined.
- **The kl frontier is single-seed.** Every frontier number is one draw. The
  frontier spans 0.46–0.91 against a real-molecule control at 0.879, far wider
  than the single-run sd of 0.0202, so its shape holds — but individual points
  do not support comparison. `kl_coef=1.0` is the knee, not an optimum: 3.0 buys
  scaffold 0.906 for `C` 0.781.
- **The reward-seed and warm-start axes are at n=2.** `C(aspirin)` spans 0.2408
  (seed 8) to 0.7917 (seed 5) across the ten folds. The scope is narrower than
  that implies for *training* — `group_advantages` standardizes within the
  group, and post-standardization the seeds agree at r ≈ 0.9 — but `postfilter`
  ranks by **absolute** `C` with no standardization, so seed choice moves the
  candidate ordering directly. That is why the seed is recorded with the set.
- **Aromaticity is not guaranteed by construction.** Type compatibility plus an
  explicit bond order guarantees connectivity and valence; two type-compatible
  aromatic fragments can still produce a ring system RDKit cannot kekulize.
  Untrained that is 4.6% of samples. The warm start removes it, which means part
  of what those epochs teach is toolkit avoidance rather than chemistry — do not
  report the validity gain as "the policy learned valid chemistry".
- **No synthesizability claim.** BRICS assembly gives a retrosynthetic
  plausibility floor by construction; nothing here tests it.

**Two method rules, both earned the hard way**

1. **Use Levene (median-centred) for a variance comparison, never the F-test**,
   on any statistic that is bounded or piled up. The F-test's p-value was wrong
   by four orders of magnitude on this data.
2. **Leave-one-out before quoting any variance ratio or any max.** Both are
   one-point statistics. A ratio that moves from 18.6 to 2.3 on removing one of
   twelve observations is describing that observation, not the populations.

`documentation/limitations.md` holds all of this in full, with the retractions.

---

## 10. The shortlist ranking (branch `ranking-v2`)

`candidates.csv` ranked by absolute `C`, and that was the last place the
permeability objective's bias was still operative. The population is fine — the
ranking was not.

| | logP | TPSA | QED | CNS-MPO | `C` |
|---|---|---|---|---|---|
| all 11329 survivors | 3.12 | 50.1 | 0.68 | 0.714 | 0.826 |
| **top 25 by `C`** (old) | **5.15** | **18.9** | 0.64 | **0.487** | 0.997 |
| **top 25 by `mpo_gated`** (new) | 1.24 | 52.3 | **0.79** | **1.000** | 0.988 |
| real BBB+ drugs | 2.67 | 54.1 | 0.67 | 0.753 | — |

**This is a ranking defect, not a generator or reward defect.** The C-gate arm
shows the same drift (+2.02 logP) despite never optimizing `C`, so no reward
change fixes it. Only the sort does.

`mpo_gated` requires `C ≥ 0.615` — the real-BBB+ 5th percentile, the same
threshold the reward gate uses — then orders by CNS-MPO (Wager 5 of 6, from
`hybrid_features.CNSMPO`, normalized by 5 because the basic-pKa term is omitted
rather than fabricated). Permeability becomes the hurdle and drug-likeness the
objective, which is the same move the reward gate makes.

**CNS-MPO saturates too, and that had to be handled.** 950 of the 9722
survivors clearing the `C` hurdle score a perfect 1.000, so "top 25 by MPO"
would have been an arbitrary 25 of 950 — precisely the defect being fixed. The
sort is `(MPO, C)` descending, breaking the tie on permeability *within* ideal
drug-space. The result keeps `C` at 0.988 while moving MPO from 0.487 to 1.000.

The new shortlists are recognizable medicinal chemistry — pyrimidinyl-
piperazines, trimethoxybenzamides, morpholino-propanones — rather than the
greasy extreme.

```bash
python -m src.postfilter results/brics_cgate/* --from-step 150 \
    --rank-by mpo_gated --out results/candidates_mpo/brics_cgate
```

`--rank-by` defaults to `c_score`, so every existing candidate file reproduces
byte-identically; the re-ranked sets live in `results/candidates_mpo/` and the
originals under `results/candidates/` are untouched.

**One caveat.** CNS-MPO is a desirability composite, not a measurement. Ranking
by it selects molecules that sit in the textbook-ideal band — logP 1.24 against
real drugs' 2.67 — so the shortlist is *more* idealized than real CNS drugs,
not merely more drug-like. That is a defensible target for a hypothesis list
and it is still a model of desirability, not evidence about any molecule.

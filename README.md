# BBB Permeability Prediction with Graph Neural Networks

Binary classification of blood–brain-barrier (BBB) permeability from molecular
structure, plus a molecule-generation experiment that uses the trained
classifiers as a reward.

```
SMILES → RDKit molecular graph → GNN → P(BBB permeable)
```

**Scope.** The core comparison — **GCN, GraphSAGE, GIN, GAT** — uses *only*
features derived from the RDKit molecular graph (39-dim atom features; no
descriptors, no fingerprints). The repo also contains four extensions, each
kept separate so the core comparison stays clean:

| family | what it adds | code |
|---|---|---|
| **base** | the four operators, identical skeleton | `src/models.py`, `src/train.py` |
| **edge ablation** | bond features via `gine`, `gat_edge`, `sage_edge` | `src/models_edge.py`, `src/train_edge.py` |
| **hybrid** | graph embedding **+ 8 RDKit descriptors** (incl. a CNS-MPO proxy) | `src/models_hybrid.py`, `src/hybrid_features.py`, `src/train_hybrid.py` |
| **descriptor baseline** | LightGBM on the same descriptors, same scaffold folds | `baselines/descriptor_baseline_v2.py` |

> The hybrid family **does** use hand-crafted descriptors. The "no descriptors"
> statement applies to the base and edge families only.

> **Paper.** This repository accompanies the RECOMB 2027 submission *Group-Relative
> Reinforcement Learning for Blood-Brain-Barrier Molecule Design: What Restrains
> Reward Hacking, and What Does Not*. The paper, its supplement, every figure and
> the scripts that regenerate them are in [`paper/`](paper/). Everything reported
> there is reproduced by `python -m src.pipeline` (see "Running" below).

> **The pipeline and the evidence for every part of it:**
> [`documentation/pipeline.md`](documentation/pipeline.md).

## Results at a glance

Test ROC-AUC, mean ± std over **10 seeds**. **Each seed is a different scaffold
split**, so ± includes split variance, not just initialization variance. Full
tables, with PR-AUC / balanced accuracy / F1 / MCC, are in
[`results/master_comparison.csv`](results/master_comparison.csv)
(regenerate with `python -m src.collect_all`).

| family (best member) | BBBP | B3DB |
|---|---|---|
| descriptor baseline (LightGBM, 8 descriptors) | 0.881 ± 0.039 | 0.852 ± 0.040 |
| base GNN (best of GCN/SAGE/GIN/GAT) | 0.895 ± 0.038 (GCN) | 0.893 ± 0.051 (SAGE) |
| edge ablation (best of 3) | 0.893 ± 0.031 (SAGE-edge) | 0.889 ± 0.052 (SAGE-edge) |
| hybrid GNN + descriptors | 0.909 ± 0.029 (GAT) | 0.900 ± 0.052 (SAGE) |
| hybrid + BatchNorm (best overall) | **0.921 ± 0.032** (GCN) | **0.905 ± 0.047** (GIN) |

How to read this:

- **Most differences are still inside seed noise.** `paired_analysis` separates
  **13 of 55 model pairs** at 95% with n=10; the largest effect is hybrid-SAGE
  over base-SAGE at +0.018. Run `python -m src.paired_analysis` before claiming
  any architecture ranking.
- Edge features give no consistent gain over the plain GNNs.
- The hybrid family is the strongest **in-distribution** by ~0.015, and that
  advantage **does not survive on the external holdout** (below).
- **These are lower than this README's earlier 3-seed figures (hybrid GAT
  0.961), and the drop is attributed rather than mysterious.** Two deliberate
  changes cost accuracy and bought correctness: the Murcko+Tanimoto split
  (~0.037 on BBBP), which stopped acyclic molecules being silently excluded
  from every test fold, and moving off BatchNorm (~0.015), which removed a
  reward-stationarity hazard. Both are measured in
  [`documentation/limitations.md`](documentation/limitations.md) 5.0. The old
  figures were optimistic, not wrong by accident.
- Absolute numbers are **not comparable** to the 0.65–0.70 BBBP scaffold
  figures in the literature; see "The scaffold split is balanced" below.

### External holdout (Adenot, Wang)

Every checkpoint (220 total) was also scored on two external sets that were
filtered to be scaffold-disjoint from **both** BBBP and B3DB. The leak filter
removes **~96.5%** of each source, leaving only **59 molecules (Adenot)** and
**55 molecules (Wang)** — see `data/external_holdout/leak_report.json`.

| family | Adenot (n=59) | Wang (n=55) |
|---|---|---|
| base | 0.985 ± 0.018 | 0.838 ± 0.048 |
| edge ablation | 0.987 ± 0.014 | 0.838 ± 0.054 |
| hybrid | 0.992 ± 0.007 | **0.831 ± 0.048** |

- **Adenot is saturated** — every family scores ~0.99, so it cannot
  discriminate between architectures at all.
- **Wang is the informative set, and everything falls to ~0.84.** That is the
  honest figure for novel chemistry; the ~0.90 above is in-distribution.
- **The hybrid family's lead does not transfer.** It is the best of the three
  on BBBP (0.909) and the *worst* of the three on Wang (0.831). Descriptors
  help on molecules resembling the training set and stop helping beyond it —
  which is the single most important caveat on the headline table.
- At n ≈ 55–59, single-digit error counts swing MCC. Treat rankings as
  suggestive; bootstrap CIs are still to do.

Write-up: [`results/external_holdout_eval/FINDINGS.md`](results/external_holdout_eval/FINDINGS.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric ≥ 2.3 uses `torch.scatter_reduce` natively, so the compiled
`torch-scatter` / `torch-sparse` extensions are **not** required.

Raw datasets live in `data/raw/` and are read from there by every module.

## Repository layout

```
.
├── data/
│   ├── raw/                 BBBP.csv, B3DB_classification.tsv, Adenot_final.csv, Wang_final.csv, BBB.csv*
│   ├── processed/           cached PyG graphs + cleaning reports (regenerate: python -m src.datasets)
│   └── external_holdout/    leak-filtered Adenot/Wang sets + leak_report.json
├── src/                     all library + CLI modules (table below)
├── baselines/
│   ├── descriptor_baseline_v2.py   LightGBM baseline on identical folds (current)
│   └── legacy/                     first-draft baseline; folds do NOT match the GNN runs
├── notebooks/results.ipynb  EDA, tables, ROC/PR curves
├── results/                 per-run outputs, aggregated tables, committed checkpoints
├── documentation/           design notes and diagrams (see "Further reading")
└── requirements.txt
```

\* `BBB.csv` is not read by any module.

| module | role |
|---|---|
| `featurize.py` | SMILES → PyG `Data`; 39-dim atom features, bond features stored |
| `datasets.py` | load, clean, deduplicate, cache both datasets |
| `split.py` | Bemis–Murcko scaffold split + leak assertions |
| `models.py`, `train.py`, `run_all.py` | base four-operator comparison (4 × 2 × 3 = 24 runs) |
| `evaluate.py` | metrics; threshold chosen on validation only |
| `ensemble.py` | soft vote, rank average, logistic stacking (within-seed) |
| `models_edge.py`, `train_edge.py` | edge-aware ablation (60 runs) |
| `brics.py` | BRICS fragments + motif graphs; shared with the generator |
| `models_hybrid.py`, `hybrid_features.py`, `train_hybrid.py` | GNN + descriptor hybrid (80 runs) |
| `external_holdout.py`, `eval_external_holdout.py` | build leak-free holdouts; score every checkpoint on them |
| `paired_analysis.py` | paired-seed architecture comparison |
| `collect_all.py` | merge every result table into `results/master_comparison.csv` |
| `reward.py`, `grpo.py`, `generator.py`, `train_gan.py` | molecule generation (GRPO) |
| `brics_assembly.py`, `brics_generator.py` | the settled generator: fragment assembly, valid by construction |
| `postfilter.py` | screen and rank generated molecules into a candidate set |
| `pipeline.py` | runs every stage above in order, resumable |
| `diversity.py`, `analyze_drift.py` | scaffold / Tanimoto / descriptor-drift diagnostics |
| `classifier_ood.py`, `mc_dropout.py`, `kl_sweep.py` | is the reward trustworthy, and can KL hold back the collapse? |

## Running

```bash
# --- smoke tests (no training) ---
python -m src.featurize                    # featurizer + FEATURE_ID
python -m src.models                       # shapes, norms, batch independence
python -m src.brics                        # fragment decompose/reassemble
python -m src.split                        # grouping + the leak-assertion trap
python -m src.datasets                     # build caches, verify splits

# --- core comparison ---
python -m src.train --model gcn --dataset bbbp --seed 0     # one run
python -m src.run_all                      # all 80 runs (10 seeds)
python -m src.run_all --threads 6          # pin the reduction order
python -m src.ensemble                     # combine predictions (within seed)

# --- extensions ---
python -m src.train_edge --run-all         # 60 runs: gine / gat_edge / sage_edge
python -m src.train_hybrid --run-all       # 80 runs, GraphNorm
python -m src.train_hybrid --run-all --norm batch   # 80 runs -> results/hybrid_batch
python baselines/descriptor_baseline_v2.py # LightGBM baseline (no torch needed)

# --- external evaluation + aggregation ---
python -m src.external_holdout             # rebuild leak-free holdouts (RDKit only)
python -m src.eval_external_holdout        # score all checkpoints on them
python -m src.paired_analysis --all-sources
python -m src.collect_all                  # -> results/master_comparison.csv

# --- the whole pipeline, resumable (see documentation/full_pipeline.md) ---
python -m src.pipeline --list              # what would run, what would skip
python -m src.pipeline                     # classifier -> reward -> generator -> screen

# --- generation, by hand ---
python -m src.reward && python -m src.grpo # self-checks; reward is the gate
python -m src.brics_generator --pretrain   # MLE warm start (~9 min, NOT optional)
python -m src.train_gan --generator brics --steps 250 --threads 6 \
    --seed 0 --reward-seed 0 --kl-coef 1.0 \
    --warm-start results/generator/bbbp_brics_pretrained.pt \
    --out-dir results/brics_kl1/rs0_ws0
python -m src.postfilter results/brics_kl1/* --from-step 150 \
    --out results/candidates/brics10

jupyter lab notebooks/results.ipynb
```

`--kl-coef 1.0` is mandatory for the BRICS generator. The repo default of 0.3
was validated for the SELFIES action space and does not transfer — at 0.3 BRICS
collapses to scaffold diversity 0.643 with 26% duplicates per group.

Runs default to CPU — these graphs are small enough that MPS kernel-launch
overhead makes it slower. `--device mps` is available to test that.

## Design decisions worth knowing

**One skeleton, four operators.** Depth, hidden width, normalization, readout,
classifier head and training loop are identical across all four models. Only the
convolution differs, so a performance gap is attributable to the aggregation
scheme rather than to incidental capacity differences. (Measured caveat: the
virtual-node arm shows part of what this comparison attributes to aggregation
is really *reach* -- with `num_layers=3` the operators differ in how far they
propagate, and a global channel largely equalizes them. See
`documentation/limitations.md` 5.0b.) `GATConv` uses
`hidden // heads` channels per head so its concatenated output matches the
others rather than inflating fourfold.

**The scaffold split is balanced, not size-sorted.** The classic DeepChem
`ScaffoldSplitter` sorts scaffold groups by size and fills train first, leaving
val/test with only singleton scaffolds. On `BBBP.csv` that is degenerate: the
file is ordered so its entire back half is class 1, and both folds come out
**100% positive**, making ROC-AUC undefined. The balanced split (Chemprop-style)
shuffles scaffold groups under a per-seed RNG instead. `src/split.py` keeps the
size-sorted variant behind `balanced=False` so the artifact can be reproduced.

**Acyclic molecules are clustered, not merged.** `MurckoScaffoldSmiles` returns
`''` for anything with no ring, so every acyclic molecule used to collapse into
a single group -- and because groups are never split across folds, that group
always landed in train and those molecules were never tested (95/1965 BBBP,
311/7805 B3DB). They are now grouped by Morgan/Tanimoto similarity (Butina,
cutoff 0.6). `split.group_keys` is the single definition of group identity and
is called by both the splitter and the leak assertion, which otherwise disagree
and report a leak that does not exist.

Consequence: absolute numbers are **not** comparable to the widely quoted
BBBP scaffold figures near 0.65–0.70, which come from the size-sorted protocol.
The comparable published range for the balanced protocol is around 0.90.

**The seed reseeds the split.** Each seed defines a different scaffold
partition, so the reported ± captures split variance as well as initialization
variance. It also means models can only be ensembled *within* a seed, where all
four share an identical fold.

**Bond features are computed but unused.** `Data.edge_attr` carries bond type,
conjugation and ring membership, but no model consumes it — GCN and GraphSAGE
structurally cannot, so feeding it to GIN/GAT alone would confound the
architecture comparison with an input-signal advantage. That follow-up ablation
now lives in `src/models_edge.py` (`gine`, `gat_edge`, `sage_edge`) and writes
to `results/edge_ablation/`, deliberately kept out of `MODELS` so the four-model
comparison stays edge-blind.

**The datasets overlap.** B3DB aggregates BBBP as one of its sources, and some
shared molecules carry contradictory labels. The two are therefore trained and
evaluated independently; training on one and testing on the other would be a
leaked evaluation.

## Molecule generation (GRPO)

The classifier above is the *scoring* half of a second project: a GAN generator
trained with GRPO to propose molecules that are predicted permeable, drug-like
and synthesizable. `src/reward.py` and `src/grpo.py` implement the parts of that
math formulation which do not depend on the generator architecture;
`src/generator.py` and `src/train_gan.py` add the generator, `D_phi` and the
outer loop.

```
generator -> SMILES -> RDKit sanitization -> frozen classifier ensemble
                                          -> QED + SA
                                          -> discriminator     } -> R(m) -> advantage -> clipped update
```

**The settled generator is BRICS fragment assembly, not SELFIES.** It is valid
by construction, and `src/brics_generator.py` supersedes `src/generator.py` for
anything new; the SELFIES generator is kept because the frontier comparison
cites it.

```bash
python -m src.pipeline                 # the whole thing, resumable
```

Each module runs its own self-check with no arguments (`python -m src.grpo`).

The classifier checkpoints the reward depends on are committed under
`results/`, as are the MLE warm start and policy seeds 0-2, so a default
pipeline run trains only the seven remaining policy seeds (~42 min). On CPU
BRICS GRPO is ~1.46 s/step at group 64, so 250 steps is ~6 min.

**If you retrain the classifier, nothing re-points the reward** — it reads
whatever is at those checkpoint paths. `python -m src.reward` is the gate that
catches a mismatch, and `python -m src.pipeline` runs it before any generator
work.

Reward (Eq. 5) is a weighted sum of four terms: the discriminator score, the
frozen permeability classifier, QED, and a synthesizability score. `assemble()`
takes the discriminator score as an argument rather than owning `D_phi` —
`D_phi` trains alongside the generator and is the only non-stationary term, so
keeping it out lets the three stationary terms be tested on their own.

```python
from src.reward import load_frozen_classifier, score_terms, weights_at, assemble

classify = load_frozen_classifier(dataset="bbbp", models=("gin", "gat", "gine"), seed=0)
terms = score_terms(smiles_list, classify)
w = weights_at(step, w0, wf, schedule="linear", k_anneal=1000)
rewards = assemble(terms, d_scores, w)
```

**"Frozen" has to mean `eval()`, not just `requires_grad_(False)`.** This used
to be about `BatchNorm1d`, whose output depends on the batch it is scored in --
so the same molecule earned a different reward depending on which candidates
shared its group, and the reward stopped being a function of the molecule. The
default is now **GraphNorm**, which is batch-independent by construction, so
that hazard is gone rather than guarded. `eval()` remains load-bearing for
`nn.Dropout`, which is still stochastic in train mode. The alone-vs-in-company
assertion stays because it is what catches a regression back to BatchNorm.

**Ensemble members must share a scaffold split.** The seed reseeds the split, so
averaging checkpoints from different seeds averages models that trained on each
other's held-out molecules. `results/edge_ablation/` reuses the core split
seed-for-seed, which is what makes a `gin`/`gat`/`gine` ensemble legitimate;
`load_frozen_classifier` asserts the saved fold indices match on **every
load**, and also that members share a norm -- a mixed-norm ensemble is not a
leak, so the fold check cannot see it, but it does mean a checkout caught
mid-update. Both previously lived only in `_self_check`, hardcoded to one model
triple at one seed.

**Validity means sanitization, not grammar.** A SELFIES string always parses to
*some* molecule, so a generator's syntactic guarantee says nothing about whether
RDKit will sanitize the result. Only sanitization gates the reward; invalid
molecules collapse to a flat penalty and never reach the classifier.

**Weight schedules.** `weights_at` covers the fixed baseline, linear and cosine
annealing (Eq. 6), and the validity-gated switch (Eq. 7) that trades a step
count for a threshold on measured validity rate. It is a pure function of the
step and the caller's own validity history, so no schedule carries hidden state
between steps.

**One update rule for both generator designs.** The formulation leaves open
whether the generator is autoregressive (SELFIES, a sequence of `T` token
actions) or one-shot (MolGAN-style atom/bond tensors, a single joint action).
`clipped_loss` accepts `(N, T)` or `(N,)` log-probs, because the one-shot case
is exactly the `T = 1` degenerate case of the sequential one — a test asserts
the two agree. That fork therefore only has to be decided in the generator; the
reward and the update do not branch on it.

**The fork is resolved to Case A.** `src/generator.py` is a GRU over SELFIES
tokens conditioned on `z` through the initial hidden state. Case B was not
chosen on accuracy grounds — it needs a graph decoder plus valence repair,
where Case A needs a library call, and `selfies` was already a dependency.
Reverting means replacing that one file: nothing downstream branches on it.

**The MLE warm start is not optional.** From a uniform babbler every group is
all-invalid, every reward is the same flat penalty, the group standard
deviation is zero, and `group_advantages` correctly returns no gradient at all.
GRPO would wait for validity to appear by chance. `--pretrain` reaches ~99%
sanitization validity on BBBP in 30 epochs, which is where GRPO can start.

**`D_phi` reuses the classifier skeleton.** Real-vs-fake is the same
graph→logit problem, so `Discriminator` wraps `build_model("gin")` rather than
introducing a second architecture. It is scored in `eval()` for exactly the
reason the frozen classifier is: with BatchNorm in train mode, `D(m)` would
depend on which candidates shared its group, and the reward would stop being a
function of the molecule.

**Two orderings from Section 3 are structural, not advisory.** The `n_D`
discriminator steps run strictly *after* the `K`-epoch PPO block, never
interleaved, so every advantage in a group is computed against one fixed `phi`
snapshot (Section 3.4) — a self-check asserts `phi` does not move during the
PPO window. `Var_i[D(m_i)]` is logged per step as the Section 3.3 diagnostic:
as `D` sharpens, that term's variance drifts relative to the fixed-variance `C`
term, changing their effective weight on `A_i` even with `w_D, w_C` held
literally constant.

**Reward hacking is real, not hypothetical.** An early 40-step smoke run looked
healthy (mean predicted permeability 0.81 → 0.96, 100% validity). Longer runs
show the opposite: over 200 steps, in all 3 seeds, scaffold diversity falls
from ~0.8 to 0.07–0.14, the spread of the classifier term collapses, and
~100% of generated molecules score above 0.95 — while validity stays at 100%
and the old `unique SMILES` metric lags the true collapse by 88–154 steps.
Of the four KL coefficients swept (0.02, 0.1, 0.3, 1.0), only 0.3 and 1.0 meet
the sweep's retention criterion, and with small reward gain. Full analysis:
`documentation/reward_hacking_investigation.md`.

**The reward is zero-inflated, so the advantage is a two-part estimator.**
`assemble` sends invalid molecules to a flat penalty and valid ones to a
positive continuous sum — two populations, not one sample. Pooling them to get
a mean and a standard deviation lets a handful of zeros set the scale the valid
molecules are ranked on. Measured on BBBP: an invalid molecule sits ~8.5 sd
below the valid cluster, and **one invalid in a group of 64 costs ~31% of the
ranking signal, three cost ~51%**. `group_advantages(valid=...)` standardizes
within the valid subgroup and hands invalid molecules a fixed `floor`, which
holds the spread at 1.0 regardless of how many zeros land in the group:

| invalid in 64 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| pooled | 1.00 | 0.70 | 0.60 | 0.49 |
| hurdle | 1.00 | 1.00 | 1.00 | 1.00 |

Ranking is not what was being lost — standardization is affine, so
`corr(adv, C)` is identical either way. What was lost is *scale*: the valid
molecules collapse toward a uniform "you sanitized" push with ranking as the
remainder. Rank-based (5% drop) and median/MAD (9%) fix it too and need no
hyperparameter, but `floor` being explicit is the point — under pooling the
effective penalty for invalidity is whatever else happened to be in the group,
which is nobody's decision. It pairs with `assemble(invalid_reward=...)`: that
sets the reward, `floor` sets the advantage the reward becomes.

Two traps the self-checks pin down, both of which fail silently rather than
loudly:

- A group whose rewards are all equal — every molecule invalid, which is the
  normal early-training state — has zero standard deviation. Its advantages are
  forced to exactly zero instead of dividing by an epsilon and amplifying float
  noise into a gradient.
- Padded positions in a `(N, T)` batch hold arbitrary values, so `exp()`
  overflows to `inf` there and `inf * 0` is `nan`. Masking has to happen on the
  log-probability difference, *before* the exponential; masking the ratio
  afterwards cannot recover it.

## Data

### Raw files (`data/raw/`)

| file | used by | notes |
|---|---|---|
| `BBBP.csv` | training; holdout leak filter | label column `p_np`; ordered so its back half is all class 1 |
| `B3DB_classification.tsv` | training; holdout leak filter | aggregates BBBP among its sources |
| `Adenot_final.csv`, `Wang_final.csv` | `src/external_holdout.py` | external sets; **never used for training** |
| `BBB.csv` | — | not read by any module |

Processed caches and cleaning reports are in `data/processed/` and can be
rebuilt with `python -m src.datasets`.

### External holdouts

`src/external_holdout.py` removes every external molecule that (a) matches a
training molecule by canonical SMILES, or (b) shares a Bemis–Murcko scaffold
with one, then drops whole scaffold groups. That is deliberately strict.

| source | rows in | direct overlap | scaffold groups tainted | rows out (pos / neg) |
|---|---|---|---|---|
| Adenot | 1650 | 1559 | 810 | **59** (15 / 44) |
| Wang | 1592 | 1469 | 761 | **55** (38 / 17) |

The surviving sets are small. See "Known limitations".

### Cleaning (BBBP, B3DB)

| | rows in | invalid SMILES | dupes merged | label conflicts | rows out |
|---|---|---|---|---|---|
| BBBP | 2050 | 11 | 54 | 20 | **1965** (76.3% positive) |
| B3DB | 7807 | 2 | 0 | 0 | **7805** (63.5% positive) |

Deduplication is on the *canonical* SMILES, not the raw string. Molecules whose
duplicate copies disagree on the label are dropped entirely rather than guessed.

## Known limitations

- **Ten seeds** (was three), and it is still not enough to rank architectures:
  `paired_analysis` separates only 13 of 55 pairs at 95%. Paired bootstrap CIs
  are still to do.
- **The in-distribution and out-of-distribution orderings disagree.** Hybrid is
  best on BBBP and worst of the three families on Wang. Any claim about which
  family is "better" has to say on which distribution.
- **The normalization choice is recorded, not settled.** GraphNorm is the
  default; BatchNorm scores ~0.015 higher on BBBP but makes a frozen
  classifier's output depend on batch composition. See
  `documentation/limitations.md` 5.0.
- **The virtual node adds nothing under the current default.** +0.0124 under
  LayerNorm, +0.0003 +/- 0.0023 under GraphNorm -- the two were fixing the same
  thing, since GraphNorm already carries graph-level statistics into every
  node. 5.0b.
- **Tiny external holdouts** (n = 55–59). Adenot is saturated and uninformative;
  Wang results swing on single-digit error counts.
- **Hybrid model uses descriptors**, including a CNS-MPO *proxy* (5 of 6 terms;
  CLogD is proxied by CLogP and the pKa term is omitted).
- **Fixed 0.5 threshold for stacking.** Single models and soft-vote/rank-average
  use a validation-tuned Youden threshold; the logistic stacker does not, so
  F1/MCC are not directly comparable across ensemble methods. ROC-AUC is.
- **The generator does not yet produce validated molecules.** Under the current
  reward it hacks the classifier (above). Treat generated molecules as
  unvalidated.
- **Domain bias.** BBBP and the external sets contain many charged / β-lactam
  compounds that are easy negatives, which inflates headline AUC.
- **Committed artifacts are large** (`data/processed/*.pt`, model checkpoints
  under `results/` and `results_layernorm/`); a Git LFS migration would shrink
  clones. The processed cache alone is ~73 MB and GitHub warns on push.
- **No license file** has been added; the repository owner needs to choose one.

## Further reading (`documentation/`)

| file | contents |
|---|---|
| **`full_pipeline.md`** | **the whole recipe end to end — start here** |
| `pipeline.md` | the classifier half, and the evidence for every part of it |
| `generator_pipeline.md` | the generation half, at the three-seed configuration |
| `brics_generator_findings.md` | what was learned building the BRICS generator |
| `limitations.md` | every known failure, measured, including the retractions |
| `reward_hacking_investigation.md` | how the collapse was found, measured and diagnosed |
| `cold_and_warm_start.md` | why the generator uses an MLE warm start and a two-part advantage |
| `open_items.pdf` / `.tex` | formalization of the open items |
| `bbp.pdf` | background reference |
| `mermaid-diagram-*.png` | architecture / pipeline diagrams |
| `BBB_Tools*.png` | reference table of online SMILES-based BBB/ADME tools (excerpted from a published source; not used by code) |

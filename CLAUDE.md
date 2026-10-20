# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # requirements.lock pins the verified versions
```

PyG ≥ 2.3 uses `torch.scatter_reduce` natively — the compiled `torch-scatter` /
`torch-sparse` extensions are *not* needed. Everything defaults to CPU; these
graphs are small enough that MPS kernel-launch overhead is a net loss.

## There is no test suite

No pytest, no CI, no Makefile. Instead **every module in `src/` is runnable with
no arguments and runs its own `_self_check()`** — assertions over the invariant
that module exists to protect (e.g. `python -m src.reward` asserts a molecule
scores identically alone and in a group; `python -m src.grpo` asserts the
masking survives padded log-probs). That is the test suite.

Follow the convention when adding a module: a `_self_check()` under
`if __name__ == "__main__"` that fails loudly if the module's core invariant
breaks. `python -m src.train_gan` with no `--steps` runs the self-check rather
than training.

```bash
python -m src.featurize      # featurizer
python -m src.models         # forward-pass shapes
python -m src.datasets       # build caches + verify splits
python -m src.reward && python -m src.grpo
python -m src.brics          # decompose/reassemble round-trip
python -m src.split          # grouping + the leak-assertion trap
```

## Common commands

```bash
# everything, in order, resuming by artifact on disk -- prefer this
python -m src.pipeline --list          # what would run, what would skip
python -m src.pipeline                 # classifier -> reward -> generator -> screen
python -m src.pipeline --only reward    # one stage
python -m src.pipeline --from generator # that stage onward

# classifier
python -m src.train --model gcn --dataset bbbp --seed 0   # one run
python -m src.run_all                                     # all 80 base runs
python -m src.train_edge --run-all                        # 60
python -m src.train_hybrid --run-all                      # 80, GraphNorm
python -m src.train_hybrid --run-all --norm batch          # 80 -> results/hybrid_batch
python -m src.run_all --threads 6 --force                 # pin reduction order
python baselines/descriptor_baseline_v2.py                # LightGBM, no torch

# evaluation / aggregation
python -m src.external_holdout        # rebuild leak-free holdouts (RDKit only)
python -m src.eval_external_holdout   # score every checkpoint on them
python -m src.paired_analysis --all-sources
python -m src.collect_all             # -> results/master_comparison.csv

# generation -- BRICS is the settled generator; src/generator.py is SELFIES legacy
python -m src.reward                   # the gate. If it fails, STOP.
python -m src.brics_generator --pretrain   # MLE warm start, ~9 min CPU, NOT optional
python -m src.train_gan --generator brics --steps 250 --threads 6 \
    --seed 0 --reward-seed 0 --kl-coef 1.0 \
    --warm-start results/generator/bbbp_brics_pretrained.pt \
    --out-dir results/brics_kl1/rs0_ws0
python -m src.postfilter results/brics_kl1/* --from-step 150 \
    --out results/candidates/brics10
```

**`--kl-coef 1.0` is mandatory for BRICS.** The repo default 0.3 was validated
for the SELFIES action space and does not transfer — the same coefficient
yields ~10.7x the KL here, and at 0.3 BRICS collapses to scaffold 0.643 with
26% duplicates per group. Never "restore the default".

`documentation/full_pipeline.md` is the single recipe; `pipeline.md` and
`generator_pipeline.md` are the per-half evidence.

Runs write to `results/<family>/.../seed<n>/`; classifier checkpoints are
committed, so the reward half never needs retraining.

## Architecture

```
SMILES → RDKit graph (39-dim atom features) → GNN → P(BBB permeable)
                                                  ↘ frozen → reward for a GRPO generator
```

Two halves. The **classifier** is a controlled four-operator comparison
(GCN / GraphSAGE / GIN / GAT) sharing one skeleton — depth, width, norm,
readout, head and training loop identical, only the conv differs, so a gap is
attributable to aggregation rather than capacity. Extensions each live in their
own `models_*.py` + `train_*.py` pair (`_edge`, `_hybrid`) and write to
their own `results/` subtree, deliberately kept out of `MODELS` so the core
comparison stays edge-blind and descriptor-blind. D-MPNN was dropped (no
consistent gain, and it cost 60 runs at 10 seeds).

The **generator** assembles BRICS fragments (`brics_assembly.py`,
`brics_generator.py`), so its output is valid by construction; it is trained by
GRPO against the frozen classifier ensemble plus QED, SA and a discriminator.
`generator.py`'s GRU over SELFIES is the superseded arm, kept because the
frontier comparison cites it.

**The classifier is trained twice, on purpose.** `--norm batch` →
`results/hybrid_batch/` is the reported model (BBBP 0.921, Wang 0.850). The
GraphNorm default is what a frozen reward must use: BatchNorm takes its
statistics from the batch, so `C(m)` would depend on which molecules shared
m's GRPO group. The reward ensemble is *not* a hybrid model — it is base/edge
`gin`/`gat`/`gine` at seed 0 under GraphNorm, which is what every committed
generator result was trained against.

`src/datasets.py` owns `ROOT` (repo root) and the processed-graph cache; every
module reads data through it rather than building paths itself.

### The run contract � read this before touching the featurizer

Checkpoints here are bare `state_dict`s, and `build_model`'s **default
arguments were the only thing making them loadable** � no caller ever passed
`in_dim`. So changing a default silently invalidated all 73 committed
checkpoints, and `load_dataset` keyed its cache on the dataset name alone, so a
featurizer change could return stale graphs and never raise.

- `featurize.FEATURE_ID` is a **content hash of the featurizer's own output** on
  a fixed probe set � derived, not declared. If you add a feature no probe
  exercises, extend `_PROBE` in the same diff or the guard goes blind.
- `train.py` records it in `metrics.json` with `node_dim`, `hidden`,
  `num_layers`, `dropout`, `heads`, `norm`, `virtual_node`, `split`.
- `featurize.run_contract(run_dir)` validates it and **returns the recorded
  architecture**, so `reward.load_frozen_classifier` and
  `eval_external_holdout` rebuild a checkpoint the way it was trained rather
  than the way today's defaults happen to be.
- `datasets.cache_path` keys on `FEATURE_ID`, so a mismatched cache is not
  detected � it simply isn't found, and rebuilds.
- Pre-contract checkpoints live in `results_v1/` and are **refused by design**.
  `results/generator/` was deliberately left in place: those checkpoints are
  self-describing and feature-independent.

`src/brics.py` is the shared fragment substrate for the classifier's motif
family and the generator's assembly engine. Its `vocab()` takes the caller's
own corpus: the classifier must build it from **training-fold molecules only**
(a full-dataset vocabulary is transductive leakage under a scaffold split),
while the generator needs a lower floor because pruning is a reconstruction
failure there.

### Invariants that are load-bearing, not stylistic

Violating any of these silently invalidates results rather than raising:

- **The seed reseeds the scaffold split.** Each seed is a different partition,
  so reported ± includes split variance, and models may only be ensembled
  *within* a seed. `results/edge_ablation/` deliberately reuses the core split
  seed-for-seed so a `gin`/`gat`/`gine` ensemble is legitimate; `reward.py`'s
  self-check asserts the saved fold indices match rather than trusting it.
- **The split is balanced, not size-sorted** (`src/split.py`, `balanced=True`).
  DeepChem's size-sorted splitter makes `BBBP.csv` degenerate — the file's back
  half is all class 1, so val/test come out 100% positive and ROC-AUC is
  undefined. Consequence: absolute numbers are **not** comparable to the
  widely quoted 0.65–0.70 BBBP figures. Comparable range is ~0.90.
- **Normalization defaults to LayerNorm, and that is a correctness choice.**
  `BatchNorm1d` made a molecule's logit depend on what shared its batch � fine
  for a classifier, a bug for a frozen reward, since the same molecule earned a
  different reward depending on its GRPO group. Measured by
  `python -m src.models` on a 4-graph batch in train mode: **batch drifts
  3.97e-02, layer 3.7e-09, graph 6.0e-08**. `eval()` is *still* load-bearing
  for `nn.Dropout`, and `reward.py`'s alone-vs-crowded assertion stays because
  it is what catches a regression back to BatchNorm.
- **`split.group_keys` is the single definition of group identity.**
  `scaffold_split` and `check_split` must group identically or the leak
  assertion fires on every run. Acyclic molecules used to collapse into one
  `""` group that always landed in train (95/1965 BBBP, 311/7805 B3DB � never
  tested); they now cluster by Tanimoto. If a change makes `check_split` fire,
  fix the grouping � **do not weaken the assertion**, or every number in the
  repo becomes unverifiable.
- **Thread count is part of the configuration.** OpenMP changes float reduction
  order and the GRPO loop is chaotic. `OMP_NUM_THREADS` is unreliable (36
  silently binds 18 on the dev machine); pass `--threads` so
  `torch.set_num_threads()` sets it in-process. A driver that sets the env var
  and trusts it contaminated the first term ablation.
- **Paired designs.** Warm-start variance is ~2× GRPO-seed variance, so arm
  comparisons pair on *both* the seed and the per-seed warm start. Pairing on
  only the seed is what made the VAE comparison read as a clean regression when
  it was noise. Run `python -m src.paired_analysis` before claiming any ranking.
- **BBBP and B3DB overlap** (B3DB aggregates BBBP, with some contradictory
  labels). Train and evaluate them independently; cross-testing is leakage.
- **Adenot/Wang are never training data.** The leak filter drops ~96.5% of each,
  leaving n = 59 / 55.

### Defaults encode validated findings

`--kl-coef 0.3` and `--min-heavy-atoms 10` are defaults because they are the
only two interventions that survived validation; everything else that was tried
is off for a recorded reason (logit transform, Tox21 term, alert penalty,
geometric aggregation each made some measured instrument worse). Don't "clean
up" a default without reading the reason — `documentation/limitations.md` is a
numbered ledger where every item carries a status tag
(**[measured] / [mitigated] / [suspected] / [untested]**) and nothing is marked
resolved.

Reward hacking here is measured, not hypothetical: unguarded, scaffold
diversity falls 0.8 → 0.07–0.14 over 200 steps in all 3 seeds while validity
stays at 100%. **Generated molecules are unvalidated and are not synthesis
candidates.** Term ablation found SA, not the classifier, drives the collapse.
Structural screening lives in `src/postfilter.py` *after* training on purpose —
a filter inside the reward taught the generator the checklist, not the
chemistry (targeted SMARTS 0.339 → 0.203 while BRENK stayed flat).

## Writing conventions

Module docstrings here are long and carry the *why*, including the measurement
that motivated the module and what it explicitly does not fix. Several open with
a retraction of an earlier result and the reason it was void. Match that: when a
claim is in a docstring or in `documentation/`, it is expected to be backed by a
number produced in this repo.

Stale: `cns_mpo_reward.py` (repo root) and `data/raw/BBB.csv` are imported and
read by nothing. `baselines/legacy/` uses folds that do **not** match the GNN
runs — never compare against it.

# Handoff to system-2 — de-novo generation on BRICS

You own the generation half. System-1 owns the classifier and has frozen the
interface the two halves meet at. This document is that interface, plus the
measurements that should change what you build.

Read it before writing code. Several things here contradict the earlier design
discussion, and each contradiction is backed by a number from this repo.

---

## 0. First five minutes

Your clone probably points only at `sankarhariharan2007-kali`. The work is on
the fork, so add it, then branch off the contract — **do not commit to
`classifier-v2`**, system-1 owns that branch and two machines pushing to it
will collide.

```bash
git remote add myfork https://github.com/JSR996/bbb-gnn-permeability 2>/dev/null
git fetch myfork --tags
git checkout -b brics-generator contract-v2      # your branch, off the frozen contract
git config user.name  "JSR996"
git config user.email "shrenikrjnasa@gmail.com"
```

Then confirm the contract holds on this machine:

```bash
python -m src.featurize      # prints FEATURE_ID -- must be c0c3bbb8
python -m src.brics          # fragment substrate self-check
python -m src.reward         # THE GATE: must pass
python -m src.grpo
```

Push with `git push -u myfork brics-generator`. Pull system-1's later work with
`git fetch myfork && git merge myfork/classifier-v2` when you need the 10-seed
ensemble (section 8).

If `python -m src.reward` passes, your reward is wired to a classifier that
matches this code. If it raises a `feature_id` error, **stop** — do not work
around it. That error means the checkpoints and the featurizer disagree, and
anything you train against them is meaningless. Pull again, or ask system-1.

You never need to poll for system-1. Everything you need is already pushed.

---

## 1. The run contract

The two halves used to be coupled by nothing but `build_model`'s default
arguments. No caller ever passed `in_dim`, checkpoints are bare `state_dict`s,
and `metrics.json` recorded no architecture at all — so changing a default
silently invalidated all 73 committed checkpoints, and the failure surfaced as
`size mismatch for convs.0` far from its cause. `load_dataset` also keyed its
cache on the dataset name alone, so a featurizer change could return stale
graphs and never raise.

That is now closed at one chokepoint:

| piece | where | what it does |
|---|---|---|
| `FEATURE_ID` | `featurize.py` | sha1 of the featurizer's own output on a fixed probe set. **Derived, not declared** — a constant you must remember to bump is one that will be wrong exactly once. Currently `c0c3bbb8`. |
| contract fields | `train.py` → `metrics.json` | `feature_id`, `node_dim`, `edge_dim`, `hidden`, `num_layers`, `dropout`, `heads`, `norm`, `virtual_node`, `split` |
| `run_contract(run_dir)` | `featurize.py` | raises on a `feature_id` mismatch, **returns the recorded architecture** |
| cache key | `datasets.cache_path` | `bbbp-<FEATURE_ID>.pt` — a stale cache is not detected, it simply isn't found |

`reward.load_frozen_classifier` and `eval_external_holdout` both call
`run_contract` and rebuild each checkpoint the way it was *trained*, not the
way today's defaults happen to be. Verified both directions: a `results_v1/`
checkpoint is refused by name, the current ones load.

**If you add a feature that no entry in `featurize._PROBE` exercises, extend
`_PROBE` in the same diff.** Otherwise the hash does not move and the guard
goes blind to exactly the change it exists to catch.

### Frozen architecture

```
NODE_DIM = 39      EDGE_DIM = 7      FEATURE_ID = c0c3bbb8
hidden = 128       num_layers = 3    dropout = 0.3     heads = 4
norm = "graph"     virtual_node = False
split = "murcko+tanimoto@0.6"
reward ensemble = ("gin", "gat", "gine") on bbbp, seed 0
```

**The featurizer did not change.** It is still the same 39 dimensions. What
changed is the normalization, the split, and the fact that all of it is now
recorded. So your side breaks only if you change `featurize.py`.

---

## 2. LayerNorm replaced BatchNorm, and it matters most to you

`BatchNorm1d` made a molecule's logit depend on which molecules shared its
batch. For a classifier that is a training detail. For **your** reward it was a
correctness bug: the same molecule earned a different reward depending on which
candidates happened to land in its GRPO group, so the reward was not a function
of the molecule at all.

Measured by `python -m src.models`, gin on a 4-graph batch, train mode:

| norm | logit drift vs scoring alone |
|---|---|
| batch | **3.97e-02** |
| layer | 3.7e-09 |
| graph | 6.0e-08 |

**`eval()` is still load-bearing**, for a smaller reason: `nn.Dropout` keeps the
forward pass stochastic in train mode. Keep every `eval()` call and keep
`reward._self_check`'s alone-vs-crowded assertion — it is cheap, and it is what
catches a regression back to BatchNorm.

**The default is now GraphNorm** (was LayerNorm), still batch-independent.
BatchNorm costs are unchanged and it remains the higher scorer. BatchNorm vs LayerNorm, 10 seeds, paired on seed:

| dataset | gcn | sage | gin | gat |
|---|---|---|---|---|
| BBBP | +0.0215 | +0.0459 | +0.0269 | +0.0168 |
| B3DB | +0.0030 | −0.0018 | +0.0096 | −0.0007 |

Pooled over 80 paired runs, batch − layer = **+0.0152 ± 0.0042 SEM**. BatchNorm
wins in all four BBBP cells and ties in all four B3DB cells — a small-dataset
effect, since BBBP trains on 1572 molecules where batch statistics regularize.
And the hazard above was *already* contained by the `eval()` guards, so
LayerNorm buys structural immunity rather than fixing a live bug.

`documentation/limitations.md` 5.0 records this as **unresolved**. If it flips,
the reward checkpoints get retrained and you re-pull — that is the only way it
reaches you, and `run_contract` makes it loud. Nothing you build on top needs
to change.

---

## 3. `src/brics.py` — the shared substrate

System-1 owns this file; you import it. One definition of "a fragment" on both
sides, or you optimize a vocabulary the classifier cannot read.

```python
from src.brics import decompose, reassemble, motif_graph, vocab, UNK

frags, junctions = decompose(mol)   # junction = (frag_a, dummy_a, frag_b, dummy_b, bond_type)
parent = reassemble(frags, junctions)
nodes, edges = motif_graph(mol)     # fragment SMILES + the cut bonds
v = vocab(smiles_list, min_count=3) # UNK is always id 0
```

`decompose` is **lossless**: 2039/2039 BBBP and 7805/7805 B3DB molecules
round-trip to an identical canonical SMILES.

Three measured facts that contradict the earlier design chat:

- **`BRICS.BRICSDecompose` has no `keepChiral` argument.** Verified against the
  installed RDKit. Chirality is preserved by default anyway (~97% of assigned
  centers survive).
- **"~150 fragments covers 95%" does not hold here.** At `min_count=3`: BBBP
  keeps 316 fragments covering **75.1%** of occurrences, B3DB 1178 covering
  **81.3%**.
- **12.8% of BBBP and 9.3% of B3DB molecules yield zero BRICS cuts.** The
  whole-molecule scaffold-core fallback is one molecule in ten, not an edge
  case. Your assembly engine must be able to emit a core with no junctions.

Also: the junction carries a **bond type**, and it is not decoration. BRICS cuts
type-7 alkene linkages, not only single bonds. Rebuilding every junction as
`SINGLE` silently hydrogenates 91 of BBBP's 2039 molecules —
`CN(C)CCC=C1c2ccccc2CCc2ccccc21` comes back as `...CCCC1...`. That bug was
caught by the round-trip assertion; do not reintroduce it in your assembler.

### `vocab()` takes YOUR corpus, on purpose

There is no committed shared vocabulary file, because the two sides need
different ones:

- **Classifier** (system-1): built from **training-fold molecules only**, per
  (dataset, seed). A vocabulary over the full dataset is transductive leakage
  under a scaffold split — the split exists to give test molecules unseen
  cores, and a global vocabulary hands every test fragment its own embedding
  slot. Rare fragments map to `UNK`.
- **Generator** (you): the whole corpus is fine, you make no held-out claim,
  and the floor is **`min_count=1`, i.e. no pruning at all**. Frequency pruning
  that is a harmless long tail for a classifier is a *reconstruction failure*
  for a generator, which cannot emit a fragment it has no token for.

  The decisive number is not how many fragments a floor drops, it is how many
  **molecules stop being reconstructable** — those leave the MLE corpus, and
  the warm start is not optional (§5.4):

  | floor | BBBP vocab | BBBP molecules fully covered | B3DB vocab | B3DB covered |
  |---|---|---|---|---|
  | **1** | **2625** | **1965 (100%)** | **8060** | **7805 (100%)** |
  | 2 | 726 | 456 (**23.2%**) | 2409 | 2802 (35.9%) |
  | 3 | 439 | 216 (11.0%) | 1520 | 1735 (22.2%) |
  | 5 | 260 | 84 (4.3%) | 858 | 791 (10.1%) |

  `min_count=2` looks cheap — it drops "only" the singletons — but it leaves
  **23% of BBBP reconstructable**. It would delete three quarters of the warm
  start to save a softmax over 2625 classes, which is nothing at this scale.
  Singletons are most of the vocabulary and almost none of the occurrences,
  but they are spread thinly across nearly every molecule, so pruning them
  hits the corpus far harder than the occurrence share suggests.

  Consequence: with no floor, `UNK` is never needed on the generator side.
  Keep the token (the classifier uses it) but assert it is never emitted.

Record whichever you used.

---

## 4. Three decisions that change what you build

### 4.1 Assembly-first. Do not generate a graph and then decompose it.

The action space **is** sequential assembly: the open attachment sites are held
in a deterministic queue, and at each step the policy chooses only **what to
put on the current site** — a fragment from the vocabulary, or CAP to close it
— until no open site remains. The molecule is the *output*.

**The policy does not choose the site.** An earlier draft of this document said
the action was "(attachment point, fragment)", which read as a joint choice
over both. It is not, and the difference matters:

- The MLE warm start is **not optional** in this project (§5.4) and needs one
  target per state. If the policy picks the site, a molecule with *k* open
  sites has up to *k!* valid trajectories, so you need a canonical ordering to
  choose a teacher-forcing target anyway — the ordering problem comes straight
  back, having also cost you a larger action space.
- GRPO standardizes rewards **within a group**. If one molecule is reachable by
  many trajectories it can appear twice in a group with different log-probs,
  and `group_advantages` is then standardizing over something that is not a
  clean sample of distinct candidates.
- Action space is `|vocab| + 2` instead of `|open sites| x |compatible
  fragments|`, which matters at 1572 BBBP training molecules.

CAP keeps most of the expressiveness: the policy can decline to grow a site, it
simply cannot reorder them.

**Two implementation notes, both places where this goes quietly wrong:**

1. Rank open sites by a property of the *partial* molecule (RDKit canonical
   rank of each dummy's anchor atom is the natural choice), and recompute the
   ranking each step. Canonical ranks shift as atoms are added, so a queue
   built once at the start drifts out of agreement with itself.
1b. **The incoming fragment's slot is part of the action, not a tie-break.**
   An earlier revision of this document said "lowest compatible dummy wins"
   and marked it as a shortcut to be measured. It was measured and it is
   lossy: reconstruction over BBBP was **63.5%** under that rule and **81.0%**
   once the slot became part of the action. Do not reintroduce it.
   (Stripping stereo at the corpus boundary took the remaining 81.0% to
   **2039/2039 exact** -- all 387 residual failures were stereo-only. That is
   free on the reward side: the featurizer has no chirality features, and
   `C(m)` on a stereo and a stereo-stripped molecule is bit-identical,
   verified.)

2. Linearize the training molecules with **the identical rule**. Decompose a
   real molecule with `src.brics.decompose`, then replay the assembly choosing
   the queue head each step and taking the fragment that molecule actually has
   there. If the rule used to linearize differs from the rule used at sampling
   time by even a tie-break, the warm start teaches a policy that the sampler
   cannot follow — and it fails as mediocre validity, not as an error.
3. **The root fragment needs a canonical rule too, and it is easy to miss.**
   Site ordering alone does not make a trace unique: a molecule with *k*
   fragments has *k* candidate roots. Measured on BBBP, **336 of 397 molecules
   have more than one fragment** (mean 4.1, max 19), so without a root rule
   most of the corpus has several valid traces and the MLE target is ambiguous
   again — the same problem as site choice, through a different door.
   Canonicalize the molecule *before* `decompose` and take `frags[0]`: after
   canonicalization, atom order is canonical and `GetMolFrags` orders by first
   atom index, so `frags[0]` is the fragment holding canonical atom 0. That is
   deterministic — but it depends entirely on the canonicalization happening
   first, so assert it rather than leaving it implicit.

The entire reason BRICS justifies this rebuild is that assembly is valid by
construction, which buys validity, a compact action space and a
synthesizability floor in one move.

**"Valid by construction" is narrower than it sounds — measured.** Type
compatibility plus an explicit bond order makes every junction legal, so
connectivity and valence are guaranteed. **Aromaticity is not**: joining two
type-compatible aromatic fragments can produce a ring system RDKit cannot
kekulize. Measured sanitization rates: untrained policy **95.4%**, MLE warm
start **100%**, trained policy **100%**, and every inspected failure was a
`KekulizeException`, never a valence error. So the floor is ~95% rather than
100%, the warm start closes the gap, and unkekulizable molecules route through
the invalidity hurdle like any other — the correct direction, and one more
reason the warm start is not optional. Generating an atom-level graph first and
decomposing it afterwards pays the full cost of the rebuild while keeping every
problem it was meant to solve: invalid graphs still reachable, no fragment
action space, and atom spam still available to the policy. It sounds like it
gives you both representations. It gives neither guarantee.

### 4.2 Stereochemistry is dropped. Do not build a chiral-pool reward.

Measured on this repo's data, 2026-10-10:

| | BBBP | B3DB |
|---|---|---|
| ≥1 assigned stereocenter | 31.2% | 54.6% |
| **genuine opposite-label stereoisomer pairs** | **0** | **57 groups / 7805 molecules** |
| specified-vs-unspecified annotation artifacts | 3 | 73 |

There is plenty of stereo *annotation* and almost no stereo *signal*. The
featurizer has no chirality features and will not get them.

**The trap, specifically.** `load_frozen_classifier` defaults to
`dataset="bbbp"`, and `train_gan.py` uses that default. A chiral-pool generator
scored by the BBBP ensemble is being rewarded by a model with **zero**
discriminating examples. You would get a flat reward curve and conclude
"stereochemistry doesn't help", when what actually happened is the reward
cannot see it. Either run any stereo arm against the **b3db** ensemble, or do
not run it. Decide before building the chiral pool, not after.

Emitting defined stereochemistry is still fine as a *correctness* property —
`decompose` preserves it, so fragments carry it for free. Just do not expect the
reward to select on it.

### 4.3 One featurization path, already consolidated

There used to be four entry points on generated SMILES: `reward.py`
canonicalized first, `train_gan.py` featurized the raw generator output in three
places. They agreed only by accident. They now all route through
`featurize.graphs_from_smiles(list)`, which canonicalizes and sanitizes once.

**Keep it that way.** If your new generator emits SMILES through a different
path, the reward and the discriminator start seeing different molecules, and it
presents as a training instability rather than as a bug.

---

## 5. What you own, and what you must not touch

**Yours:** `generator.py`, `train_gan.py`, `grpo.py`, `diversity.py`,
`postfilter.py`, `analyze_drift.py`, `kl_sweep.py`, and whatever new modules the
BRICS assembly engine needs.

**System-1's — a change here silently breaks the contract on the other
machine:** `featurize.py`, `split.py`, `models.py`, `models_edge.py`,
`models_hybrid.py`, `brics.py`, `train.py`.

If you need something changed in those, say so rather than editing — the whole
point of `FEATURE_ID` is that a divergence is loud, but it is only loud at the
moment someone reloads a checkpoint, which may be hours after the edit.

Suggested build order, none of which depends on the classifier:

1. BRICS assembly engine — attachment bookkeeping, fragment compatibility by
   BRICS type, bond formation, the **deterministic site queue** of §4.1 (so one
   molecule has one trajectory, not *k!*), CAP, terminal capping.
2. Hierarchical policy: core selection, then one categorical over
   `vocab + {CAP}` per dequeued site. The policy does **not** select the site
   — see §4.1.
3. **Terminal-only reward.** Never score a molecule with open `[*]` tags; the
   classifier's prediction on a partial graph is meaningless.
4. MLE warm start on assembly traces decomposed from real molecules. The warm
   start is not optional — from a uniform policy every group is all-invalid,
   the group std is zero, and `group_advantages` correctly returns no gradient.

---

## 6. Things in this repo that will bite you

- **`results_v1/`** is the pre-contract archive. Nothing reads it and
  `run_contract` refuses it. Do not "fix" a load error by pointing at it.
- **`results/generator/`** was deliberately left in place. Those checkpoints are
  self-describing (they store their own vocab and dims) and feature-independent,
  so your existing warm starts still load.
- **D-MPNN was removed** (`models_dmpnn.py`, `train_dmpnn.py`) along with its
  references in `eval_external_holdout`, `paired_analysis` and `collect_all`.
- **The ensemble must share a fold.** `reward._self_check` asserts the
  `test_preds.npz` `idx` arrays are element-wise identical across `gin`, `gat`
  and `gine`. The seed reseeds the split, so members from different seeds have
  trained on each other's held-out molecules.
- **Reward hacking here is measured, not hypothetical.** Unguarded, scaffold
  diversity falls 0.8 → 0.07–0.14 over 200 steps in all 3 seeds while validity
  stays at 100%. Term ablation found **SA**, not the classifier, drives it.
  `kl_coef=0.3` and `min_heavy_atoms=10` are defaults because they are the only
  two interventions that survived validation.
- **Structural screening belongs in `postfilter.py`, after training.** A
  reactive-group penalty inside the reward taught the generator the checklist,
  not the chemistry: the targeted SMARTS it was penalized on fell 0.339 → 0.203
  while BRENK stayed flat.

---

## 7. Usage protocol

Both Claude sessions bill the same Pro account with 5-hour windows. Training
burns CPU, not tokens; writing code burns tokens. **Run one session at a time.**
System-1 used the overnight window to land this contract and launch its sweep;
that sweep needs no session attached.

---

## 8. Checkpoint status — the reward ensemble is on a favourable split

The 10-seed sweep is **complete** (220 runs). The `gin`/`gat`/`gine` bbbp
seed-0 checkpoints you score against are real runs under the frozen contract
(`feature_id=c0c3bbb8`, `norm=layer`, `split=murcko+tanimoto@0.6`), identical in
configuration to every sweep run. There is no pending swap.

**All ten seeds are available.** Every seed 0-9 has `gin`, `gat` and `gine`
present with **element-wise identical fold indices** (verified), so any single
seed is a legitimate ensemble. Mean ROC-AUC of the three members, by seed:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean | **0.910** | 0.867 | 0.901 | 0.885 | 0.873 | 0.838 | 0.855 | 0.914 | 0.827 | 0.818 |

Seed 0 is on the high side — 2nd of 10, against a range of 0.818-0.914.

**Default to seed 0 anyway, and do not pick a seed by this table.** Choosing
the reward by a test statistic is selecting your instrument on test-fold
information; "a seed nearer the mean" is as much a selection as "the best
seed". Seed 0 is the right default precisely because it was fixed before
anyone looked at these numbers.

An earlier revision of this section said seed 0 makes `C(m)` "more confident
than the model is on average". That was loose and is withdrawn: a high test
ROC-AUC means stronger *ranking on that fold*, not higher confidence on novel
generated chemistry, and the two are not the same thing. The sound caution is
narrower:

1. **Never quote a seed-0 number as the classifier's accuracy.** The headline
   figures are the 10-seed means in `results/master_comparison.csv`.
2. **Your reward is one draw from a spread of 0.818-0.914.** A generation
   result that holds only under one reward seed is a property of that
   classifier, not of your method.

So: **expose `--reward-seed` (default 0) and treat it as a robustness axis,
not a tuning knob.** `load_frozen_classifier` already takes `seed=`; it just
needs threading through `train_gan.py`'s CLI and into `config.json`. Run the
headline generation result at **two or more reward seeds** and report both.
That is the same discipline `paired_analysis` enforces on the classifier side,
and it costs one extra run rather than an argument about which seed is fair.

**One seed per ensemble is structural, not laziness.** The seed reseeds the
scaffold split, so members must share a fold or they have trained on each
other's held-out molecules — `reward._self_check` asserts the
`test_preds.npz` `idx` arrays match element-wise. Vary the seed *between*
runs; never mix seeds *within* an ensemble.

**If the norm default flips** (see §2 and `documentation/limitations.md` 5.0),
these three get retrained and you re-pull. `run_contract` replays whichever
norm a checkpoint recorded, so nothing breaks silently either way.

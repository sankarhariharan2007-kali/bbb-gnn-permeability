# The pipeline, and the evidence for every part of it

Everything below is measured on this repo at 10 seeds, paired on seed. Each
seed is a different scaffold split, so every `±` includes split variance.
`documentation/limitations.md` holds the failures in full.

---

## 1. There are two pipelines, because the classifier has two jobs

They differ in exactly one line, and the reason is not accuracy.

### 1a. Standalone classifier — the best numbers

```bash
python -m src.train_hybrid --model gcn --dataset bbbp --seed 0 --norm batch
```

```
SMILES
  → clean: canonical-SMILES dedup, drop label conflicts  (datasets.py)
  → RDKit atom graph, 39-dim atom features, no chirality (featurize.py)
  → split: Murcko scaffold, acyclics by Tanimoto cluster (split.py)
  → 3-layer GNN, hidden 128, BatchNorm, dropout 0.3      (models_hybrid.py)
  → readout: global mean + max over atoms
  → concat 8 RDKit descriptors at the head               (hybrid_features.py)
  → 2-layer MLP head → 1 logit
  → threshold tuned on validation only                   (evaluate.py)
```

**BBBP 0.921 · B3DB 0.905 · Wang (novel chemistry) 0.850**

### 1b. Classifier as a frozen reward — use GraphNorm

```bash
python -m src.train_hybrid --model gcn --dataset bbbp --seed 0   # norm=graph is the default
```

**BBBP 0.909 · B3DB 0.900 · Wang 0.839**

Costs ~0.010 ROC-AUC. Buy it anyway when the checkpoint will be frozen and used
as a GRPO reward: `BatchNorm1d` makes a molecule's logit depend on which other
molecules share its batch, so the same molecule earns a different reward
depending on its group and the reward stops being a function of the molecule.
Measured by `python -m src.models` on a 4-graph batch in train mode: batch
drifts **3.97e-02**, graph 6.0e-08, layer 3.7e-09. `eval()` guards contain this
in practice, but GraphNorm removes the failure mode rather than guarding it.

Operator choice is free: GCN / SAGE / GAT land within 0.01 of each other. Use
GCN, it is the cheapest.

---

## 2. What every part is worth

Pooled over 4 operators x 2 datasets x 10 seeds, paired, vs the plain base:

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

Descriptors and BatchNorm are the only two positive effects, and they stack:
predicted +0.0173 if independent, measured **+0.0195 ± 0.0024**.

---

## 3. The one sentence that explains the table

**BBB permeability behaves like a property of bulk composition, not of
long-range topology.**

Everything that adds compositional or whole-molecule information helps:
descriptors, a wide atom-level readout, batch statistics. Everything that adds
structural reasoning is neutral or harmful: depth, recurrence, cross-fragment
connectivity, fragment hierarchy, bond features.

That is why LightGBM on 8 descriptors reaches 0.880, why the four GNN operators
are statistically indistinguishable, and why both HRM and the motif model
failed. It is a more useful statement about this problem than any row above.

---

## 4. Report ~0.85, not ~0.92

| | BBBP (in-distribution) | Wang (n=55, novel) |
|---|---|---|
| hybrid + BatchNorm | 0.921 | 0.850 |
| edge ablation | 0.893 | **0.853** |
| base | 0.895 | 0.846 |
| hybrid + GraphNorm | 0.909 | 0.839 |

Adenot (n=59) is saturated at ~0.99 for every family and discriminates nothing.
Wang is the informative set, and **everything falls to ~0.85 there.**

**In-distribution ranking does not predict generalization.** Across 11 models
the correlation between BBBP and Wang score is **r = −0.45 (p = 0.16)** —
suggestive rather than established, but there is certainly no evidence *for*
transfer. Plain hybrid is 1st-3rd in-distribution and 7th-9th on Wang.

BatchNorm is the exception that makes 1a defensible: its gain is **+0.0101
in-distribution and +0.0108 on Wang**, the same direction and magnitude. It
moves the hybrid family from last to second on Wang.

---

## 5. Protocol, and why it is stricter than the published numbers

**The split clusters acyclic molecules.** `MurckoScaffoldSmiles` returns `''`
for any ring-free molecule, so they used to collapse into one group that,
being large, always landed in train — 95/1965 BBBP and 311/7805 B3DB molecules
were **never in a test fold**. They are now grouped by Morgan/Tanimoto (Butina,
0.6). This costs ~0.037 ROC-AUC on BBBP and is the reason the headline is lower
than this repo's earlier figures. `split.group_keys` is the single definition of
group identity and both the splitter and the leak assertion call it.

**Ten seeds, not three.** The true seed-to-seed spread is 0.03–0.06, not the
~0.015 three seeds suggested. Even at n=10, `paired_analysis` separates only
**13 of 55 model pairs** at 95%. Do not rank operators on this data.

**Numbers are not comparable to the 0.65–0.70 BBBP literature figures**, which
come from the size-sorted DeepChem protocol. That protocol is degenerate here —
BBBP's back half is all class 1, so val/test come out 100% positive and ROC-AUC
is undefined.

---

## 6. Reproducing it

**The whole pipeline, including the generator half, is
`documentation/full_pipeline.md` — use that.** `python -m src.pipeline --list`
shows what would run and what is already on disk. What follows is the
classifier half alone, for when that is all you want.

```bash
python -m src.datasets          # build caches, verify splits
python -m src.run_all           # base grid, 80 runs
python -m src.train_hybrid --run-all              # GraphNorm arm
python -m src.train_hybrid --run-all --norm batch # BatchNorm arm, i.e. 1a
python -m src.train_edge --run-all
python -m src.eval_external_holdout
python -m src.collect_all       # -> results/master_comparison.csv
python -m src.paired_analysis --all-sources
```

`--norm batch` routes to `results/hybrid_batch/`. Until that flag existed the
arm holding the best numbers in this repo could not be regenerated by any
committed command — it was produced ad hoc while `eval_external_holdout` read
it.

Every module self-checks with no arguments; `python -m src.reward` is the gate
that the classifier and the generator agree. `featurize.FEATURE_ID`
(`c0c3bbb8`) is recorded in every `metrics.json` and validated on checkpoint
load, so a mismatched classifier fails loudly rather than scoring silently.

Result trees: `results/` (GraphNorm, current), `results/hybrid_batch/`
(1a above), `results_layernorm/` and `results_v1/` (superseded, kept for the
comparisons that cite them).

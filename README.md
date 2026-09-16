# BBB Permeability Prediction with Graph Neural Networks

Binary classification of blood–brain-barrier permeability from SMILES alone.
Four message-passing architectures — **GCN**, **GraphSAGE**, **GIN**, **GAT** —
are compared on two datasets under a scaffold split, then combined into ensembles.

```
SMILES → RDKit molecular graph → GNN → P(BBB permeable)
```

The only input feature is the SMILES string. No descriptors, no fingerprints —
every feature the models see is derived from the RDKit molecular graph.

## Setup

```bash
.venv/bin/pip install -r requirements.txt
```

PyTorch Geometric ≥ 2.3 uses `torch.scatter_reduce` natively, so the compiled
`torch-scatter` / `torch-sparse` extensions are **not** required.

## Running

```bash
python -m src.featurize                              # featurizer smoke test
python -m src.models                                 # forward-pass shape check
python -m src.datasets                               # build caches, verify splits
python -m src.train --model gcn --dataset bbbp --seed 0   # one run
python -m src.run_all                                # all 24 runs
python -m src.ensemble                               # combine predictions
jupyter lab notebooks/results.ipynb                  # tables and figures
```

Runs default to CPU — these graphs are small enough that MPS kernel-launch
overhead makes it slower. `--device mps` is available to test that.

## Layout

| path | role |
|---|---|
| `src/featurize.py` | SMILES → PyG `Data`; 39-dim atom features, bond features stored but unused |
| `src/datasets.py` | load, clean, deduplicate, and cache both datasets |
| `src/split.py` | Bemis–Murcko scaffold split + leak assertions |
| `src/models.py` | one skeleton, four convolution operators |
| `src/train.py` | one `(model, dataset, seed)` run |
| `src/evaluate.py` | metrics; threshold chosen on validation only |
| `src/run_all.py` | 4 models × 2 datasets × 3 seeds |
| `src/ensemble.py` | soft vote, rank average, logistic stacking |
| `notebooks/results.ipynb` | EDA, results tables, ROC/PR curves |

Results land in `results/<dataset>/<model>/seed<N>/`, aggregated into
`results/all_runs.csv` and `results/summary.csv`.

## Design decisions worth knowing

**One skeleton, four operators.** Depth, hidden width, normalization, readout,
classifier head and training loop are identical across all four models. Only the
convolution differs, so a performance gap is attributable to the aggregation
scheme rather than to incidental capacity differences. `GATConv` uses
`hidden // heads` channels per head so its concatenated output matches the
others rather than inflating fourfold.

**The scaffold split is balanced, not size-sorted.** The classic DeepChem
`ScaffoldSplitter` sorts scaffold groups by size and fills train first, leaving
val/test with only singleton scaffolds. On `BBBP.csv` that is degenerate: the
file is ordered so its entire back half is class 1, and both folds come out
**100% positive**, making ROC-AUC undefined. The balanced split (Chemprop-style)
shuffles scaffold groups under a per-seed RNG instead. `src/split.py` keeps the
size-sorted variant behind `balanced=False` so the artifact can be reproduced.

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
architecture comparison with an input-signal advantage. Swapping `GINConv` for
`GINEConv` is the natural follow-up ablation.

**The datasets overlap.** B3DB aggregates BBBP as one of its sources, and some
shared molecules carry contradictory labels. The two are therefore trained and
evaluated independently; training on one and testing on the other would be a
leaked evaluation.

## Data cleaning

| | rows in | invalid SMILES | dupes merged | label conflicts | rows out |
|---|---|---|---|---|---|
| BBBP | 2050 | 11 | 54 | 20 | **1965** (76.3% positive) |
| B3DB | 7807 | 2 | 0 | 0 | **7805** (63.5% positive) |

Deduplication is on the *canonical* SMILES, not the raw string. Molecules whose
duplicate copies disagree on the label are dropped entirely rather than guessed.

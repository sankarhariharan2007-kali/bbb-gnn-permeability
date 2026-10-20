# Paper (RECOMB 2027 submission)

The paper is `recomb2027_main.pdf`; its LaTeX source (`main.tex`) was removed from the repository.

| file | what it is |
|---|---|
| `recomb2027_main.pdf` | main paper (title page + 10 pages + references) |
| `recomb2027_supplement.pdf` | supplementary material (single PDF) |
| `supp.tex` | LaTeX source of the supplement (`supp.tex` is generated from `supp_template.tex` by `make_supp.py`) |
| `figs/` | every figure of the paper (PDF, plus `shortlist.png`) |
| `make_figs.py` | regenerates all figures from `results/` |
| `make_supp.py` | regenerates the data tables of the supplement from `results/` |
| `scaffold_overlap.py` | scaffold-overlap statistics quoted in the paper (23.5% / 34.8%) |
| `check_format.py` | audits a PDF: page size, margins, font sizes, per-page fill |
| `data/seed{0,1,2}_history.csv` | original unconstrained-baseline histories (commit `e0e5da8b`), used by the collapse figure: `results/gan/seed*/history.csv` were later overwritten by a regenerated run |

Rebuild everything from the repository root (needs `matplotlib`, `scipy`, `rdkit`, `pandas`, and a LaTeX install):

```bash
python paper/make_figs.py
python paper/make_supp.py
python paper/scaffold_overlap.py
cd paper && pdflatex supp.tex && pdflatex supp.tex
```

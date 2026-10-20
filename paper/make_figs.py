"""Regenerate every figure of the paper from the repository (branch ranking-v2).
Usage: python paper/make_figs.py [repo]   (default: repository root)   -> writes figs/*.pdf, figs/shortlist.png
Figures are drawn at their final printed width so that every label is >= 8 pt on the page.
The collapse figure uses the ORIGINAL baseline histories (commit e0e5da8b, saved in data/): the files
results/gan/seed*/history.csv on the branch were later overwritten by a regenerated run."""
import sys, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

REPO = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent
R = REPO / "results"; HERE = pathlib.Path(__file__).parent; OUT = HERE / "figs"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "axes.titleweight": "bold", "axes.linewidth": 0.7})
BLUE, GREEN, ORANGE, RED, GREY = "#2b6cb0", "#2f855a", "#dd8a1d", "#c53030", "#8a8f98"
FAM_COL = {"LightGBM": GREY, "base GNN": BLUE, "edge-aware": GREEN, "hybrid, GraphNorm": ORANGE, "hybrid, BatchNorm": RED}
W = 6.5  # text width in inches

# ---------------------------------------------------------------- classifier
m = pd.read_csv(R / "master_comparison.csv")
fam = {"descriptor_gbm": "LightGBM", "base_gnn": "base GNN", "edge_ablation": "edge-aware", "hybrid": "hybrid, GraphNorm", "hybrid_batch": "hybrid, BatchNorm"}
m["family"] = m.source.map(fam)
m = m[m.model != "lightgbm_raw"]                       # keep the 8-descriptor variant (raw + 2 MPO features)
m["fo"] = m.family.map({f: i for i, f in enumerate(FAM_COL)})
bb = m[m.dataset == "bbbp"].sort_values(["fo", "model"]).reset_index(drop=True)
b3 = m[m.dataset == "b3db"].set_index(["family", "model"])
runs = pd.read_csv(R / "external_holdout_eval" / "external_holdout_runs.csv")
w = runs[(runs.holdout == "wang") & (runs.train_dataset == "bbbp")].groupby(["family", "model"]).roc_auc.mean()
s2f = {"base": "base GNN", "edge_ablation": "edge-aware", "hybrid": "hybrid, GraphNorm", "hybrid_batch": "hybrid, BatchNorm"}
wang = {(s2f[f], mm): v for (f, mm), v in w.items()}

fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.15), gridspec_kw={"width_ratios": [1.2, 1]})
ypos = np.array([len(bb) - i - 0.7 * g for i, g in enumerate(bb.fo.values)])
for yy, (_, r) in zip(ypos, bb.iterrows()):
    c = FAM_COL[r.family]
    a.errorbar(r.test_roc_auc_mean, yy + 0.14, xerr=r.test_roc_auc_std, fmt="o", ms=3.8, color=c, elinewidth=1.1, capsize=1.5)
    k = (r.family, r.model)
    if k in b3.index:
        q = b3.loc[k]; a.errorbar(q.test_roc_auc_mean, yy - 0.14, xerr=q.test_roc_auc_std, fmt="s", ms=3.3, mfc="white", color=c, elinewidth=0.9, capsize=1.5)
a.set_yticks(ypos); a.set_yticklabels([r.model.replace("lightgbm_raw+mpo_5of6", "LightGBM") for _, r in bb.iterrows()])
for f in list(FAM_COL)[1:]:
    idx = bb.index[bb.family == f]; a.text(0.672, ypos[idx].mean(), f, color=FAM_COL[f], fontsize=8, fontweight="bold", va="center")
a.set_xlim(0.665, 1.0); a.set_xlabel("test ROC-AUC (mean ± sd, 10 seeds)")
a.set_title("In-distribution accuracy"); a.plot([], [], "o", color="k", ms=3.8, label="BBBP"); a.plot([], [], "s", mfc="white", color="k", ms=3.3, label="B3DB")
a.legend(loc="upper left", frameon=False, borderaxespad=0.1); a.xaxis.grid(True, lw=0.3, alpha=0.5); a.set_axisbelow(True)
D = pd.DataFrame([(r.family, r.model, r.test_roc_auc_mean, wang[(r.family, r.model)]) for _, r in bb.iterrows() if r.family != "LightGBM"], columns=["family", "model", "id", "wang"])
G = D[D.family != "hybrid, BatchNorm"]; H = D[D.family == "hybrid, BatchNorm"]
rr, pp = stats.pearsonr(G.id, G.wang); sl, ic = np.polyfit(G.id, G.wang, 1)
xs = np.linspace(G.id.min() - .002, G.id.max() + .002, 10); b.plot(xs, sl * xs + ic, color=GREY, lw=1, ls="--")
for f in ["base GNN", "edge-aware", "hybrid, GraphNorm"]:
    g = G[G.family == f]; b.scatter(g.id, g.wang, s=22, color=FAM_COL[f], label=f, zorder=3)
b.scatter(H.id, H.wang, s=26, facecolor="white", edgecolor=RED, lw=1.2, label="hybrid, BatchNorm\n(not in $r$)", zorder=3)
b.text(0.04, 0.05, f"$r={rr:.2f}$, $p={pp:.2f}$", transform=b.transAxes)
b.set_xlabel("BBBP test ROC-AUC"); b.set_ylabel("Wang ROC-AUC ($n=55$)"); b.set_title("Rank does not transfer")
b.legend(loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.1); b.set_ylim(0.82, 0.935); b.grid(True, lw=0.3, alpha=0.5); b.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "classifier.pdf"); plt.close(fig)
print("classifier: r=%.3f p=%.3f n=%d" % (rr, pp, len(G)))

# ---------------------------------------------------------------- components (documentation/full_pipeline.md, section 6)
comp = [("descriptors + BatchNorm (together)", 0.0195, 0.0024, "pos"), ("8 RDKit descriptors", 0.0099, 0.0019, "pos"),
        ("BatchNorm over GraphNorm", 0.0074, 0.0023, "pos"), ("BRICS motif graph, intra + atom readout", 0.0064, 0.0059, "null"),
        ("virtual node", 0.0003, 0.0023, "null"), ("edge (bond) features", 0.0, None, "null"),
        ("BRICS motif graph, added on top", -0.0042, 0.0026, "null"), ("long-range atom connectivity", -0.0194, None, "neg"),
        ("HRM two-timescale recurrence", -0.0381, 0.0059, "neg"), ("BRICS motif graph replacing atoms", -0.1161, 0.0107, "neg")]
cc = {"pos": GREEN, "null": GREY, "neg": RED}
fig, ax = plt.subplots(figsize=(W, 2.2)); yy = np.arange(len(comp))[::-1]; XMIN = -0.05; LBL = 0.03
for y_, (lab, d, s_, v) in zip(yy, comp):
    clipped = d < XMIN
    ax.barh(y_, max(d, XMIN), color=cc[v], height=0.62, alpha=0.9)
    if s_ and not clipped: ax.errorbar(d, y_, xerr=s_, color="k", capsize=2, lw=0.9)
    if clipped: ax.annotate("", xy=(XMIN, y_), xytext=(XMIN + 0.012, y_), arrowprops=dict(arrowstyle="->", color="white", lw=1))
    txt = ("\u2248 0 (no SEM)" if lab.startswith("edge") else f"{d:+.4f}" + (f" \u00b1 {s_:.4f}" if s_ else " (no SEM)")) + (" \u2014 off scale" if clipped else "")
    ax.text(LBL, y_, txt, va="center", fontsize=8)
ax.axvline(0, color="k", lw=0.8); ax.axvline(0.0173, color=GREEN, lw=0.9, ls=":")
ax.text(0.0166, len(comp) - 0.4, "independent prediction (+0.0173)", fontsize=8, color=GREEN, va="bottom", ha="right")
ax.set_yticks(yy); ax.set_yticklabels([c[0] for c in comp]); ax.set_xlim(XMIN, 0.092); ax.set_xticks([-0.04, -0.02, 0, 0.02])
ax.set_xlabel("paired \u0394 ROC-AUC (4 operators \u00d7 2 datasets \u00d7 10 seeds)", loc="left"); ax.set_ylim(-0.6, len(comp) + 0.4)
fig.tight_layout(); fig.savefig(OUT / "components.pdf"); plt.close(fig)

# ---------------------------------------------------------------- collapse (ORIGINAL baseline histories)
H0 = [pd.read_csv(HERE / "data" / f"seed{s}_history.csv") for s in range(3)]; cols = [BLUE, ORANGE, GREEN]
sm = lambda x: x.rolling(9, center=True, min_periods=1).mean()
def cross(d, c):
    base = d[c].iloc[:10].mean(); i = np.where(sm(d[c]).values < 0.8 * base)[0]; return int(d.step.iloc[i[0]])
fig, axs = plt.subplots(1, 3, figsize=(W, 1.95))
for ax, (c, t, mark) in zip(axs, [("scaffold_frac", "scaffold fraction", True), ("unique_smiles_frac", "distinct-SMILES fraction", True), ("reward_mean", "mean reward", False)]):
    for s, d in enumerate(H0):
        ax.plot(d.step, sm(d[c]), color=cols[s], lw=1.3, label=f"seed {s}")
        if mark: cs = cross(d, c); ax.plot(cs, sm(d[c]).iloc[cs], "v", color=cols[s], ms=5)
    ax.set_title(t); ax.set_xlabel("GRPO step"); ax.grid(True, lw=0.3, alpha=0.5)
axs[0].set_ylim(0, 1); axs[1].set_ylim(0, 1.03); axs[0].legend(frameon=False, loc="upper right", handlelength=1.2)
axs[1].text(0.03, 0.04, "\u25bc 20% below start", transform=axs[1].transAxes)
fig.tight_layout(); fig.savefig(OUT / "collapse.pdf"); plt.close(fig)
print("collapse crossings scaffold:", [cross(d, "scaffold_frac") for d in H0], "unique:", [cross(d, "unique_smiles_frac") for d in H0])

# ---------------------------------------------------------------- term ablation
t = pd.read_csv(R / "term_ablation" / "term_ablation.csv")
names = [("none", "none"), ("D", "$-D$"), ("Q", "$-Q$"), ("C", "$-C$"), ("S", "$-S$")]
fig, ax = plt.subplots(figsize=(3.2, 2.15))
for i, (k, lab) in enumerate(names):
    v = t[t.term_dropped == k].sort_values("seed").scaffold_frac.values
    ax.bar(i, v.mean(), color=GREY if k == "none" else (GREEN if k == "S" else BLUE), alpha=0.8, width=0.62)
    ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color="k", s=9, zorder=3)
ax.axhline(0.30, color=RED, ls="--", lw=0.9); ax.text(-0.45, 0.315, "collapse screen", color=RED, fontsize=8, va="bottom")
ax.set_xticks(range(len(names))); ax.set_xticklabels([n[1] for n in names]); ax.set_ylabel("endpoint scaffold fraction"); ax.set_ylim(0, 0.9)
ax.grid(axis="y", lw=0.3, alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "ablation.pdf"); plt.close(fig)
print("ablation d_scaffold means:", t.groupby("term_dropped").d_scaffold_frac.mean().round(3).to_dict())

# ---------------------------------------------------------------- rejected scores (gate arm)
d = pd.read_csv(R / "pubchem" / "rejected_scores.csv"); g = d[d.arm == "cgate"]
s = g.groupby("reason").c_all.agg(["mean", "sem", "count"])
ordr = ["undersized", "in training data", "survivor", "catalogue alert", "targeted alert", "structural defect"]; s = s.loc[ordr]
short = ["under-\nsized", "in\ntrain.", "survivor", "cata-\nlogue", "tar-\ngeted", "struct.\ndefect"]
fig, ax = plt.subplots(figsize=(3.2, 2.15))
ax.bar(range(len(s)), s["mean"], yerr=1.96 * s["sem"], color=[RED, GREY, GREEN, BLUE, BLUE, BLUE], alpha=0.85, capsize=2, width=0.68)
for i, mu in enumerate(s["mean"]): ax.text(i, mu + 0.035, f"{mu:.2f}", ha="center", fontsize=8)
ax.axhline(s.loc["survivor", "mean"], color=GREEN, ls=":", lw=0.9)
ax.set_xticks(range(len(s))); ax.set_xticklabels(short, fontsize=8); ax.set_ylabel("mean reward-ensemble $C$"); ax.set_ylim(0, 1.12)
ax.grid(axis="y", lw=0.3, alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "rejected.pdf"); plt.close(fig)
print("rejected (gate arm):", s["mean"].round(3).to_dict(), "n:", s["count"].to_dict())

# ---------------------------------------------------------------- gate, two corpora
def load(p): return pd.read_csv(REPO / "results" / p / "screened.csv")
def st(a, gt):
    sv = lambda x: int((x.passes & ~x.in_training_flat).sum())
    return {"pooled distinct\noutput": 100 * (len(gt) / len(a) - 1), "survivors of\nthe screen": 100 * (sv(gt) / sv(a) - 1),
            "training molecules\nreproduced": 100 * (gt.in_training_flat.sum() / a.in_training_flat.sum() - 1)}
bbbp = st(load("candidates/brics10"), load("candidates/brics_cgate")); b3d = st(load("candidates/b3db_a"), load("candidates/b3db_cgate"))
fig, ax = plt.subplots(figsize=(4.3, 2.3)); keys = list(bbbp); x = np.arange(len(keys)); wd = 0.36
ax.bar(x - wd / 2, [bbbp[k] for k in keys], wd, color=BLUE, label="BBBP"); ax.bar(x + wd / 2, [b3d[k] for k in keys], wd, color=ORANGE, label="B3DB")
for xi, k in zip(x, keys):
    for off, dd in ((-wd / 2, bbbp), (wd / 2, b3d)):
        v = dd[k]; ax.text(xi + off, v + (2 if v >= 0 else -3), f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_ylabel("gate vs. $C$-as-term (%)")
ax.set_ylim(-38, 92); ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y", lw=0.3, alpha=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "gate.pdf"); plt.close(fig)
print("gate bbbp", {k.replace("\n", " "): round(float(v), 1) for k, v in bbbp.items()}, "b3db", {k.replace("\n", " "): round(float(v), 1) for k, v in b3d.items()})

# ---------------------------------------------------------------- baselines
bl = pd.read_csv(R / "pubchem" / "generator_baselines.csv"); bl = bl[bl.arm != "random policy"].reset_index(drop=True)
bl["lab"] = ["W", "A", "G", "R"]; real = bl[bl.arm.str.startswith("real")].iloc[0]
fig, axs = plt.subplots(1, 4, figsize=(W, 1.95))
for ax, (c, ttl) in zip(axs, [("c", "$C$ (in reward)"), ("qed", "QED (in reward)"), ("mpo", "CNS-MPO (not)"), ("scaffold", "scaffold frac.")]):
    ax.bar(range(len(bl)), bl[c], color=[GREY, BLUE, GREEN, "#444"], alpha=0.9, width=0.72); ax.axhline(real[c], color=RED, ls="--", lw=0.9)
    for i, v in enumerate(bl[c]): ax.text(i, 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="white", rotation=90)
    ax.set_title(ttl); ax.set_xticks(range(len(bl))); ax.set_xticklabels(bl.lab, fontsize=8); ax.set_ylim(0, 1.0)
axs[0].set_ylabel("8 groups of 64")
fig.tight_layout(); fig.savefig(OUT / "baselines.pdf"); plt.close(fig)

# ---------------------------------------------------------------- shortlist (RDKit)
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
sl = pd.read_csv(R / "candidates_mpo" / "brics_cgate" / "candidates.csv").head(12)
mols = [Chem.MolFromSmiles(x) for x in sl.smiles]
leg = [f"{i+1}: C={r.c_score:.3f} MPO={r.cns_mpo:.2f} QED={r.qed:.2f}" for i, r in enumerate(sl.itertuples())]
dr = rdMolDraw2D.MolDraw2DCairo(1440, 600, 360, 300); dr.drawOptions().legendFontSize = 30
dr.DrawMolecules(mols, legends=leg); dr.FinishDrawing(); open(OUT / "shortlist.png", "wb").write(dr.GetDrawingText())
print("shortlist rows:", len(sl))

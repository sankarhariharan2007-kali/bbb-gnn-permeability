"""Build supp.tex (Supplementary Material) from the repository (branch ranking-v2).  usage: python paper/make_supp.py [repo]   (default: repository root)"""
import sys, pathlib, json, numpy as np, pandas as pd
REPO = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent; R = REPO / "results"
HERE = pathlib.Path(__file__).parent
f3 = lambda x: f"{x:.3f}"
def pm(m, s): return f"${m:.3f}\\pm{s:.3f}$"
esc = lambda s: s.replace("_", "\\_")

# ---- S1 per-model results ---------------------------------------------------------------
mc = pd.read_csv(R / "master_comparison.csv")
runs = pd.read_csv(R / "external_holdout_eval" / "external_holdout_runs.csv")
fam = [("descriptor_gbm", "LightGBM", "LightGBM"), ("base_gnn", "base GNN", "base"), ("edge_ablation", "edge-aware", "edge_ablation"),
       ("hybrid", "hybrid, GraphNorm", "hybrid"), ("hybrid_batch", "hybrid, BatchNorm", "hybrid_batch")]
rows = []
for src, label, hf in fam:
    sub = mc[(mc.source == src)]
    if src == "descriptor_gbm": sub = sub[sub.model == "lightgbm_raw+mpo_5of6"]
    for mdl in sorted(sub.model.unique()):
        r = {}
        for ds in ("bbbp", "b3db"):
            q = sub[(sub.dataset == ds) & (sub.model == mdl)].iloc[0]; r[ds] = pm(q.test_roc_auc_mean, q.test_roc_auc_std)
        def ext(h, tr):
            if src == "descriptor_gbm": return "--"
            g = runs[(runs.family == hf) & (runs.model == mdl) & (runs.holdout == h) & (runs.train_dataset == tr)].roc_auc
            return pm(g.mean(), g.std())
        rows.append((label, "LightGBM (8 desc.)" if src == "descriptor_gbm" else mdl.replace("_", "-"), r["bbbp"], r["b3db"], ext("wang", "bbbp"), ext("wang", "b3db"), ext("adenot", "bbbp")))
S1 = ["\\begin{tabular}{@{}llccccc@{}}\\toprule", "family & model & BBBP & B3DB & Wang$\\leftarrow$BBBP & Wang$\\leftarrow$B3DB & Adenot$\\leftarrow$BBBP\\\\\\midrule"]
last = None
for r in rows:
    if last and r[0] != last: S1.append("\\midrule")
    S1.append(f"{r[0] if r[0]!=last else ''} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]}\\\\"); last = r[0]
S1.append("\\bottomrule\\end{tabular}")

# ---- S4 B3DB replication ----------------------------------------------------------------
def L(p): return pd.read_csv(R / p / "screened.csv")
def summ(a, g, n_corpus):
    sv = lambda x: int((x.passes & ~x.in_training_flat).sum())
    return [len(a), len(g), sv(a), sv(g), int(a.in_training_flat.sum()), int(g.in_training_flat.sum()), n_corpus]
bb = summ(L("candidates/brics10"), L("candidates/brics_cgate"), 1965); b3 = summ(L("candidates/b3db_a"), L("candidates/b3db_cgate"), 7805)
def rowfmt(name, v):
    d = lambda a, b: f"{100*(b/a-1):+.1f}\\%"
    return (f"{name} & {v[0]:,} & {v[1]:,} & {d(v[0],v[1])} & {v[2]:,} & {v[3]:,} & {d(v[2],v[3])} & {100*v[4]/v[6]:.1f}\\%$\\to${100*v[5]/v[6]:.1f}\\% \\\\").replace(",", "{,}")
S4 = ["\\begin{tabular}{@{}lccccccc@{}}\\toprule", "corpus & pooled A & pooled gate & change & surv. A & surv. gate & change & corpus covered\\\\\\midrule", rowfmt("BBBP ($\\tau=0.615$)", bb), rowfmt("B3DB ($\\tau=0.476$)", b3), "\\bottomrule\\end{tabular}"]

# ---- S5 term ablation per seed ---------------------------------------------------------
t = pd.read_csv(R / "term_ablation" / "term_ablation.csv")
S5 = ["\\begin{tabular}{@{}llccccl@{}}\\toprule", "dropped & seed & scaffold & $\\Delta$ scaffold & Tanimoto & mean $C$ & collapsed\\\\\\midrule"]
for k in ["none", "D", "Q", "C", "S"]:
    for _, r in t[t.term_dropped == k].sort_values("seed").iterrows():
        dsc = "--" if pd.isna(r.d_scaffold_frac) else f"{r.d_scaffold_frac:+.3f}"
        S5.append(f"{'--' if k=='none' else '$'+k+'$'} & {int(r.seed)} & {r.scaffold_frac:.3f} & {dsc} & {r.tanimoto_dist:.3f} & {r.mean_C:.3f} & {'yes' if r.collapsed else 'no'}\\\\")
    if k != "S": S5.append("\\midrule")
S5.append("\\bottomrule\\end{tabular}")

# ---- S6 screening verdicts per arm --------------------------------------------------------
d = pd.read_csv(R / "pubchem" / "rejected_scores.csv")
order = ["survivor", "in training data", "undersized", "structural defect", "targeted alert", "catalogue alert"]
S6 = ["\\begin{tabular}{@{}lrrrrrr@{}}\\toprule", "& \\multicolumn{2}{c}{arm A} & \\multicolumn{2}{c}{arm B (hybrid-BN reward)} & \\multicolumn{2}{c}{gate arm}\\\\ verdict & $n$ & mean $C$ & $n$ & mean $C$ & $n$ & mean $C$\\\\\\midrule"]
for o in order:
    cells = []
    for arm in ("a", "b", "cgate"):
        q = d[(d.arm == arm) & (d.reason == o)].c_all; cells += [f"{len(q):,}".replace(",", "{,}"), f"{q.mean():.3f}"]
    S6.append(f"{o} & " + " & ".join(cells) + "\\\\")
S6.append("\\bottomrule\\end{tabular}")

# ---- S10 run configuration ---------------------------------------------------------------
c = json.load(open(R / "brics_cgate" / "pseed0" / "config.json"))
keep = ["steps", "group_size", "ppo_epochs", "n_d", "lr_g", "lr_d", "clip_eps", "kl_coef", "invalid_floor", "schedule", "k_anneal", "max_len", "min_heavy_atoms", "min_c", "drop", "reward_family", "reward_seed", "generator", "threads_requested"]
S10 = ["\\begin{tabular}{@{}ll@{}}\\toprule", "setting & value (gate arm, policy seed 0)\\\\\\midrule"]
for k in keep: S10.append(f"\\texttt{{{esc(k)}}} & \\texttt{{{esc(str(c.get(k)))}}}\\\\")
S10.append("\\bottomrule\\end{tabular}")

tex = open(HERE / "supp_template.tex").read()
for key, val in (("@@S1@@", S1), ("@@S4@@", S4), ("@@S5@@", S5), ("@@S6@@", S6), ("@@S10@@", S10)): tex = tex.replace(key, "\n".join(val))
open(HERE / "supp.tex", "w").write(tex); print("wrote supp.tex")

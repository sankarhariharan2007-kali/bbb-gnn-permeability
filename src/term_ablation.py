"""Which reward term drives the scaffold collapse? Zero one weight at a time.

The reward is R = w_D*D + w_C*C + w_Q*QED + w_S*SA. The investigation inferred
that the collapse is driven by C from the saturation data, but never ablated
the terms (reward_hacking_investigation.md, Section 8). This does.

Each run is the baseline configuration with ONE weight zeroed, for each seed.
The baselines are the committed results/gan/seed{0,1,2} runs, so the ablation
costs 4 terms x 3 seeds = 12 runs, not 15.

Read the output with three things in mind:

  * "Zeroed" is not "deleted". The term is still scored and logged, so mean_C
    on the generated molecules stays observable when w_C = 0. If mean_C still
    climbs with C out of the reward, the classifier's confidence is a side
    effect of what QED/SA/D select for, not something the policy chased.

  * Zeroing is not a clean marginal contribution. group_advantages standardizes
    the TOTAL reward inside the valid subgroup, so dropping a high-variance
    term inflates the effective weight of everything left. Read a row as "what
    happens when this term does not steer", not "this term's share".

  * w_D = 0 also removes the realism anchor. D still trains (the n_D steps
    still run) but nothing consumes it.

Same seed gives the same warm-start sample at step 0 in every arm, so the
comparison is paired. With n=3 seeds nothing here is a significance test: the
summary reports per-seed deltas and how many seeds agree on the sign. Seed 0
collapses earlier and harder than 1 and 2 (investigation, Section 2), so look
at the seeds separately before averaging them.

    python -m src.term_ablation --check-baseline 30    # does the code still reproduce the baseline?
    python -m src.term_ablation --jobs 3               # all 12 runs, 3 at a time
    python -m src.term_ablation --summarize-only       # re-read what exists
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import ROOT
from .kl_sweep import c_spread, summarize_run
from .train_gan import OUT_DIR as BASE_DIR

ABL_DIR = ROOT / "results" / "term_ablation"
TERMS = ("C", "D", "Q", "S")


def run_dir(term: str, seed: int) -> Path:
    return ABL_DIR / f"drop_{term}" / f"seed{seed}"


def base_dir(seed: int) -> Path:
    return BASE_DIR / f"seed{seed}"


def is_done(rdir: Path) -> bool:
    # history.csv is rewritten every step, so its existence says nothing about
    # completion. config.json is written once, after the last step.
    return (rdir / "config.json").exists()


def _cmd(term: str, seed: int, steps: int, group_size: int, device: str) -> list[str]:
    return [sys.executable, "-m", "src.train_gan", "--steps", str(steps),
            "--seed", str(seed), "--group-size", str(group_size),
            "--device", device, "--drop", term,
            "--out-dir", str(run_dir(term, seed))]


def _launch(job: tuple[str, int], steps: int, group_size: int, device: str,
            threads: int) -> tuple[str, int, int]:
    term, seed = job
    rdir = run_dir(term, seed)
    rdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    with open(rdir / "train.log", "w") as log:
        rc = subprocess.run(_cmd(term, seed, steps, group_size, device),
                            stdout=log, stderr=subprocess.STDOUT, env=env,
                            cwd=ROOT).returncode
    print(f"  drop_{term}/seed{seed}: {'ok' if rc == 0 else f'FAILED rc={rc} (see train.log)'}",
          flush=True)
    return term, seed, rc


def check_baseline(n_steps: int, group_size: int, device: str,
                   seed: int = 0) -> bool:
    """Does today's code still reproduce the committed baseline?

    Every ablation delta is taken against results/gan/seed<s>. If the training
    loop has drifted since those runs were made, the deltas measure the drift.
    Compare the first `n_steps` rows of a fresh default-config run.
    """
    from .train_gan import train

    ref = pd.read_csv(base_dir(seed) / "history.csv").head(n_steps)
    tmp = ABL_DIR / "_baseline_check"
    out = train(steps=n_steps, group_size=group_size, device=device, seed=seed,
                out_dir=tmp, log=False)["history"]
    new = pd.DataFrame(out)
    ok = True
    for col in ("reward_mean", "scaffold_frac", "c_mean"):
        diff = float(np.abs(new[col].to_numpy() - ref[col].to_numpy()).max())
        # CPU float reductions are not bitwise-stable across torch builds, so
        # demand agreement, not equality; rows that diverge by more than this
        # are a different run, not noise.
        flag = diff < 1e-3
        ok &= flag
        print(f"  {col:<14} max|diff| over {n_steps} steps = {diff:.2e}  "
              f"{'ok' if flag else 'DIVERGES'}")
    print("baseline " + ("reproduces" if ok else "DOES NOT reproduce")
          + f" (seed {seed}, {n_steps} steps)")
    return ok


def first_drop(x: pd.Series, frac: float = 0.8, open_n: int = 10,
               smooth: int = 9) -> float:
    """Step where the smoothed series first falls `frac` x its own opening level.

    Mirrors the investigation's lead-time definition (20% below opening,
    9-step smoothing). NaN if it never does.
    """
    opening = x.head(open_n).mean()
    sm = x.rolling(smooth, center=True, min_periods=1).mean()
    below = np.flatnonzero(sm.to_numpy() < frac * opening)
    return float(below[0]) if below.size else float("nan")


def one_row(rdir: Path, term: str, seed: int, dataset: str, device: str,
            tail: int) -> dict:
    h = pd.read_csv(rdir / "history.csv")
    row = {"term_dropped": term, "seed": seed, "steps": len(h),
           **summarize_run(rdir, tail=tail),
           **c_spread(rdir, dataset, device),
           "t_scaffold_drop": first_drop(h.scaffold_frac),
           "t_tpsa_spread_drop": first_drop(h.tpsa_spread_ratio)}
    return row


def summarize(seeds: list[int], terms: list[str], dataset: str, device: str,
              tail: int, collapse_below: float) -> pd.DataFrame:
    rows = []
    for s in seeds:
        if (base_dir(s) / "history.csv").exists():
            rows.append(one_row(base_dir(s), "none", s, dataset, device, tail))
        else:
            print(f"  (no baseline at {base_dir(s)}; deltas for seed {s} skipped)")
        for t in terms:
            if is_done(run_dir(t, s)):
                rows.append(one_row(run_dir(t, s), t, s, dataset, device, tail))
    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing to summarize yet")
        return df

    base = df[df.term_dropped == "none"].set_index("seed")
    metrics = ["scaffold_frac", "tanimoto_dist", "tpsa_spread_ratio",
               "sd_C", "mean_C", "frac_over_95"]
    for m in metrics:
        df[f"d_{m}"] = df.apply(
            lambda r: r[m] - base[m].get(r.seed, np.nan)
            if r.term_dropped != "none" else np.nan, axis=1)
    df["collapsed"] = df.scaffold_frac < collapse_below

    ABL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ABL_DIR / "term_ablation.csv", index=False)

    show = ["term_dropped", "seed", "scaffold_frac", "d_scaffold_frac",
            "tanimoto_dist", "tpsa_spread_ratio", "sd_C", "mean_C",
            "t_scaffold_drop", "collapsed"]
    print("\n" + "=" * 100)
    print(f"TERM ABLATION -- endpoint = mean of last {tail} steps; "
          f"d_* = (term dropped) - (baseline), same seed")
    print("=" * 100)
    print(df[show].round(3).to_string(index=False))

    print("\nAcross seeds (n is small: read the sign agreement, not the mean):")
    agg = []
    for t, g in df[df.term_dropped != "none"].groupby("term_dropped"):
        agg.append({
            "term_dropped": t, "n_seeds": len(g),
            "d_scaffold_frac_mean": g.d_scaffold_frac.mean(),
            "seeds_scaffold_up": f"{int((g.d_scaffold_frac > 0).sum())}/{len(g)}",
            "d_sd_C_mean": g.d_sd_C.mean(),
            "mean_C_mean": g.mean_C.mean(),
            "n_collapsed": f"{int(g.collapsed.sum())}/{len(g)}",
        })
    print(pd.DataFrame(agg).round(3).to_string(index=False))
    print(f"\ncollapsed = endpoint scaffold_frac < {collapse_below} "
          "(an arbitrary screen: baseline ends at 0.07-0.14, BBBP at matched "
          "n is ~0.6-0.76).")
    print("If 'seeds_scaffold_up' is 3/3 for a term, removing it helps on every "
          "seed. If collapse persists with C dropped, C is not the driver.")
    print(f"wrote {ABL_DIR / 'term_ablation.csv'}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default=",".join(TERMS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=200,
                    help="must match the baseline (200) or deltas are confounded")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel training processes; threads are split evenly")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--collapse-below", type=float, default=0.30)
    ap.add_argument("--check-baseline", type=int, default=0, metavar="N",
                    help="reproduce the first N baseline steps, then exit")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    terms = [t for t in args.terms.split(",") if t]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    bad = set(terms) - set(TERMS)
    if bad:
        raise SystemExit(f"unknown terms {sorted(bad)}; choose from {TERMS}")

    if args.check_baseline:
        sys.exit(0 if check_baseline(args.check_baseline, args.group_size,
                                     args.device) else 1)

    if not args.summarize_only:
        todo = [(t, s) for s in seeds for t in terms if not is_done(run_dir(t, s))]
        print(f"{len(todo)} runs to do, {len(terms) * len(seeds) - len(todo)} already done")
        threads = max(1, (os.cpu_count() or 1) // max(1, args.jobs))
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            results = list(pool.map(
                lambda j: _launch(j, args.steps, args.group_size, args.device, threads),
                todo))
        failed = [(t, s) for t, s, rc in results if rc != 0]
        if failed:
            print(f"\n{len(failed)} run(s) failed: {failed}. Re-run to retry only those.")

    summarize(seeds, terms, args.dataset, args.device, args.tail, args.collapse_below)


if __name__ == "__main__":
    main()

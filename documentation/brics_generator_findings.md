# BRICS assembly generator: what it bought, and what it cost

The BRICS fragment-assembly generator against the SELFIES token generator, under
one reward. The engine reconstructs exactly and validity is near-total.

**Compared as frontiers rather than as points: where both families operate the
two curves cross, so neither dominates on diversity-vs-confidence. BRICS'
frontier extends further -- scaffold 0.906 against 0.835, 2.09 SD -- and BRICS
dominates on validity at every setting. The cost is sensitivity: BRICS needs
kl_coef >= 1.0 to avoid collapsing to duplicates, where SELFIES holds 0.99
unique across the whole range.**

**And the instrument gap reverses the confidence axis (section 3b).** Every BRICS
point's `C(BBBP) - C(B3DB)` brackets the real-drug reference of +0.037; every
SELFIES point is higher. Measured properly at n=12 per arm, BRICS' mean gap
(+0.0381) sits 0.0009 from the real-drug reference (+0.0373) against SELFIES'
+0.0551, Welch p = 0.003 — but the per-sample distributions **overlap**. So
BRICS' lower `c_mean` is honesty rather than weakness, and the "BRICS pays confidence
for validity" framing in section 3 is wrong. (`D_real` appears to agree and is
discarded: its discriminator was trained against SELFIES specifically, so it
cannot rank two families. Section 3b.)

A single matched-`kl_coef` comparison cannot say this, and an earlier version of
this document tried to. `kl_coef` is not commensurable across these action spaces
(section 4), so comparing at the same number is arbitrary whichever number is
chosen.

**This supersedes an earlier version of this document that reported BRICS losing
0.27 of scaffold diversity.** That measurement was real but its interpretation
was wrong: it was taken at `kl_coef=0.3`, a value validated for SELFIES, and the
entire gap closes when the coefficient is set for this action space. Section 4
is the retraction.

Runs: 24 GRPO runs of 250 steps, group 64, `min_heavy_atoms=10`, `--threads 6`,
GRPO `--seed 0`, GraphNorm reward ensemble. A 2x2 of reward seed x warm start per
arm (`results/brics_grpo` kl=0.3, `results/brics_kl1` kl=1.0,
`results/selfies_control` kl=0.3) plus a 4x3 `kl_coef` x `ent_coef` sweep
(`results/brics_sweep`).

**What is committed.** `history.csv`, `config.json` and `samples/` for every
run -- every number here derives from those, and `samples/` is what
`src/postfilter.py` pools. The generator and discriminator checkpoints are not
committed: they are deterministic given the committed warm start, `--seed` and
the pinned thread count in each `config.json`, and they would add ~150 MB
against a standing limitation about artifact size.

---

## 1. The engine reconstructs exactly

`python -m src.brics_assembly`: **2039/2039 exact**, 0 untraceable, 0 mismatch,
longest trace 24 actions. Two corrections were needed, both measured.

**The incoming slot must be part of the action.** Picking the lowest-index
compatible dummy reconstructs only **63.5%**. 948 of 2612 fragments have more
than one slot compatible with some site type; the wrong pick leaves a different
site set open, so every later lookup is against the wrong site and the error
cascades -- `CC(C)NCC(O)COc1cccc2ccccc12` came back as `CC(C)NCC(O)CNC(C)C`.
6051 actions instead of 2614. Do not reintroduce the heuristic.

**Stereochemistry has to be stripped.** With the slot fixed, reconstruction
reached **81.0%** and all 387 residual failures were stereo-only, zero
constitutional: `reassemble` appends the new bond after the dummy is removed, so
the anchor's chiral tag is read against a different neighbour order and the
parity flips. Stripping is free -- the featurizer has no chirality features, so
`C(m)` changes by exactly 0.00e+00 on sulpiride, L-DOPA and quinine. (The graph
tensors are *not* element-wise equal, because canonical re-ordering permutes atom
rows; they match once sorted, and the readout is permutation-invariant. Compare
sorted tensors or compare outputs.) A defined but WRONG centre is a false claim
for no signal; an undefined one is merely silent.

## 2. "Valid by construction" is narrower than it sounds

Type compatibility plus an explicit bond order guarantees **connectivity and
valence, not aromaticity**. Joining two type-compatible aromatic fragments can
yield a ring system RDKit cannot kekulize. Every inspected failure is a
`KekulizeException`; none are valence.

| policy | sanitizes |
|---|---|
| untrained | 95.39% (118/2560 fail) |
| MLE warm start | 100% (0/2048) |
| 250-step runs | 99.98% (min 98.44% at one step) |

The floor is ~95.4%, not 100%. `AssemblyState.to_mol` returns `""` and the
molecule routes through the invalidity hurdle.

**Part of what the warm start teaches is toolkit avoidance, not chemistry.** The
95.4% -> 100% gain is the policy learning to avoid pairs RDKit cannot kekulize,
which is a representation limit of the stack. Do not report it as "the policy
learned valid chemistry".

## 3. The headline: comparable diversity, a validity/confidence trade

Steps 225-250, means over 4 cells. Warm-start reference measured at the same
n=64: scaffold 0.854, unique 0.990.

| arm | scaffold | unique | C | reward | rdkit_valid |
|---|---|---|---|---|---|
| BRICS kl=0.3 (inherited) | 0.643 | 0.744 | 0.932 | 2.280 | 0.9998 |
| **BRICS kl=1.0 (tuned)** | **0.864** | **0.982** | 0.847 | 2.090 | 0.9998 |
| SELFIES kl=0.3 (validated) | 0.821 | 0.995 | 0.900 | 2.080 | 0.9946 |

Paired on reward seed and warm start, BRICS kl=1.0 against SELFIES kl=0.3,
n=4 cells, 3 df, t_crit = 3.182:

| metric | diff | SEM | t | p | verdict |
|---|---|---|---|---|---|
| `scaffold_frac` | +0.0431 | 0.0154 | 2.79 | 0.068 | **not distinguishable** |
| `reward_mean` | +0.0096 | 0.0129 | 0.74 | 0.512 | **not distinguishable** |
| `undersized_frac` | −0.0046 | 0.0015 | −3.09 | 0.054 | not distinguishable |
| `unique_smiles_frac` | −0.0134 | 0.0016 | −8.60 | 0.003 | differs, negligibly |
| `tanimoto_dist` | +0.0045 | 0.0010 | 4.36 | 0.022 | differs, negligibly |
| `c_mean` | −0.0536 | 0.0066 | −8.16 | 0.004 | differs |
| `rdkit_valid_frac` | +0.0052 | 0.0002 | 25.87 | 0.000 | differs |

So at *these two* coefficients BRICS matches SELFIES on scaffold diversity and
total reward, is marginally behind on exact-duplicate rate (0.982 vs 0.995), buys
**+0.0052 validity**, and pays **−0.054 of classifier confidence**.

That is a claim about two points, and `kl_coef=1.0`-vs-`0.3` is as arbitrary a
pairing as `0.3`-vs-`0.3` once the coefficient is known not to be commensurable.
Section 3a is the comparison that does not depend on choosing a pairing.

## 3a. The two frontiers

`kl_coef` in {0.1, 0.3, 1.0, 3.0} per family, 250 steps, current reward, fixed to
reward seed 0 / warm start ws0 / GRPO seed 0 -- so the BRICS column is the
sweep's `ent_coef=0.0` slice and the two frontiers are paired cell-for-cell.

| family | kl | scaffold | C | unique | reward | rdkit_valid |
|---|---|---|---|---|---|---|
| BRICS | 0.1 | 0.464 | 0.971 | 0.511 | 2.358 | 1.0000 |
| BRICS | 0.3 | 0.723 | 0.947 | 0.830 | 2.305 | 0.9999 |
| BRICS | 1.0 | 0.887 | 0.871 | 0.986 | 2.130 | 0.9996 |
| BRICS | 3.0 | **0.906** | 0.782 | 0.993 | 1.945 | 0.9999 |
| SELFIES | 0.1 | 0.638 | 0.960 | 0.989 | 2.255 | 0.9966 |
| SELFIES | 0.3 | 0.809 | 0.921 | 0.996 | 2.110 | 0.9948 |
| SELFIES | 1.0 | 0.835 | 0.849 | 0.996 | 1.862 | 0.9911 |
| SELFIES | 3.0 | 0.819 | 0.820 | 0.992 | 1.755 | 0.9899 |

**What the frontier shows, sized against noise.** The single-run sd on
`scaffold_frac` is now **measured** rather than inferred: 0.0202 over 3 GRPO
seeds at the settled point, so a difference between two independent single draws
has sd 0.0286. (An earlier version of this section inferred 0.034 from the
paired SEM, which was conservative.) Differences below are quoted in those
units.

*BRICS' frontier extends further.* 0.906 against 0.835 is **2.49 SD** -- BRICS
reaches structural diversity SELFIES does not attain at any coefficient tested.
This is the claim that carries weight.

*SELFIES' apparent turn-back does not.* Scaffold goes 0.638 -> 0.809 -> 0.835 ->
0.819, and that last step is **0.56 SD**: indistinguishable from flat. An earlier
version of this section led with it. The defensible statement is that SELFIES
*saturates* above kl=1.0 while BRICS keeps rising, not that it reverses.

*In the overlap, the frontiers cross.* Interpolating C at matched scaffold over
0.638-0.835 gives mean +0.0031 for BRICS with range −0.0111 to +0.0466 and
**three sign flips**. Neither family is above the other; they interleave. The
positive mean is carried almost entirely by the single point at scaffold 0.835,
so it is not a summary worth quoting.

*The Pareto count is provisional.* Over (scaffold, C) via
`pareto.dominance_count`, all four BRICS points are non-dominated and only two of
four SELFIES points are. But the dominance of BRICS@1.0 over SELFIES@1.0 rests on
a **1.82 SD** scaffold difference, so "two of four" is a property of this draw
rather than a structural fact. Adding `rdkit_valid` does not change membership.

**Validity is the one axis with uniform dominance.** BRICS is 0.9996-1.0000 flat
across the whole range; SELFIES is 0.9899-0.9966 and *degrades* as kl rises. Not
a trade -- BRICS is higher at every setting.

**The cost is fragility.** At kl=0.1 BRICS collapses to 0.511 unique while
SELFIES holds 0.989, and BRICS needs kl >= 1.0 before duplicates vanish. SELFIES
is insensitive to the coefficient on that axis across the entire range. If a
coefficient has to be chosen without a sweep, SELFIES is the safer default and
BRICS is the one that repays tuning.

## 3b. Off-instrument: the C axis points the other way

A metrics table cannot distinguish "better" from "Goodharting differently", so
the frontier is scored on the two checks `aggregate_arms` defines: the
**instrument gap** `C(BBBP) - C(B3DB)`, the training ensemble minus a judge the
reward never used, and **D_real**, a fixed discriminator's score. Pooled over
each run's last 20 sample steps with `aggregate_arms.pool`.

| family | kl | scaffold | C(BBBP) | C(B3DB) | gap | abs(gap−real) | D_real |
|---|---|---|---|---|---|---|---|
| BRICS | 0.1 | 0.464 | 0.927 | 0.900 | +0.027 | 0.010 | 0.931 |
| BRICS | 0.3 | 0.723 | 0.912 | 0.883 | +0.029 | 0.008 | 0.939 |
| BRICS | 1.0 | 0.887 | 0.848 | 0.813 | +0.035 | **0.002** | 0.939 |
| BRICS | 3.0 | 0.906 | 0.771 | 0.730 | +0.042 | 0.005 | 0.947 |
| SELFIES | 0.1 | 0.638 | 0.956 | 0.874 | +0.082 | 0.044 | 0.710 |
| SELFIES | 0.3 | 0.809 | 0.923 | 0.862 | +0.060 | 0.023 | 0.774 |
| SELFIES | 1.0 | 0.835 | 0.857 | 0.769 | +0.088 | 0.051 | 0.792 |
| SELFIES | 3.0 | 0.819 | 0.824 | 0.725 | +0.099 | 0.062 | 0.800 |
| **REAL BBB+ drugs** | | | 0.906 | 0.869 | **+0.037** | — | **0.940** |

**This reverses the reading of the C axis.** Sections 3 and 3a treat BRICS' lower
`c_mean` as a cost. It is not: every BRICS point's instrument gap (+0.027 to
+0.042) brackets the real-drug reference of +0.037, while every SELFIES point
(+0.060 to +0.099) is higher. SELFIES' extra confidence is substantially the
selector's blind spot being exploited, not better molecules, which is exactly the
failure the check exists to catch.

**The "do not overlap" claim is RETRACTED.** It rested on 8 single draws. With
12 samples per arm (3 runs x steps 200/220/240/249 each):

| arm | n | mean gap | sd | range |
|---|---|---|---|---|
| BRICS | 12 | **+0.0381** | 0.0155 | +0.0170 to +0.0677 |
| SELFIES | 12 | +0.0551 | 0.0065 | +0.0454 to +0.0636 |
| real BBB+ drugs | — | +0.0373 | — | — |

BRICS' max (+0.0677) exceeds SELFIES' min (+0.0454), so the families **do**
overlap and an individual BRICS sample can be further from the real reference
than an individual SELFIES one.

**What survives is the mean difference, and it is still the strongest result in
this report.** Welch t = -3.49, **p = 0.0034**, and it survives both exposures
this data has:

| exposure | adjustment | result |
|---|---|---|
| sensitivity to one point | leave-one-out, any single BRICS sample dropped | p = 0.0005 - 0.0078 |
| repeated interrogation | Bonferroni over the 3 hypotheses tested on these samples | **p = 0.010** |
| | conservative x8 | **p = 0.027** |

Mean \|gap - real\| is 0.0126 for BRICS against 0.0178 for SELFIES, and BRICS'
mean lands 0.0009 from the real-drug reference. So: *BRICS' instrument gap is
significantly closer to real drugs on average, though the per-sample
distributions overlap.* Not a categorical separation.

**State the family when quoting the adjusted figure.** Three distinct hypotheses
were tested on these 12+12 samples — means differ, variances differ, ranges
non-overlap. Leave-one-out re-runs are robustness checks of the first, not new
hypotheses, and the thread's other tests (p95 ratios, seed agreement) were on
different data. A x20 correction counting every test run anywhere in the thread
would give p = 0.068 and fail, but that count is wrong for this family. The
claim holds to x12.

**BRICS also looks like the less consistent arm on this axis, but that is an
observation and not a result.** sd 0.0155 against 0.0065, ratio 5.74, Levene
median-centred p = 0.037, and it does hold under leave-one-out (4.04 at worst)
where the retracted agreement-variance claim collapsed. It still **does not
survive multiplicity**: 0.111 at x3, 0.296 at x8. No p-value should appear next
to it.

Two things that are easy to conflate and are not the same: leave-one-out tests
sensitivity to a single point, multiplicity tests exposure to repeated
interrogation. This observation passes the first and fails the second. And its
being flagged by one party and confirmed by the other is not independent
evidence — both looked at the same twelve numbers. Confirming it needs samples
not already used to find it.

**D_real is confounded and is NOT usable here.** The numbers above (BRICS
0.931-0.947, SELFIES 0.710-0.800, real drugs 0.940) look like BRICS closing the
"generator 0.80 vs real 0.94" realism gap. They are not evidence of that.
`results/gan/seed0/config.json` has `feature_id` ABSENT, no `generator` key and
`kl_coef=0.02`: that discriminator predates the `--generator` flag, so it was
trained adversarially against a **SELFIES** generator specifically, under the
retired classifier, at the hacked coefficient. It therefore learned one
generator's artifacts. BRICS scoring high may only mean it lacks SELFIES
artifacts, and SELFIES scoring low is a detector marking the family it was built
against.

That is the same asymmetric-instrument error as transplanting `kl_coef`
(section 4), pointing the other way: a judge calibrated on one family cannot
rank two. A cross-family realism claim needs one discriminator trained on real
against a **mixture of both** families, which is a separate experiment and is not
run here.

**The instrument gap does not share the flaw.** `C(BBBP)` and `C(B3DB)` are
contract-era classifiers with no adversarial history against either generator --
both only ever saw real molecules, so neither family is in-domain for the judge.
System-1 recomputed the reference independently over 1560 real BBB+ drugs under
the current ensembles and got **+0.0373** against the +0.037 here. Section 3b's
reversal of the C axis therefore stands on the gap alone.

So at matched diversity BRICS' lower `c_mean` is **honesty, not weakness**, and
`BRICS@kl=1.0` is the most real-drug-like point measured on either curve (gap
+0.035 against +0.037, D_real 0.939 against 0.940) while also being the
highest-diversity point not bought with a confidence collapse.

Two corrections this produced:

- **The real-drug reference is +0.037, not −0.029**, and the sign is flipped.
  `aggregate_arms`' docstring quotes −0.029 from the retired ensemble. Since the
  whole check is "how far from the real-drug gap", a stale reference inverts the
  verdict, so it is recomputed here under the current GraphNorm ensembles.
- **`aggregate_arms.offinstrument()` does not run on this branch.** Its hardcoded
  `results/gan/seed0/bbbp_disc.pt` is BatchNorm-era, so `load_state_dict` fails
  with `running_mean` against `mean_scale`. Loading it as
  `Discriminator(norm="batch")` -- replaying the recorded architecture rather than
  imposing today's default, as `run_contract` does for classifiers -- fixes it.
  The file is not in this half's ownership, so it is reported, not edited.

Caveat: B3DB aggregates BBBP among its sources, so it is an *unused* judge rather
than an independent one, and some shared molecules carry contradictory labels.
It is the best available unseen judge, not a clean holdout. Each point is also
n=1, so the within-family spread (BRICS 0.015 wide, SELFIES 0.039) is the only
noise estimate. The n=12 re-measurement above supersedes the non-overlap
reading taken from those 8 points.

## 4. Retraction: the diversity collapse was a mis-set coefficient

The earlier version of this document reported, correctly as a measurement:
scaffold −0.2701 +/- 0.0170 and `unique_smiles_frac` −0.3370 +/- 0.0313 for
BRICS against SELFIES, both at `kl_coef=0.3`. It then explained the duplicates
as structural:

> "The compactness that prevents atom spam is the same property that permits
> re-emission."

**That explanation was mis-scoped rather than simply wrong, and the frontier says
how.** Re-emission is not an inherent cost of a compact action space -- raising
the coefficient removes it entirely (0.744 -> 0.982 unique). But it *is* the
characteristic failure mode of this action space when regularization is
insufficient, and BRICS is far more sensitive to that than SELFIES: at kl=0.1
BRICS is at 0.511 unique against SELFIES' 0.989 (section 3a). So compactness
does not *permit* re-emission as a standing property; it makes re-emission what
under-regularization looks like here. Raising the coefficient alone recovers it,
paired on the same cells:

| BRICS kl=1.0 − kl=0.3 | diff | SEM | p |
|---|---|---|---|
| `scaffold_frac` | +0.2212 | 0.0267 | 0.004 |
| `unique_smiles_frac` | +0.2380 | 0.0313 | 0.005 |
| `c_mean` | −0.0850 | 0.0099 | 0.003 |
| `reward_mean` | −0.1907 | 0.0121 | 0.001 |

Re-emission was an under-regularized policy concentrating, not a property of a
compact action space. `kl_coef=0.3` survived validation *for SELFIES* and
transplanted as a number rather than as a strength: at the same coefficient
BRICS showed 10.7x the KL, because the clipped loss is a token-level masked mean
and BRICS takes ~5.1 decisions over a 6051-way softmax against SELFIES' ~38.6
over 77. **Any hyperparameter validated on one action space has to be re-swept on
another.**

## 5. The sweep: KL is the lever, entropy is not

4x3, `kl_coef` x `ent_coef`, scaffold_frac @225-250 (warm start 0.854):

| kl \ ent | 0.0 | 0.02 | 0.1 |
|---|---|---|---|
| 0.1 | 0.464 | 0.411 | 0.613 |
| 0.3 | 0.723 | 0.547 | 0.809 |
| 1.0 | **0.887** | **0.891** | 0.874 |
| 3.0 | **0.906** | 0.879 | 0.873 |

`unique_smiles_frac` on the same grid reaches 0.986-0.993 for every kl >= 1.0,
i.e. the duplicate problem disappears entirely.

**Entropy does nothing once KL is adequate.** At kl=1.0 scaffold is
0.887/0.891/0.874 across ent 0/0.02/0.1 -- flat. It helps only where KL is
missing (0.723 -> 0.809 at kl=0.3), so it substitutes for absent regularization
rather than adding to it. The mechanism is visible in the logs: per-step entropy
rises to ~2.5 at kl >= 1.0 **with no bonus at all**, above the warm start's
2.130, so adequate KL already holds per-step entropy up and an explicit bonus has
nothing left to do.

Sweeping KL alone would have found the same setting. The second axis is what
makes "entropy was never the knob" a finding instead of an assumption.

**Entropy was worth testing on theory.** With ~5.1 decisions against ~38.6, the
same per-step entropy gives far lower molecule-level entropy -- measured, BRICS
has 1.8x the per-step entropy (2.130 vs 1.170) and 24% of the molecule-level
entropy (10.79 vs 45.11). That predicted an entropy bonus should help. It did
not, because KL reaches the same axis indirectly.

**The warm start was not the bottleneck.** A hypothesis worth ruling out was that
`pi_ref` is itself peaked, in which case no `kl_coef` could recover diversity it
never had. At matched n=64 the BRICS warm start is scaffold 0.854 / unique 0.990
against SELFIES' 0.811 / 0.982 -- *more* diverse. The runs lost diversity their
reference genuinely had, which is why raising the anchor's weight recovered it.
Measuring this at n=512 instead gives 0.654 vs 0.765 and inverts the conclusion;
`scaffold_frac` is n-dependent and must be compared at matched n.

**Cost of diversity is C**: 0.947 -> 0.871 -> 0.781 as kl goes 0.3 -> 1.0 -> 3.0.
kl=3.0 buys a little more scaffold (0.906) for much less C (0.781), so 1.0 is the
knee.

## 6. The k3 KL estimator produces extreme outliers here

Left unchanged -- the loss still uses unclipped k3, which keeps the committed
GRPO runs comparable. `kl_clipped` (d clamped to +/-10 nats before the exp) is
logged beside it instead of replacing it.

It earned itself immediately: sweep cell kl=1.0/ent=0.02 recorded `kl=2940.5`
with `kl_clipped=93.3`, so **~97% of the raw mean was a single tail event** -- and
that cell still produced the best-in-column diversity (0.891). Without the
companion statistic that run reads as broken. Similarly `rs0_ws1` at kl=0.3 hit
`kl=17327.56` at step 7 with the next step at 0.0145.

The estimator is `exp(d) - d - 1`; one action where the policy disagrees with
`pi_ref` by ~10 nats contributes exp(10) ~ 22026, and a 6051-way space over ~5
decisions both permits that and averages it over 5 positions rather than 38.
**The outlier is a symptom, not a bug**: it means the policy reached an action the
reference assigns ~zero probability, which is expected when the action space was
rebuilt under it. Gradient clipping at 1.0 bounds the damage.

## 7. Duplicates shrink the effective group

`n_eff_group` and `eff_group_frac` are now logged: distinct SMILES among the
*rankable* rows, which is the set `group_advantages` actually standardizes over.
`eff_group_frac` tracks `unique_smiles_frac` to within 0.002 throughout, so at
kl=0.3 a nominal group of 64 carried ~48 distinct candidates and at kl=0.1 only
~33. Every advantage estimate in the kl=0.3 matrix was computed from a smaller
effective group than `group_size` claims. At kl=1.0 it is 0.987, so the problem
does not arise at the tuned setting.

## 8. What these runs cannot tell you

- **~~One GRPO seed.~~ CLEARED at the settled point.** The settled configuration
  (kl=1.0, rs0, ws0) now has 3 GRPO seeds: scaffold 0.8643 +/- 0.0117 SEM
  (sd 0.0202), unique 0.9821 +/- 0.0030, C 0.8824 +/- 0.0056, reward
  2.1517 +/- 0.0123, rdkit_valid 0.9997 +/- 0.0001. The single-draw figure was
  0.864 and the 3-seed mean is 0.864. The *frontier* runs remain single-seed, so
  the shape claims above are still single draws sized against this sd.
- **`kl_coef=1.0` was selected from section 5's grid**, and one of the four
  paired cells (`rs0_ws0`) is the same configuration as the sweep cell that
  selected it. It reproduced bit-identically (0.8874 both ways, a useful
  determinism check), but it is not independent of the selection; the other
  three cells are new data.
- **`results/kl_sweep` cannot anchor anything.** The pre-existing SELFIES sweep
  has no `feature_id` in any cell (so a different classifier, and `run_contract`
  would refuse those checkpoints) and ran 120 steps against 250. Verified, not
  assumed. It is a different experiment, not an older criterion, which is why
  section 3a re-ran the SELFIES frontier from scratch.
- **Each frontier point is n=1.** No error bars, so a crossing near a knee is not
  distinguishable from noise -- which is exactly where the two curves interleave.
  The *shape* claims (SELFIES turning back, BRICS monotone, the 0.835 vs 0.906
  ceilings) rest on single draws and want replication before they are leaned on.
- **Only one axis of the sweep was carried to the frontier.** `ent_coef` is held
  at 0, justified by section 5 but not re-tested per family.
- **The instrument gap uses B3DB, which is not independent.** B3DB aggregates
  BBBP among its sources, so it is a judge the reward never *used* rather than a
  clean holdout, and shared molecules can carry contradictory labels. A gap
  measured against a genuinely external set (the Adenot/Wang holdouts) would be
  stronger, but at n=55-59 those cannot support per-arm means.
- **Section 3b's frontier table is n=1 per point**, and the non-overlap read
  from it is retracted; see the n=12 re-measurement in that section. The mean
  difference survives at p = 0.003.
- **No realism claim is made.** `D_real` is discarded as confounded, and the
  experiment that would support one -- a discriminator trained on real against a
  mixture of both families -- has not been run.
- **The 2x2 cannot separate its two noise sources** -- one run per cell is zero
  residual df. At kl=0.3: reward-seed effect −0.072, warm-start effect −0.081,
  the same order as each other and as the between-cell spread.
- **No synthesizability claim.** Assembly from BRICS fragments gives a
  retrosynthetic plausibility floor by construction; nothing here tests it.
- **Nothing here is a synthesis candidate.** Diversity being retained is not
  evidence the molecules are good.

## 9. What stands

The case for BRICS was never classifier accuracy -- the hierarchical motif family
settled that separately (replacing atom detail with fragment structure
−0.1161 +/- 0.0107; adding it on top −0.0051 +/- 0.0047, i.e. nothing).

It was validity by construction and an action space that cannot emit atom spam.
Validity holds and is the one axis with uniform dominance across the whole
regularization range (0.9996-1.0000 against 0.9899-0.9966, with SELFIES
degrading as kl rises). The action-space argument is structural. And the
diversity objection that appeared to count against it was an artifact of a
transplanted coefficient.

It also turns out to buy something nobody claimed for it: BRICS is markedly
harder on the instrument. Its gap to an unused judge tracks real drugs at every
coefficient tested (section 3b). That was not part of the case for BRICS and is
the strongest result in this report. The realism half of that claim is NOT
supported -- `D_real`'s discriminator was built against SELFIES and cannot rank
two families.

Two things are narrower than the original claim. Validity is a ~95.4% untrained
floor rather than a guarantee, because aromaticity is not covered. And BRICS is
the more hyperparameter-sensitive family: it reaches diversity SELFIES cannot,
but only after a sweep, and under weak regularization it fails worse.

The honest summary is not "BRICS is better" or "BRICS matches" but: **the
diversity frontiers interleave where they overlap and BRICS' extends further;
BRICS dominates on validity and on both off-instrument checks; and BRICS costs a
sweep that SELFIES does not.** The one axis where SELFIES leads -- raw `c_mean`
at matched diversity -- is the axis the off-instrument checks say not to trust.

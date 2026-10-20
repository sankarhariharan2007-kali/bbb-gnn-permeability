"""BRICS assembly state machine: (site, fragment) -> molecule. No torch.

Assembly-first, per handoff section 4.1. The action space IS sequential
assembly -- pick a fragment for the current attachment point, let RDKit form
the bond -- so the molecule is the output and validity is by construction.
Generating an atom-level graph and decomposing it afterwards would pay the full
cost of this rebuild while keeping every problem it was meant to solve.

Four things here are load-bearing, and each exists because the alternative
fails silently.

**State lives in `brics.reassemble`'s own coordinates.** `frags` is append-only,
so a list index IS the `frag_idx` a junction tuple names, and an open site is
already `(frag_idx, local_dummy_idx, brics_type)` -- exactly what `reassemble`
consumes. There is therefore never a global attachment id to translate into
local form. Hand-rolling RWMol bond formation would buy nothing and fork the
assembly logic away from the module whose round-trip is verified on 2039/2039.

**The molecule is materialized once, at the terminal step.** The automaton is
purely symbolic (action ids plus BRICS types) and masking is type
compatibility, which needs no RDKit at all. So one `reassemble` and one
sanitize per rollout rather than per step, and `log_probs` can replay the walk
in pure Python with no chemistry.

**An action names the incoming slot, not just the fragment.** This was measured,
not assumed. Choosing the lowest-index compatible dummy on the incoming
fragment reconstructs only 63.5% of BBBP: 948 of 2612 fragments have more than
one slot compatible with some site type, and picking the wrong one leaves a
different set of sites open, so every later lookup is against the wrong site and
the error cascades into a different molecule
(`CC(C)NCC(O)COc1cccc2ccccc12` came back as `CC(C)NCC(O)CNC(C)C`). The slot is
information only the policy can supply, so it is part of the action. That costs
one action per (fragment, slot) pair -- 6402 actions instead of 2614 -- and buys
exact reconstruction.

**Canonical trajectory ordering.** Open sites are a FIFO deque and a placed
fragment pushes its remaining dummies in ascending local index order.
Vocabulary keys are canonical fragment SMILES, so re-parsing one already yields
canonical atom order -- ascending atom index in the re-parsed fragment IS the
canonical rank, and no `CanonicalRankAtoms` call is needed. Site order alone
does not make a trace unique, though: a molecule with k fragments has k
candidate roots. `trace` pins the root by canonicalizing BEFORE decomposing,
which puts canonical atom 0 in `frags[0]`; that is asserted rather than left as
a side effect of a line written for something else.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

from .brics import decompose, reassemble

# Dummy atoms are stripped before scoring, never after.
_DUMMY = Chem.MolFromSmarts("[#0]")

MAX_TYPE = 16


def _compat() -> dict[int, frozenset[int]]:
    """BRICS attachment-type compatibility, derived not declared.

    `reactionDefs` distinguishes 7a from 7b but `FindBRICSBonds` emits '7' for
    both ends, so the sub-variants are collapsed to match what `brics.decompose`
    actually records in a junction.
    """
    pairs: dict[int, set[int]] = {}
    for group in BRICS.reactionDefs:
        for a, b, _rxn in group:
            ta, tb = int(a.rstrip("ab")), int(b.rstrip("ab"))
            pairs.setdefault(ta, set()).add(tb)
            pairs.setdefault(tb, set()).add(ta)
    return {t: frozenset(v) for t, v in pairs.items()}


COMPAT = _compat()


def bond_order(ta: int, tb: int) -> Chem.BondType:
    """('7a','7b','=') is the ONLY non-single reaction def in BRICS.

    Rebuilding a 7-7 junction as SINGLE silently hydrogenates 91 of BBBP's 2039
    molecules -- `CN(C)CCC=C1c2ccccc2CCc2ccccc21` comes back as `...CCCC1...`.
    The junction carries a bond type because of this; it is not decoration.
    """
    return Chem.BondType.DOUBLE if (ta, tb) == (7, 7) else Chem.BondType.SINGLE


def largest_component(mol: Chem.Mol) -> Chem.Mol | None:
    """Largest connected component, canonicalized.

    105 BBBP entries are multi-component (salts, hydrates). `decompose` returns
    a disconnected molecule as a single dummy-free "fragment" whose SMILES
    carries the counter-ion, so the vocabulary and the trace must agree on this
    preprocessing or 8 molecules come back out of vocabulary.
    """
    parts = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    best = max(parts, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolFromSmiles(Chem.MolToSmiles(best))


def corpus(smiles_list: list[str], strip_stereo: bool = True) -> list[str]:
    """Raw SMILES -> the canonical, single-component, flat corpus used everywhere.

    `vocab` and `trace` MUST be fed from this, or a fragment the trace needs has
    no token.

    **Stereochemistry is stripped, and that is a correctness decision rather
    than a shortcut.** Keeping it reconstructs only 81.0% of BBBP, and every one
    of the 387 failures is stereo-only -- zero constitutional errors. The cause
    is `reassemble` appending the new bond after the dummy is removed, so the
    chiral tag on the anchor is read against a different neighbour order and the
    parity flips. Handoff section 4.2 already drops stereo: the featurizer has
    no chirality features and will not get them.

    Verified here on sulpiride, L-DOPA and quinine: stripping stereo changes
    `C(m)` by exactly 0.00e+00. Note what is NOT true -- the graph tensors are
    not element-wise equal, because canonical re-ordering permutes the atom rows
    (quinine's `x` differs element-wise, matches once sorted). The readout is
    permutation-invariant, so the OUTPUT is identical. Compare sorted tensors or
    compare outputs; comparing `x` directly will mislead you.

    So the reward cannot see chirality, and emitting a DEFINED but WRONG centre
    would be a false claim about a molecule for no signal at all, where an
    undefined centre is merely silent. Flat in, flat out, exact.
    """
    out = []
    for s in smiles_list:
        if not isinstance(s, str):
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        lcc = largest_component(mol)
        if lcc is None:
            continue
        if strip_stereo:
            Chem.RemoveStereochemistry(lcc)
        out.append(Chem.MolToSmiles(lcc))
    return out


class FragSpec:
    """One vocabulary entry: the fragment, and where it can be attached.

    `sites` are indices into the RE-PARSED canonical fragment, which is the
    coordinate system every junction tuple this module emits refers to.
    """

    __slots__ = ("smiles", "mol", "sites")

    def __init__(self, smiles: str) -> None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"unparseable fragment: {smiles!r}")
        self.smiles = smiles
        self.mol = mol
        self.sites = tuple(
            (a.GetIdx(), a.GetIsotope())
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 0
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FragSpec({self.smiles!r}, sites={self.sites})"


def build_specs(stoi: dict[str, int]) -> list[FragSpec | None]:
    """Vocabulary -> id-aligned specs. Id 0 is UNK and stays None forever."""
    specs: list[FragSpec | None] = [None] * len(stoi)
    for smi, idx in stoi.items():
        if idx == 0:
            continue
        specs[idx] = FragSpec(smi)
    return specs


class ActionTable:
    """The action space: root placements, (fragment, slot) placements, CAP, STOP.

    Built once per vocabulary and shared by every rollout, so the legality masks
    are computed once rather than per step.
    """

    def __init__(self, specs: list[FragSpec | None]) -> None:
        self.specs = specs
        frag_of: list[int] = []
        slot_of: list[int] = []
        type_of: list[int] = []
        self.root_action: dict[int, int] = {}
        self.slot_action: dict[tuple[int, int], int] = {}

        for fid, spec in enumerate(specs):
            if spec is None:
                continue
            self.root_action[fid] = len(frag_of)
            frag_of.append(fid)
            slot_of.append(-1)
            type_of.append(0)
            for idx, t in spec.sites:
                self.slot_action[(fid, idx)] = len(frag_of)
                frag_of.append(fid)
                slot_of.append(idx)
                type_of.append(t)

        self.frag_of = np.array(frag_of, dtype=np.int32)
        self.slot_of = np.array(slot_of, dtype=np.int32)
        self.type_of = np.array(type_of, dtype=np.int32)
        n = len(frag_of)
        self.n_place = n
        self.cap_id = n
        self.stop_id = n + 1
        # PAD is a real action id rather than an overloaded slot, so a frozen
        # row after STOP is distinguishable from "placed fragment 0" in a
        # stored action tensor.
        self.pad_id = n + 2
        self.n_actions = n + 3

        # Row 0 is the root step: only root placements. Row ta: only slot
        # placements whose incoming dummy is compatible with ta, plus CAP/STOP.
        self.legal_by_type = np.zeros((MAX_TYPE + 1, self.n_actions), dtype=bool)
        self.legal_by_type[0, :n] = self.slot_of == -1
        for ta in range(1, MAX_TYPE + 1):
            ok = np.isin(self.type_of, list(COMPAT.get(ta, frozenset())))
            self.legal_by_type[ta, :n] = ok & (self.slot_of >= 0)
            self.legal_by_type[ta, self.cap_id] = True
            self.legal_by_type[ta, self.stop_id] = True

        # Nothing open: the only honest move is to finish.
        self._legal_empty = np.zeros(self.n_actions, dtype=bool)
        self._legal_empty[self.stop_id] = True
        # Frozen rows still need a finite softmax; `mask` zeroes the
        # contribution, so which slot is "legal" here is immaterial.
        self._legal_done = np.zeros(self.n_actions, dtype=bool)
        self._legal_done[self.pad_id] = True


class AssemblyState:
    """One rollout. Fully determined by its action sequence."""

    def __init__(self, table: ActionTable) -> None:
        self.t = table
        self.frags: list[Chem.Mol] = []
        self.junctions: list[tuple] = []
        self.queue: deque[tuple[int, int, int]] = deque()
        self.done = False

    @property
    def site_type(self) -> int:
        """BRICS type of the open site about to be filled. 0 iff root step."""
        if not self.frags or not self.queue:
            return 0
        return self.queue[0][2]

    def legal(self) -> np.ndarray:
        if self.done:
            return self.t._legal_done
        if not self.frags:
            return self.t.legal_by_type[0]
        if not self.queue:
            return self.t._legal_empty
        return self.t.legal_by_type[self.site_type]

    def apply(self, action: int) -> None:
        if self.done:
            return
        t = self.t
        if action == t.stop_id:
            self.queue.clear()
            self.done = True
            return
        if action == t.cap_id:
            # The dummy stays in `frags` and survives `reassemble`; `to_mol`
            # strips it to an implicit hydrogen.
            self.queue.popleft()
            return

        fid = int(t.frag_of[action])
        slot = int(t.slot_of[action])
        spec = t.specs[fid]
        new_idx = len(self.frags)

        if not self.frags:
            if slot != -1:
                raise ValueError(f"action {action} is a slot placement, not a root")
            self.frags.append(Chem.Mol(spec.mol))
            for idx, ty in spec.sites:
                self.queue.append((new_idx, idx, ty))
            return

        if slot < 0:
            raise ValueError(f"action {action} is a root placement, but a site is open")
        fa, da, ta = self.queue.popleft()
        tb = int(t.type_of[action])
        if tb not in COMPAT.get(ta, frozenset()):
            raise ValueError(f"type {tb} is not compatible with site type {ta}")
        self.frags.append(Chem.Mol(spec.mol))
        self.junctions.append((fa, da, new_idx, slot, bond_order(ta, tb)))
        for idx, ty in spec.sites:
            if idx != slot:
                self.queue.append((new_idx, idx, ty))

    def to_mol(self) -> Chem.Mol | None:
        """Materialize, strip caps, sanitize. None on any chemical failure.

        **"Valid by construction" covers connectivity and valence, not
        aromaticity.** Type compatibility plus an explicit bond order makes
        every junction a chemically legal bond, but joining two type-compatible
        AROMATIC fragments can still produce a ring system RDKit cannot
        kekulize. Measured failure modes over random-policy rollouts: 118/118
        are `KekulizeException`, none are valence. Measured rates: an untrained
        policy fails 4.61% (118/2560), the MLE warm start 0/2048, the 20-step
        GRPO policy 0/2048, and the 20-step run itself 1/1280 -- so a drifting
        policy can still reach one. That failure returns "" here and routes
        through the invalidity hurdle like any other invalid molecule, which is
        the correct direction; it is one more reason the warm start is not
        optional.
        """
        if not self.frags:
            return None
        try:
            mol = reassemble(self.frags, self.junctions)
            mol = Chem.DeleteSubstructs(mol, _DUMMY)
            Chem.SanitizeMol(mol)
        except Exception:
            # reassemble sanitizes unguarded, so a bad valence raises rather
            # than returning None; IndexError covers reassemble([], []).
            return None
        return mol

    def smiles(self) -> str:
        """Terminal SMILES, or "" if the molecule is not scoreable.

        Terminal-only reward, per handoff section 5.3. Since the merge of
        classifier-v2 both `featurize.smiles_to_graph` and `reward.sanitizable`
        reject dummy atoms, so a leak now scores as invalid rather than silently
        earning a meaningless C. This stays as the first line of defence
        regardless -- a partial graph's prediction is meaningless and should
        never reach the classifier at all.
        """
        mol = self.to_mol()
        if mol is None:
            return ""
        smi = Chem.MolToSmiles(mol)
        return "" if "*" in smi else smi


def _canon_perm(frag: Chem.Mol) -> list[int]:
    """Map an original atom index to its index in the re-parsed canonical SMILES.

    `_smilesAtomOutputOrder` is only set as a side effect of `MolToSmiles`, so
    it must be read after that call and never before.
    """
    Chem.MolToSmiles(frag)
    raw = frag.GetProp("_smilesAtomOutputOrder").strip("[],")
    order = [int(x) for x in raw.split(",") if x != ""]
    perm = [0] * len(order)
    for pos, orig in enumerate(order):
        perm[orig] = pos
    return perm


def trace(
    smiles: str,
    table: ActionTable,
    stoi: dict[str, int],
    max_len: int = 64,
) -> list[int] | None:
    """Oracle action sequence for a real molecule, ending in STOP.

    None if unparseable, out of vocabulary, or longer than `max_len`. The oracle
    drives the SAME `AssemblyState.apply` the sampler uses, so the warm-start
    distribution and the sampler cannot diverge.
    """
    # Must match `corpus()` exactly, or a fragment the trace needs has no token.
    prepped = corpus([smiles])
    if not prepped:
        return None
    mol = Chem.MolFromSmiles(prepped[0])
    if mol is None:
        return None

    dfrags, djunctions = decompose(mol)
    # Root uniqueness: GetMolFrags orders components by first atom occurrence,
    # so after the canonicalization in `largest_component`, dfrags[0] holds
    # canonical atom 0. Without that canonicalization the root -- and therefore
    # the whole trajectory -- would depend on input spelling.
    assert any(a.GetIdx() == 0 for a in dfrags[0].GetAtoms()), "root is not canonical atom 0"

    perms = [_canon_perm(f) for f in dfrags]
    ids = [stoi.get(Chem.MolToSmiles(f)) for f in dfrags]
    if any(i is None or i == 0 for i in ids):
        return None  # OOV: with min_count=1 over `corpus(...)` this is empty

    # (decompose frag, canonical dummy index) -> (partner frag, partner dummy)
    link: dict[tuple[int, int], tuple[int, int]] = {}
    for fa, da, fb, db, _bt in djunctions:
        ca, cb = perms[fa][da], perms[fb][db]
        link[(fa, ca)] = (fb, cb)
        link[(fb, cb)] = (fa, ca)

    root = table.root_action.get(ids[0])
    if root is None:
        return None
    state = AssemblyState(table)
    state.apply(root)
    actions = [root]
    asm2dec = [0]
    # BRICS only cuts acyclic bonds, so the motif graph of a connected molecule
    # is a tree. Consuming each edge once makes this walk terminate by
    # construction rather than by the max_len bound.
    seen: set[tuple[int, int]] = set()

    while state.queue and len(actions) < max_len - 1:
        fa_asm, da, _ta = state.queue[0]
        key = (asm2dec[fa_asm], da)
        partner = link.get(key)
        if partner is None or key in seen or partner in seen:
            actions.append(table.cap_id)
            state.apply(table.cap_id)
            continue
        seen.add(key)
        seen.add(partner)
        dec_fb, db = partner
        act = table.slot_action.get((ids[dec_fb], db))
        if act is None:
            return None
        actions.append(act)
        state.apply(act)
        asm2dec.append(dec_fb)

    if state.queue:
        return None  # did not finish inside max_len
    actions.append(table.stop_id)
    return actions


def replay(actions, table: ActionTable) -> AssemblyState:
    """Drive a fresh state through an action sequence."""
    state = AssemblyState(table)
    for a in actions:
        if state.done:
            break
        state.apply(int(a))
    return state


def _self_check() -> None:
    import pandas as pd
    from rdkit import RDLogger

    from .brics import vocab

    RDLogger.DisableLog("rdApp.*")

    # --- the compatibility table and the bond rule -----------------------
    for ta, partners in COMPAT.items():
        for tb in partners:
            assert ta in COMPAT[tb], f"COMPAT asymmetric at {ta}/{tb}"
    assert bond_order(7, 7) == Chem.BondType.DOUBLE
    assert bond_order(3, 16) == Chem.BondType.SINGLE
    assert bond_order(1, 3) == Chem.BondType.SINGLE
    print(f"COMPAT symmetric over {len(COMPAT)} types; bond_order(7,7)=DOUBLE")

    df = pd.read_csv("data/raw/BBBP.csv")
    canon = corpus(list(df["smiles"]))
    stoi = vocab(canon, min_count=1)
    specs = build_specs(stoi)
    table = ActionTable(specs)
    print(f"corpus {len(canon)} molecules; vocab {len(stoi)} fragments; "
          f"{table.n_actions} actions "
          f"({int((table.slot_of == -1).sum())} root + "
          f"{int((table.slot_of >= 0).sum())} slot + CAP + STOP + PAD)")

    # --- the alkene regression, explicitly -------------------------------
    alkene = "CN(C)CCC=C1c2ccccc2CCc2ccccc21"
    acts = trace(alkene, table, stoi)
    assert acts is not None, "alkene not traceable"
    got = replay(acts, table).smiles()
    want = corpus([alkene])[0]
    assert got == want, f"type-7 junction lost:\n  got  {got}\n  want {want}"
    print("type-7 alkene round-trips; double bond survives assembly")

    # --- reconstruction over the whole corpus ----------------------------
    ok = untraceable = mismatch = 0
    longest = 0
    bad = None
    for s in canon:
        acts = trace(s, table, stoi)
        if acts is None:
            untraceable += 1
            continue
        assert acts[-1] == table.stop_id, "trace does not end in STOP"
        longest = max(longest, len(acts))
        if replay(acts, table).smiles() == s:
            ok += 1
        else:
            mismatch += 1
            if bad is None:
                bad = (s, replay(acts, table).smiles())
    n = len(canon)
    rate = ok / n
    print(f"reconstruction {ok}/{n} = {rate:.4f}  "
          f"(untraceable {untraceable}, mismatch {mismatch}); "
          f"longest trace {longest} actions")
    if bad:
        print(f"  first mismatch: {bad[0]}\n               -> {bad[1]}")

    # --- the terminal guard ----------------------------------------------
    multi = next(fid for fid, sp in enumerate(specs)
                 if sp is not None and len(sp.sites) >= 2)
    st = AssemblyState(table)
    st.apply(table.root_action[multi])
    while st.queue:
        st.apply(table.cap_id)
    capped = st.smiles()
    assert capped and "*" not in capped, f"capped molecule leaked a dummy: {capped!r}"
    assert AssemblyState(table).smiles() == "", "empty state must not be scoreable"
    print(f"capped site -> dummy-free SMILES ({capped}); empty state -> \"\"")

    # --- legality invariants ---------------------------------------------
    st = AssemblyState(table)
    assert not st.legal()[table.stop_id], "STOP legal on an empty molecule"
    assert st.legal()[:table.n_place][table.slot_of >= 0].sum() == 0, "slot action legal at root"
    st.apply(table.root_action[multi])
    assert st.legal()[table.stop_id] and st.legal()[table.cap_id]
    assert st.legal()[:table.n_place][table.slot_of == -1].sum() == 0, "root action legal mid-assembly"
    ta = st.site_type
    for a in np.flatnonzero(st.legal()[:table.n_place]):
        assert int(table.type_of[a]) in COMPAT[ta], "illegal type offered"
    print("root/slot actions never cross; every offered slot is type-compatible")

    # Exact is achievable, so anything less is a regression, not a tolerance.
    assert ok == n, f"reconstruction {ok}/{n} is not exact"
    assert untraceable == 0 and mismatch == 0
    print("\nbrics_assembly self-check passed")


if __name__ == "__main__":
    _self_check()

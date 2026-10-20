"""GRU policy over BRICS assembly actions. Satisfies the existing roll contract.

The automaton lives in `brics_assembly`; this module only scores it. The policy
is a GRU conditioned on `z` through the initial hidden state, which is the same
shape as `SelfiesGenerator` -- so `train_gan`'s loop, `grpo.py`, `reward.py` and
`diversity.py` need no knowledge that the action space changed.

`sample()` returns exactly `{"z", "actions", "mask", "logp_old", "smiles"}` and
`log_probs(actions, z)` re-scores the SAME stored actions. That is the whole
contract `train_gan` depends on, so matching it verbatim keeps the diff to a
generator registry.

Three things worth knowing before editing.

**The legality mask is replayed, not stored.** `log_probs` re-walks the
automaton in pure Python to recover each step's site type and legal set. The
roll dict therefore stays at five keys, but it means sampling and re-scoring
must agree exactly or the PPO ratio is not 1 at epoch 0 and every advantage is
scaled by a silent bug. `_self_check` asserts `allclose(log_probs(...),
logp_old)` for precisely this reason -- it is the single most load-bearing
assertion in the file.

**The site type is fed, not inferred.** Without it the GRU cannot tell which
attachment point it is filling, since the previous action alone does not
identify the open site.

**There is no forced STOP at `max_len`.** A truncated rollout keeps its open
dummies, and `AssemblyState.to_mol` strips every dummy to an implicit hydrogen,
so truncation degrades to implicit capping and still yields a terminal,
dummy-free molecule. Forcing STOP would make the legal mask depend on `T`, which
is a property of the batch rather than of the prefix, and replay could not
reproduce it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger

from .brics import vocab as build_frag_vocab
from .brics_assembly import ActionTable, AssemblyState, build_specs, corpus, trace
from .datasets import ROOT, _load_raw
from .generator import CKPT_DIR

RDLogger.DisableLog("rdApp.*")

MAX_LEN = 64  # longest real BBBP trace is 24 actions


class BricsGenerator(nn.Module):
    """pi_theta over BRICS assembly actions, conditioned on z."""

    def __init__(
        self,
        stoi: dict[str, int],
        emb: int = 128,
        hidden: int = 256,
        layers: int = 1,
        z_dim: int = 64,
        type_emb: int = 32,
    ) -> None:
        super().__init__()
        self.stoi = dict(stoi)
        self.specs = build_specs(self.stoi)
        self.table = ActionTable(self.specs)
        self.n_actions = self.table.n_actions
        self.bos = self.n_actions  # one extra embedding row, never an output
        self.z_dim, self.hidden, self.layers, self.emb_dim = z_dim, hidden, layers, emb

        self.emb_act = nn.Embedding(self.n_actions + 1, emb)
        self.emb_type = nn.Embedding(17, type_emb)  # 0 = root step, 1..16 = BRICS type
        self.z2h = nn.Linear(z_dim, layers * hidden)
        self.gru = nn.GRU(emb + type_emb, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, self.n_actions)

    # --- plumbing --------------------------------------------------------
    def _h0(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.z2h(z)).view(-1, self.layers, self.hidden).transpose(0, 1).contiguous()

    def _step_logits(self, prev: torch.Tensor, site_type: torch.Tensor, h):
        x = torch.cat([self.emb_act(prev), self.emb_type(site_type)], dim=-1).unsqueeze(1)
        o, h = self.gru(x, h)
        return self.out(o.squeeze(1)), h

    def _replay(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover each step's site type and legal set from the action prefix.

        Pure Python over the deterministic automaton; no chemistry. Must match
        what `sample` saw at the same step, or the ratio is not 1 at epoch 0.
        """
        n, t = actions.shape
        acts = actions.detach().cpu().numpy()
        types = np.zeros((n, t), dtype=np.int64)
        legal = np.zeros((n, t, self.n_actions), dtype=bool)
        for i in range(n):
            st = AssemblyState(self.table)
            for k in range(t):
                types[i, k] = st.site_type
                legal[i, k] = st.legal()
                st.apply(int(acts[i, k]))
        dev = actions.device
        return torch.from_numpy(types).to(dev), torch.from_numpy(legal).to(dev)

    def logits(self, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Teacher-forced, legality-masked logits. (N,T) -> (N,T,A)."""
        types, legal = self._replay(actions)
        prev = torch.cat(
            [torch.full((actions.shape[0], 1), self.bos, dtype=torch.long, device=actions.device),
             actions[:, :-1]],
            dim=1,
        )
        x = torch.cat([self.emb_act(prev), self.emb_type(types)], dim=-1)
        o, _ = self.gru(x, self._h0(z))
        lg = self.out(o)
        return lg.masked_fill(~legal, float("-inf"))

    def log_probs(self, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        lg = torch.log_softmax(self.logits(actions, z), dim=-1)
        return lg.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    # --- rollout ---------------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, max_len: int = MAX_LEN, device: str = "cpu") -> dict:
        was_training = self.training
        self.eval()
        z = torch.randn(n, self.z_dim, device=device)
        h = self._h0(z)
        prev = torch.full((n,), self.bos, dtype=torch.long, device=device)
        states = [AssemblyState(self.table) for _ in range(n)]

        acts_t, mask_t, logp_t = [], [], []
        for _ in range(max_len):
            live = np.array([not s.done for s in states])
            if not live.any():
                break
            types = torch.tensor([s.site_type for s in states], dtype=torch.long, device=device)
            legal = torch.from_numpy(np.stack([s.legal() for s in states])).to(device)
            lg, h = self._step_logits(prev, types, h)
            lg = lg.masked_fill(~legal, float("-inf"))
            logp = torch.log_softmax(lg, dim=-1)
            a = torch.multinomial(logp.exp(), 1).squeeze(-1)

            acts_t.append(a)
            logp_t.append(logp.gather(-1, a.unsqueeze(-1)).squeeze(-1))
            mask_t.append(torch.from_numpy(live.astype(np.float32)).to(device))
            for i, st in enumerate(states):
                st.apply(int(a[i]))
            prev = a

        if was_training:
            self.train()
        return {
            "z": z,
            "actions": torch.stack(acts_t, dim=1),
            "mask": torch.stack(mask_t, dim=1),
            "logp_old": torch.stack(logp_t, dim=1).detach(),
            "smiles": [s.smiles() for s in states],
        }

    # --- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                # "kind" is a trust boundary, not decoration: without it a
                # SELFIES checkpoint loads as a BRICS policy and trains against
                # the wrong action semantics in silence.
                "kind": "brics",
                "vocab": self.stoi,
                "state": self.state_dict(),
                "z_dim": self.z_dim,
                "hidden": self.hidden,
                "layers": self.layers,
                "emb": self.emb_dim,
                "type_emb": self.emb_type.embedding_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "BricsGenerator":
        ck = torch.load(path, map_location=device, weights_only=False)
        kind = ck.get("kind")
        if kind != "brics":
            raise ValueError(
                f"{path} is a {kind or 'selfies'} checkpoint, not a BRICS one. "
                f"Build one with: python -m src.brics_generator --pretrain"
            )
        gen = cls(
            ck["vocab"],
            emb=ck["emb"],
            hidden=ck["hidden"],
            layers=ck["layers"],
            z_dim=ck["z_dim"],
            type_emb=ck.get("type_emb", 32),
        ).to(device)
        gen.load_state_dict(ck["state"])
        return gen


def load_generator(path: Path, device: str = "cpu"):
    """Load whichever generator family a checkpoint belongs to.

    `train_gan.GENERATORS` maps a NAME to a class for `--generator`; this maps a
    FILE to an instance, which is what anything reading a finished run needs.
    The `kind` key is the discriminator -- a SELFIES checkpoint predates it and
    simply has none.
    """
    from .generator import SelfiesGenerator

    ck = torch.load(path, map_location="cpu", weights_only=False)
    return (BricsGenerator if ck.get("kind") == "brics" else SelfiesGenerator).load(path, device)


def build_vocab(dataset: str = "bbbp", min_count: int = 1) -> tuple[dict[str, int], list[str]]:
    """Fragment vocabulary and the matching corpus.

    `min_count=1` by design. The decisive quantity is not fragments dropped but
    molecules that stop being reconstructable, since those leave the MLE corpus:
    at a floor of 1 every BBBP molecule is covered, at 2 only 23.2% and at 3
    only 11.0%. Singletons are most of the vocabulary and almost none of the
    occurrences, but they are spread thinly across nearly every molecule. UNK is
    kept at id 0 for parity with the classifier's vocabulary and is never a
    legal action here.
    """
    smiles = _load_raw(dataset)["smiles"].dropna().tolist()
    canon = corpus(smiles)
    return build_frag_vocab(canon, min_count=min_count), canon


def pretrain(
    dataset: str = "bbbp",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    max_len: int = MAX_LEN,
    device: str = "cpu",
    seed: int = 0,
    min_count: int = 1,
    out_name: str | None = None,
) -> BricsGenerator:
    """MLE warm start on assembly traces decomposed from real molecules.

    Not optional, per handoff section 5.4: from a uniform policy every group is
    all-invalid, the group standard deviation is zero and `group_advantages`
    correctly returns no gradient, so GRPO would wait for validity to appear by
    chance. Assembly guarantees connectivity and valence but not aromaticity, so
    the warm start is also what puts the policy on drug-like chemistry rather
    than on arbitrary legal assemblies.

    **Do not report the validity gain as "the policy learned valid chemistry".**
    Untrained assembly sanitizes at 95.4% and the warm start takes it to 100%,
    but every inspected failure is a `KekulizeException` -- an RDKit
    representation limit, not a chemical one. Part of what those epochs teach is
    "avoid fragment pairs the toolkit cannot kekulize", which is an artifact of
    the cheminformatics stack. Harmless, but it is not chemistry and should not
    be counted as such.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    stoi, canon = build_vocab(dataset, min_count=min_count)
    gen = BricsGenerator(stoi).to(device)
    traces = [t for t in (trace(s, gen.table, stoi, max_len=max_len) for s in canon) if t]
    print(f"{dataset}: {len(traces)}/{len(canon)} traceable, "
          f"|V|={len(stoi)} fragments, |A|={gen.n_actions} actions, "
          f"longest trace {max(len(t) for t in traces)}")

    t_max = max(len(t) for t in traces)
    data = torch.full((len(traces), t_max), gen.table.pad_id, dtype=torch.long)
    for i, tr in enumerate(traces):
        data[i, : len(tr)] = torch.tensor(tr, dtype=torch.long)
    data = data.to(device)

    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        gen.train()
        perm = torch.randperm(len(data), device=device)
        total = nseen = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[perm[i : i + batch_size]]
            z = torch.randn(len(batch), gen.z_dim, device=device)
            # The same legality mask the sampler uses is already applied inside
            # `logits`, so pretrain and RL score under one distribution.
            lp = gen.log_probs(batch, z)
            keep = (batch != gen.table.pad_id).float()
            nll = -(lp * keep).sum() / keep.sum()
            opt.zero_grad()
            nll.backward()
            nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            opt.step()
            total += float(nll.detach()) * len(batch)
            nseen += len(batch)
        probe = gen.sample(200, max_len=max_len, device=device)["smiles"]
        valid = sum(1 for s in probe if s) / len(probe)
        print(f"  epoch {ep:2d}   nll={total / nseen:.4f}   valid={valid:.1%}")

    path = CKPT_DIR / (out_name or f"{dataset}_brics_pretrained.pt")
    gen.save(path)
    print(f"saved {path}")
    return gen


def _self_check() -> None:
    import pandas as pd

    df = pd.read_csv(ROOT / "data" / "raw" / "BBBP.csv")
    canon = corpus(list(df["smiles"])[:400])
    stoi = build_frag_vocab(canon, min_count=1)
    torch.manual_seed(0)
    gen = BricsGenerator(stoi, hidden=64, emb=32, z_dim=16)
    print(f"|V|={len(stoi)} fragments -> |A|={gen.n_actions} actions")

    roll = gen.sample(16, max_len=24)
    assert set(roll) == {"z", "actions", "mask", "logp_old", "smiles"}, set(roll)
    n, t = roll["actions"].shape
    assert roll["z"].shape == (16, 16)
    assert roll["mask"].shape == (n, t) and roll["logp_old"].shape == (n, t)
    assert not roll["logp_old"].requires_grad, "logp_old must be detached"
    assert len(roll["smiles"]) == 16
    print(f"roll keys/shapes ok: actions {tuple(roll['actions'].shape)}")

    # Terminal-only: no rollout may carry an open attachment point.
    assert all("*" not in s for s in roll["smiles"]), "a dummy atom leaked into a reward SMILES"
    valid = sum(1 for s in roll["smiles"] if s)
    print(f"{valid}/16 sampled molecules sanitize; no dummy leaked")

    # The mask is contiguous: live region, then nothing.
    m = roll["mask"]
    assert torch.all(m[:, 0] == 1), "first step must be live"
    for i in range(n):
        row = m[i].tolist()
        assert row == sorted(row, reverse=True), f"mask not contiguous: {row}"
    # Padded positions carry PAD and contribute exactly zero log-prob.
    pad = roll["actions"][m == 0]
    if pad.numel():
        assert torch.all(pad == gen.table.pad_id), "frozen rows must hold PAD"
        assert torch.allclose(roll["logp_old"][m == 0],
                              torch.zeros_like(roll["logp_old"][m == 0]))
    print("mask contiguous; frozen rows hold PAD at logp 0")

    # THE assertion: re-scoring the stored actions must reproduce the sampling
    # log-probs exactly. If the replayed legality mask drifts from the sampled
    # one, the PPO ratio is not 1 at epoch 0 and every advantage is silently
    # rescaled.
    lp = gen.log_probs(roll["actions"], roll["z"])
    assert torch.allclose(lp, roll["logp_old"], atol=1e-5), \
        f"replay drift: max |d| = {(lp - roll['logp_old']).abs().max():.3e}"
    print(f"sample/rescore agree: max |d| = {(lp - roll['logp_old']).abs().max():.2e}")

    # Gradient reaches the head, and only through the live region.
    loss = -(lp * roll["mask"]).sum() / roll["mask"].sum()
    loss.backward()
    assert gen.out.weight.grad is not None and gen.out.weight.grad.abs().sum() > 0
    print("gradient reaches out.weight")

    # Illegal actions are impossible, not merely unlikely.
    types, legal = gen._replay(roll["actions"])
    chosen = legal.gather(-1, roll["actions"].unsqueeze(-1)).squeeze(-1)
    assert torch.all(chosen), "an illegal action was sampled"
    full = gen.logits(roll["actions"], roll["z"])
    assert torch.all(torch.isinf(full[~legal])), "illegal actions are not masked to -inf"
    print("every sampled action was legal; illegal logits are -inf")

    # A real molecule's trace must score finitely under the policy.
    tr = trace(canon[0], gen.table, stoi, max_len=64)
    assert tr is not None
    a = torch.tensor([tr], dtype=torch.long)
    assert torch.isfinite(gen.log_probs(a, torch.randn(1, 16))).all(), \
        "an oracle trace is unreachable under the policy's own mask"
    print("oracle trace is reachable under the policy mask")

    # Round trip, and the kind guard.
    tmp = CKPT_DIR / "_brics_selfcheck.pt"
    gen.save(tmp)
    again = BricsGenerator.load(tmp)
    lp2 = again.log_probs(roll["actions"], roll["z"])
    assert torch.allclose(lp, lp2, atol=1e-6), "save/load changed the log-probs"
    try:
        from .generator import SelfiesGenerator
        sg = CKPT_DIR / "bbbp_pretrained.pt"
        if sg.exists():
            try:
                BricsGenerator.load(sg)
            except ValueError as e:
                assert "selfies" in str(e)
                print("a SELFIES checkpoint is refused by name")
        _ = SelfiesGenerator
    finally:
        tmp.unlink(missing_ok=True)
    print("save/load reproduces log-probs")

    print("\nbrics_generator self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrain", action="store_true")
    ap.add_argument("--dataset", default="bbbp")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.pretrain:
        pretrain(dataset=args.dataset, epochs=args.epochs, min_count=args.min_count,
                 seed=args.seed, out_name=args.out_name, device=args.device)
    else:
        _self_check()


if __name__ == "__main__":
    main()

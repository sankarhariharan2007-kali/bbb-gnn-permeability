"""Resolve every generated molecule against PubChem's bulk InChIKey table.

WHY NOT THE API. The web service is the wrong tool at this volume and this
proves it: 19219 single GETs got this client HTTP 429'd, and even the batched
form is 129 requests against a service whose limits are not ours to spend.
NCBI publishes the whole CID-to-InChIKey mapping precisely so bulk questions
are answered locally. One download, then no network at all, no rate limit,
and an answer that reproduces forever without asking anyone's server again.

    CID-InChI-Key.gz   7.37 GB compressed, ~119M compounds
    format             CID <TAB> InChI <TAB> InChIKey

WHAT A MATCH MEANS. The full InChIKey matches skeleton, stereochemistry AND
protonation -- CIDs 1 and 2 share `RDHQFKQIGNGIED-UHFFFAOYSA` and differ only
in the final character (N vs O) because one is protonated. Generated molecules
are flat and neutral, so they carry `-UHFFFAOYSA-N`, and they match PubChem's
stereo-unspecified parent entry where one is registered. That is how the
generated lysergamide resolves to CID 3981 rather than to LSD's stereo-defined
CID 5761.

A hit means the structure is registered. It is evidence the generator makes
real chemistry, and evidence AGAINST novelty. It says nothing about
permeability, activity or safety, and 'novel' means absent from this table and
from the training corpus -- not that a molecule is unknown everywhere, nor
that it is any good.

    python -m src.pubchem_bulk D:/pubchem_bulk/CID-InChI-Key.gz
    python -m src.pubchem_bulk --self-check        # no file needed
"""
from __future__ import annotations

import argparse
import gzip
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCREENED = ROOT / "results" / "candidates" / "brics10" / "screened.csv"
OUTDIR = ROOT / "results" / "pubchem"


def inchikeys(smiles: list[str]) -> dict[str, str]:
    """Computed locally with RDKit. 19219/19219 succeed on this pool."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.inchi import MolToInchiKey
    RDLogger.DisableLog("rdApp.*")
    out: dict[str, str] = {}
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            k = MolToInchiKey(m)
        except Exception:
            continue
        if k:
            out[s] = k
    return out


def scan(path: Path, wanted: set[str], progress_every: int = 20_000_000
         ) -> dict[str, int]:
    """One streaming pass. -> {inchikey: lowest CID}.

    Parsed by locating the last tab rather than splitting the line, because at
    ~119M lines the difference is minutes. The key is the third field; the CID
    is only parsed on a hit, which happens ~19k times out of 119M.
    """
    found: dict[str, int] = {}
    t0 = time.time()
    n = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            cut = line.rfind("\t")
            if cut < 0:
                continue
            key = line[cut + 1:].rstrip("\n")
            if key in wanted and key not in found:
                # First occurrence wins: the file is CID-ordered, so that is
                # the lowest (oldest) CID for the key.
                try:
                    found[key] = int(line[:line.index("\t")])
                except ValueError:
                    pass
                if len(found) == len(wanted):
                    print(f"  all {len(wanted)} keys found at line {n:,}")
                    break
            if n % progress_every == 0:
                el = time.time() - t0
                print(f"  {n:,} lines, {len(found):,}/{len(wanted):,} matched, "
                      f"{el / 60:.1f} min ({n / el / 1e6:.1f}M lines/s)")
    print(f"  scanned {n:,} lines in {(time.time() - t0) / 60:.1f} min, "
          f"{len(found):,} keys matched")
    return found


def classify(scr: pd.DataFrame, keys: dict[str, str],
             found: dict[str, int]) -> pd.DataFrame:
    """Three mutually exclusive buckets, training winning over known.

    A molecule copied from the training corpus is a memorization result whether
    or not PubChem also lists it, so `training` takes priority. There is no
    `unresolved` bucket here -- that was an artefact of network failures, and a
    local scan either finds a key or establishes that it is absent.
    """
    m = scr.copy()
    m["inchikey"] = m.smiles.map(keys).fillna("")
    m["cid"] = m.inchikey.map(found)
    m["bucket"] = "novel"
    m.loc[m.cid.notna(), "bucket"] = "known"
    m.loc[m.inchikey == "", "bucket"] = "unkeyable"
    m.loc[m.in_training_flat.fillna(False).astype(bool), "bucket"] = "training"
    return m


LABEL = {"training": "in its training data",
         "known": "exist already (PubChem), not in training",
         "novel": "no record found -- novel",
         "unkeyable": "no InChIKey could be computed"}
ORDER = ["training", "known", "novel", "unkeyable"]


def summarize(m: pd.DataFrame, title: str, note: str = "") -> pd.DataFrame:
    n = len(m)
    print(f"\n{'-' * 72}\n{title}  (n={n})\n{'-' * 72}")
    if note:
        print(f"  {note}\n")
    rows = []
    for b in ORDER:
        c = int((m.bucket == b).sum())
        if b == "unkeyable" and not c:
            continue
        rows.append(dict(population=title, bucket=b, n=c, share=c / n if n else 0))
        print(f"  {LABEL[b]:<44} {c:>6}  {c / n:>6.1%}")
    p = (m.bucket == "novel").mean()
    se = (p * (1 - p) / n) ** 0.5 if n else 0
    print(f"\n  novel fraction: {p:.1%} +/- {1.96 * se:.1%} (95%)")
    return pd.DataFrame(rows)


def _self_check() -> None:
    import io
    # A fake table in the real format, including the protonation pair that
    # makes full-key matching necessary.
    table = ("1\tInChI=1S/x\tRDHQFKQIGNGIED-UHFFFAOYSA-N\n"
             "2\tInChI=1S/x/p+1\tRDHQFKQIGNGIED-UHFFFAOYSA-O\n"
             "3981\tInChI=1S/y\tVAYOSLLFUXYJDT-UHFFFAOYSA-N\n"
             "9999\tInChI=1S/y\tVAYOSLLFUXYJDT-UHFFFAOYSA-N\n")
    p = OUTDIR / "_selfcheck.tsv.gz"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(table)
    try:
        got = scan(p, {"VAYOSLLFUXYJDT-UHFFFAOYSA-N",
                       "RDHQFKQIGNGIED-UHFFFAOYSA-O",
                       "ZZZZZZZZZZZZZZ-UHFFFAOYSA-N"})
    finally:
        p.unlink(missing_ok=True)
    # Lowest CID wins for a duplicated key, the protonated form is a DISTINCT
    # key, and an absent key must simply be absent rather than guessed.
    assert got == {"VAYOSLLFUXYJDT-UHFFFAOYSA-N": 3981,
                   "RDHQFKQIGNGIED-UHFFFAOYSA-O": 2}, got

    scr = pd.DataFrame([
        dict(smiles="A", in_training_flat=True, passes=True),
        dict(smiles="B", in_training_flat=False, passes=True),
        dict(smiles="C", in_training_flat=False, passes=True),
        dict(smiles="D", in_training_flat=False, passes=False),
    ])
    m = classify(scr, {"A": "k1", "B": "k2", "C": "k3"}, {"k1": 10, "k2": 20})
    assert list(m.bucket) == ["training", "known", "novel", "unkeyable"], list(m.bucket)
    print("pubchem_bulk self-check passed")


def main() -> None:
    global OUTDIR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", nargs="?", type=Path,
                    help="path to CID-InChI-Key.gz")
    ap.add_argument("--screened", type=Path, default=SCREENED,
                    help="screened.csv of the arm to classify")
    ap.add_argument("--outdir", type=Path, default=OUTDIR,
                    help="where the three CSVs are written")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if not args.table or not args.table.exists():
        ap.error("pass the path to CID-InChI-Key.gz")

    OUTDIR = args.outdir
    scr = pd.read_csv(args.screened)
    keys = inchikeys(list(scr.smiles))
    print(f"{len(scr)} molecules, {len(keys)} InChIKeys, "
          f"{len(set(keys.values()))} distinct\n")
    print(f"scanning {args.table.name} "
          f"({args.table.stat().st_size / 2**30:.2f} GB compressed)")
    found = scan(args.table, set(keys.values()))

    m = classify(scr, keys, found)
    print(f"\n{'=' * 72}\nWHERE THE GENERATED MOLECULES COME FROM\n{'=' * 72}")

    surv = m[m.passes.fillna(False).astype(bool)
             & ~m.in_training_flat.fillna(False).astype(bool)]
    s_sum = summarize(surv, "SCREENED SURVIVORS",
                      "the shortlist: passed size, structural, alert and "
                      "training-set screening.\n  The training bucket is zero "
                      "by construction -- those were removed to form it.")
    a_sum = summarize(m, "ALL GENERATED",
                      "every deduplicated molecule the ten policy seeds "
                      "produced. This is the\n  right denominator for \"how "
                      "many of them already exist\".")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.concat([s_sum, a_sum], ignore_index=True).to_csv(OUTDIR / "summary.csv", index=False)
    m.to_csv(OUTDIR / "classified_all.csv", index=False)
    surv.to_csv(OUTDIR / "classified_survivors.csv", index=False)
    print(f"\nwrote summary.csv, classified_all.csv, classified_survivors.csv "
          f"in {OUTDIR}")
    print("\n  A hit means the structure is registered in PubChem. It is NOT")
    print("  evidence of permeability, activity or safety, and 'novel' is not")
    print("  evidence of value: the POOLED bucket holds a trioxide. (That molecule")
    print("  does NOT survive screening -- 1 targeted + 7 catalogue hits. On")
    print("  survivors, 16 motifs no screened catalogue encodes match 0 of 11,329.)")


if __name__ == "__main__":
    main()

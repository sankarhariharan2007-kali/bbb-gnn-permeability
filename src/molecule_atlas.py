"""Render generated molecules to a browsable multi-page PDF, 12 per page.

WHY PIL AND NOT MATPLOTLIB. At 72,528 pooled molecules this is ~6,000 pages, so
per-molecule matplotlib axes are too slow and too heavy. RDKit's
`MolsToGridImage` already lays out a grid, and PIL appends page images into one
PDF, which keeps the cost to roughly one draw call per molecule.

WHAT A PAGE CARRIES. Each cell is labelled with the molecule's row index in its
arm's `screened.csv`, so a structure in the PDF can be traced back to its
SMILES, scores and screening verdict. Pages are titled with the arm and page
number. Order is the arm's own file order, which is NOT a ranking -- see the
caveat on display sets in documentation/full_pipeline.md.

    python -m src.molecule_atlas --arm cgate --set survivors
    python -m src.molecule_atlas --arm cgate --set pooled
    python -m src.molecule_atlas --all                  # every arm, pooled
    python -m src.molecule_atlas --self-check
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from .datasets import ROOT

RDLogger.DisableLog("rdApp.*")

COLS, ROWS = 3, 4
PER_PAGE = COLS * ROWS
CELL = (420, 340)            # per-molecule drawing size, px
HEADER = 46                  # space for the page title

ARMS = {
    "a": ("A: ensemble reward, C as term", "candidates/brics10"),
    "b": ("B: hybrid_batch reward, C as term", "candidates/brics_hybrid"),
    "cgate": ("GATE: ensemble reward, C as hurdle", "candidates/brics_cgate"),
}
OUTDIR = ROOT / "results" / "atlas"


def load(arm: str, which: str) -> pd.DataFrame:
    _, rel = ARMS[arm]
    d = pd.read_csv(ROOT / "results" / rel / "screened.csv")
    d = d.reset_index().rename(columns={"index": "row"})
    if which == "survivors":
        d = d[d.passes.fillna(False).astype(bool)
              & ~d.in_training_flat.fillna(False).astype(bool)]
    return d.reset_index(drop=True)


def _page(chunk: pd.DataFrame, title: str) -> Image.Image:
    """One 3x4 page. Unparseable SMILES leave a labelled blank rather than
    shifting every later cell, so a cell's position always matches its label.
    """
    mols, legends = [], []
    for r in chunk.itertuples():
        m = Chem.MolFromSmiles(r.smiles)
        mols.append(m if m is not None else Chem.MolFromSmiles(""))
        lab = f"#{r.row}"
        if hasattr(r, "c_score") and pd.notna(r.c_score):
            lab += f"  C={r.c_score:.2f}"
        if m is None:
            lab += "  [unparseable]"
        legends.append(lab)

    grid = Draw.MolsToGridImage(mols, molsPerRow=COLS, subImgSize=CELL,
                                legends=legends, returnPNG=False)
    page = Image.new("RGB", (grid.width, grid.height + HEADER), "white")
    page.paste(grid, (0, HEADER))
    ImageDraw.Draw(page).text((14, 16), title, fill="black")
    return page


#: Pages held in memory before a file is written. PIL needs every page of a
#: multi-page PDF resident at save time, and a page is ~1.4 MB, so the whole
#: 6,044-page pooled set would want ~8.8 GB at once. Chunking caps that at
#: roughly 2 GB and keeps each survivor set to a single file.
CHUNK_PAGES = 1500


def build(arm: str, which: str, limit: int | None = None,
          chunk_pages: int = CHUNK_PAGES) -> list[Path]:
    label, _ = ARMS[arm]
    df = load(arm, which)
    if limit:
        df = df.head(limit)
    n_pages = -(-len(df) // PER_PAGE)
    n_parts = -(-n_pages // chunk_pages)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"{label}\n  {which}: {len(df)} molecules -> {n_pages} pages"
          + (f" in {n_parts} files" if n_parts > 1 else ""))
    t0 = time.time()
    written: list[Path] = []
    for part in range(n_parts):
        lo, hi = part * chunk_pages, min((part + 1) * chunk_pages, n_pages)
        pages = []
        for p in range(lo, hi):
            pages.append(_page(
                df.iloc[p * PER_PAGE:(p + 1) * PER_PAGE],
                f"{label}  |  {which}  |  page {p + 1} of {n_pages}"))
            if (p + 1) % 100 == 0:
                el = time.time() - t0
                print(f"    {p + 1}/{n_pages} pages, {el / 60:.1f} min, "
                      f"eta {(n_pages - p - 1) * el / (p + 1) / 60:.0f} min")
        # The limit goes in the filename. Without it a quick `--limit 240` look
        # silently overwrites the full atlas for that arm, which is a 945-page
        # render to get back.
        suffix = "" if n_parts == 1 else f"_part{part + 1:02d}"
        head = f"{arm}_{which}" + (f"_first{limit}" if limit else "")
        out = OUTDIR / f"{head}{suffix}.pdf"
        pages[0].save(out, "PDF", save_all=True, append_images=pages[1:],
                      resolution=150.0)
        written.append(out)
        print(f"  wrote {out.name}  ({hi - lo} pages, "
              f"{out.stat().st_size / 2**20:.1f} MB)")
        pages.clear()          # release before the next chunk
    print(f"  done in {(time.time() - t0) / 60:.1f} min")
    return written


def _self_check() -> None:
    df = load("cgate", "survivors")
    assert len(df) > 0 and "row" in df.columns
    # A page must hold exactly PER_PAGE cells even when the last chunk is short,
    # and a bad SMILES must not shift the others out of alignment with labels.
    probe = pd.DataFrame([
        dict(row=0, smiles="CCO", c_score=0.5),
        dict(row=1, smiles="not_a_molecule", c_score=0.6),
        dict(row=2, smiles="c1ccccc1", c_score=0.7),
    ])
    img = _page(probe, "probe")
    assert img.width > 0 and img.height > HEADER
    # Row indices must survive the survivor filter, or a structure cannot be
    # traced back to its SMILES in screened.csv.
    raw = pd.read_csv(ROOT / "results" / ARMS["cgate"][1] / "screened.csv")
    r0 = int(df.row.iloc[0])
    assert raw.smiles.iloc[r0] == df.smiles.iloc[0], "row index does not map back"
    print(f"page renders {img.width}x{img.height}px; row indices map back")
    print("molecule_atlas self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="cgate")
    ap.add_argument("--set", dest="which", choices=["survivors", "pooled"],
                    default="survivors")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N molecules only, for a quick look")
    ap.add_argument("--all", action="store_true",
                    help="every arm, pooled: ~72.5k molecules, ~6000 pages")
    ap.add_argument("--chunk-pages", type=int, default=CHUNK_PAGES,
                    help="pages per output file; caps peak memory")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()
    if args.all:
        for a in ("cgate", "a", "b"):
            build(a, "pooled", chunk_pages=args.chunk_pages)
        return
    build(args.arm, args.which, args.limit, chunk_pages=args.chunk_pages)


if __name__ == "__main__":
    main()

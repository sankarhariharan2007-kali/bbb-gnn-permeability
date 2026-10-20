"""Format audit: page count, page size, margins, font sizes, and per-page fill (body pages)."""
import sys, fitz
f = sys.argv[1]; doc = fitz.open(f)
print(f"{f}: {len(doc)} pages, size {doc[0].rect.width:.0f}x{doc[0].rect.height:.0f} pt")
sizes = {}; 
for i, pg in enumerate(doc, 1):
    minx = miny = 1e9; maxx = maxy = -1; txt_sizes = []
    for b in pg.get_text("dict")["blocks"]:
        if b["type"] == 0:
            for l in b["lines"]:
                for s in l["spans"]:
                    if not s["text"].strip(): continue
                    x0, y0, x1, y1 = s["bbox"]; txt_sizes.append(round(s["size"], 1))
                    # ignore the page number line (below the text block) for margin check
                    if y0 > 792 - 72: continue
                    minx, miny, maxx, maxy = min(minx, x0), min(miny, y0), max(maxx, x1), max(maxy, y1)
    for im in pg.get_image_info():
        x0, y0, x1, y1 = im["bbox"]; minx, miny, maxx, maxy = min(minx, x0), min(miny, y0), max(maxx, x1), max(maxy, y1)
    for d in pg.get_drawings():
        r = d["rect"]
        if r.width > 1 or r.height > 1:
            minx, miny, maxx, maxy = min(minx, r.x0), min(miny, r.y0), max(maxx, r.x1), max(maxy, r.y1)
    body = [s for s in txt_sizes]
    print(f"p{i:2d}: content x[{minx:5.1f},{maxx:5.1f}] y[{miny:5.1f},{maxy:5.1f}]  (limits 72..540, 72..720)  fill={(maxy-72)/(720-72)*100:5.1f}%  text sizes min={min(body) if body else '-'}")

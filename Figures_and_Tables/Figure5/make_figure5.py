#!/usr/bin/env python3
"""Figure 5 — the Semblance 2.0 application UI (two-panel screenshot composite).

Unlike Figure1-4 (data plots), Figure 5 is a faithful screenshot of the live app
(`ctrp-gdsc-ccle-ml/7_app`). It is regenerated in three steps; this script documents and
performs steps 2-3 (compose), given the two raw screenshots from step 1.

STEP 1 — capture (run the real app, screenshot both tabs with Playwright/Chromium):
    # backend engine (Signature Match) — CPU only, atlas bundled, BioLORD lazy:
    cd 7_app/backend && PORT=8080 python3 main.py            # GET /healthz -> {"engine":"real"}
    # static frontend (Cell Line Lookup):
    cd 7_app/frontend && python3 -m http.server 8000
    # then, in a headless Chromium (device_scale_factor=2, viewport width 1240):
    #   Tab A: select cell line ACH-000219 (A375, BRAF-mutant melanoma) -> full_page png -> tabA.png
    #   Tab B: click 'Load an example melanoma signature', set Top K = 10, Search -> full_page png -> tabB.png
    # config.js BACKEND_URL must point at http://localhost:8080 for Tab B to render live matches.

STEP 2-3 — compose (this script): trim each panel, drop panel B's duplicated header, place the
two panels SIDE BY SIDE (landscape) with (a)/(b) labels, and emit Figure5.png + Figure5.pdf.
Expects tabA.png / tabB.png beside it.
"""
import pathlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np

D = pathlib.Path(__file__).parent
A = Image.open(D / "tabA.png").convert("RGB")
B = Image.open(D / "tabB.png").convert("RGB")


def trim_bottom(im, pad=28):
    a = np.asarray(im)
    nonwhite = (a < 245).any(axis=2).any(axis=1)
    last = np.where(nonwhite)[0]
    bot = (last.max() + pad) if len(last) else im.height
    return im.crop((0, 0, im.width, min(bot, im.height)))


def border(im, col=(210, 216, 226), w=2):
    ImageDraw.Draw(im).rectangle([0, 0, im.width - 1, im.height - 1], outline=col, width=w)
    return im


def header_cut(im, pad=50):
    """y at which to crop panel B so the tab bar is kept but the title/disclaimer are not.

    Found from the active tab's indigo underline rather than hard-coded, because the offset moves
    with the capture's device scale factor (it was ~455 for the original 2x/1240 shots and ~300
    for the current ones). Falls back to no crop if the underline isn't found.
    """
    a = np.asarray(im.convert("RGB")).astype(int)
    blue = (a[:, :, 2] > 140) & (a[:, :, 2] - a[:, :, 0] > 50) & (a[:, :, 2] - a[:, :, 1] > 40)
    rows = np.where(blue.sum(1) > 60)[0]
    rows = rows[rows < im.height // 2]
    if not len(rows):
        return 0
    # the tab label sits just above its underline; keep both plus a little breathing room
    return max(0, int(rows[0]) - pad)


# Panel A: full page, trim trailing whitespace. Panel B: drop the duplicated header/disclaimer
# above the tab bar, then trim.
A = border(trim_bottom(A))
B = border(trim_bottom(B.crop((0, header_cut(B), B.width, B.height))))


def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


F = font(64)
PAD, GAP, LBL = 46, 60, 104
colW = max(A.width, B.width)
H = max(A.height, B.height)
canvas = Image.new("RGB", (PAD + colW + GAP + colW + PAD, PAD + LBL + H + PAD), "white")
d = ImageDraw.Draw(canvas)
xA, xB, ytop = PAD, PAD + colW + GAP, PAD
d.text((xA, ytop + 18), "a", fill=(20, 28, 48), font=F)
d.text((xB, ytop + 18), "b", fill=(20, 28, 48), font=F)
canvas.paste(A, (xA, ytop + LBL))
canvas.paste(B, (xB, ytop + LBL))

canvas.save(D / "Figure5.png")
canvas.save(D / "Figure5.pdf", "PDF", resolution=300.0)
print("Figure5", canvas.size, "aspect", round(canvas.width / canvas.height, 3))

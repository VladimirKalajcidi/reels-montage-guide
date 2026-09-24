"""Контакт-лист ключевых кадров ролика 31 — смысловая проверка графики глазами."""
import os
import sys

import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard31 import SRC, SHOTS, shot_at
from style import FPS
import render31

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/claude-501/-Users-vladimirkalajcidi-reels-good/91139b87-4ecc-4df3-b81a-62353af7a1fa/scratchpad/contact31"
TIMES = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None

if TIMES is None:
    TIMES = []
    for t0, t1, kind, _ in SHOTS:
        if kind == "A1":
            continue
        TIMES += [round(t0 + 0.10, 2), round(t0 + (t1 - t0) * 0.55, 2), round(t1 - 0.06, 2)]

os.makedirs(OUTDIR, exist_ok=True)
cap = cv2.VideoCapture(SRC)
for t in TIMES:
    fno = int(round(t * FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ok, fr = cap.read()
    t0, t1, kind, prm = shot_at(t)

    def read_stock():
        c = cv2.VideoCapture(f"{render31.STOCK_DIR}/stock_{prm['clip']}.mp4")
        c.set(cv2.CAP_PROP_POS_MSEC, (prm.get("ss", 0) + (t - t0)) * 1000)
        _, s = c.read()
        c.release()
        return s

    canvas = render31.compose(kind, prm, fr, t, t0, read_stock if kind == "stock" else None)
    canvas.convert("RGB").resize((432, 768), Image.LANCZOS).save(
        f"{OUTDIR}/{t:06.2f}_{kind}.jpg", quality=88)
cap.release()
print("кадров:", len(TIMES), "->", OUTDIR)

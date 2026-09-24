import os, sys
import cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard30 import SRC, SHOTS, shot_at
from style import FPS
import render30 as R

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/claude-501/-Users-vladimirkalajcidi-reels-good/2cd08acb-89e0-4900-8db2-cb5894ed5f61/scratchpad/probe30"
os.makedirs(OUTDIR, exist_ok=True)

TIMES = []
for t0, t1, kind, _ in SHOTS:
    TIMES += [round(t0 + 0.05, 2), round(t0 + (t1 - t0) * 0.6, 2), round(t1 - 0.05, 2)]

cap = cv2.VideoCapture(SRC)
for t in TIMES:
    fno = int(round(t * FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ok, fr = cap.read()
    t0, t1, kind, prm = shot_at(t)

    def read_stock():
        c = cv2.VideoCapture(f"{R.STOCK_DIR}/stock_{prm['clip']}.mp4")
        c.set(cv2.CAP_PROP_POS_MSEC, (prm.get("ss", 0) + (t - t0)) * 1000)
        _, s = c.read()
        c.release()
        return s

    canvas = R.compose(kind, prm, fr, t, t0, read_stock if kind == "stock" else None)
    canvas.convert("RGB").resize((432, 768), Image.LANCZOS).save(
        f"{OUTDIR}/{t:06.2f}_{kind}.jpg", quality=88)
cap.release()
print("кадров:", len(TIMES), "->", OUTDIR)

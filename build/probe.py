"""Пробный рендер отдельных кадров — проверка до полного прогона."""
import sys, os, math
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render as R
from storyboard import SHOTS

TIMES = [float(x) for x in sys.argv[1:]] or [0.9, 3.9, 7.5, 14.0, 17.5, 21.5, 26.6, 30.0,
                                             38.2, 42.6, 47.0, 51.0, 57.4, 60.5, 63.5, 66.5, 70.0]
CW, CH = CARD_A[2], CARD_A[3]
cap = {"A1": cv2.VideoCapture(R.A1), "A2": cv2.VideoCapture(R.A2)}
maskA = rounded_mask(CW, CH, R_A)

for t in TIMES:
    f = int(round(t * FPS))
    si = 0
    while si < len(SHOTS) - 1 and t >= SHOTS[si][1]:
        si += 1
    t0, t1, kind, prm = SHOTS[si]
    lt = t - t0
    c = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in ("A1", "A2"):
        v = cap[kind]; v.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = v.read()
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        c.paste(pil, (CARD_A[0], CARD_A[1]), maskA)
    else:
        c.paste(grid_canvas(CW, CH, phase=f * 0.02), (CARD_A[0], CARD_A[1]), maskA)
    L, T = R.shot_layers(kind, prm, lt, t1 - t0, t)
    for l in L:
        c.alpha_composite(l)
    items = T + R.caption_items(t)
    if items:
        c.alpha_composite(text_layer((W, H), items))
    c.convert("RGB").save(f"test/p_{t:05.1f}_{kind}.jpg", quality=90)
    print(f"{t:6.2f}  {kind}")

"""QA ролика 9.

1. яркие пиксели вне карточки          (по готовому mp4)
2. пересечение строк внутри блока      (по раскадровке)
3. графика пересекает текст            (по слоям, до рендера)
4. графика вылезает из своей зоны      (по слоям, до рендера)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard9 import OUT, SHOTS, CAPS, DUR, FACE_KINDS, GFX_ZONE, GFX_X, shot_at
from style import CARD_A, CARD_B, FPS
import render9

ALPHA = 40          # порог видимости пикселя


def bright_outside():
    cap = cv2.VideoCapture(OUT)
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        kind = shot_at(f / FPS)[2]
        rect = CARD_B if kind == "stock" else CARD_A
        mask = np.zeros(img.shape[:2], np.uint8)
        x, y, w, h = rect
        mask[y:y + h, x:x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:8]


def line_overlaps():
    bad = 0
    for i in range(len(CAPS) - 1):
        if CAPS[i + 1][0] - CAPS[i][1] <= 0.30:
            if shot_at(CAPS[i][0]) is not shot_at(CAPS[i + 1][0]):
                continue
            s1 = max(sz for _, _, sz in CAPS[i][2])
            s2 = max(sz for _, _, sz in CAPS[i + 1][2])
            slot = render9.SLOTS[render9.slot_for(shot_at(CAPS[i][0])[2])]
            if slot["step"] < (s1 + s2) * 0.50:
                bad += 1
    return bad


def gfx_text_overlap():
    """Слой графики и слой субтитров не должны иметь общих пикселей."""
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        t0, _, kind, _ = shot_at(t)
        gl = render9.graphics_layer(kind, t - t0)
        if gl is None:
            continue
        cl = render9.caption_layer(t)
        if cl is None:
            continue
        ga = np.array(gl.split()[3])
        ca = np.array(cl.split()[3])
        n = int(np.count_nonzero((ga > ALPHA) & (ca > ALPHA)))
        if n:
            bad += 1
            sample.append((round(t, 2), kind, n))
    return bad, sample[:8]


def gfx_in_zone():
    """Элементы графики лежат в y 700…1560, x 175…905."""
    y0, y1 = GFX_ZONE
    x0, x1 = GFX_X
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        ts, _, kind, _ = shot_at(t)
        gl = render9.graphics_layer(kind, t - ts)
        if gl is None:
            continue
        a = np.array(gl.split()[3])
        ys, xs = np.nonzero(a > ALPHA)
        if len(ys) == 0:
            continue
        if ys.min() < y0 or ys.max() > y1 or xs.min() < x0 or xs.max() > x1:
            bad += 1
            sample.append((round(t, 2), kind,
                           int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return bad, sample[:8]


if __name__ == "__main__":
    o, os_ = gfx_text_overlap()
    print(f"gfx_text_overlap_frames={o} sample={os_}")
    z, zs = gfx_in_zone()
    print(f"gfx_out_of_zone_frames={z} sample={zs}")
    print(f"line_overlap_pairs={line_overlaps()}")
    if os.path.exists(OUT):
        b, bs = bright_outside()
        print(f"bright_outside_card_frames={b} sample={bs}")

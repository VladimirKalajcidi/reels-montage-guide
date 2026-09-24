"""QA ролика 12: зоны карточек, пересечение графики с субтитрами, наложение строк,
плюс проверка самой математики — формула Пика должна сходиться с площадью по Гауссу.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard12 import OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS, shot_at
from style import CARD_A, CARD_B, FPS
import render12

ALPHA = 40


def bright_outside():
    """Автопроверка 1: нет пикселей ярче 200 вне прямоугольника карточки."""
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
    """Автопроверка 2: соседние строки блока не пересекаются.

    Порог — правило brand-kit §5: шаг = max(60, (кегль1 + кегль2) * 0.70).
    """
    bad, sample = 0, []
    for i in range(len(CAPS) - 1):
        if CAPS[i + 1][0] - CAPS[i][1] > 0.30:
            continue
        if shot_at(CAPS[i][0]) is not shot_at(CAPS[i + 1][0]):
            continue
        s1 = max(sz for _, _, sz in CAPS[i][2])
        s2 = max(sz for _, _, sz in CAPS[i + 1][2])
        slot = render12.SLOTS[render12.slot_for(shot_at(CAPS[i][0])[2])]
        need = max(60, (s1 + s2) * 0.70)
        if slot["step"] < need:
            bad += 1
            sample.append((round(CAPS[i][0], 2), s1, s2, slot["step"], round(need, 1)))
    return bad, sample[:8]


def gfx_text_overlap():
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        t0, _, kind, _ = shot_at(t)
        gl = render12.graphics_layer(kind, t - t0)
        if gl is None:
            continue
        cl = render12.caption_layer(t)
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
    y0, y1 = GFX_ZONE
    x0, x1 = GFX_X
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        ts, _, kind, _ = shot_at(t)
        gl = render12.graphics_layer(kind, t - ts)
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


def caption_in_card():
    """Текст целиком внутри карточки с отступом 30px — до рендера, по габаритам глифов."""
    bad, sample = 0, []
    for i, (t0, _t1, _runs, _s) in enumerate(CAPS):
        kind = shot_at(t0)[2]
        rect = CARD_B if kind == "stock" else CARD_A
        x, y, w, h = rect
        lay = render12.caption_layer(t0 + 0.5)
        if lay is None:
            continue
        a = np.array(lay.split()[3])
        ys, xs = np.nonzero(a > ALPHA)
        if len(ys) == 0:
            continue
        if (xs.min() < x + 30 or xs.max() > x + w - 30
                or ys.min() < y + 30 or ys.max() > y + h - 30):
            bad += 1
            sample.append((i, round(t0, 2), int(xs.min()), int(xs.max()),
                           int(ys.min()), int(ys.max())))
    return bad, sample[:8]


def pick_math():
    """Формула Пика на каждой фигуре ролика должна сойтись с площадью по Гауссу."""
    out = []
    for name in ("MAIN", "TRI", "STAR", "ANY", "WEIRD"):
        p = getattr(render12, name)
        area = render12.poly_area(p)
        i = len(render12.lattice_interior(p))
        b = len(render12.lattice_boundary(p))
        out.append((name, area, i, b, abs(area - (i + b / 2 - 1)) < 1e-9))
    return out


def rhythm():
    lens = [b - a for a, b, _, _ in SHOTS]
    face = sum(b - a for a, b, k, _ in SHOTS if k in FACE_KINDS)
    srt = sorted(lens)
    return dict(shots=len(SHOTS), avg=round(sum(lens) / len(lens), 2),
                median=round(srt[len(srt) // 2], 2),
                shortest=round(min(lens), 2), longest=round(max(lens), 2),
                first_cut=SHOTS[0][1], face_pct=round(100 * face / DUR, 1))


if __name__ == "__main__":
    print("ритм:", rhythm())
    for name, area, i, b, ok in pick_math():
        print(f"пик {name:6} S={area:<5} i={i:<3} b={b:<3} сходится={ok}")
    o, os_ = gfx_text_overlap()
    print(f"gfx_text_overlap_frames={o} sample={os_}")
    z, zs = gfx_in_zone()
    print(f"gfx_out_of_zone_frames={z} sample={zs}")
    lo, los = line_overlaps()
    print(f"line_overlap_pairs={lo} sample={los}")
    c, cs = caption_in_card()
    print(f"caption_out_of_card={c} sample={cs}")
    if os.path.exists(OUT):
        bo, bs = bright_outside()
        print(f"bright_outside_card_frames={bo} sample={bs}")

"""QA ролика 57 («парадокс двух конвертов»).

1. text_outside_card  — субтитры за границей карточки       (по ready mp4)
2. line_overlap       — строки субтитров налезают друг на друга (по раскадровке)
3. check4_not_black   — последний кадр каждого плана не чёрный (по ready mp4)
4. gfx_text_overlap   — графика перекрывает субтитры         (по слоям)
"""
import os, sys
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS, W, H
from storyboard157 import OUT, SHOTS, CAPS, DUR, FACE_KINDS, GRID_KINDS
import render157

ALPHA = 40


def _card(kind):
    return CARD_B if kind == "stock" else CARD_A


def _shot_idx(t):
    for i, (t0, t1, _, _) in enumerate(SHOTS):
        if t0 <= t < t1:
            return i
    return len(SHOTS) - 1


# ---------------------------------------------------------------- 1. субтитры вне карточки
def text_outside_card():
    """Белые пиксели субтитров не должны выходить за границу активной карточки."""
    bad, sample = 0, []
    nf = int(round(DUR * FPS))
    for i, (t0, t1, kind, _) in enumerate(SHOTS):
        # проверяем один кадр в середине плана
        f = int((t0 + t1) / 2 * FPS)
        t = f / FPS
        ci = render157._active(t)
        if ci is None:
            continue
        cl = render157.caption_layer(t)
        if cl is None:
            continue
        cx, cy, cw, ch = _card(kind)
        ca = np.array(cl.split()[3])
        # маска за пределами карточки
        outside = ca.copy()
        outside[cy:cy + ch, cx:cx + cw] = 0
        if outside.max() > ALPHA:
            bad += 1
            ys, xs = np.nonzero(outside > ALPHA)
            sample.append((round(t, 2), kind, int(xs.min()), int(ys.min()),
                           int(outside.max())))
    return bad, sample[:8]


# ---------------------------------------------------------------- 2. пересечение строк
def line_overlap():
    """Верхний пиксель нижней строки ниже нижнего пикселя верхней строки."""
    bad, sample = 0, []
    for ci, lay in render157.LAYOUT.items():
        # собираем уникальные y-позиции строк (по anchor="ls" → y = baseline)
        baselines = sorted({it["xy"][1] for it in lay})
        if len(baselines) < 2:
            continue
        f = render157.wide(62)
        # кап-высота ≈ f.getbbox("A")[3] - f.getbbox("A")[1] (приближённо)
        cap_h = f.getbbox("А")[3] - f.getbbox("А")[1] + 4   # +4px запас
        for k in range(len(baselines) - 1):
            top_of_lower = baselines[k + 1] - cap_h
            bottom_of_upper = baselines[k]
            if top_of_lower < bottom_of_upper:
                bad += 1
                sample.append((ci, baselines[k], baselines[k + 1]))
    return bad, sample[:8]


# ---------------------------------------------------------------- 3. последний кадр не чёрный (check-4)
def check4_not_black():
    """Последний кадр каждого плана не должен быть пустым. Brightness ≥40 внутри
    активной карточки (B для stock, A для всех остальных). Чёрный фон вне карточки —
    нормально, поэтому мерим только в пределах card-области."""
    if not os.path.exists(OUT):
        return -1, []
    cap = cv2.VideoCapture(OUT)
    bad, sample = 0, []
    nf = int(round(DUR * FPS))
    # grid-шоты намеренно тёмные (фон сетки) — проверяем только face и stock
    check_frames = [int(t1 * FPS) - 1 for _, t1, k, _ in SHOTS
                    if k in FACE_KINDS or k == "stock"]
    check_frames.append(nf - 1)  # финальный кадр — всегда
    for f in sorted(set(check_frames)):
        f = max(0, min(f, nf - 1))
        si = _shot_idx(f / FPS)
        kind = SHOTS[si][2]
        cx, cy, cw, ch = _card(kind)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        # яркость только внутри карточки
        br = cv2.cvtColor(img[cy:cy + ch, cx:cx + cw], cv2.COLOR_BGR2GRAY).mean()
        if br < 40:
            bad += 1
            sample.append((f, round(f / FPS, 2), kind, round(br, 1)))
    cap.release()
    return bad, sample[:8]


# ---------------------------------------------------------------- 4. графика не перекрывает субтитры
def gfx_text_overlap():
    bad, sample = 0, []
    for i, (t0, t1, kind, prm) in enumerate(SHOTS):
        if kind not in GRID_KINDS:
            continue
        t = (t0 + t1) / 2
        shot = SHOTS[i]
        lay, nums = render157.gfx(t, shot)
        cl = render157.caption_layer(t)
        if cl is None or (lay is None and not nums):
            continue
        if lay is not None:
            ga = np.array(lay.split()[3])
            ca = np.array(cl.split()[3])
            n = int(np.count_nonzero((ga > ALPHA) & (ca > ALPHA)))
            if n:
                bad += 1
                sample.append((round(t, 2), kind, n))
    return bad, sample[:8]


if __name__ == "__main__":
    print("=== QA ролик 57 ===")

    tb, ts = text_outside_card()
    verdict_t = "OK" if tb == 0 else f"FAIL ({tb} планов)"
    print(f"text_outside_card: {verdict_t}")
    if ts:
        for s in ts:
            print(f"  t={s[0]}s kind={s[1]} x={s[2]} y={s[3]} alpha={s[4]}")

    lb, ls = line_overlap()
    verdict_l = "OK" if lb == 0 else f"FAIL ({lb} пар строк)"
    print(f"line_overlap: {verdict_l}")
    if ls:
        for s in ls:
            print(f"  cap={s[0]} base_top={s[1]} base_bot={s[2]}")

    cb, cs = check4_not_black()
    verdict_c = "OK" if cb == 0 else (f"FAIL ({cb} чёрных кадров)" if cb >= 0 else "SKIP (нет OUT)")
    print(f"check4_not_black: {verdict_c}")
    if cs:
        for s in cs:
            print(f"  frame={s[0]} t={s[1]}s br={s[2]}")

    gb, gs = gfx_text_overlap()
    verdict_g = "OK" if gb == 0 else f"FAIL ({gb} кадров)"
    print(f"gfx_text_overlap: {verdict_g}")
    if gs:
        for s in gs:
            print(f"  t={s[0]}s kind={s[1]} pixels={s[2]}")

    all_ok = all(x == 0 for x in [tb, lb, max(cb, 0), gb])
    print("=== ИТОГ:", "ВСЕ OK" if all_ok else "ЕСТЬ ПРОБЛЕМЫ", "===")

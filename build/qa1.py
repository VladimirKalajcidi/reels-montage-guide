"""Автопроверки ролика 1 — delivery-specs.md §6. Обе должны дать ноль."""
import os, sys
import numpy as np
import cv2
from PIL import ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS
from storyboard1 import SHOTS, DUR
import render1 as R

OUT = "/Users/vladimirkalajcidi/reels_good2/videos/1/kvant_edit.mp4"


def shot_at(t):
    for t0, t1, kind, prm in SHOTS:
        if t0 <= t < t1:
            return kind
    return SHOTS[-1][2]


def card_for(kind, t=None):
    if kind in ("A1", "money"):
        return CARD_A
    x, y, w, h = CARD_B
    # R7: подпись-лейбл санкционированно живёт ПОД карточкой (shot-recipes.md R7)
    if t is not None:
        for t0, t1, _ in R.LABELS:
            if t0 <= t < t1:
                return (x, y, w, h + 80)
    return CARD_B


# ---------------------------------------------------------- 1. текст за карточкой
def check_text_outside_card():
    cap = cv2.VideoCapture(OUT)
    nf = int(round(DUR * FPS))
    bad = 0
    mask = np.zeros((1920, 1080), np.uint8)
    for f in range(0, nf, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        t = f / FPS
        kind = shot_at(t)
        x, y, w, h = card_for(kind, t)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        outside = gray.copy()
        outside[y:y + h, x:x + w] = 0
        if outside.max() > 200:
            bad += 1
            ys, xs = np.where(outside > 200)
            if bad <= 5:
                print(f"  кадр {f} ({t:.2f}s, {kind}): {len(xs)} px вне карточки, "
                      f"пример ({xs[0]},{ys[0]})={outside[ys[0],xs[0]]}")
    cap.release()
    print(f"текст за карточкой: {bad} кадров из {nf // 3 + 1} проверенных")
    return bad


# ---------------------------------------------------------- 2. наложение строк
def check_line_overlap():
    bad = 0
    checked = 0
    times = sorted(set([round(t0 + 0.05, 3) for t0, t1, runs, slot in R.CAPS] +
                        [round(t0 + (t1 - t0) / 2, 3) for t0, t1, runs, slot in R.CAPS]))
    for t in times:
        kind = shot_at(t)
        items = R.caption_items(t, kind)
        boxes = []
        for it in items:
            f = it["font"]
            txt = it["text"]
            bbox = f.getbbox(txt)
            if bbox is None:
                continue
            x0, y0 = it["xy"]
            an = it.get("anchor", "la")
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ax = {"l": 0, "m": w / 2, "r": w}[an[0]]
            ay = {"a": 0, "m": h / 2, "d": h, "t": 0}.get(an[1], 0)
            box = (x0 - ax, y0 - ay, x0 - ax + w, y0 - ay + h)
            boxes.append(box)
        checked += 1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                if ix > 0 and iy > 0:
                    bad += 1
                    if bad <= 5:
                        print(f"  t={t:.2f}s: пересечение строк {a} × {b}")
    print(f"наложение строк: {bad} пар из {checked} проверенных моментов")
    return bad


if __name__ == "__main__":
    print("=== QA ролик 1 ===")
    n1 = check_text_outside_card()
    n2 = check_line_overlap()
    print()
    print("ИТОГО:", "ПРОШЕЛ" if n1 == 0 and n2 == 0 else f"брак: outside={n1} overlap={n2}")

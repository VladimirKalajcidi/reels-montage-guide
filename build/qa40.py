"""Автопроверки ролика 8 («аналитик спроса») — delivery-specs.md §6. Обе должны дать ноль."""
import os, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS, W, H, grid_canvas, text_layer
from storyboard40 import SHOTS, DUR
import render40 as R

OUT = "/Users/vladimirkalajcidi/reels_good2/videos/8/analyst_edit.mp4"


def shot_at(t):
    for t0, t1, kind, prm in SHOTS:
        if t0 <= t < t1:
            return kind
    return SHOTS[-1][2]


def card_for(kind, t=None):
    if kind in ("A1", "num"):
        return CARD_A
    return CARD_B


# ---------------------------------------------------------- 1. текст за карточкой
def check_text_outside_card():
    cap = cv2.VideoCapture(OUT)
    nf = int(round(DUR * FPS))
    bad = 0
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


# ---------------------------------------------------------- 3. R6 (крупное слово) поверх текста
def check_topword_vs_caption():
    bad = 0
    checked = 0
    for t0, t1, kind, prm in SHOTS:
        tw = prm.get("topword")
        if not tw:
            continue
        for frac in (0.05, 0.3, 0.6, 0.9):
            t = t0 + (t1 - t0) * frac
            checked += 1
            f = R.ghost_font(tw)
            bbox = f.getbbox(tw)
            tw_w, tw_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tw_box = (540 - tw_w / 2, 440 - tw_h / 2, 540 + tw_w / 2, 440 + tw_h / 2)

            items = R.caption_items(t, kind)
            for it in items:
                ff = it["font"]
                txt = it["text"]
                cb = ff.getbbox(txt)
                if cb is None:
                    continue
                x0, y0 = it["xy"]
                an = it.get("anchor", "la")
                w, h = cb[2] - cb[0], cb[3] - cb[1]
                ax = {"l": 0, "m": w / 2, "r": w}[an[0]]
                ay = {"a": 0, "m": h / 2, "d": h, "t": 0}.get(an[1], 0)
                cap_box = (x0 - ax, y0 - ay, x0 - ax + w, y0 - ay + h)
                ix = max(0, min(tw_box[2], cap_box[2]) - max(tw_box[0], cap_box[0]))
                iy = max(0, min(tw_box[3], cap_box[3]) - max(tw_box[1], cap_box[1]))
                if ix > 0 and iy > 0:
                    bad += 1
                    if bad <= 5:
                        print(f"  t={t:.2f}s: R6 «{tw}» пересекается с субтитром «{txt}»")
    print(f"R6 поверх субтитра: {bad} моментов из {checked} проверенных")
    return bad


if __name__ == "__main__":
    print("=== QA ролик 8 («аналитик спроса») ===")
    n1 = check_text_outside_card()
    n2 = check_line_overlap()
    n3 = check_topword_vs_caption()
    print()
    ok = n1 == 0 and n2 == 0 and n3 == 0
    print("ИТОГО:", "ПРОШЕЛ" if ok else f"брак: outside={n1} overlap={n2} topword_vs_caption={n3}")

"""Автопроверки ролика 2 («актуарий») — delivery-specs.md §6. Обе должны дать ноль."""
import os, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS, W, H, grid_canvas, text_layer
from storyboard3 import SHOTS, DUR
import render3 as R

OUT = "/Users/vladimirkalajcidi/reels_good2/videos/2/aktuary_edit.mp4"


def shot_at(t):
    for t0, t1, kind, prm in SHOTS:
        if t0 <= t < t1:
            return kind
    return SHOTS[-1][2]


def card_for(kind, t=None):
    if kind in ("A1", "num"):
        return CARD_A
    x, y, w, h = CARD_B
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


# ---------------------------------------------------------- 3. графика поверх текста
# (для составной графики уровня R10 — в этом ролике не используется, см. status.md:
# по прямой просьбе кастомная графика ограничена простыми числами-ревилами R5a/R5b,
# остальное — сток. Проверка оставлена рабочей на будущее: любой SHOTS-kind с функцией
# "<kind>_layer" в render3.py проверяется автоматически.)
def check_graphic_vs_text():
    """Слой графики (без сетки-подложки) не должен иметь общих пикселей с текстом (порог альфы 40)."""
    kinds_used = {kind for _, _, kind, _ in SHOTS}
    layer_fn = {k: getattr(R, f"{k}_layer") for k in kinds_used if hasattr(R, f"{k}_layer") and k != "num"}
    bad = 0
    checked = 0
    CX, CY = CARD_A[0], CARD_A[1]
    for t0, t1, kind, prm in SHOTS:
        if kind not in layer_fn:
            continue
        for frac in (0.15, 0.4, 0.65, 0.9):
            t = t0 + (t1 - t0) * frac
            checked += 1
            canvas, titems_graphic = layer_fn[kind](t, (t0, t1, kind, prm))
            # маска графики: то, чем канвас с графикой отличается от чистой сетки той же фазы
            base_grid = grid_canvas(canvas.width, canvas.height, phase=t * 6.0).convert("RGBA")
            arr_g = np.asarray(canvas).astype(np.int16)
            arr_b = np.asarray(base_grid).astype(np.int16)
            diff = np.abs(arr_g - arr_b).sum(axis=2)
            graphic_mask = diff > 40
            # плюс встроенный в графику текст (цифры внутри плашек) — тоже часть графики, не субтитра
            cap_items = R.caption_items(t, kind)
            if not cap_items:
                continue
            cap_layer = text_layer((W, H), cap_items)
            cap_alpha = np.asarray(cap_layer.split()[3])
            full_mask = np.zeros((H, W), dtype=bool)
            full_mask[CY:CY + canvas.height, CX:CX + canvas.width] = graphic_mask
            overlap = full_mask & (cap_alpha > 40)
            n = int(overlap.sum())
            if n > 0:
                bad += 1
                if bad <= 5:
                    ys, xs = np.where(overlap)
                    print(f"  t={t:.2f}s ({kind}): {n} пикселей пересечения графики и текста, "
                          f"пример ({xs[0]},{ys[0]})")
    print(f"графика поверх текста: {bad} моментов из {checked} проверенных")
    return bad


if __name__ == "__main__":
    print("=== QA ролик 2 («актуарий») ===")
    n1 = check_text_outside_card()
    n2 = check_line_overlap()
    n3 = check_graphic_vs_text()
    print()
    ok = n1 == 0 and n2 == 0 and n3 == 0
    print("ИТОГО:", "ПРОШЕЛ" if ok else f"брак: outside={n1} overlap={n2} graphic_vs_text={n3}")

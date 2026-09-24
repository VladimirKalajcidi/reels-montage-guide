"""Автопроверки ролика 10 («задача Монти Холла») — START-HERE.md, delivery-specs.md §6.
1. текст за карточкой, 2. наложение строк — обязательные, обе должны дать ноль.
3. графика × субтитры (пиксельно по альфе >40) — для планов с сеткой, тоже ноль."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS, W, H, text_layer
from storyboard43 import SHOTS, DUR, CAPS, GRID_KINDS
import render43 as R


def card_for(kind):
    return CARD_B if kind == "stock" else CARD_A


def check_text_outside_card():
    cap = cv2.VideoCapture(R.OUT)
    nf = int(round(DUR * FPS))
    bad = n = 0
    for f in range(0, nf, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        n += 1
        t = f / FPS
        kind = SHOTS[R._shot_idx(t)][2]
        x, y, w, h = card_for(kind)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray[y:y + h, x:x + w] = 0
        if gray.max() > 200:
            bad += 1
            ys, xs = np.where(gray > 200)
            if bad <= 5:
                print(f"  кадр {f} ({t:.2f}s, {kind}): {len(xs)} px вне карточки, ({xs[0]},{ys[0]})")
    cap.release()
    print(f"1. текст за карточкой: {bad} кадров из {n} проверенных")
    return bad


def _boxes(items):
    out = []
    for it in items:
        bb = it["font"].getbbox(it["text"], anchor=it.get("anchor", "la"))
        x0, y0 = it["xy"]
        out.append((x0 + bb[0], y0 + bb[1], x0 + bb[2], y0 + bb[3]))
    return out


def check_line_overlap():
    bad = checked = 0
    times = sorted(set([round(c[0] + 0.6, 3) for c in CAPS] + [round(c[1] - 0.02, 3) for c in CAPS]
                       + [round(R.BLOCKS[i][1] - 0.02, 3) for i in range(len(CAPS))]))
    for t in times:
        # на полном раскрытии: все строки блока допечатаны
        boxes = _boxes(R.caption_items(t))
        checked += 1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    bad += 1
                    if bad <= 5:
                        print(f"  t={t:.2f}s: пересечение {a} × {b}")
    print(f"2. наложение строк: {bad} пар в {checked} проверенных моментах")
    return bad


def check_gfx_vs_text():
    bad = n = 0
    for f in range(0, int(round(DUR * FPS)), 3):
        t = f / FPS
        shot = SHOTS[R._shot_idx(t)]
        if shot[2] not in GRID_KINDS:
            continue
        cl = R.caption_layer(t)       # субтитры вместе с тенью
        if cl is None:
            continue
        n += 1
        lay, nums = R.gfx(t, shot)
        ga = np.zeros((H, W), np.uint8)
        if lay is not None:
            ga = np.maximum(ga, np.array(lay.split()[3]))
        if nums:
            ga = np.maximum(ga, np.array(text_layer((W, H), nums).split()[3]))
        ta = np.array(cl.split()[3])
        k = int(((ga > 40) & (ta > 40)).sum())
        if k:
            bad += 1
            if bad <= 5:
                print(f"  t={t:.2f}s ({shot[2]}): {k} общих пикселей")
    print(f"3. графика × субтитры: {bad} кадров из {n} проверенных")
    return bad


if __name__ == "__main__":
    print("=== QA ролик 10 («задача Монти Холла») ===")
    n1 = check_text_outside_card()
    n2 = check_line_overlap()
    n3 = check_gfx_vs_text()
    ok = n1 == 0 and n2 == 0 and n3 == 0
    print("ИТОГО:", "ПРОШЁЛ" if ok else f"брак: outside={n1} overlap={n2} gfx={n3}")

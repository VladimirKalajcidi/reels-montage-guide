"""Автопроверки ролика 56 (формула шнуровки, слот 156) — START-HERE.md, delivery-specs.md §6.
1. текст за карточкой, 2. наложение строк — обязательные, обе должны дать ноль.
3. графика × субтитры (пиксельно по альфе >40) — для планов с сеткой, тоже ноль.
4. карточка не пустая (ролик 25): на планах с лицом и стоком средняя яркость внутри карточки ≥40 —
   проверки 1–3 пропустили чёрные планы A2 из недописанного A-roll."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS, W, H, text_layer
from storyboard156 import SHOTS, DUR, CAPS, GRID_KINDS
import render156 as R


def card_for(kind):
    return CARD_B if kind in ("stock", "photo") else CARD_A


def check_text_outside_card():
    cap = cv2.VideoCapture(R.OUT)
    nf = int(round(DUR * FPS))
    bad = n = 0
    for f in sorted(set(range(0, nf, 3)) | {nf - 1}):      # урок 51: последний кадр — всегда
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            print(f"  кадр {f} не читается")
            bad += 1
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


def check_card_not_black():
    cap = cv2.VideoCapture(R.OUT)
    bad = n = 0
    nf = int(round(DUR * FPS))
    # урок ролика 51: последний кадр (nf − 1) — всегда, шаг 3 видит его только случайно
    for f in sorted(set(range(0, nf, 3)) | {nf - 1}):
        t = f / FPS
        kind = SHOTS[R._shot_idx(t)][2]
        if kind not in ("A1", "A2", "stock", "photo"):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            print(f"  кадр {f} не читается")
            bad += 1
            continue
        n += 1
        x, y, w, h = card_for(kind)
        m = float(img[y + 40:y + h - 40, x + 40:x + w - 40].mean())
        if m < 40:
            bad += 1
            if bad <= 5:
                print(f"  кадр {f} ({t:.2f}s, {kind}): средняя яркость карточки {m:.1f}")
    cap.release()
    print(f"4. пустая карточка: {bad} кадров из {n} проверенных")
    return bad


if __name__ == "__main__":
    print("=== QA ролик 56 (формула шнуровки) ===")
    n1 = check_text_outside_card()
    n2 = check_line_overlap()
    n3 = check_gfx_vs_text()
    n4 = check_card_not_black()
    ok = n1 == 0 and n2 == 0 and n3 == 0 and n4 == 0
    print("ИТОГО:", "ПРОШЁЛ" if ok else f"брак: outside={n1} overlap={n2} gfx={n3} black={n4}")

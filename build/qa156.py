"""QA-проверки ролика 56.

Проверка 1: текст за пределами карточки — ни одного пикселя ярче 200 вне Card A/B.
Проверка 4: пустая карточка — средняя яркость внутри карточки на face/stock-планах ≥ 40.
Каждые 3 кадра + принудительно кадр nf-1 (урок ролика 51).
"""
import os, sys, cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard156 import SHOTS, DUR, FACE_KINDS

ROOT  = "/Users/vladimirkalajcidi/reels_challenge"
VIDEO = f"{ROOT}/videos/56/shoelace_edit.mp4"

# Card A (canvas): x 105-975, y 238-1618
CA_X1, CA_Y1, CA_X2, CA_Y2 = 105, 238, 975, 1618
# Card B (canvas): x 110-968, y 614-1234
CB_X1, CB_Y1, CB_X2, CB_Y2 = 110, 614, 968, 1234
# Canvas
CW, CH_C = 1080, 1920
FPS = 30


def _shot_at(t):
    for s in SHOTS:
        if s[0] <= t < s[1]:
            return s
    return SHOTS[-1]


def main():
    cap = cv2.VideoCapture(VIDEO)
    assert cap.isOpened(), f"Не открылся: {VIDEO}"
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"файл: {VIDEO}  кадров: {nf}")

    # строим маску «разрешённая зона» (Card A ∪ Card B)
    allowed = np.zeros((CH_C, CW), dtype=np.uint8)
    allowed[CA_Y1:CA_Y2, CA_X1:CA_X2] = 1
    allowed[CB_Y1:CB_Y2, CB_X1:CB_X2] = 1

    err1, err4 = [], []
    frames_to_check = sorted(set(list(range(0, nf, 3)) + [nf - 1]))

    f_idx = 0
    for fi in frames_to_check:
        while f_idx <= fi:
            ok, bgr = cap.read()
            if not ok:
                break
            f_idx += 1
        if not ok:
            break
        t = fi / FPS
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ── Проверка 1: белые пиксели вне разрешённой зоны ──────────────────
        bright = (gray > 200).astype(np.uint8)
        outside = bright * (1 - allowed)
        cnt = int(outside.sum())
        if cnt > 0:
            err1.append((fi, t, cnt))

        # ── Проверка 4: пустая карточка на face/stock планах ─────────────────
        shot = _shot_at(t)
        kind = shot[2]
        if kind in FACE_KINDS or kind == "stock":
            if kind in FACE_KINDS:
                roi = gray[CA_Y1:CA_Y2, CA_X1:CA_X2]
            else:
                roi = gray[CB_Y1:CB_Y2, CB_X1:CB_X2]
            mean_bright = float(roi.mean())
            if mean_bright < 40:
                err4.append((fi, t, mean_bright, kind))

    cap.release()

    if err1:
        print(f"\nПРОВЕРКА 1 FAILED — {len(err1)} кадров с текстом за карточкой:")
        for fi, t, cnt in err1[:10]:
            print(f"  кадр {fi} ({t:.2f}с): {cnt} пикс")
    else:
        print("\nПРОВЕРКА 1: OK — текст в карточках")

    if err4:
        print(f"\nПРОВЕРКА 4 FAILED — {len(err4)} чёрных кадров:")
        for fi, t, mb, kk in err4[:10]:
            print(f"  кадр {fi} ({t:.2f}с): яркость {mb:.1f}  ({kk})")
    else:
        print("ПРОВЕРКА 4: OK — пустых карточек нет")

    if err1 or err4:
        sys.exit(1)
    print("\nQA: ОБЕ ПРОВЕРКИ ПРОШЛИ ✓")


if __name__ == "__main__":
    main()

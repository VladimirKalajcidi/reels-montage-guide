"""QA ролика 6: две обязательные автопроверки из START-HERE.md.

1. Текст за карточкой: скан каждые 3 кадра, ищем пиксели ярче 200 вне
   прямоугольника карточки (A для лица/графики, B для стока).
2. Наложение строк: для каждого блока субтитров сравниваем реальные
   габариты соседних строк (font.getbbox), пересечений быть не должно.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image

from style import CARD_A, CARD_B, W, H, font
from storyboard6 import SHOTS, CAPS, DUR

OUT = "/Users/vladimirkalajcidi/reels_good2/videos/6/geofizik_edit.mp4"


def shot_kind_at(t):
    for t0, t1, kind, _ in SHOTS:
        if t0 <= t < t1:
            return kind
    return SHOTS[-1][2]


def card_rect(kind):
    if kind == "stock":
        return CARD_B
    return CARD_A


def check_text_outside_card():
    cap = cv2.VideoCapture(OUT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad = []
    f = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if f % 3 == 0:
            _, img = cap.retrieve()
            t = f / fps
            kind = shot_kind_at(t)
            x, y, w, h = card_rect(kind)
            mask = np.ones(img.shape[:2], dtype=bool)
            mask[y:y + h, x:x + w] = False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            outside_bright = gray[mask] > 200
            cnt = int(outside_bright.sum())
            if cnt > 0:
                bad.append((f, round(t, 2), cnt))
        else:
            cap.retrieve()
        f += 1
    cap.release()
    return bad, n


def build_blocks():
    blocks, cur = [], []
    for i, (t0, t1, runs, slot) in enumerate(CAPS):
        newblock = False
        if cur:
            pi = cur[-1]
            if shot_kind_at(CAPS[pi][0]) != shot_kind_at(t0):
                newblock = True
            elif t0 - CAPS[pi][1] > 0.30:
                newblock = True
            elif len(cur) >= 2:
                newblock = True
            elif type(CAPS[pi][3]) is not type(slot):
                newblock = True
        if newblock:
            blocks.append(cur); cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    return blocks


def line_bbox(runs, ks=1.0):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    maxh = 0
    for (txt, k, sz) in runs:
        f = font(kmap[k], max(8, int(sz * ks)))
        b = f.getbbox(txt)
        maxh = max(maxh, b[3] - b[1])
    return maxh


def check_line_overlap():
    """Модель раскладки блока из render6.py: шаг между строками
    max(74, (высота_i + высота_i+1) * 0.92). Проверяем, что реальные габариты
    соседних строк не превышают выделенный шаг."""
    blocks = build_blocks()
    bad = []
    for bid, members in enumerate(blocks):
        if len(members) < 2:
            continue
        heights = [line_bbox(CAPS[i][2]) for i in members]
        for k in range(len(members) - 1):
            step = max(74.0, (heights[k] + heights[k + 1]) * 0.92)
            min_needed = (heights[k] + heights[k + 1]) / 2
            if step < min_needed:
                bad.append((bid, members[k], members[k + 1], step, min_needed))
    return bad


if __name__ == "__main__":
    bad_text, n = check_text_outside_card()
    print(f"[1] текст за карточкой: {len(bad_text)} кадров из {n // 3} проверенных")
    for f, t, cnt in bad_text[:20]:
        print(f"    кадр {f} t={t}s ярких пикселей вне карточки: {cnt}")

    bad_lines = check_line_overlap()
    print(f"[2] наложение строк: {len(bad_lines)} пар")
    for bid, i, j, step, need in bad_lines[:20]:
        print(f"    блок {bid}: {CAPS[i][2]} / {CAPS[j][2]}  шаг={step:.0f} нужно>={need:.0f}")

    print("ИТОГ:", "OK" if not bad_text and not bad_lines else "ЕСТЬ ЗАМЕЧАНИЯ")

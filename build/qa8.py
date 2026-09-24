"""Обязательные автопроверки ролика 8 (START-HERE.md). Обе должны дать ноль."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, W, H, font
from storyboard8 import SHOTS, CAPS
import render8 as R

VID = "/Users/vladimirkalajcidi/reels_good/videos/8/imaginary_edit.mp4"


def shot_kind_at(t):
    for t0, t1, kind, _ in SHOTS:
        if t0 <= t < t1:
            return kind
    return SHOTS[-1][2]


def check_outside_card(step=3):
    """1. Ни один пиксель ярче 200 вне прямоугольника карточки."""
    cap = cv2.VideoCapture(VID)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bad, f = [], -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        f += 1
        if f % step:
            continue
        _, img = cap.retrieve()
        t = f / fps
        kind = shot_kind_at(t)
        card = CARD_B if kind == "stock" else CARD_A
        x, y, w, h = card
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.ones_like(g, bool)
        mask[y:y + h, x:x + w] = False
        n = int((g[mask] > 200).sum())
        if n:
            bad.append((f, t, kind, n))
    cap.release()
    return bad


def check_line_overlap():
    """2. Ни одна пара строк внутри блока не пересекается (по font.getbbox)."""
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    groups = {}
    for i, (k, bend, blen, bid) in R.BLOCKS.items():
        groups.setdefault(bid, []).append(i)

    bad = []
    for bid, members in groups.items():
        members.sort()
        spans = []
        for idx in members:
            t0, t1, runs, slot = CAPS[idx]
            # повторяем раскладку из render8.caption_items на момент конца блока
            t = R.BLOCKS[idx][1] - 1e-3
            kind = shot_kind_at(CAPS[idx][0])
            if kind == "stock":
                continue
            L = R.BLOCK_LAY[idx]
            if isinstance(slot, str) and slot in ("T", "B"):
                base_x, base_y, an = CARD_A[0] + 62, 372, "lm"
            elif isinstance(slot, str):
                from storyboard8 import SLOTS
                base_x, base_y, an = SLOTS[slot]
            else:
                base_x, base_y, an = slot
            card = CARD_A
            lo, hi = card[1] + R.PAD, card[1] + card[3] - R.PAD
            shift = 0.0
            if base_y + L["bot"] > hi:
                shift = hi - (base_y + L["bot"])
            if base_y + L["top"] + shift < lo:
                shift = lo - (base_y + L["top"])
            y = base_y + L["dy"] + shift
            rr = [(txt, kk, max(24, int(sz * L["ks"]))) for (txt, kk, sz) in runs]
            rr, xy, an2 = R.clamp_to_card(rr, base_x + L["dx"], y, an, card)
            # реальные вертикальные габариты по глифам
            tops, bots = [], []
            for (txt, kk, sz) in rr:
                b = font(kmap[kk], sz).getbbox(txt)
                tops.append(xy[1] - (b[3] - b[1]) / 2)
                bots.append(xy[1] + (b[3] - b[1]) / 2)
            spans.append((idx, min(tops), max(bots)))
        for a in range(len(spans)):
            for b in range(a + 1, len(spans)):
                i1, t1_, b1 = spans[a]
                i2, t2_, b2 = spans[b]
                if min(b1, b2) - max(t1_, t2_) > 0:
                    bad.append((bid, i1, i2, round(min(b1, b2) - max(t1_, t2_), 1)))
    return bad


if __name__ == "__main__":
    print("== 1. текст за карточкой ==")
    bad1 = check_outside_card()
    print(f"кадров с пикселями >200 вне карточки: {len(bad1)}")
    for r in bad1[:12]:
        print("   ", r)

    print("== 2. наложение строк в блоке ==")
    bad2 = check_line_overlap()
    print(f"пересекающихся пар строк: {len(bad2)}")
    for r in bad2[:12]:
        print("   ", r)

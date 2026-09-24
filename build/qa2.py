"""Обязательные автопроверки ролика 2 (START-HERE.md §«Обязательные автопроверки»).

1. Текст за карточкой: каждый 3-й кадр — ни одного пикселя ярче 200 вне прямоугольника
   карточки (A или B — по типу плана).
2. Наложение строк: внутри каждого блока соседние строки не пересекаются по РЕАЛЬНЫМ
   габаритам глифов (растеризуем строку и берём bbox альфы, а не кегль).

Обе проверки должны дать ноль.
"""
import os, sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import W, H, FPS, CARD_A, CARD_B, font
import render2 as R

SHOTS, CAPS = R.SHOTS, R.CAPS


# ---------------------------------------------------------------- 1. текст за карточкой
def check_outside(path, step=3, thr=200):
    cap = cv2.VideoCapture(path)
    bad, f = [], -1
    while True:
        if not cap.grab():
            break
        f += 1
        if f % step:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        t = f / FPS
        kind = R._shot_kind_at(t)
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        g = img.max(axis=2)
        g[y:y + h, x:x + w] = 0
        n = int((g > thr).sum())
        if n:
            ys, xs = np.where(g > thr)
            bad.append((f, t, n, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    cap.release()
    return bad


# ---------------------------------------------------------------- 2. наложение строк
KMAP = {"r": "sans", "i": "sans_it", "s": "serif_it"}


def _phrase_geometry(idx, t, kind):
    """Повторяет раскладку из render2.caption_items для одной фразы."""
    t0, t1, runs, slot = CAPS[idx]
    k, bend, blen, bid = R.BLOCKS[idx]
    card = CARD_B if kind == "stock" else CARD_A
    L = R.BLOCK_LAY[idx]
    if isinstance(slot, str) and slot in ("T", "B"):
        base_x, base_y, an = ((CARD_B[0] + 26, 706, "lm") if kind == "stock"
                              else (CARD_A[0] + 62, 372, "lm"))
    elif isinstance(slot, str):
        base_x, base_y, an = R.SLOTS[slot]
    else:
        base_x, base_y, an = slot
    lo, hi = card[1] + R.PAD, card[1] + card[3] - R.PAD
    shift = 0.0
    if base_y + L["bot"] > hi:
        shift = hi - (base_y + L["bot"])
    if base_y + L["top"] + shift < lo:
        shift = lo - (base_y + L["top"])
    xy = (base_x + L["dx"], base_y + L["dy"] + shift)
    runs = [(txt, kk, max(24, int(sz * L["ks"]))) for (txt, kk, sz) in runs]
    return R.clamp_to_card(runs, xy[0], xy[1], an, card)


def _rasterize(runs, xy, an):
    """Реальные вертикальные габариты строки — по альфе отрисованных глифов."""
    items = R.line_items(runs, xy, an)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    from style import text_layer
    img.alpha_composite(text_layer((W, H), [dict(it, glow_r=0) for it in items]))
    a = np.array(img.split()[3])
    ys = np.where(a.max(axis=1) > 8)[0]
    xs = np.where(a.max(axis=0) > 8)[0]
    if not len(ys):
        return None
    return (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))


def check_overlap():
    groups = {}
    for i, (k, bend, blen, bid) in R.BLOCKS.items():
        groups.setdefault(bid, []).append(i)
    bad = []
    for bid, members in sorted(groups.items()):
        members.sort()
        kind = R._shot_kind_at(CAPS[members[0]][0])
        t = max(CAPS[i][1] for i in members) - 0.02
        boxes = []
        for i in members:
            runs, xy, an = _phrase_geometry(i, t, kind)
            b = _rasterize(runs, xy, an)
            boxes.append((i, b))
        for a in range(len(boxes) - 1):
            i1, b1 = boxes[a]
            i2, b2 = boxes[a + 1]
            if not b1 or not b2:
                continue
            # пересечение и по вертикали, и по горизонтали = строки наезжают
            vy = min(b1[3], b2[3]) - max(b1[2], b2[2])
            vx = min(b1[1], b2[1]) - max(b1[0], b2[0])
            if vy > 0 and vx > 0:
                bad.append((bid, i1, i2, vy, vx,
                            "".join(r[0] for r in CAPS[i1][2]),
                            "".join(r[0] for r in CAPS[i2][2])))
    return bad


# ---------------------------------------------------------------- границы карточки в раскладке
def check_inside_card():
    """Дополнительно: строка целиком внутри карточки (проверка самой раскладки, не пикселей)."""
    bad = []
    for i, (t0, t1, runs0, slot) in enumerate(CAPS):
        kind = R._shot_kind_at(t0)
        card = CARD_B if kind == "stock" else CARD_A
        runs, xy, an = _phrase_geometry(i, t0, kind)
        b = _rasterize(runs, xy, an)
        if not b:
            continue
        x0, x1, y0, y1 = b
        if (x0 < card[0] or x1 > card[0] + card[2]
                or y0 < card[1] or y1 > card[1] + card[3]):
            bad.append((i, b, "".join(r[0] for r in runs0)))
    return bad


if __name__ == "__main__":
    print("== 2. наложение строк внутри блока")
    ov = check_overlap()
    print(f"   пересечений: {len(ov)}")
    for r in ov:
        print("   ", r)

    print("== 2b. строка вне карточки (по раскладке)")
    ic = check_inside_card()
    print(f"   нарушений: {len(ic)}")
    for r in ic:
        print("   ", r)

    vid = sys.argv[1] if len(sys.argv) > 1 else R.TMP_VIDEO
    if os.path.exists(vid):
        print(f"== 1. пиксели ярче 200 вне карточки ({os.path.basename(vid)})")
        out = check_outside(vid)
        print(f"   кадров с нарушением: {len(out)}")
        for r in out[:15]:
            print("   ", r)

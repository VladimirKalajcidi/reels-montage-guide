"""Сборка ролика 7 («инженер компьютерного зрения»). Речь не резана — резы
только по картинке. Слот 41 — см. docstring storyboard41.py (слот 7 занят
чужим роликом другого корневого проекта).

Собственная графика сведена к двум простым числам-ревилам (R5a, белые) —
по anti-patterns.md составная инфографика (R10) систематически ломает
раскладку текст/графика там, где сток закрывает смысл не хуже.
"""
import os, sys, subprocess
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from storyboard41 import SHOTS, CAPS, SLOTS, DUR, LABELS, NUMS

BUILD = os.path.dirname(os.path.abspath(__file__))
SRC = "/Users/vladimirkalajcidi/reels_good2/videos/7/source.mov"
VIDEO_DIR = os.path.dirname(SRC)
A1 = f"{BUILD}/assets/aroll_41_A1.mp4"
A2 = f"{BUILD}/assets/aroll_41_A2.mp4"
OUT = "/Users/vladimirkalajcidi/reels_good2/videos/7/cv_engineer_edit.mp4"

GRADE = ("eq=gamma=1.18:contrast=1.06:saturation=1.10:brightness=0.02,"
         "colorbalance=rm=0.04:gm=0.008:bm=-0.03,unsharp=5:5:0.30")
# источник 720x1280: A1 — почти вся ширина, A2 — крупнее (уже кроп, тот же дубль)
FRAMINGS = {"A1": (720, 1142, 0, 120), "A2": (560, 888, 80, 227)}

CX, CY, R_A_ = CARD_A[0], CARD_A[1], R_A
CW, CH = CARD_A[2], CARD_A[3]
NF = int(round(DUR * FPS))


def prep_aroll():
    for name, out in (("A1", A1), ("A2", A2)):
        if os.path.exists(out):
            continue
        w, h, x, y = FRAMINGS[name]
        vf = f"hflip,crop={w}:{h}:{x}:{y},scale={CW}:{CH}:flags=lanczos,{GRADE}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14",
                        "-preset", "medium", "-pix_fmt", "yuv420p", out], check=True)
        print("готово:", out)


BX, BY, BW, BH = CARD_B
STOCK_DIR = f"{VIDEO_DIR}/stock"
STOCK_PREP_DIR = f"{STOCK_DIR}/prepared"
STOCK_CAP_CARD = (BX, BY, BW, 230)  # верхняя треть карточки B


def prep_stock():
    """Каждая видео-вставка -> карточка B 858x620, кроп до 1.385:1, притемнение ~25-30%."""
    os.makedirs(STOCK_PREP_DIR, exist_ok=True)
    out = {}
    for i, (t0, t1, kind, prm) in enumerate(SHOTS):
        if kind != "stock":
            continue
        src = f"{STOCK_DIR}/stock_{prm['clip']}.mp4"
        dst = f"{STOCK_PREP_DIR}/sb41_{i}_{prm['clip']}.mp4"
        if not os.path.exists(dst):
            vf = ("crop='min(iw,ih*1.3839)':'min(ih,iw/1.3839)',"
                  f"scale={BW}:{BH}:flags=lanczos,"
                  "eq=brightness=-0.22:contrast=1.08:saturation=0.90,fps=30")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", str(prm.get("ss", 0)), "-t", f"{t1 - t0 + 0.2:.3f}",
                            "-i", src,
                            "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15",
                            "-preset", "medium", "-pix_fmt", "yuv420p", dst], check=True)
            print("готово:", dst)
        out[i] = dst
    return out


# ---------------------------------------------------------------- текст
def line_items(runs, xy, anchor, reveal_chars=None, opacity=1.0):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    fonts = [font(kmap[k], sz) for (_, k, sz) in runs]
    widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
    total = sum(widths)
    x, y, an = xy[0], xy[1], anchor
    if an[0] == "m":
        sx = x - total / 2
    elif an[0] == "r":
        sx = x - total
    else:
        sx = x
    items, cum, cx = [], 0, sx
    for i, (t, k, sz) in enumerate(runs):
        f = fonts[i]
        it = dict(text=t, font=f, xy=(cx, y), anchor="l" + an[1],
                  fill=WHITE, glow=WHITE,
                  glow_r=int(10 + sz * 0.10), glow_a=0.55 if k != "s" else 0.62,
                  opacity=opacity)
        if reveal_chars is not None:
            n = len(t)
            done = reveal_chars - cum
            if done <= 0:
                cum += n
                cx += widths[i]
                continue
            if done < n:
                it["reveal"] = ("wipe", cx + f.getlength(t[:int(done)]) + 2)
        cum += len(t)
        items.append(it)
        cx += widths[i] + 10
    return items


def _shot_kind_at(t):
    for t0, t1, kind, _ in SHOTS:
        if t0 <= t < t1:
            return kind
    return SHOTS[-1][2]


def _is_insert(kind):
    return kind == "stock"


def _build_blocks():
    blocks, cur = [], []
    for i, (t0, t1, runs, slot) in enumerate(CAPS):
        newblock = False
        if cur:
            pi = cur[-1]
            k_prev, k_new = _shot_kind_at(CAPS[pi][0]), _shot_kind_at(t0)
            if _is_insert(k_prev) != _is_insert(k_new):
                newblock = True
            elif t0 - CAPS[pi][1] > 0.30:
                newblock = True
            elif len(cur) >= 2:
                newblock = True
        if newblock:
            blocks.append(cur); cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    info = {}
    for bid, b in enumerate(blocks):
        end = max(CAPS[i][1] for i in b)
        for k, i in enumerate(b):
            info[i] = (k, end, len(b), bid)
    return info


BLOCKS = _build_blocks()

SCATTER_A = [(-58, -62, 0.98), (55, 0, 1.08), (-30, 62, 0.94)]
SIMPLE_A = [(0, -62, 1.0), (0, 0, 1.0), (0, 62, 1.0)]
DIM = [1.0, 0.74, 0.58]
PAD = 30


def _eff_size(runs, ks=1.0):
    return max(r[2] for r in runs) * ks


def _block_layout():
    groups = {}
    for i, (k, bend, blen, bid) in BLOCKS.items():
        groups.setdefault(bid, []).append(i)
    lay = {}
    for bid, members in groups.items():
        members.sort()
        calm = (bid % 3 == 0)
        tbl = SIMPLE_A if calm else SCATTER_A
        ys, prev, acc = [], None, 0.0
        for k, i in enumerate(members):
            sz = _eff_size(CAPS[i][2], tbl[min(k, 2)][2])
            if k:
                acc += max(74.0, (prev + sz) * 0.92)
            ys.append(acc)
            prev = sz
        total = ys[-1]
        first_h = _eff_size(CAPS[members[0]][2], tbl[0][2]) * 0.62
        last_h = _eff_size(CAPS[members[-1]][2], tbl[min(len(members) - 1, 2)][2]) * 0.62
        for k, i in enumerate(members):
            ks = tbl[min(k, 2)][2]
            dx = tbl[min(k, 2)][0]
            dy = ys[k] - total / 2
            lay[i] = dict(dx=dx, dy=dy, ks=ks, k=k,
                          top=-(total / 2) - first_h, bot=-(total / 2) + total + last_h)
    return lay


BLOCK_LAY = _block_layout()


def clamp_to_card(runs, x, y, an, card):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    cx0, cx1 = card[0] + PAD, card[0] + card[2] - PAD
    cy0, cy1 = card[1] + PAD, card[1] + card[3] - PAD
    avail = cx1 - cx0
    w = sum(font(kmap[k], sz).getlength(txt) for (txt, k, sz) in runs)
    max_sz = max(sz for (_, _, sz) in runs)
    max_h = max_sz * 1.24
    height_avail = cy1 - cy0
    if w > avail or max_h > height_avail:
        f = min(avail / max(1, w), height_avail / max(1, max_h))
        runs = [(txt, k, max(22, int(sz * f))) for (txt, k, sz) in runs]
        w = sum(font(kmap[k], sz).getlength(txt) for (txt, k, sz) in runs)
    left = x - w / 2 if an[0] == "m" else (x - w if an[0] == "r" else x)
    left = min(max(left, cx0), cx1 - w)
    half = max(font(kmap[k], sz).size for (_, k, sz) in runs) * 0.62
    y = min(max(y, cy0 + half), cy1 - half)
    return runs, (left, y), "l" + an[1]


def caption_items(t, shot_kind):
    if shot_kind == "numw":
        return []
    if _is_insert(shot_kind):
        active = []
        for idx, (t0, t1, runs, slot) in enumerate(CAPS):
            k, bend, blen, bid = BLOCKS[idx]
            if t0 <= t < bend:
                active.append(idx)
        active = active[-2:]
        out = []
        y0 = (BY + 74) if len(active) > 1 else (BY + 114)
        for pos, idx in enumerate(active):
            t0, t1, runs, slot = CAPS[idx]
            newer = len(active) - pos - 1
            dim = DIM[min(newer, len(DIM) - 1)]
            runs = [(txt, kk, min(sz, 62 if kk != "s" else 72)) for (txt, kk, sz) in runs]
            runs, xy, an = clamp_to_card(runs, 540, y0 + pos * 70, "mm", STOCK_CAP_CARD)
            lt = t - t0
            nchars = sum(len(r[0]) for r in runs)
            type_dur = min(0.55, max(0.28, nchars * 0.030))
            rc = nchars if lt >= type_dur else nchars * (lt / type_dur)
            op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
            out += line_items(runs, xy, an, reveal_chars=rc, opacity=op)
        return out

    out = []
    card = CARD_A
    for idx, (t0, t1, runs, slot) in enumerate(CAPS):
        k, bend, blen, bid = BLOCKS[idx]
        if not (t0 <= t < bend):
            continue
        start = idx - k
        newer = sum(1 for j in range(idx + 1, start + blen) if CAPS[j][0] <= t)
        dim = DIM[min(newer, len(DIM) - 1)]

        L = BLOCK_LAY[idx]
        base_x, base_y, an = SLOTS[slot][0], SLOTS[slot][1], SLOTS[slot][2]

        lo, hi = card[1] + PAD, card[1] + card[3] - PAD
        shift = 0.0
        if base_y + L["bot"] > hi:
            shift = hi - (base_y + L["bot"])
        if base_y + L["top"] + shift < lo:
            shift = lo - (base_y + L["top"])
        xy = (base_x + L["dx"], base_y + L["dy"] + shift)
        ks = L["ks"]

        runs = [(txt, kk, max(24, int(sz * ks))) for (txt, kk, sz) in runs]
        runs, xy, an = clamp_to_card(runs, xy[0], xy[1], an, card)
        lt = t - t0
        nchars = sum(len(r[0]) for r in runs)
        type_dur = min(0.55, max(0.28, nchars * 0.030))
        rc = nchars if lt >= type_dur else nchars * (lt / type_dur)
        op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
        out += line_items(runs, xy, an, reveal_chars=rc, opacity=op)
    return out


def num_items(t, t0, digits, xy, size, blue=False):
    import math
    lt = t - t0
    if lt < 0:
        return []
    dur = 0.38
    p = min(1.0, lt / dur)
    e = ease_out(p)
    sc = 0.55 + 0.45 * e
    if 0.55 < p < 1.0:
        sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    f = font("sans", max(8, int(size * sc)))
    col = BLUE if blue else WHITE
    gl = BLUE_GLOW if blue else WHITE
    return [dict(text=digits, font=f, xy=xy, anchor="mm", fill=col, glow=gl,
                 glow_r=int(size * 0.20), glow_a=1.0 if blue else 0.7,
                 opacity=min(1.0, lt / 0.12))]


def label_item(text, xy, size=50, kind="r", opacity=1.0, blue=False, anchor="mm"):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    col = BLUE if blue else WHITE
    gl = BLUE_GLOW if blue else WHITE
    return dict(text=text, font=font(kmap[kind], size), xy=xy, anchor=anchor,
                fill=col, glow=gl, glow_r=30 if blue else 14,
                glow_a=0.9 if blue else 0.55, opacity=opacity)


def label_for(t, kind):
    """R7 — служебная подпись под карточкой B (сток со счётным смыслом)."""
    for t0, t1, text in LABELS:
        if t0 <= t < t1 and _is_insert(kind):
            lt = t - t0
            op = min(1.0, lt / 0.30)
            return [label_item(text, (540, BY + BH + 46), 40, "s", opacity=op)]
    return []


def numw_layer(t, t0):
    """Белое число-ревил (R5a) — плоская тёмная карточка A вместо сетки."""
    prm = NUMS[t0]
    lt = t - t0
    L, T = [], []
    card = Image.new("RGBA", (CW, CH), (8, 8, 10, 255))
    L.append(card)
    T += num_items(t, t0 + 0.05, prm["digits"], (CX + CW // 2, CY + CH // 2 - 30),
                   prm.get("size", 150), False)
    T.append(label_item(prm["sub"], (CX + CW // 2, CY + CH // 2 + 130), 58, "r", blue=False,
                        opacity=min(1.0, max(0.0, lt - 0.30) / 0.3)))
    return L, T


# ---------------------------------------------------------------- главный цикл
def main():
    prep_aroll()
    stock_files = prep_stock()
    stock_caps = {i: cv2.VideoCapture(p) for i, p in stock_files.items()}
    stock_pos = {i: -1 for i in stock_files}

    cap1 = cv2.VideoCapture(A1)
    cap2 = cv2.VideoCapture(A2)
    maskA = rounded_mask(CW, CH, R_A_)
    maskB = rounded_mask(BW, BH, R_B)

    tmp = f"{BUILD}/assets/_video41.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    si = 0
    for f in range(NF):
        t = f / FPS
        ok1 = cap1.grab(); ok2 = cap2.grab()
        while si < len(SHOTS) - 1 and t >= SHOTS[si][1]:
            si += 1
        t0, t1, kind, prm = SHOTS[si]
        shot_idx = si
        lt = t - t0

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        layers, titems = [], []

        if kind in ("A1", "A2"):
            cap = cap1 if kind == "A1" else cap2
            okr, img = cap.retrieve()
            if not okr:
                img = np.zeros((CH, CW, 3), np.uint8)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            canvas.paste(pil, (CX, CY), maskA)
        elif kind == "stock":
            v = stock_caps[shot_idx]
            want = int(round(lt * FPS))
            while stock_pos[shot_idx] < want:
                okg = v.grab()
                if not okg:
                    break
                stock_pos[shot_idx] += 1
            okr, img = v.retrieve()
            if okr:
                canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                             (BX, BY), maskB)
        elif kind == "numw":
            layers, titems = numw_layer(t, t0)
            for L in layers:
                canvas.paste(L, (CX, CY)) if L.size == (CW, CH) else canvas.alpha_composite(L)
            layers = []

        for L in layers:
            canvas.alpha_composite(L)
        items = titems + caption_items(t, kind) + label_for(t, kind)
        if items:
            canvas.alpha_composite(text_layer((W, H), items))

        ff.stdin.write(canvas.convert("RGB").tobytes())
        if f % 150 == 0:
            print(f"  кадр {f}/{NF}  ({t:5.1f}s)")

    ff.stdin.close(); ff.wait()
    cap1.release(); cap2.release()

    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", tmp, "-i", SRC,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy",
                    "-af", "highpass=f=70,loudnorm=I=-14:TP=-1.5:LRA=7",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", OUT], check=True)
    print("готово:", OUT)


if __name__ == "__main__":
    main()

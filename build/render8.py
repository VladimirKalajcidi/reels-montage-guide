"""Сборка ролика 8 («мнимая единица»). Речь не резана — резы только по картинке."""
import os, sys, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from storyboard8 import SHOTS, CAPS, SLOTS, DUR

BUILD = os.path.dirname(os.path.abspath(__file__))
SRC = "/Users/vladimirkalajcidi/reels_good/videos/8/source.mov"
VIDEO_DIR = os.path.dirname(SRC)
A1 = f"{BUILD}/assets/aroll_8_A1.mp4"
A2 = f"{BUILD}/assets/aroll_8_A2.mp4"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/8/imaginary_edit.mp4"

GRADE = ("eq=gamma=1.18:contrast=1.06:saturation=1.10:brightness=0.02,"
         "colorbalance=rm=0.04:gm=0.008:bm=-0.03,unsharp=5:5:0.30")
# w, h, x, y — кроп из исходника 1080x1920, линия глаз ≈42% высоты карточки
FRAMINGS = {"A1": (972, 1542, 84, 202), "A2": (870, 1380, 135, 270)}

CX, CY = CARD_A[0], CARD_A[1]
CW, CH = CARD_A[2], CARD_A[3]
BX, BY, BW, BH = CARD_B
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


STOCK_DIR = f"{VIDEO_DIR}/stock"
STOCK_PREP_DIR = f"{STOCK_DIR}/prepared"
STOCK_CAP_CARD = (105, 388, 870, 196)


def prep_stock():
    """Вставка → карточка B 858x620, кроп до 1.385:1, притемнение ~25–30%."""
    os.makedirs(STOCK_PREP_DIR, exist_ok=True)
    out = {}
    for i, (t0, t1, kind, prm) in enumerate(SHOTS):
        if kind != "stock":
            continue
        src = f"{STOCK_DIR}/stock_{prm['clip']}.mp4"
        dst = f"{STOCK_PREP_DIR}/sb8_{i}_{prm['clip']}.mp4"
        if not os.path.exists(dst):
            vf = ("crop='min(iw,ih*1.3839)':'min(ih,iw/1.3839)',"
                  f"scale={BW}:{BH}:flags=lanczos,"
                  "eq=brightness=-0.22:contrast=1.08:saturation=0.90,fps=30")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", str(prm.get("ss", 0)), "-t", f"{t1 - t0 + 0.3:.3f}",
                            "-i", src, "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15",
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


def _build_blocks():
    blocks, cur = [], []
    for i, (t0, t1, runs, slot) in enumerate(CAPS):
        newblock = False
        if cur:
            pi = cur[-1]
            if _shot_kind_at(CAPS[pi][0]) != _shot_kind_at(t0):
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
    info = {}
    for bid, b in enumerate(blocks):
        end = max(CAPS[i][1] for i in b)
        for k, i in enumerate(b):
            info[i] = (k, end, len(b), bid)
    return info


BLOCKS = _build_blocks()

SCATTER_A = [(-58, -62, 0.98), (55, 0, 1.08), (-30, 62, 0.94)]
SCATTER_C = [(0, 0, 1.0), (18, 58, 1.04), (-24, 116, 0.96)]
SIMPLE_A = [(0, -62, 1.0), (0, 0, 1.0), (0, 62, 1.0)]
SIMPLE_C = [(0, 0, 1.0), (0, 58, 1.0), (0, 116, 1.0)]
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
        slot0 = CAPS[members[0]][3]
        flag = isinstance(slot0, str) and slot0 in ("T", "B")
        centered = isinstance(slot0, str) and not flag
        tbl = (SIMPLE_A if calm else SCATTER_A) if centered else (SIMPLE_C if calm else SCATTER_C)

        ys, prev, acc = [], None, 0.0
        for k, i in enumerate(members):
            sz = _eff_size(CAPS[i][2], 1.0 if flag else tbl[min(k, 2)][2])
            if k:
                acc += max(74.0, (prev + sz) * 0.92)
            ys.append(acc)
            prev = sz
        total = ys[-1]
        first_h = _eff_size(CAPS[members[0]][2], 1.0 if flag else tbl[0][2]) * 0.62
        last_h = _eff_size(CAPS[members[-1]][2],
                           1.0 if flag else tbl[min(len(members) - 1, 2)][2]) * 0.62

        for k, i in enumerate(members):
            ks = 1.0 if flag else tbl[min(k, 2)][2]
            dx = 0 if flag else tbl[min(k, 2)][0]
            dy = ys[k] - (total / 2 if centered else 0)
            lay[i] = dict(dx=dx, dy=dy, ks=ks, k=k,
                          top=-(total / 2 if centered else 0) - first_h,
                          bot=-(total / 2 if centered else 0) + total + last_h)
    return lay


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


BLOCK_LAY = _block_layout()


def caption_items(t, shot_kind=None):
    out = []
    if shot_kind == "stock":
        # brand-kit §5: поверх стоковой вставки — левым флагом от x136,
        # первая строка y706, всё внутри карточки B
        active = []
        for idx, (t0, t1, runs, slot) in enumerate(CAPS):
            k, bend, blen, bid = BLOCKS[idx]
            if t0 <= t < bend:
                active.append(idx)
        active = active[-3:]
        y, prev = 706.0, None
        rows = []
        for idx in active:
            runs = [(txt, kk, min(sz, 58 if kk != "s" else 68))
                    for (txt, kk, sz) in CAPS[idx][2]]
            sz = max(r[2] for r in runs)
            if prev is not None:
                y += max(62.0, (prev + sz) * 0.70)
            rows.append((idx, runs, y))
            prev = sz
        for pos, (idx, runs, ry) in enumerate(rows):
            t0 = CAPS[idx][0]
            newer = len(rows) - pos - 1
            dim = DIM[min(newer, len(DIM) - 1)]
            runs, xy, an = clamp_to_card(runs, 136, ry, "lm", CARD_B)
            lt = t - t0
            nchars = sum(len(r[0]) for r in runs)
            type_dur = min(0.55, max(0.28, nchars * 0.030))
            rc = nchars if lt >= type_dur else nchars * (lt / type_dur)
            op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
            out += line_items(runs, xy, an, reveal_chars=rc, opacity=op)
        return out
    card = CARD_A
    for idx, (t0, t1, runs, slot) in enumerate(CAPS):
        k, bend, blen, bid = BLOCKS[idx]
        if not (t0 <= t < bend):
            continue
        start = idx - k
        newer = sum(1 for j in range(idx + 1, start + blen) if CAPS[j][0] <= t)
        dim = DIM[min(newer, len(DIM) - 1)]

        L = BLOCK_LAY[idx]
        if isinstance(slot, str) and slot in ("T", "B"):
            base_x, base_y, an = CARD_A[0] + 62, 372, "lm"
        elif isinstance(slot, str):
            base_x, base_y, an = SLOTS[slot][0], SLOTS[slot][1], SLOTS[slot][2]
        else:
            base_x, base_y, an = slot[0], slot[1], slot[2]

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


# ---------------------------------------------------------------- элементы графики
def clamp01(v):
    return max(0.0, min(1.0, v))


def stagger(lt, n, dur=0.26, step=0.09):
    return [clamp01((lt - i * step) / dur) for i in range(n)]


def pop_item(lt, t0, text, xy, size, kind="r", blue=False, dur=3 / FPS):
    """R5a — резкий поп за 2–3 кадра."""
    p = clamp01((lt - t0) / dur)
    if lt < t0:
        return []
    sc = 0.6 + 0.4 * ease_out(p)
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    f = font(kmap[kind], max(8, int(size * sc)))
    col = BLUE if blue else WHITE
    return [dict(text=text, font=f, xy=xy, anchor="mm", fill=col,
                 glow=BLUE_GLOW if blue else WHITE,
                 glow_r=int(size * (0.20 if blue else 0.16)),
                 glow_a=1.0 if blue else 0.7,
                 opacity=clamp01((lt - t0) / 0.08))]


def blue_num(lt, t0, text, xy, size):
    """R5b — синее число: scale 0.55→1.0 с оверщутом ~3%."""
    if lt < t0:
        return []
    p = clamp01((lt - t0) / 0.38)
    e = ease_out(p)
    sc = 0.55 + 0.45 * e
    if 0.55 < p < 1.0:
        sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    f = font("sans", max(8, int(size * sc)))
    return [dict(text=text, font=f, xy=xy, anchor="mm", fill=BLUE, glow=BLUE_GLOW,
                 glow_r=int(size * 0.20), glow_a=1.0,
                 opacity=clamp01((lt - t0) / 0.12))]


def label(text, xy, size=50, kind="r", opacity=1.0, blue=False, anchor="mm"):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    return dict(text=text, font=font(kmap[kind], size), xy=xy, anchor=anchor,
                fill=BLUE if blue else WHITE, glow=BLUE_GLOW if blue else WHITE,
                glow_r=30 if blue else 14, glow_a=0.9 if blue else 0.55, opacity=opacity)


def fit_row(words, size, x0, x1, kind="r", min_gap_k=0.6):
    """Центры слов в ряд с РАВНЫМИ зазорами между габаритами глифов.
    Кегль ужимается, пока зазор не станет читаемым. Возвращает (центры, кегль)."""
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    sz = size
    while True:
        f = font(kmap[kind], sz)
        ws = [f.getlength(w) for w in words]
        gap = (x1 - x0 - sum(ws)) / max(1, len(words) - 1)
        if gap >= sz * min_gap_k or sz <= 26:
            break
        sz -= 2
    cx, centers = x0, []
    for w in ws:
        centers.append(cx + w / 2)
        cx += w + gap
    return centers, sz


def strokes(segs, color=WHITE, width=7, alpha=235):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for (x0, y0, x1, y1) in segs:
        d.line([(x0, y0), (x1, y1)], fill=color + (alpha,), width=width)
    return lay.filter(ImageFilter.GaussianBlur(0.4))


def partial(x0, y0, x1, y1, p):
    return (x0, y0, x0 + (x1 - x0) * p, y0 + (y1 - y0) * p)


def obj_card(text, xy, w=520, h=142, prog=1.0, size=72, alpha=1.0):
    """R10 — белая скруглённая плашка с мягкой тенью и чёрным текстом."""
    if prog <= 0:
        return None
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    e = ease_out(prog)
    cw, ch = w * (0.7 + 0.3 * e), h * (0.7 + 0.3 * e)
    x, y = xy[0] - cw / 2, xy[1] - ch / 2
    d.rounded_rectangle([x, y, x + cw, y + ch], radius=26,
                        fill=WHITE + (int(238 * e * alpha),))
    d.text(xy, text, font=font("sans", int(size * (0.7 + 0.3 * e))),
           fill=(0, 0, 0, int(250 * e * alpha)), anchor="mm")
    sh = lay.filter(ImageFilter.GaussianBlur(16))
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(sh)
    out.alpha_composite(lay)
    return out


def real_axis(prog=1.0, y=950, x0=225, x1=855, with_labels=True):
    """Числовая прямая — «обычные числа»."""
    L, T = [], []
    p = ease_out(clamp01(prog))
    cx = (x0 + x1) / 2
    half = (x1 - x0) / 2 * p
    segs = [(cx - half, y, cx + half, y)]
    ticks = [-2, -1, 0, 1, 2]
    stepx = (x1 - x0) / 4
    for k, v in enumerate(ticks):
        tx = x0 + k * stepx
        if abs(tx - cx) <= half + 1:
            segs.append((tx, y - 16, tx, y + 16))
    L.append(strokes(segs, WHITE, 6, 225))
    if with_labels and p > 0.75:
        op = clamp01((p - 0.75) / 0.25)
        for k, v in enumerate(ticks):
            tx = x0 + k * stepx
            T.append(label(("−" + str(-v)) if v < 0 else str(v), (tx, y + 62), 42,
                           opacity=0.85 * op))
    return L, T


def imag_axis(prog=1.0, x=540, y=950, half=178):
    """Мнимая ось — синяя, растёт вверх и вниз из центра."""
    p = ease_out(clamp01(prog))
    h = half * p
    return strokes([(x, y - h, x, y + h)], BLUE, 7, 250)


# ---------------------------------------------------------------- графика планов
def shot_layers(kind, prm, lt, dur, t):
    L, T = [], []

    if kind == "nonsense":
        T += pop_item(lt, 0.10, "√−1", (540, 940), 150)
        p = clamp01((lt - 0.55) / 0.45)
        if p > 0:
            L.append(strokes([partial(345, 1030, 735, 850, ease_out(p))], WHITE, 8, 240))

    elif kind == "no_such":
        la, ta = real_axis(clamp01(lt / 0.55))
        L += la; T += ta
        T += pop_item(lt, 0.75, "√−1", (540, 826), 92, dur=0.10)

    elif kind == "sq_pos":
        c = obj_card("2² = +4", (540, 950), prog=clamp01((lt - 0.20) / 0.34))
        if c:
            L.append(c)

    elif kind == "sq_neg":
        c1 = obj_card("2² = +4", (540, 880), prog=1.0, alpha=0.42)
        if c1:
            L.append(c1)
        c2 = obj_card("(−2)² = +4", (540, 1050), prog=clamp01((lt - 0.25) / 0.36))
        if c2:
            L.append(c2)

    elif kind == "crossed":
        T += pop_item(lt, 0.10, "√−1", (540, 950), 152)
        st = stagger(lt - 0.55, 2, 0.34, 0.20)
        segs = []
        if st[0] > 0:
            segs.append(partial(340, 1055, 740, 845, ease_out(st[0])))
        if st[1] > 0:
            segs.append(partial(340, 845, 740, 1055, ease_out(st[1])))
        if segs:
            L.append(strokes(segs, WHITE, 8, 240))

    elif kind == "mockery":
        T += pop_item(lt, 0.15, "«мнимое»", (540, 940), 104, kind="s", dur=0.10)

    elif kind == "letter_i":
        T += pop_item(lt, 0.10, "i", (540, 950), 250)

    elif kind == "assume":
        la, ta = real_axis(1.0)
        L += la; T += ta
        p = clamp01((lt - 0.35) / 0.40)
        if p > 0:
            e = ease_out(p)
            r = 20 * e
            dot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(dot).ellipse([540 - r, 830 - r, 540 + r, 830 + r],
                                        fill=BLUE + (int(255 * e),))
            glow = dot.filter(ImageFilter.GaussianBlur(26))
            out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            out.alpha_composite(glow); out.alpha_composite(dot)
            L.append(out)

    elif kind == "new_field":
        la, ta = real_axis(clamp01((lt - 0.10) / 0.55))
        L += la; T += ta

    elif kind == "plane":
        la, ta = real_axis(1.0)
        L += la; T += ta
        p = clamp01((lt - 0.30) / 0.50)
        if p > 0:
            L.append(imag_axis(p))
        T += blue_num(lt, 0.72, "i", (620, 812), 84)

    elif kind == "formula":
        st = stagger(lt - 0.30, 3, 0.24, 0.14)
        parts = [("a", 396), ("+", 486), ("b", 574)]
        for (txt, x), p in zip(parts, st):
            if p > 0:
                T.append(label(txt, (x, 980), 126, opacity=ease_out(p)))
        T += blue_num(lt, 0.86, "i", (648, 980), 132)

    elif kind == "three_uses":
        # три области → сходятся в i: каждая линия показывает связь с мнимой единицей
        words = ["ток", "сигнал", "квантовая"]
        xs, wsz = fit_row(words, 46, CARD_B[0] + 44, CARD_B[0] + CARD_B[2] - 44)
        y_lab, y_join = 848, 1008
        st = stagger(lt - 0.12, 3, 0.24, 0.13)
        segs = []
        for i, (txt, x) in enumerate(zip(words, xs)):
            if st[i] > 0:
                T.append(label(txt, (x, y_lab), wsz, opacity=0.95 * ease_out(st[i])))
            lp = clamp01((lt - (0.34 + 0.13 * i)) / 0.30)
            if lp > 0:
                segs.append(partial(x, y_lab + 42, 540, y_join, ease_out(lp)))
        if segs:
            L.append(strokes(segs, WHITE, 5, 200))
        T += blue_num(lt, 1.16, "i", (540, 1078), 92)

    return L, T


_gfc = {}


def ghost_font(word, target=800):
    if word in _gfc:
        return _gfc[word]
    sz = 150
    for _ in range(40):
        w = font("sans", sz).getlength(word)
        if abs(w - target) < 12:
            break
        sz = max(60, min(300, int(sz * target / w)))
    _gfc[word] = font("sans", sz)
    return _gfc[word]


# ---------------------------------------------------------------- главный цикл
def main():
    prep_aroll()
    stock_files = prep_stock()
    stock_caps = {i: cv2.VideoCapture(p) for i, p in stock_files.items()}
    stock_pos = {i: -1 for i in stock_files}

    cap1 = cv2.VideoCapture(A1)
    cap2 = cv2.VideoCapture(A2)
    maskA = rounded_mask(CW, CH, R_A)
    maskB = rounded_mask(BW, BH, R_B)

    grid_cache = {}

    def grid_for(f):
        k = f // 3
        if k not in grid_cache:
            grid_cache.clear()
            grid_cache[k] = grid_canvas(CW, CH, phase=k * 0.06)
        return grid_cache[k]

    tmp = f"{BUILD}/assets/_video_v8.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    si = 0
    for f in range(NF):
        t = f / FPS
        cap1.grab(); cap2.grab()
        while si < len(SHOTS) - 1 and t >= SHOTS[si][1]:
            si += 1
        t0, t1, kind, prm = SHOTS[si]
        lt = t - t0

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        if kind in ("A1", "A2"):
            cap = cap1 if kind == "A1" else cap2
            okr, img = cap.retrieve()
            if not okr:
                img = np.zeros((CH, CW, 3), np.uint8)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            canvas.paste(pil, (CX, CY), maskA)
            tw = prm.get("topword")
            if tw:
                canvas.alpha_composite(text_layer((W, H), [
                    dict(text=tw, font=ghost_font(tw), xy=(540, 440), anchor="mm",
                         fill=WHITE, glow=WHITE, glow_r=24, glow_a=0.35,
                         opacity=0.62 * ease_out(min(1, lt / 0.30)))]))
        elif kind == "stock":
            v = stock_caps[si]
            want = int(round(lt * FPS))
            while stock_pos[si] < want:
                if not v.grab():
                    break
                stock_pos[si] += 1
            okr, img = v.retrieve()
            if okr:
                canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                             (BX, BY), maskB)
        else:
            canvas.paste(grid_for(f), (CX, CY), maskA)

        layers, titems = shot_layers(kind, prm, lt, t1 - t0, t)
        for lay in layers:
            canvas.alpha_composite(lay)
        items = titems + caption_items(t, kind)
        if items:
            canvas.alpha_composite(text_layer((W, H), items))

        ff.stdin.write(canvas.convert("RGB").tobytes())
        if f % 150 == 0:
            print(f"  кадр {f}/{NF}  ({t:5.1f}s)")

    ff.stdin.close(); ff.wait()
    cap1.release(); cap2.release()
    for v in stock_caps.values():
        v.release()
    print("видео собрано:", tmp)


if __name__ == "__main__":
    main()

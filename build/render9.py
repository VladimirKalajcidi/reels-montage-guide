"""Рендер ролика 9: формула Эйлера.

Кадр собирается слоями, чтобы их можно было проверить по отдельности:
  фон (лицо / сток / сетка) → слой графики → слой субтитров
Слой графики и слой субтитров не должны иметь общих пикселей (qa9.py).
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard9 import (SRC, SHOTS, CAPS, SLOTS, DUR, GRID_KINDS, FACE_KINDS,
                         GFX_ZONE, GFX_X, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_euler.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good2/videos/9/stock"

GY0, GY1 = GFX_ZONE          # 700 … 1560
GX0, GX1 = GFX_X             # 175 … 905
GCX = (GX0 + GX1) // 2       # 540
GCY = (GY0 + GY1) // 2       # 1130


# ----------------------------------------------------------------- источник --

def cv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def source_card(frame):
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    target = CARD_A[2] / CARD_A[3]
    if iw / ih > target:
        nw = int(ih * target)
        x = (iw - nw) // 2
        fr = fr[:, x:x + nw]
    else:
        nh = int(iw / target)
        y = max(0, (ih - nh) // 2 - 30)
        y = min(y, ih - nh)
        fr = fr[y:y + nh, :]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_AREA)
    fr = cv2.convertScaleAbs(fr, alpha=1.06, beta=8)
    b, g, r = cv2.split(fr.astype(np.float32))
    r = np.clip(r * 1.04 + 3, 0, 255)
    b = np.clip(b * 0.97, 0, 255)
    fr = cv2.merge([b, g, r]).astype(np.uint8)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.10
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    fr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


def stock_frame_from_bgr(fr):
    ih, iw = fr.shape[:2]
    target = CARD_B[2] / CARD_B[3]
    if iw / ih > target:
        nw = int(ih * target)
        x = (iw - nw) // 2
        fr = fr[:, x:x + nw]
    else:
        nh = int(iw / target)
        y = (ih - nh) // 2
        fr = fr[y:y + nh, :]
    fr = cv2.resize(fr, (CARD_B[2], CARD_B[3]), interpolation=cv2.INTER_AREA)
    fr = cv2.convertScaleAbs(fr, alpha=1.02, beta=-42)   # притемнение ~27%
    return cv_to_pil(fr)


# ------------------------------------------------------------------ графика --

def gtext(text, xy, size=150, blue=False, serif=False, op=1.0, anchor="mm"):
    kind = "serif_it" if serif else "sans"
    f = font(kind, size)
    color = BLUE if blue else WHITE
    return dict(text=text, font=f, xy=xy, anchor=anchor, fill=color,
                glow=BLUE_GLOW if blue else WHITE,
                glow_r=34 if blue else 16, glow_a=0.62 if blue else 0.45, opacity=op)


def pop(lt, dur=0.09):
    return ease_out(min(1.0, lt / dur)) if lt >= 0 else 0.0


def blue_pop(lt, dur=0.38):
    if lt < 0:
        return 0.0, 1.0
    p = ease_out(min(1.0, lt / dur))
    over = 1.0 + 0.03 * math.sin(min(1.0, lt / dur) * math.pi)
    return p, over


def _layer():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def g_five(lay, lt):
    p = pop(lt, 0.09)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [gtext("5", (GCX, GCY), int(210 * (0.6 + 0.4 * p)))]))


FORMULA_PARTS = [
    ("e",  "sans",   122,   0, 0.00, False),
    ("i",  "sans_it", 68, -60, 0.68, False),
    ("π",  "sans",    68, -60, 1.80, False),
    ("+1", "sans",    96,   0, 2.08, False),
    ("=",  "sans",    88,   0, 3.12, False),
    ("0",  "sans",   178,   0, 3.32, True),
]

FORMULA_Y = 980      # план без субтитров — графика может стоять выше в своей зоне


def _formula_layout():
    """Абсолютные позиции всех частей — считаются один раз, не зависят от t."""
    widths, fonts = [], []
    for txt, kind, sz, dy, at, blue in FORMULA_PARTS:
        f = font(kind, sz)
        fonts.append(f)
        widths.append(f.getlength(txt))
    gaps = [8, 6, 30, 28, 32]      # e|i  i|π  π|+1  +1|=  =|0
    total = sum(widths) + sum(gaps)
    x = GCX - total / 2
    xs = []
    for i, w in enumerate(widths):
        xs.append(x)
        x += w + (gaps[i] if i < len(gaps) else 0)
    return xs, widths, fonts


FX, FW, FF = _formula_layout()


def g_formula(lay, lt):
    base_y = FORMULA_Y
    items = []
    for i, (txt, kind, sz, dy, at, blue) in enumerate(FORMULA_PARTS):
        if lt < at:
            continue
        if blue:
            p, over = blue_pop(lt - at, 0.38)
            s = max(1, int(sz * (0.55 + 0.45 * p) * over))
            items.append(gtext(txt, (FX[i] + FW[i] / 2, base_y + dy), s, blue=True, anchor="mm"))
        else:
            op = ease_out(min(1.0, (lt - at) / 0.28))
            items.append(dict(text=txt, font=FF[i], xy=(FX[i], base_y + dy), anchor="lm",
                              fill=WHITE, glow=WHITE, glow_r=16, glow_a=0.45, opacity=op))
    if lt >= 3.72:
        op = ease_out(min(1.0, (lt - 3.72) / 0.35))
        items.append(gtext("формула эйлера", (GCX, base_y + 210), 44, serif=True, op=op * 0.9))
    lay.alpha_composite(text_layer((W, H), items))


CONSTANTS = ["e", "i", "π", "0", "1"]


def g_constants(lay, lt):
    items = []
    for i, ch in enumerate(CONSTANTS):
        p = pop(lt - i * 0.085, 0.10)
        if p <= 0:
            continue
        x = GCX + (i - 2) * 150
        items.append(gtext(ch, (x, GCY), int(112 * (0.6 + 0.4 * p)),
                           serif=(ch in ("i",))))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_e_symbol(lay, lt):
    p = pop(lt, 0.10)
    items = [gtext("e", (350, GCY), int(200 * (0.6 + 0.4 * p)))]
    d = ImageDraw.Draw(lay)
    if lt > 0.25:
        prog = ease_out(min(1.0, (lt - 0.25) / 0.85))
        pts = []
        n = 40
        for i in range(int(n * prog) + 1):
            u = i / n
            pts.append((640 + u * 230, 1330 - (math.exp(u * 2.2) - 1) * 42))
        if len(pts) > 1:
            d.line(pts, fill=BLUE + (230,), width=6, joint="curve")
    lay.alpha_composite(text_layer((W, H), items))


def g_growth(lay, lt):
    d = ImageDraw.Draw(lay)
    prog = ease_out(min(1.0, lt / 1.45))
    pts = []
    n = 60
    for i in range(int(n * prog) + 1):
        u = i / n
        pts.append((230 + u * 640, 1450 - (math.exp(u * 2.3) - 1) * 55))
    if len(pts) > 1:
        d.line(pts, fill=WHITE + (225,), width=6, joint="curve")
        hx, hy = pts[-1]
        d.ellipse([hx - 13, hy - 13, hx + 13, hy + 13], fill=BLUE + (255,))


def g_imaginary(lay, lt):
    d = ImageDraw.Draw(lay)
    cx, cy, r = GCX, GCY + 20, 200
    d.line([(cx - r - 45, cy), (cx + r + 45, cy)], fill=WHITE + (150,), width=3)
    d.line([(cx, cy - r - 45), (cx, cy + r + 45)], fill=WHITE + (150,), width=3)
    prog = ease_out(min(1.0, lt / 1.0))
    pts = []
    steps = max(2, int(30 * prog))
    for i in range(steps + 1):
        a = -i / 30 * math.pi / 2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    if len(pts) > 1:
        d.line(pts, fill=BLUE + (225,), width=5, joint="curve")
    px = cx + r * math.cos(-prog * math.pi / 2)
    py = cy + r * math.sin(-prog * math.pi / 2)
    d.ellipse([px - 14, py - 14, px + 14, py + 14], fill=BLUE + (255,))
    lay.alpha_composite(text_layer((W, H), [
        gtext("1", (cx + r + 78, cy), 46, op=0.75),
        gtext("i", (cx - 42, cy - r - 40), 52, op=0.9),
    ]))


def g_pi_circle(lay, lt):
    d = ImageDraw.Draw(lay)
    cx, cy, r = 330, 950, 110
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WHITE + (225,), width=6)
    d.line([(cx - r, cy), (cx + r, cy)], fill=BLUE + (225,), width=6)
    base_y, seg, x0 = 1330, 2 * r, 190
    d.line([(x0, base_y), (x0 + 3.15 * seg, base_y)], fill=WHITE + (120,), width=3)
    filled = min(3.14159, 3.14159 * (lt / 2.2)) if lt < 2.6 else 3.14159
    # непрерывная синяя развёртка от начала линии + засечки по целым диаметрам
    d.line([(x0, base_y), (x0 + seg * filled, base_y)], fill=BLUE + (235,), width=9)
    for k in range(int(filled) + 1):
        x = x0 + k * seg
        d.line([(x, base_y - 18), (x, base_y + 18)], fill=WHITE + (205,), width=4)
    if lt > 1.7:
        op = ease_out(min(1.0, (lt - 1.7) / 0.45))
        lay.alpha_composite(text_layer((W, H), [gtext("≈ 3,14", (700, 1150), 62, blue=True, op=op)]))


def g_zero_one(lay, lt):
    items = []
    for i, ch in enumerate(("0", "1")):
        p = pop(lt - i * 0.12, 0.09)
        if p > 0:
            items.append(gtext(ch, (GCX - 110 + i * 220, GCY), int(200 * (0.6 + 0.4 * p))))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_converge(lay, lt):
    """Три области сходятся в диаграмму Венна — общая середина, но круги различимы."""
    d = ImageDraw.Draw(lay)
    R = 150
    starts = [(GCX, 862), (340, 1400), (740, 1400)]
    ends = [(GCX, GCY - 78), (GCX - 88, GCY + 68), (GCX + 88, GCY + 68)]
    for i, (st, en) in enumerate(zip(starts, ends)):
        p = ease_out(min(1.0, max(0.0, (lt - i * 0.55) / 0.65)))
        if p <= 0:
            continue
        cx = st[0] + (en[0] - st[0]) * p
        cy = st[1] + (en[1] - st[1]) * p
        col = BLUE if i == 0 else WHITE
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col + (205,), width=5)


GFX = {
    "five": g_five, "formula": g_formula, "constants": g_constants,
    "e_symbol": g_e_symbol, "growth": g_growth, "imaginary": g_imaginary,
    "pi_circle": g_pi_circle, "zero_one": g_zero_one, "converge": g_converge,
}


def graphics_layer(kind, lt):
    """Только элементы переднего плана, без сетки-подложки."""
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    return lay


# ---------------------------------------------------------------- субтитры --

def line_items(runs, xy, anchor, opacity=1.0, reveal_chars=None, max_w=None):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    sizes = [sz for _, _, sz in runs]
    fonts = [font(kmap[k], sz) for (_, k, sz) in runs]
    widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
    total = sum(widths) + 8 * (len(runs) - 1)
    if max_w and total > max_w:                       # ужать кегль, детерминированно
        sc = max_w / total
        sizes = [max(20, int(sz * sc)) for sz in sizes]
        fonts = [font(kmap[runs[i][1]], sizes[i]) for i in range(len(runs))]
        widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
        total = sum(widths) + 8 * (len(runs) - 1)
    x, y = xy
    cx = x - total / 2 if anchor[0] == "m" else x
    out, cum = [], 0
    for i, (txt, k, _) in enumerate(runs):
        f = fonts[i]
        it = dict(text=txt, font=f, xy=(cx, y), anchor="l" + anchor[1],
                  fill=WHITE, glow=WHITE, glow_r=int(12 + sizes[i] * .08),
                  glow_a=0.58 if k == "s" else 0.48, opacity=opacity)
        if reveal_chars is not None:
            done = reveal_chars - cum
            if done <= 0:
                cx += widths[i] + 8
                cum += len(txt)
                continue
            if done < len(txt):
                it["reveal"] = ("wipe", cx + f.getlength(txt[:int(done)]) + 2)
        out.append(it)
        cx += widths[i] + 8
        cum += len(txt)
    return out


def build_blocks():
    """Блок: до 2 фраз, рвётся на паузе >0.30с и на смене плана."""
    blocks, cur = [], []
    for i, c in enumerate(CAPS):
        if cur:
            prev = CAPS[cur[-1]]
            same_shot = shot_at(c[0]) is shot_at(prev[0])
            if c[0] - prev[1] > 0.30 or len(cur) >= 2 or not same_shot:
                blocks.append(cur)
                cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    info = {}
    for bid, b in enumerate(blocks):
        end = max(CAPS[i][1] for i in b)
        for pos, i in enumerate(b):
            info[i] = (bid, pos, len(b), end)
    return info


BLOCKS = build_blocks()
DIM = [1.0, 0.74, 0.58]


def caption_layer(t):
    kind = shot_at(t)[2]
    slot = SLOTS[slot_for(kind)]
    active = [i for i, c in enumerate(CAPS) if c[0] <= t < BLOCKS[i][3]]
    items = []
    for idx in active[-3:]:
        t0, _, runs = CAPS[idx]
        bid, pos, _, _ = BLOCKS[idx]
        newer = sum(1 for j in active if j > idx and BLOCKS[j][0] == bid)
        dim = DIM[min(newer, 2)]
        bx, by = slot["x"], slot["y"] + pos * slot["step"]
        if slot["scatter"]:
            calm = bid % 3 == 0
            bx += (0 if calm else [-45, 38, -20][min(pos, 2)])
        lt = t - t0
        nchars = sum(len(r[0]) for r in runs)
        type_dur = min(0.55, max(0.30, nchars * 0.030))
        reveal = nchars if lt >= type_dur else nchars * lt / type_dur
        op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
        max_w = slot["x_max"] - bx if slot["anchor"][0] == "l" else \
            2 * min(bx - slot["x_min"], slot["x_max"] - bx)
        items += line_items(runs, (bx, by), slot["anchor"], opacity=op,
                            reveal_chars=reveal, max_w=max_w)
    if not items:
        return None
    return text_layer((W, H), items)


# ------------------------------------------------------------------- сборка --

def background(kind, prm, fr, t, stock_reader=None):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        card = source_card(fr)
        canvas.paste(card, (CARD_A[0], CARD_A[1]), rounded_mask(CARD_A[2], CARD_A[3], R_A))
    elif kind == "stock":
        sfr = stock_reader()
        card = stock_frame_from_bgr(sfr)
        canvas.paste(card, (CARD_B[0], CARD_B[1]), rounded_mask(CARD_B[2], CARD_B[3], R_B))
    else:
        x, y, w, h = CARD_A
        g = grid_canvas(w, h, phase=t * 0.7)
        canvas.paste(g, (x, y), rounded_mask(w, h, R_A))
    return canvas


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    tmp = f"{BUILD}/assets/_video_euler_raw.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    nf = int(round(DUR * FPS))
    stock_cap, stock_key = None, None

    for fno in range(nf):
        ok, fr = cap.read()
        if not ok:
            break
        t = fno / FPS
        t0, t1, kind, prm = shot_at(t)
        lt = t - t0

        def read_stock():
            nonlocal stock_cap, stock_key
            key = (t0, prm["clip"])
            if key != stock_key:
                if stock_cap is not None:
                    stock_cap.release()
                stock_cap = cv2.VideoCapture(f"{STOCK_DIR}/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
                stock_key = key
            ok_s, sfr = stock_cap.read()
            if not ok_s:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok_s, sfr = stock_cap.read()
            return sfr

        canvas = background(kind, prm, fr, t, read_stock)
        gl = graphics_layer(kind, lt)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        wr.write(pil_to_cv(canvas.convert("RGB")))
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}")

    wr.release()
    cap.release()
    if stock_cap is not None:
        stock_cap.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp,
        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-pix_fmt", "yuv420p", VID,
    ], check=True)
    print("готово:", VID)


if __name__ == "__main__":
    main()

"""Сборка ролика 29 («Санкт-Петербургский парадокс»).

Слойность: A-roll / сетка -> предметная инфографика -> субтитры.
Речь добавляется без резки в sfx29.py. Вся анимация вычисляется только из
локального времени плана, поэтому любой кадр воспроизводим и seek-safe.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard29 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, STOCK_KINDS,
                          STOCK_SOURCES, GFX_ZONE, GFX_X, TAG, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# 34 детекции по дублю после hflip: cx 404±7, глаза y691±5, лицо 285±8px.
# Оба кропа ставят глаза на 42% высоты карточки. A1 ограничен правым краем.
FRAMINGS = {
    "A1": dict(w=632, h=1002, x=88, y=270),
    "A2": dict(w=576, h=914, x=116, y=307),
}


def clamp01(v):
    return max(0.0, min(1.0, v))


def stagger(lt, n, t0=0.08, step=0.085, dur=0.38):
    return [ease_out(clamp01((lt - t0 - i * step) / dur)) for i in range(n)]


def cv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def source_card(frame, kind):
    """Штатный A-roll: hflip, кроп, ресайз. Вкусового грейда нет."""
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    f = FRAMINGS[kind]
    w, h = f["w"], f["h"]
    x = max(0, min(f["x"], iw - w))
    y = max(0, min(f["y"], ih - h))
    fr = fr[y:y + h, x:x + w]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_CUBIC)
    return cv_to_pil(fr)


def stock_card(frame):
    """Сток в Card B: crop-to-fill и спокойное затемнение под белую типографику."""
    img = fit_cover(cv_to_pil(frame), CARD_B[2], CARD_B[3])
    img = ImageEnhance.Color(img).enhance(.92)
    img = ImageEnhance.Brightness(img).enhance(.70)
    return img


class StockReader:
    """Последовательно читает два коротких стоковых фрагмента без рывков времени."""
    def __init__(self):
        self.kind = None
        self.cap = None
        self.fps = 0.0
        self.next_frame = 0
        self.last = None

    def _open(self, kind):
        self.release()
        path, ss = STOCK_SOURCES[kind]
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.next_frame = int(round(ss * self.fps))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.next_frame)
        self.kind = kind
        self.last = None

    def frame(self, kind, lt):
        if self.kind != kind:
            self._open(kind)
        _, ss = STOCK_SOURCES[kind]
        target = int(math.floor((ss + lt) * self.fps + 1e-6))
        while self.next_frame <= target:
            ok, fr = self.cap.read()
            if not ok:
                break
            self.last = fr
            self.next_frame += 1
        if self.last is None:
            raise RuntimeError(f"Не удалось прочитать сток {kind}")
        return self.last

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.kind = None


def _layer():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def _col(color, alpha):
    return tuple(color) + (int(round(255 * clamp01(alpha))),)


def seg(d, a, b, p=1.0, alpha=1.0, width=5, color=WHITE):
    p = clamp01(p)
    if p <= .002 or alpha <= .002:
        return
    q = (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)
    d.line([a, q], fill=_col(color, alpha), width=width)


def glow_layer(layer, radius=14, color=WHITE, strength=.42):
    a = layer.split()[3]
    blur = a.filter(ImageFilter.GaussianBlur(radius))
    gl = Image.new("RGBA", layer.size, tuple(color) + (0,))
    gl.putalpha(blur.point(lambda v: int(v * strength)))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(gl)
    out.alpha_composite(layer)
    return out


def draw_center_text(lay, text, xy, size, color=WHITE, kind="sans",
                     alpha=1.0, scale=1.0, glow=None, glow_a=.50):
    if alpha <= .003 or scale <= .02:
        return
    f = font(kind, max(12, int(round(size * scale))))
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    bb = probe.textbbox((0, 0), text, font=f, anchor="mm")
    pad = max(24, int(size * .34))
    sw, sh = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    sub = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    ImageDraw.Draw(sub).text((sw / 2, sh / 2), text, font=f,
                             fill=_col(color, alpha), anchor="mm")
    sub = glow_layer(sub, max(8, int(size * .13)), glow or color, glow_a)
    lay.alpha_composite(sub, (int(xy[0] - sw / 2), int(xy[1] - sh / 2)))


_ZMASK = None


def zone_mask():
    global _ZMASK
    if _ZMASK is None:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([GX0 + 14, GY0 + 14, GX1 - 14, GY1 - 14], fill=255)
        _ZMASK = m.filter(ImageFilter.GaussianBlur(4))
    return _ZMASK


# --- монета ---------------------------------------------------------------

def draw_coin(lay, cx, cy, r, heads=True, p=1.0, alpha=1.0, flip=0.0,
              color=WHITE):
    """Живая монета: сторона орла помечена «О», сторона решки — «Р»."""
    p = clamp01(p)
    if p <= .003:
        return
    sx = max(.08, abs(math.cos(flip * math.pi)))
    rr = r * (.62 + .38 * ease_out(p))
    sub = Image.new("RGBA", (int(2 * r + 48), int(2 * r + 48)), (0, 0, 0, 0))
    d = ImageDraw.Draw(sub)
    ox, oy = sub.size[0] / 2, sub.size[1] / 2
    box = [ox - rr * sx, oy - rr, ox + rr * sx, oy + rr]
    d.ellipse(box, fill=_col(BLACK, alpha * p), outline=_col(color, alpha * p), width=6)
    if sx > .25:
        label = "О" if heads else "Р"
        lf = font("sans", max(18, int(rr * 1.05)))
        label_layer = Image.new("RGBA", sub.size, (0, 0, 0, 0))
        ImageDraw.Draw(label_layer).text(
            (ox, oy - rr * .03), label, font=lf, fill=_col(color, alpha * p), anchor="mm")
        if sx < .98:
            nw = max(1, int(round(label_layer.width * sx)))
            label_layer = label_layer.resize((nw, label_layer.height), Image.LANCZOS)
            sub.alpha_composite(label_layer, (int((sub.width - nw) / 2), 0))
        else:
            sub.alpha_composite(label_layer)
    sub = glow_layer(sub, 10, color, .34)
    lay.alpha_composite(sub, (int(cx - sub.size[0] / 2), int(cy - sub.size[1] / 2)))


def arrow(d, a, b, p=1.0, color=WHITE, alpha=1.0, width=5):
    p = clamp01(p)
    seg(d, a, b, p, alpha, width, color)
    if p > .75:
        q = (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        for s in (-1, 1):
            aa = ang + s * 2.55
            d.line([q, (q[0] + 22 * math.cos(aa), q[1] + 22 * math.sin(aa))],
                   fill=_col(color, alpha), width=width)


def coin_row(lay, states, y, pvals, r=66, x0=None, x1=None):
    n = len(states)
    x0 = x0 if x0 is not None else 260
    x1 = x1 if x1 is not None else 820
    xs = [540] if n == 1 else np.linspace(x0, x1, n)
    for i, heads in enumerate(states):
        draw_coin(lay, float(xs[i]), y, r, heads, pvals[i], flip=(1 - pvals[i]) * .55)
    return xs


def payout_scene(lay, lt, states, value):
    ps = stagger(lt, len(states), .06, .13, .34)
    coin_row(lay, states, 940, ps, r=62, x0=330, x1=750)
    p = ease_out(clamp01((lt - .48) / .20))
    s = .60 + .40 * p
    draw_center_text(lay, f"{value} ₽", (540, 1308), 150, WHITE,
                     alpha=p, scale=s, glow_a=.55)


def probability_scene(lay, lt, states, value, frac):
    ps = stagger(lt, len(states), .05, .10, .30)
    coin_row(lay, states, 880, ps, r=48, x0=350, x1=730)
    pv = ease_out(clamp01((lt - .28) / .30))
    pp = ease_out(clamp01((lt - .55) / .34))
    draw_center_text(lay, f"{value} ₽", (360, 1268), 105, WHITE, alpha=pv,
                     scale=.65 + .35 * pv)
    draw_center_text(lay, frac, (720, 1268), 116, BLUE, alpha=pp,
                     scale=.55 + .48 * pp, glow=BLUE_GLOW, glow_a=.72)


# --- графические планы ----------------------------------------------------

def g_infiniteWin(lay, lt):
    ps = stagger(lt, 4, .02, .10, .30)
    xs = [275, 410, 545, 680]
    for j, (x, p) in enumerate(zip(xs, ps), 1):
        for i in range(j):
            draw_coin(lay, x, 1390 - i * 92, 38, True, p, alpha=.92)
    p = ease_out(clamp01((lt - .48) / .14))
    draw_center_text(lay, "∞", (788, 1050), 180, WHITE, alpha=p,
                     scale=.60 + .40 * p, glow_a=.62)


def g_coinLaunch(lay, lt):
    p = clamp01(lt / .85)
    x = 320 + 440 * p
    y = 1325 - 430 * math.sin(math.pi * p)
    draw_coin(lay, x, y, 74, heads=p > .52, p=min(1, p * 4), flip=p * 5.5)
    d = ImageDraw.Draw(lay)
    pts = [(320 + 440 * q, 1325 - 430 * math.sin(math.pi * q)) for q in np.linspace(0, p, 34)]
    if len(pts) > 1:
        d.line(pts, fill=_col(WHITE, .34), width=4)


def g_coinLoop(lay, lt):
    ps = stagger(lt, 5, .03, .29, .28)
    states = [False, False, False, False, True]
    xs = coin_row(lay, states, 1120, ps, r=52, x0=250, x1=830)
    d = ImageDraw.Draw(lay)
    for i in range(4):
        arrow(d, (xs[i] + 58, 1120), (xs[i + 1] - 58, 1120), ps[i + 1], alpha=.60, width=4)


def g_ruleHeads(lay, lt):
    p1, p2, p3 = stagger(lt, 3, .05, .14, .34)
    draw_coin(lay, 310, 1090, 86, True, p1)
    d = ImageDraw.Draw(lay)
    arrow(d, (410, 1090), (640, 1090), p2, alpha=.82)
    # открытая «ячейка выплаты», число появится только на следующем плане
    d.rounded_rectangle([660, 965, 850, 1215], radius=34,
                        outline=_col(WHITE, p3), width=6)
    for i in range(3):
        d.ellipse([704 + i * 34, 1072 - i * 22, 770 + i * 34, 1138 - i * 22],
                  outline=_col(WHITE, p3 * .82), width=4)


def g_doubling(lay, lt):
    vals = ["2", "4", "8", "16"]
    ps = stagger(lt, 4, .04, .18, .30)
    d = ImageDraw.Draw(lay)
    xs = [250, 435, 620, 805]
    for i, (x, v, p) in enumerate(zip(xs, vals, ps)):
        draw_center_text(lay, v, (x, 1120), 102 if i < 3 else 92, WHITE,
                         alpha=p, scale=.62 + .38 * p)
        if i:
            arrow(d, (xs[i - 1] + 58, 1120), (x - 58, 1120), p, alpha=.60, width=4)
    draw_center_text(lay, "₽", (810, 1328), 60, WHITE, alpha=ps[-1] * .72)


def g_expectTerms(lay, lt):
    rows = [("2 × 1/2", "1"), ("4 × 1/4", "1"), ("8 × 1/8", "1")]
    ps = stagger(lt, 3, .04, .34, .34)
    ys = [850, 1110, 1370]
    d = ImageDraw.Draw(lay)
    for (lhs, rhs), y, p in zip(rows, ys, ps):
        draw_center_text(lay, lhs, (410, y), 72, WHITE, alpha=p)
        arrow(d, (585, y), (670, y), p, alpha=.55, width=4)
        draw_center_text(lay, rhs, (770, y), 108, WHITE, alpha=p,
                         scale=.58 + .42 * p, glow_a=.58)


def g_infiniteOutcomes(lay, lt):
    d = ImageDraw.Draw(lay)
    p = ease_out(clamp01(lt / .42))
    root = (245, 1120)
    draw_coin(lay, *root, 42, False, p)
    levels = 5
    for i in range(levels):
        q = ease_out(clamp01((lt - .12 - i * .18) / .30))
        x0, x1 = 290 + i * 105, 380 + i * 105
        spread = 210 / (i + 1)
        y0 = 1120
        seg(d, (x0, y0), (x1, y0 - spread), q, .74, 4)
        seg(d, (x0, y0), (x1, y0 + spread), q, .74, 4)
        draw_coin(lay, x1, y0 - spread, max(20, 36 - i * 3), True, q, alpha=.80)
        draw_coin(lay, x1, y0 + spread, max(20, 36 - i * 3), False, q, alpha=.58)
    draw_center_text(lay, "…", (850, 1120), 110, WHITE,
                     alpha=ease_out(clamp01((lt - 1.05) / .28)))


def g_infiniteSum(lay, lt):
    vals = ["1", "+", "1", "+", "1", "+", "…", "=", "∞"]
    xs = np.linspace(235, 820, len(vals))
    ps = stagger(lt, len(vals), .05, .19, .25)
    for i, (v, x, p) in enumerate(zip(vals, xs, ps)):
        blue = i == len(vals) - 1
        draw_center_text(lay, v, (float(x), 1130), 116 if v in {"1", "∞"} else 72,
                         BLUE if blue else WHITE, alpha=p,
                         scale=(.55 + .48 * p) if blue else (.72 + .28 * p),
                         glow=BLUE_GLOW if blue else WHITE,
                         glow_a=.72 if blue else .42)


def g_priceRise(lay, lt):
    d = ImageDraw.Draw(lay)
    p = ease_out(clamp01(lt / .44))
    seg(d, (430, 1450), (430, 780), p, .90, 6)
    arrow(d, (430, 1450), (430, 780), p, alpha=.90, width=6)
    levels = [(1360, 1), (1200, 2), (1040, 4), (880, 8)]
    ps = stagger(lt, 4, .10, .18, .28)
    for (y, n), q in zip(levels, ps):
        d.line([(410, y), (470, y)], fill=_col(WHITE, q), width=5)
        for j in range(n):
            xx = 555 + (j % 4) * 56
            yy = y - (j // 4) * 52
            d.ellipse([xx - 22, yy - 22, xx + 22, yy + 22],
                      outline=_col(WHITE, q * .86), width=4)
    draw_center_text(lay, "…", (715, 790), 100, WHITE,
                     alpha=ease_out(clamp01((lt - .85) / .32)))


def g_tinyTail(lay, lt):
    d = ImageDraw.Draw(lay)
    ps = stagger(lt, 7, .02, .13, .25)
    xs = np.linspace(250, 825, 7)
    for i, (x, p) in enumerate(zip(xs, ps)):
        r = max(22, 52 - i * 4)
        draw_coin(lay, float(x), 1080, r, False, p, alpha=max(.25, 1 - i * .11))
        if i:
            w = max(1, 7 - i)
            seg(d, (xs[i - 1] + 45, 1080), (x - 42, 1080), p,
                max(.18, .76 - i * .09), w)
    draw_center_text(lay, "…", (845, 1315), 94, WHITE, alpha=ps[-1] * .65)


def g_tinyWin(lay, lt):
    p1, p2, p3 = stagger(lt, 3, .02, .18, .30)
    d = ImageDraw.Draw(lay)
    # почти исчезнувшая ветвь слева направо
    seg(d, (230, 1120), (595, 1120), p1, .38, 2)
    draw_center_text(lay, "1/2ⁿ", (330, 1010), 54, WHITE, alpha=p1 * .65)
    # огромная выплата, но добраться до неё почти невозможно
    for i in range(6):
        for j in range(i + 1):
            d.ellipse([635 + j * 36, 1415 - i * 72, 701 + j * 36, 1481 - i * 72],
                      outline=_col(WHITE, p2 * .78), width=4)
    draw_center_text(lay, "2ⁿ ₽", (730, 850), 96, WHITE, alpha=p3,
                     scale=.64 + .36 * p3)


def g_stockSpb(lay, lt):
    p1 = ease_out(clamp01((lt - .04) / .40))
    p2 = ease_out(clamp01((lt - .34) / .40))
    draw_center_text(lay, "санкт-петербургский", (540, 860), 54,
                     WHITE, kind="serif_it", alpha=p1, scale=.70 + .30 * p1)
    draw_center_text(lay, "ПАРАДОКС", (540, 1035), 82,
                     WHITE, alpha=p2, scale=.56 + .44 * p2, glow_a=.58)


GFX = {
    "infiniteWin": g_infiniteWin,
    "coinLaunch": g_coinLaunch,
    "coinLoop": g_coinLoop,
    "ruleHeads": g_ruleHeads,
    "payout2": lambda lay, t: payout_scene(lay, t, [True], 2),
    "payout4": lambda lay, t: payout_scene(lay, t, [False, True], 4),
    "payout8": lambda lay, t: payout_scene(lay, t, [False, False, True], 8),
    "doubling": g_doubling,
    "prob2": lambda lay, t: probability_scene(lay, t, [True], 2, "1/2"),
    "prob4": lambda lay, t: probability_scene(lay, t, [False, True], 4, "1/4"),
    "prob8": lambda lay, t: probability_scene(lay, t, [False, False, True], 8, "1/8"),
    "expectTerms": g_expectTerms,
    "infiniteOutcomes": g_infiniteOutcomes,
    "infiniteSum": g_infiniteSum,
    "tinyTail": g_tinyTail,
    "tinyWin": g_tinyWin,
    "stockSpb": g_stockSpb,
}


def graphics_layer(kind, lt):
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    if kind in STOCK_KINDS:
        x, y, w, h = CARD_B
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([x + 18, y + 18, x + w - 18, y + h - 18], fill=255)
        lay.putalpha(ImageChops.multiply(lay.split()[3], m.filter(ImageFilter.GaussianBlur(3))))
    else:
        lay.putalpha(ImageChops.multiply(lay.split()[3], zone_mask()))
    return lay


# --- субтитры --------------------------------------------------------------

def text_layer_fast(size, items):
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    for it in items:
        txt, f, xy = it["text"], it["font"], it["xy"]
        anchor = it.get("anchor", "mm")
        bbox = probe.textbbox(xy, txt, font=f, anchor=anchor)
        gr = int(it.get("glow_r", 14)); pad = max(8, gr * 3)
        left = max(0, int(bbox[0] - pad)); top = max(0, int(bbox[1] - pad))
        right = min(size[0], int(bbox[2] + pad)); bottom = min(size[1], int(bbox[3] + pad))
        if right <= left or bottom <= top:
            continue
        sub = Image.new("RGBA", (right-left, bottom-top), (0,0,0,0))
        d = ImageDraw.Draw(sub)
        d.text((xy[0]-left, xy[1]-top), txt, font=f, fill=it["fill"]+(255,), anchor=anchor)
        a = sub.split()[3]
        blur = a.filter(ImageFilter.GaussianBlur(gr))
        gl = Image.new("RGBA", sub.size, it.get("glow", it["fill"])+(0,))
        gl.putalpha(blur.point(lambda v: int(v * it.get("glow_a", .50))))
        comp = Image.new("RGBA", sub.size, (0,0,0,0)); comp.alpha_composite(gl); comp.alpha_composite(sub)
        sub = comp
        rev = it.get("reveal")
        if rev and rev[0] == "wipe":
            rx = int(rev[1] - left)
            if rx <= 0: continue
            if rx < sub.size[0]:
                m = Image.new("L", sub.size, 0)
                ImageDraw.Draw(m).rectangle([0,0,rx,sub.size[1]], fill=255)
                sub.putalpha(Image.composite(sub.split()[3], Image.new("L",sub.size,0), m))
        op = it.get("opacity", 1.0)
        if op < 1:
            sub.putalpha(sub.split()[3].point(lambda v: int(v * op)))
        layer.alpha_composite(sub, (left, top))
    return layer


def line_items(runs, xy, anchor, opacity=1.0, reveal_chars=None, max_w=None):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    sizes = [sz for _, _, sz in runs]
    fonts = [font(kmap[k], sz) for _, k, sz in runs]
    widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
    total = sum(widths) + 8 * (len(runs)-1)
    if max_w and total > max_w:
        sc = max_w / total
        sizes = [max(22, int(s * sc)) for s in sizes]
        fonts = [font(kmap[runs[i][1]], sizes[i]) for i in range(len(runs))]
        widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
        total = sum(widths) + 8 * (len(runs)-1)
    x, y = xy; cx = x-total/2 if anchor[0] == "m" else x
    out=[]; cum=0
    for i,(txt,k,_) in enumerate(runs):
        f=fonts[i]
        it=dict(text=txt,font=f,xy=(cx,y),anchor="l"+anchor[1],fill=WHITE,
                glow=WHITE,glow_r=int(12+sizes[i]*.08),
                glow_a=.60 if k=="s" else .48,opacity=opacity)
        if reveal_chars is not None:
            done=reveal_chars-cum
            if done<=0:
                cx += widths[i]+8; cum += len(txt); continue
            if done<len(txt): it["reveal"]=("wipe",cx+f.getlength(txt[:int(done)])+2)
        out.append(it); cx += widths[i]+8; cum += len(txt)
    return out


def build_blocks():
    blocks=[]; cur=[]
    for i,c in enumerate(CAPS):
        if cur:
            prev=CAPS[cur[-1]]
            if c[0]-prev[1]>.30 or len(cur)>=3 or shot_at(c[0]) is not shot_at(prev[0]):
                blocks.append(cur); cur=[]
        cur.append(i)
    if cur: blocks.append(cur)
    info={}
    for bid,b in enumerate(blocks):
        end=max(CAPS[i][1] for i in b)
        sizes=[max(sz for _,_,sz in CAPS[i][2]) for i in b]
        offs=[]; acc=0.0
        for k in range(len(b)):
            if k: acc += max(60.0,(sizes[k-1]+sizes[k])*.70)
            offs.append(acc)
        for pos,i in enumerate(b): info[i]=(bid,pos,len(b),end,offs[pos])
    return info


BLOCKS=build_blocks(); DIM=[1.0,.74,.58]


def caption_layer(t):
    kind=shot_at(t)[2]; slot=SLOTS[slot_for(kind)]
    active=[i for i,c in enumerate(CAPS) if c[0]<=t<BLOCKS[i][3]]
    items=[]
    for idx in active[-3:]:
        t0,_,runs,_=CAPS[idx]; bid,pos,_,_,yoff=BLOCKS[idx]
        newer=sum(1 for j in active if j>idx and BLOCKS[j][0]==bid)
        bx,by=slot["x"],slot["y"]+yoff
        if slot["scatter"] and bid%3!=0: bx += [-46,38,-22][min(pos,2)]
        lt=t-t0; nchars=sum(len(r[0]) for r in runs)
        dur=min(.55,max(.30,nchars*.030)); reveal=nchars if lt>=dur else nchars*lt/dur
        op=(.35+.65*ease_out(min(1,lt/.40)))*DIM[min(newer,2)]
        max_w=slot["x_max"]-bx if slot["anchor"][0]=="l" else 2*min(bx-slot["x_min"],slot["x_max"]-bx)
        items += line_items(runs,(bx,by),slot["anchor"],op,reveal,max_w)
    return text_layer_fast((W,H),items) if items else None


def background(kind, fr, t, lt, stocks):
    canvas=Image.new("RGBA",(W,H),(0,0,0,255))
    if kind in FACE_KINDS:
        card=source_card(fr,kind)
        canvas.paste(card,(CARD_A[0],CARD_A[1]),rounded_mask(CARD_A[2],CARD_A[3],R_A))
    elif kind in STOCK_KINDS:
        card = stock_card(stocks.frame(kind, lt))
        canvas.paste(card, (CARD_B[0], CARD_B[1]), rounded_mask(CARD_B[2], CARD_B[3], R_B))
    else:
        x,y,w,h=CARD_A; key=int(t*10)
        if key not in GRID_CACHE:
            if len(GRID_CACHE)>8: GRID_CACHE.clear()
            GRID_CACHE[key]=grid_canvas(w,h,phase=(key/10)*.7)
        canvas.paste(GRID_CACHE[key],(x,y),rounded_mask(w,h,R_A))
    return canvas


def compose(kind, fr, t, t0, stocks):
    canvas=background(kind,fr,t,t-t0,stocks)
    gl=graphics_layer(kind,t-t0)
    if gl is not None: canvas.alpha_composite(gl)
    cl=caption_layer(t)
    if cl is not None: canvas.alpha_composite(cl)
    return canvas


def main():
    cap=cv2.VideoCapture(SRC); os.makedirs(f"{BUILD}/assets",exist_ok=True)
    stocks=StockReader()
    nf=int(round(DUR*FPS))
    enc=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y",
        "-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r",str(FPS),
        "-i","-","-an","-c:v","libx264","-crf","15","-preset","medium",
        "-pix_fmt","yuv420p","-color_range","tv","-colorspace","bt709",
        "-color_primaries","bt709","-color_trc","bt709",
        "-bsf:v","h264_metadata=colour_primaries=1:transfer_characteristics=1:"
                 "matrix_coefficients=1:video_full_range_flag=0",VID],stdin=subprocess.PIPE)
    last=None
    for fno in range(nf):
        ok,fr=cap.read(); last=fr if ok else last; fr=last
        t=fno/FPS; t0,_,kind,_=shot_at(t)
        canvas=compose(kind,fr,t,t0,stocks)
        enc.stdin.write(pil_to_cv(canvas.convert("RGB")).tobytes())
        if fno and fno%300==0: print(f"frame {fno}/{nf}",flush=True)
    enc.stdin.close(); enc.wait(); cap.release(); stocks.release()
    if enc.returncode: raise SystemExit(enc.returncode)
    print("готово:",VID)


if __name__=="__main__":
    main()

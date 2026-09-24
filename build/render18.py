"""Сборка ролика 18 («правило високосного года»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx18.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Отличия тракта, унаследованные из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп (ролик 16: подтяжка гаммы к 130/112/104
  делает лицо вымытым и поднимает шум).
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v).

Новое в этом ролике: исходник 720x1280, а не 1080x1920, поэтому FRAMINGS
пересчитаны под него — числа из render17.py сюда не переносятся.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard18 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, CENTURIES, LEAP_TICK_YEARS,
                          YEAR_J, YEAR_G, YEAR_J_SHORT, DIFF,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_leap.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/18/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GRID_CACHE = {}

# Крупность A-roll. Исходник 720x1280; детектор даёт лицо стабильно
# (x~235 y~513 d~303 после hflip), линия глаз y~640. Кроп ставит глаза
# на 41% высоты карточки. Соотношение кропа = 870/1380 карточки A.
FRAMINGS = {
    "A1": dict(w=667, h=1058, x=52, y=206),
    "A2": dict(w=587, h=931, x=94, y=258),
}


def cv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def fit_bgr(fr, w, h):
    ih, iw = fr.shape[:2]
    target = w / h
    if iw / ih > target:
        nw = int(ih * target)
        x = (iw - nw) // 2
        fr = fr[:, x:x + nw]
    else:
        nh = int(iw / target)
        y = max(0, min((ih - nh) // 2, ih - nh))
        fr = fr[y:y + nh, :]
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)


def source_card(frame, kind):
    """A-roll: hflip + кроп под карточку A. Грейда нет — см. шапку модуля.

    hflip безусловный: рука с петличкой входит в сырой кадр справа — так же,
    как в ролике 17, где зеркальность подтверждена надписью на футболке.
    """
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    f = FRAMINGS[kind]
    w, h = f["w"], f["h"]
    x = max(0, min(f["x"], iw - w))
    y = max(0, min(f["y"], ih - h))
    fr = fr[y:y + h, x:x + w]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_AREA)
    return cv_to_pil(fr)


STOCK_GAIN = 0.72      # притемнение вставки на 28% по экспозиции (brand-kit §6: 25-30%)


def stock_card(fr):
    """Вставка в карточке B: притемнение умножением + лёгкое обесцвечивание."""
    fr = fit_bgr(fr, CARD_B[2], CARD_B[3])
    fr = np.clip(fr.astype(np.float32) * STOCK_GAIN, 0, 255)
    hsv = cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255)
    fr = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


def clamp01(v):
    return max(0.0, min(1.0, v))


def lerp(a, b, t):
    return a + (b - a) * t


def gtext(text, xy, size=80, blue=False, serif=False, op=1.0, anchor="mm",
          color=None, glow_r=None):
    kind = "serif_it" if serif else "sans"
    fill = color if color is not None else (BLUE if blue else WHITE)
    glow = BLUE_GLOW if blue else (color if color is not None else WHITE)
    gr = glow_r if glow_r is not None else (34 if blue else 15)
    return dict(text=text, font=font(kind, size), xy=xy, anchor=anchor,
                fill=fill, glow=glow,
                glow_r=gr, glow_a=0.86 if blue else 0.48,
                opacity=op)


def pop(lt, dur=0.10):
    return ease_out(clamp01(lt / dur)) if lt >= 0 else 0.0


def blue_pop(lt, dur=0.38):
    """R5b: 0.55 -> 1.0 с микро-оверщутом ~3%."""
    if lt < 0:
        return 0.0, 1.0
    p = clamp01(lt / dur)
    return ease_out(p), 1.0 + 0.03 * math.sin(p * math.pi)


def _layer():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def text_layer(size, items):
    """Быстрая версия style.text_layer: размывается только bbox строки с запасом под glow."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    pd = ImageDraw.Draw(probe)
    for it in items:
        txt, f, xy = it["text"], it["font"], it["xy"]
        anchor = it.get("anchor", "mm")
        bbox = pd.textbbox(xy, txt, font=f, anchor=anchor)
        gr = int(it.get("glow_r", 14))
        pad = max(8, gr * 3)
        left = max(0, int(math.floor(bbox[0] - pad)))
        top = max(0, int(math.floor(bbox[1] - pad)))
        right = min(size[0], int(math.ceil(bbox[2] + pad)))
        bottom = min(size[1], int(math.ceil(bbox[3] + pad)))
        if right <= left or bottom <= top:
            continue
        sub = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
        d = ImageDraw.Draw(sub)
        d.text((xy[0] - left, xy[1] - top), txt, font=f,
               fill=it["fill"] + (255,), anchor=anchor)
        if gr:
            a = sub.split()[3]
            blur = a.filter(ImageFilter.GaussianBlur(gr))
            gl = Image.new("RGBA", sub.size, it.get("glow", it["fill"]) + (0,))
            gl.putalpha(blur.point(lambda v: int(v * it.get("glow_a", 0.55))))
            comp = Image.new("RGBA", sub.size, (0, 0, 0, 0))
            comp.alpha_composite(gl)
            comp.alpha_composite(sub)
            sub = comp
        rev = it.get("reveal")
        if rev and rev[0] == "wipe":
            rx = int(rev[1] - left)
            if rx <= 0:
                continue
            if rx < sub.size[0]:
                m = Image.new("L", sub.size, 0)
                ImageDraw.Draw(m).rectangle([0, 0, rx, sub.size[1]], fill=255)
                m = m.filter(ImageFilter.GaussianBlur(1.2))
                sub.putalpha(Image.composite(sub.split()[3], Image.new("L", sub.size, 0), m))
        op = it.get("opacity", 1.0)
        if op < 1.0:
            sub.putalpha(sub.split()[3].point(lambda v: int(v * op)))
        layer.alpha_composite(sub, (left, top))
    return layer


# --- примитивы -------------------------------------------------------------

def dot(d, x, y, r, alpha=210, color=WHITE):
    if r <= 0 or alpha <= 0:
        return
    d.ellipse([x - r, y - r, x + r, y + r], fill=color + (int(alpha),))


def arrow(d, a, b, alpha=215, width=7, head=24, color=WHITE):
    d.line([a, b], fill=color + (int(alpha),), width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (1, -1):
        d.line([b, (b[0] - head * math.cos(ang - s * 0.45),
                    b[1] - head * math.sin(ang - s * 0.45))],
               fill=color + (int(alpha),), width=width)


def check_mark(d, p, a, b, c, width=14, alpha=238):
    """Галка a->b->c, рисуется на долю p."""
    p = clamp01(p)
    if p <= 0:
        return
    f1 = min(1.0, p / 0.40)
    d.line([a, (a[0] + (b[0] - a[0]) * f1, a[1] + (b[1] - a[1]) * f1)],
           fill=WHITE + (int(alpha),), width=width)
    if p > 0.40:
        f2 = (p - 0.40) / 0.60
        d.line([b, (b[0] + (c[0] - b[0]) * f2, b[1] + (c[1] - b[1]) * f2)],
               fill=WHITE + (int(alpha),), width=width)


def cross_mark(d, cx, cy, s, p=1.0, width=13, alpha=240):
    """Косой крест: две палки, вторая после первой."""
    p = clamp01(p)
    if p <= 0:
        return
    e1 = ease_out(min(1.0, p / 0.55))
    d.line([(cx - s, cy - s), (cx - s + 2 * s * e1, cy - s + 2 * s * e1)],
           fill=WHITE + (int(alpha),), width=width)
    if p > 0.45:
        e2 = ease_out((p - 0.45) / 0.55)
        d.line([(cx + s, cy - s), (cx + s - 2 * s * e2, cy - s + 2 * s * e2)],
               fill=WHITE + (int(alpha),), width=width)


def check_at(d, cx, cy, s, p=1.0, width=None, alpha=238):
    """Галка, вписанная в квадрат 2s вокруг (cx, cy)."""
    w = width if width is not None else max(4, int(s * 0.34))
    check_mark(d, p, (cx - s, cy), (cx - s * 0.28, cy + s * 0.62),
               (cx + s, cy - s * 0.72), width=w, alpha=alpha)


def verdict(d, cx, cy, s, ok, p=1.0, alpha=238):
    if ok:
        check_at(d, cx, cy, s, p, alpha=alpha)
    else:
        cross_mark(d, cx, cy, s * 0.82, p, width=max(4, int(s * 0.30)), alpha=alpha)


def mark_box(d, cx, cy, w, h, p=1.0, alpha=225, width=4, radius=12):
    p = clamp01(p)
    if p <= 0:
        return
    e = 0.72 + 0.28 * ease_out(p)
    hw, hh = w * e / 2, h * e / 2
    d.rounded_rectangle([cx - hw, cy - hh, cx + hw, cy + hh], radius=radius,
                        outline=WHITE + (int(alpha),), width=width)


def dashed(d, a, b, alpha=150, width=3, dash=16, gap=12):
    dx, dy = b[0] - a[0], b[1] - a[1]
    ln = math.hypot(dx, dy)
    if ln < 1:
        return
    ux, uy = dx / ln, dy / ln
    t = 0.0
    while t < ln:
        t2 = min(ln, t + dash)
        d.line([(a[0] + ux * t, a[1] + uy * t), (a[0] + ux * t2, a[1] + uy * t2)],
               fill=WHITE + (int(alpha),), width=width)
        t += dash + gap


def number_items(text, cx, cy, size, prog, blue=False, base_op=1.0):
    """Строка-число посимвольно; prog — сколько символов уже проявилось (дробно)."""
    f = font("sans", size)
    total = f.getlength(text)
    x = cx - total / 2
    out = []
    for i, ch in enumerate(text):
        w = f.getlength(ch)
        a = clamp01(prog - i)
        if a > 0.005 and ch != " ":
            out.append(gtext(ch, (x + w / 2, cy), size, blue=blue,
                             op=base_op * ease_out(a)))
        x += w
    return out


def span_x(text, cx, size, i0, i1):
    """Границы символов [i0, i1) в отцентрованной по cx строке."""
    f = font("sans", size)
    x0 = cx - f.getlength(text) / 2
    return x0 + f.getlength(text[:i0]), x0 + f.getlength(text[:i1])


# --- геометрия ---------------------------------------------------------------

TICK_X0, TICK_PITCH, N_TICKS = 252, 48, 13
AXIS_Y = 1120
DAY_Y = 1010          # квадратик-«день» над шкалой
DAY_S = 36
LABEL_Y = 1215

CAL_C, CAL_R = 7, 5
CELL_W, CELL_H = 88, 72
CAL_X0 = 540 - CAL_C * CELL_W // 2       # 232
CAL_Y0 = 900

CENT_Y, CENT_MARK_Y, CENT_SZ = 1000, 1165, 36
CENT_PITCH = 144

YEAR_Y = 850
CHIP_Y = (1030, 1140, 1250)
CHIP_LX, CHIP_MX = 380, 660
VERDICT_Y = 1430


def tick_x(i):
    return TICK_X0 + i * TICK_PITCH


def cent_x(i):
    return 540 + (i - (len(CENTURIES) - 1) / 2) * CENT_PITCH


# --- графика планов ---------------------------------------------------------

def g_div4(lay, lt):
    """«год високосный если делится на четыре»: шкала лет, каждый четвёртый
    получает высокую засечку, подпись года и квадратик добавленного дня."""
    d = ImageDraw.Draw(lay)
    items = []
    p = clamp01(lt / 0.45)
    if p > 0.01:
        x0 = tick_x(0) - 18
        x1 = tick_x(N_TICKS - 1) + 18
        d.line([(x0, AXIS_Y), (x0 + (x1 - x0) * ease_out(p), AXIS_Y)],
               fill=WHITE + (170,), width=4)
    for i in range(N_TICKS):
        u = clamp01((lt - 0.35 - i * 0.05) / 0.18)
        if u <= 0:
            continue
        e = ease_out(u)
        d.line([(tick_x(i), AXIS_Y - 18 * e), (tick_x(i), AXIS_Y + 18 * e)],
               fill=WHITE + (int(165 * u),), width=4)
    for k, i in enumerate(range(0, N_TICKS, 4)):
        u = clamp01((lt - 1.05 - k * 0.16) / 0.24)
        if u <= 0:
            continue
        e = ease_out(u)
        d.line([(tick_x(i), AXIS_Y - 18 - 24 * e), (tick_x(i), AXIS_Y + 18 + 24 * e)],
               fill=WHITE + (int(235 * u),), width=6)
        items.append(gtext(str(LEAP_TICK_YEARS[k]), (tick_x(i), LABEL_Y), 34, op=e))
    for k, i in enumerate(range(0, N_TICKS, 4)):
        u = clamp01((lt - 1.55 - k * 0.16) / 0.22)
        if u <= 0:
            continue
        e = ease_out(u)
        s = DAY_S * e / 2
        d.rounded_rectangle([tick_x(i) - s, DAY_Y - s, tick_x(i) + s, DAY_Y + s],
                            radius=6, fill=WHITE + (int(235 * u),))
        d.line([(tick_x(i), DAY_Y + s + 4), (tick_x(i), AXIS_Y - 44)],
               fill=WHITE + (int(110 * u),), width=3)
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_noexc(lay, lt):
    """«если бы им пользовались без исключений»: та же шкала едет влево,
    и каждый четвёртый год без единого пропуска получает свой день."""
    d = ImageDraw.Draw(lay)
    items = []
    fade_in = ease_out(clamp01(lt / 0.16))
    off = -lt * 78.0
    d.line([(GX0 + 20, AXIS_Y), (GX1 - 20, AXIS_Y)],
           fill=WHITE + (int(170 * fade_in),), width=4)
    for i in range(-2, 34):
        x = tick_x(i) + off
        if x < GX0 + 20 or x > GX1 - 20:
            continue
        edge = min(1.0, (x - (GX0 + 20)) / 70.0, ((GX1 - 20) - x) / 70.0)
        a = fade_in * max(0.0, edge)
        if a <= 0.01:
            continue
        leap = i % 4 == 0
        if leap:
            d.line([(x, AXIS_Y - 42), (x, AXIS_Y + 42)],
                   fill=WHITE + (int(235 * a),), width=6)
            s = DAY_S / 2
            d.rounded_rectangle([x - s, DAY_Y - s, x + s, DAY_Y + s],
                                radius=6, fill=WHITE + (int(235 * a),))
            d.line([(x, DAY_Y + s + 4), (x, AXIS_Y - 44)],
                   fill=WHITE + (int(110 * a),), width=3)
            if GX0 + 80 < x < GX1 - 80:
                items.append(gtext(str(2020 + i), (x, LABEL_Y), 34, op=a))
        else:
            d.line([(x, AXIS_Y - 18), (x, AXIS_Y + 18)],
                   fill=WHITE + (int(165 * a),), width=4)
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_drift(lay, lt):
    """«календарь со временем начал бы немного съезжать»: месячная сетка,
    тонкая рамка — где дата должна стоять, залитая клетка уезжает от неё."""
    d = ImageDraw.Draw(lay)
    for r in range(CAL_R):
        for c in range(CAL_C):
            u = clamp01((lt - 0.03 - (r * CAL_C + c) * 0.008) / 0.20)
            if u <= 0:
                continue
            x, y = CAL_X0 + c * CELL_W, CAL_Y0 + r * CELL_H
            d.rounded_rectangle([x + 3, y + 3, x + CELL_W - 3, y + CELL_H - 3],
                                radius=8, outline=WHITE + (int(135 * u),), width=3)
    home = 1 * CAL_C + 1
    hx = CAL_X0 + (home % CAL_C) * CELL_W
    hy = CAL_Y0 + (home // CAL_C) * CELL_H
    u = clamp01((lt - 0.55) / 0.25)
    if u > 0:
        d.rounded_rectangle([hx + 3, hy + 3, hx + CELL_W - 3, hy + CELL_H - 3],
                            radius=8, outline=WHITE + (int(215 * u),), width=5)
    v = clamp01((lt - 0.80) / 0.20)
    if v > 0:
        pos = home + max(0.0, (lt - 0.80) * 2.6)
        col = pos % CAL_C
        row = int(pos // CAL_C)
        row = min(row, CAL_R - 1)
        fx = CAL_X0 + col * CELL_W
        fy = CAL_Y0 + row * CELL_H
        d.rounded_rectangle([fx + 7, fy + 7, fx + CELL_W - 7, fy + CELL_H - 7],
                            radius=7, fill=WHITE + (int(230 * ease_out(v)),))
        gap_a, gap_b = hx + CELL_W / 2, fx + CELL_W / 2
        if gap_b - gap_a > 26 or row > home // CAL_C:
            dashed(d, (gap_a, 1300), (max(gap_a + 24, gap_b), 1300),
                   alpha=int(160 * ease_out(v)), width=3)
            arrow(d, (max(gap_a + 24, gap_b) - 22, 1300),
                  (max(gap_a + 26, gap_b), 1300),
                  alpha=int(160 * ease_out(v)), width=3, head=14)


def g_y36525(lay, lt):
    """«год длится неровно 365,25 дня»: белое число (R5a) и квадрат,
    у которого закрашена одна четверть — это те самые ,25.

    Субтитра на плане нет: число звучит в речи и пишется графикой.
    """
    d = ImageDraw.Draw(lay)
    prog = 0.0
    if lt >= 0.02:
        prog = 3 * ease_out(clamp01((lt - 0.02) / 0.10))
    if lt >= 0.55:
        prog = 3 + 3 * ease_out(clamp01((lt - 0.55) / 0.12))
    if prog > 0:
        s = 0.62 + 0.38 * ease_out(clamp01((lt - 0.02) / 0.09))
        lay.alpha_composite(text_layer(
            (W, H), number_items(YEAR_J_SHORT, 540, 900, int(150 * s), prog)))
    qx, qy, qs = 540, 1230, 110
    for k, (cx, cy) in enumerate(((qx - qs / 2, qy - qs / 2), (qx + qs / 2, qy - qs / 2),
                                  (qx - qs / 2, qy + qs / 2), (qx + qs / 2, qy + qs / 2))):
        u = clamp01((lt - 1.05 - k * 0.08) / 0.20)
        if u <= 0:
            continue
        e = ease_out(u)
        h = qs * e / 2
        d.rounded_rectangle([cx - h, cy - h, cx + h, cy + h], radius=8,
                            outline=WHITE + (int(200 * u),), width=4)
    f = clamp01((lt - 1.50) / 0.26)
    if f > 0:
        puls = 0.90 + 0.10 * math.sin((lt - 1.50) * 4.2)
        h = qs * ease_out(f) / 2 - 8
        d.rounded_rectangle([qx - qs / 2 - h, qy - qs / 2 - h,
                             qx - qs / 2 + h, qy - qs / 2 + h],
                            radius=6, fill=WHITE + (int(225 * f * puls),))


BLUE_DIGIT_T = [(3, 0.80), (4, 1.05), (5, 1.50), (6, 1.95), (7, 2.40)]


def g_y3652425(lay, lt):
    """«примерно 365,2425»: синее число-итог (R5b), разряды набираются
    синхронно речи. Субтитра на плане нет."""
    d = ImageDraw.Draw(lay)
    p, o = blue_pop(lt, 0.38)
    if p <= 0:
        return
    size = int(104 * (0.55 + 0.45 * p) * o)
    prog = 3 * ease_out(clamp01(lt / 0.30))
    for _, t0 in BLUE_DIGIT_T:
        prog += ease_out(clamp01((lt - t0) / 0.20))
    lay.alpha_composite(text_layer(
        (W, H), number_items(YEAR_G, 540, 940, size, prog, blue=True)))
    n = max(0, min(len(YEAR_G), int(prog)))
    if n:
        x0, x1 = span_x(YEAR_G, 540, size, 0, n)
        d.line([(x0, 1030), (x1, 1030)], fill=BLUE_GLOW + (170,), width=5)


def g_diff(lay, lt):
    """«из-за этой небольшой разницы»: две длины года, выровненные по разрядам,
    расхождение в рамке, и синяя разница."""
    d = ImageDraw.Draw(lay)
    sz = 70
    items = []
    a1 = ease_out(clamp01(lt / 0.16))
    a2 = ease_out(clamp01((lt - 0.10) / 0.16))
    if a1 > 0:
        items += number_items(YEAR_J, 540, 860, sz, len(YEAR_J), base_op=0.55 * a1)
    if a2 > 0:
        items += number_items(YEAR_G, 540, 950, sz, len(YEAR_G), base_op=0.95 * a2)
    if items:
        lay.alpha_composite(text_layer((W, H), items))
    u = clamp01((lt - 0.30) / 0.24)
    if u > 0:
        lo, hi = span_x(YEAR_J, 540, sz, 5, 8)
        puls = 0.86 + 0.14 * math.sin((lt - 0.30) * 4.6)
        mark_box(d, (lo + hi) / 2, 905, (hi - lo) + 26, 156, u,
                 alpha=int(225 * u * puls), width=4, radius=14)
    p, o = blue_pop(lt - 0.55, 0.38)
    if p > 0:
        lay.alpha_composite(text_layer((W, H), [
            gtext(DIFF, (540, 1200), int(136 * (0.55 + 0.45 * p) * o), blue=True),
        ]))


OV_CELL, OV_GAP = 120, 24
OV_X0 = 540 - (4 * OV_CELL + 3 * OV_GAP) // 2      # 264
OV_Y0, OV_Y1 = 980, 1100
OV_OVER = 26                    # на столько «день» выше клетки — это и есть перебор
OV_BIN = (540, 1400)            # копилка излишков, её же наполняет план accum


def ov_cx(i):
    return OV_X0 + i * (OV_CELL + OV_GAP) + OV_CELL / 2


def g_over4(lay, lt):
    """«каждый четвёртый год слегка перебарщивает с поправкой»: четыре года,
    в четвёртый кладут целый день — он оказывается чуть выше клетки,
    и этот излишек уходит в копилку."""
    d = ImageDraw.Draw(lay)
    for i in range(4):
        u = clamp01((lt - 0.02 - i * 0.12) / 0.20)
        if u <= 0:
            continue
        e = ease_out(u)
        hw, hh = OV_CELL * e / 2, (OV_Y1 - OV_Y0) * e / 2
        cx, cy = ov_cx(i), (OV_Y0 + OV_Y1) / 2
        d.rounded_rectangle([cx - hw, cy - hh, cx + hw, cy + hh], radius=12,
                            outline=WHITE + (int(195 * u),), width=4)
    cx = ov_cx(3)
    hw = OV_CELL / 2 - 8
    day_h = (OV_Y1 - OV_Y0) - 12 + OV_OVER
    cut = clamp01((lt - 1.95) / 0.16)        # излишек уже отделился
    a = clamp01((lt - 0.90) / 0.18)
    if a > 0:
        drop = ease_out(clamp01((lt - 1.25) / 0.30))
        bottom = lerp(930, OV_Y1 - 6, drop)
        top = bottom - day_h + (OV_OVER + 4) * cut
        e = ease_out(a)
        d.rounded_rectangle([cx - hw * e, top, cx + hw * e, bottom], radius=10,
                            fill=WHITE + (int(235 * a),))
        v = clamp01((lt - 1.62) / 0.22)
        if v > 0 and cut <= 0.02:
            puls = 0.80 + 0.20 * math.sin((lt - 1.62) * 5.4)
            d.rounded_rectangle([cx - hw - 10, top - 10, cx + hw + 10, OV_Y0 - 2],
                                radius=8, outline=WHITE + (int(230 * v * puls),),
                                width=4)
    b = clamp01((lt - 1.80) / 0.24)
    if b > 0:
        e = ease_out(b)
        d.rounded_rectangle([OV_BIN[0] - 86, OV_BIN[1] - 30 * e,
                             OV_BIN[0] + 86, OV_BIN[1] + 30 * e],
                            radius=8, outline=WHITE + (int(200 * b),), width=4)
    s = clamp01((lt - 2.00) / 0.36)
    if s > 0:
        e = ease_out(s)
        sx = lerp(cx, OV_BIN[0], e)
        sy = lerp(OV_Y0 - 16, OV_BIN[1], e)
        w = lerp(hw, 56, e)
        d.rounded_rectangle([sx - w, sy - 10, sx + w, sy + 10], radius=5,
                            fill=WHITE + (235,))


ACC_X0, ACC_X1 = 480, 600
ACC_TOP, ACC_BOT = 1000, 1300


def g_accum(lay, lt):
    """«раз в несколько сотен лет накапливается лишний день»: излишки сыплются
    в накопитель, он переполняется — и оттуда выпрыгивает целый день."""
    d = ImageDraw.Draw(lay)
    u = clamp01(lt / 0.20)
    if u > 0:
        d.rounded_rectangle([ACC_X0, ACC_TOP, ACC_X1, ACC_BOT], radius=10,
                            outline=WHITE + (int(205 * u),), width=5)
    # линия лет: по ней излишки съезжают в накопитель, она же держит композицию по центру
    g = clamp01(lt / 0.26)
    if g > 0:
        d.line([(266, 880), (266 + (814 - 266) * ease_out(g), 880)],
               fill=WHITE + (int(120 * g),), width=3)
    fill = clamp01((lt - 0.25) / 1.95)
    burst = clamp01((lt - 2.25) / 0.20)
    if fill > 0 and burst < 1.0:
        top = ACC_BOT - 6 - (ACC_BOT - ACC_TOP - 16) * fill
        if ACC_BOT - 6 - top >= 16:
            d.rounded_rectangle([ACC_X0 + 8, top, ACC_X1 - 8, ACC_BOT - 6], radius=7,
                                fill=WHITE + (int(210 * (1 - burst)),))
    run = clamp01((lt - 0.25) / 1.95)
    if 0 < run < 1:
        cx = 266 + (814 - 266) * run
        d.line([(cx, 852), (cx, 908)], fill=WHITE + (215,), width=5)
    for k in range(22):
        t0 = 0.25 + k * 0.088
        e = (lt - t0) / 0.34
        if e <= 0 or e >= 1:
            continue
        ox = 266 + (814 - 266) * ((t0 - 0.25) / 1.95)
        e = ease_out(e)
        sx = lerp(ox, 540, e)
        sy = lerp(886, ACC_TOP + 16, e)
        d.rounded_rectangle([sx - 30, sy - 5, sx + 30, sy + 5], radius=4,
                            fill=WHITE + (225,))
    if burst > 0:
        e = ease_out(burst)
        h = 60 * e
        cy = lerp(ACC_TOP, 820, e)
        d.rounded_rectangle([540 - h, cy - h, 540 + h, cy + h], radius=10,
                            fill=WHITE + (int(240 * burst),))
    if lt > 2.55:
        puls = 0.5 + 0.5 * math.sin((lt - 2.55) * 5.0)
        d.rounded_rectangle([540 - 74, 820 - 74, 540 + 74, 820 + 74], radius=14,
                            outline=WHITE + (int(90 + 90 * puls),), width=3)


def _centuries(lay, d, items, marks, year_op, t_mark, stagger, lt, box=()):
    """Общая отрисовка ряда веков. marks: список 'check' | 'cross' | None."""
    for i, y in enumerate(CENTURIES):
        items.append(gtext(str(y), (cent_x(i), CENT_Y), CENT_SZ, op=year_op(i)))
    for i, m in enumerate(marks):
        if m is None:
            continue
        u = clamp01((lt - t_mark - i * stagger) / 0.26)
        if u <= 0:
            continue
        verdict(d, cent_x(i), CENT_MARK_Y, 26, m == "check", u, alpha=int(238 * u))
    for i, t0 in box:
        u = clamp01((lt - t0) / 0.26)
        if u > 0:
            puls = 0.85 + 0.15 * math.sin((lt - t0) * 4.6)
            mark_box(d, cent_x(i), CENT_Y, 112, 74, u,
                     alpha=int(215 * u * puls), width=4)


def g_rule4(lay, lt):
    """«год високосный если делится на четыре»: все века делятся на 4 — всем галка."""
    d = ImageDraw.Draw(lay)
    items = []
    _centuries(lay, d, items,
               ["check"] * len(CENTURIES),
               lambda i: ease_out(clamp01((lt - 0.02 - i * 0.10) / 0.22)),
               0.90, 0.11, lt)
    lay.alpha_composite(text_layer((W, H), items))


def g_rule100(lay, lt):
    """«но не является високосным если делится на сто»: галки гаснут, приходят кресты."""
    d = ImageDraw.Draw(lay)
    items = []
    fade = ease_out(clamp01(lt / 0.14))
    for i, y in enumerate(CENTURIES):
        off = clamp01((lt - 0.25 - i * 0.11) / 0.22)
        items.append(gtext(str(y), (cent_x(i), CENT_Y), CENT_SZ,
                           op=fade * (1.0 - 0.30 * off)))
        old = clamp01((lt - 0.25 - i * 0.11) / 0.18)
        if old < 1.0:
            check_at(d, cent_x(i), CENT_MARK_Y, 26, 1.0,
                     alpha=int(238 * fade * (1.0 - old)))
        u = clamp01((lt - 0.30 - i * 0.11) / 0.26)
        if u > 0:
            cross_mark(d, cent_x(i), CENT_MARK_Y, 22, u, width=8, alpha=int(240 * u))
    lay.alpha_composite(text_layer((W, H), items))


BACK_400 = [3]      # из этого ряда на 400 делится только 2000


def g_rule400(lay, lt):
    """«кроме случаев когда он всё же делится на четыреста»: 1600 и 2000 возвращаются."""
    d = ImageDraw.Draw(lay)
    items = []
    for i, y in enumerate(CENTURIES):
        back = i in BACK_400
        u = clamp01((lt - 0.30 - BACK_400.index(i) * 0.22) / 0.26) if back else 0.0
        items.append(gtext(str(y), (cent_x(i), CENT_Y), CENT_SZ,
                           op=0.62 + 0.38 * u))
        if not back:
            cross_mark(d, cent_x(i), CENT_MARK_Y, 22, 1.0, width=8, alpha=190)
            continue
        if u < 1.0:
            cross_mark(d, cent_x(i), CENT_MARK_Y, 22, 1.0, width=8,
                       alpha=int(190 * (1.0 - u)))
        if u > 0:
            check_at(d, cent_x(i), CENT_MARK_Y, 26, u, alpha=int(238 * u))
            puls = 0.85 + 0.15 * math.sin((lt - 0.30) * 4.6)
            mark_box(d, cent_x(i), CENT_Y, 112, 74, u,
                     alpha=int(215 * u * puls), width=4)
    lay.alpha_composite(text_layer((W, H), items))


CHIPS = (("÷ 4", 4), ("÷ 100", 100), ("÷ 400", 400))


def _year_card(lay, lt, year):
    """Год, три проверки делимости и вердикт. Решающая строка — ÷400."""
    d = ImageDraw.Draw(lay)
    items = []
    yp = pop(lt - 0.02, 0.12)
    if yp > 0:
        items += number_items(str(year), 540, YEAR_Y, int(130 * (0.68 + 0.32 * yp)),
                              len(str(year)))
    for k, (label, div) in enumerate(CHIPS):
        u = clamp01((lt - 0.35 - k * 0.27) / 0.22)
        if u <= 0:
            continue
        items.append(gtext(label, (CHIP_LX, CHIP_Y[k]), 54, op=ease_out(u),
                           anchor="lm"))
        m = clamp01((lt - 0.51 - k * 0.27) / 0.24)
        if m > 0:
            verdict(d, CHIP_MX, CHIP_Y[k], 24, year % div == 0, m, alpha=int(238 * m))
    b = clamp01((lt - 1.12) / 0.26)
    if b > 0:
        puls = 0.85 + 0.15 * math.sin((lt - 1.12) * 4.6)
        mark_box(d, 530, CHIP_Y[2], 400, 96, b, alpha=int(210 * b * puls),
                 width=4, radius=16)
    v = clamp01((lt - 1.40) / 0.30)
    if v > 0:
        ok = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)
        verdict(d, 540, VERDICT_Y, 46, ok, v, alpha=int(240 * v))
    lay.alpha_composite(text_layer((W, H), items))


def g_y2000(lay, lt):
    """«год две тысячи является високосным»: год пишет графика, вердикт — галка."""
    _year_card(lay, lt, 2000)


def g_y1900(lay, lt):
    """«а год тысяча девятьсот не является високосным»: ÷400 не проходит."""
    _year_card(lay, lt, 1900)


GFX = {
    "div4": g_div4, "noexc": g_noexc, "drift": g_drift, "y36525": g_y36525,
    "y3652425": g_y3652425, "diff": g_diff, "over4": g_over4, "accum": g_accum,
    "rule4": g_rule4, "rule100": g_rule100, "rule400": g_rule400,
    "y2000": g_y2000, "y1900": g_y1900,
}


def graphics_layer(kind, lt):
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    return lay


def line_items(runs, xy, anchor, opacity=1.0, reveal_chars=None, max_w=None):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    sizes = [sz for _, _, sz in runs]
    fonts = [font(kmap[k], sz) for (_, k, sz) in runs]
    widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
    total = sum(widths) + 8 * (len(runs) - 1)
    if max_w and total > max_w:
        sc = max_w / max(1, total)
        sizes = [max(22, int(sz * sc)) for sz in sizes]
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
                  glow_a=0.60 if k == "s" else 0.48, opacity=opacity)
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
    """Блок = до 2 фраз в пределах одного плана, разрыв по паузе >0.30с."""
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
        t0, _, runs, _ = CAPS[idx]
        bid, pos, _, _ = BLOCKS[idx]
        newer = sum(1 for j in active if j > idx and BLOCKS[j][0] == bid)
        dim = DIM[min(newer, 2)]
        bx, by = slot["x"], slot["y"] + pos * slot["step"]
        if slot["scatter"]:
            calm = bid % 3 == 0                      # каждый третий блок — спокойный
            bx += (0 if calm else [-48, 40, -22][min(pos, 2)])
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


def background(kind, prm, fr, t, stock_reader=None):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        card = source_card(fr, kind)
        canvas.paste(card, (CARD_A[0], CARD_A[1]), rounded_mask(CARD_A[2], CARD_A[3], R_A))
    elif kind == "stock":
        card = stock_card(stock_reader())
        canvas.paste(card, (CARD_B[0], CARD_B[1]), rounded_mask(CARD_B[2], CARD_B[3], R_B))
    else:
        x, y, w, h = CARD_A
        key = int(t * 10)
        if key not in GRID_CACHE:
            if len(GRID_CACHE) > 8:
                GRID_CACHE.clear()
            GRID_CACHE[key] = grid_canvas(w, h, phase=(key / 10) * 0.7)
        canvas.paste(GRID_CACHE[key], (x, y), rounded_mask(w, h, R_A))
    return canvas


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    nf = int(round(DUR * FPS))
    # кадры уходят в ffmpeg сырыми: один проход кодека вместо mp4v + x264
    enc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-pix_fmt", "yuv420p", VID,
    ], stdin=subprocess.PIPE)
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
        enc.stdin.write(pil_to_cv(canvas.convert("RGB")).tobytes())
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}")

    enc.stdin.close()
    enc.wait()
    cap.release()
    if stock_cap is not None:
        stock_cap.release()
    print("готово:", VID)


if __name__ == "__main__":
    main()

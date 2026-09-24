"""Сборка ролика 17 («алгоритм Луна»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx17.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Два отличия от старых рендеров, оба по разбору прошлых роликов:
* A-roll не грейдится — только hflip и кроп. Цифры из гайда (130/112/104)
  на этом тёмном вечернем дубле дают неестественно вымытый кадр.
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v):
  два кодека подряд — лишнее пережатие.
"""
import math
import os
import random
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard17 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, CARD, CARD_SWAPPED, CARD_SUM, CARD_SUM_BAD,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_luhn.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/17/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GRID_CACHE = {}

# крупность A-roll: детектор даёт лицо стабильно (x~309 y~591 d~479 на 1080x1920
# после hflip), линия глаз y~790. Кроп ставит глаза на 41% высоты карточки.
FRAMINGS = {
    "A1": dict(w=1000, h=1586, x=46, y=124),
    "A2": dict(w=880, h=1394, x=106, y=205),
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

    hflip безусловный: надпись на футболке в исходнике читается зеркально —
    проверено по кадру, дубль снят фронталкой.
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

def circ(d, cx, cy, r, p=1.0, width=8, alpha=235, color=WHITE, start=-90):
    p = clamp01(p)
    if p <= 0.001:
        return
    if p >= 0.999:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color + (alpha,), width=width)
    else:
        d.arc([cx - r, cy - r, cx + r, cy + r], start, start + 360 * p,
              fill=color + (alpha,), width=width)


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


def bez_arrow(d, a, c, b, p=1.0, alpha=215, width=6, color=WHITE, head=20):
    p = clamp01(p)
    if p <= 0.01:
        return
    n = 24
    pts = []
    for i in range(n + 1):
        u = (i / n) * p
        x = (1 - u) ** 2 * a[0] + 2 * (1 - u) * u * c[0] + u ** 2 * b[0]
        y = (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * c[1] + u ** 2 * b[1]
        pts.append((x, y))
    d.line(pts, fill=color + (int(alpha),), width=width, joint="curve")
    if p > 0.92:
        ang = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
        for s in (1, -1):
            d.line([pts[-1], (pts[-1][0] - head * math.cos(ang - s * 0.45),
                              pts[-1][1] - head * math.sin(ang - s * 0.45))],
                   fill=color + (int(alpha),), width=width)


def check_mark(d, p, a, b, c, width=14, alpha=238):
    """Галка a->b->c, рисуется на долю p."""
    p = clamp01(p)
    if p <= 0:
        return
    f1 = min(1.0, p / 0.40)
    d.line([a, (a[0] + (b[0] - a[0]) * f1, a[1] + (b[1] - a[1]) * f1)],
           fill=WHITE + (alpha,), width=width)
    if p > 0.40:
        f2 = (p - 0.40) / 0.60
        d.line([b, (b[0] + (c[0] - b[0]) * f2, b[1] + (c[1] - b[1]) * f2)],
               fill=WHITE + (alpha,), width=width)


def cross_mark(d, cx, cy, s, p=1.0, width=13, alpha=240):
    """Косой крест: две палки, вторая после первой."""
    p = clamp01(p)
    if p <= 0:
        return
    e1 = ease_out(min(1.0, p / 0.55))
    d.line([(cx - s, cy - s), (cx - s + 2 * s * e1, cy - s + 2 * s * e1)],
           fill=WHITE + (alpha,), width=width)
    if p > 0.45:
        e2 = ease_out((p - 0.45) / 0.55)
        d.line([(cx + s, cy - s), (cx + s - 2 * s * e2, cy - s + 2 * s * e2)],
               fill=WHITE + (alpha,), width=width)


def tick_under(d, cx, y, w, p=1.0, alpha=235, width=5):
    """Подчёркивание под цифрой. Кольцо на шаге 38px залезает на соседний глиф —
    проверено на кадре: 4539 превращалось в цепочку сросшихся кружков."""
    p = clamp01(p)
    if p <= 0:
        return
    e = ease_out(p) * w / 2
    d.line([(cx - e, y), (cx + e, y)], fill=WHITE + (int(alpha),), width=width)


def mark_box(d, cx, cy, w, h, p=1.0, alpha=225, width=4):
    """Рамка вокруг цифры или пары цифр — вместо кольца, чтобы не задеть соседей."""
    p = clamp01(p)
    if p <= 0:
        return
    e = 0.72 + 0.28 * ease_out(p)
    hw, hh = w * e / 2, h * e / 2
    d.rounded_rectangle([cx - hw, cy - hh, cx + hw, cy + hh], radius=10,
                        outline=WHITE + (int(alpha),), width=width)


# --- геометрия номера карты -------------------------------------------------

PITCH, GAP = 38, 24
ROW_W = 16 * PITCH + 3 * GAP            # 680
ROW_X0 = 540 - ROW_W // 2               # 200
ROW_Y = 900                             # строка цифр номера
DBL_Y = 1030                            # строка удвоенных значений
DIG_SZ = 46

DOT_R = 8
GRID_TOP = 1120
GRID_DY = 30
PACK_DX = 40                            # компактная укладка (sumrow / sum80)
SPREAD_DX = 78                          # разнесённые десятки (div10 / nodiv)
SPREAD_TOP = 1080


def digit_x(i):
    return ROW_X0 + i * PITCH + (i // 4) * GAP + PITCH // 2


def luhn_parts(num):
    """[(цифра, удвоенное или None, итоговое)] — удваивается каждая вторая с конца."""
    out = []
    for i, ch in enumerate(num):
        d = int(ch)
        if i % 2 == 0:
            v = d * 2
            out.append((d, v, v - 9 if v > 9 else v))
        else:
            out.append((d, None, d))
    return out


PARTS = luhn_parts(CARD)
DBL_IDX = [i for i in range(16) if i % 2 == 0]          # слева направо
DBL_IDX_R = list(reversed(DBL_IDX))                     # с конца номера
BIG_IDX = [i for i in DBL_IDX if PARTS[i][1] > 9]       # удвоение дало больше 9


def digits_row(items, num=CARD, y=ROW_Y, size=DIG_SZ, alpha=1.0, only=None,
               reveal=None):
    """Строка номера. only — множество индексов, reveal(i) -> 0..1 прозрачность."""
    for i, ch in enumerate(num):
        if only is not None and i not in only:
            continue
        op = alpha if reveal is None else alpha * reveal(i)
        if op <= 0.01:
            continue
        items.append(gtext(ch, (digit_x(i), y), size, op=op))


# --- графика планов ---------------------------------------------------------

def g_scan(lay, lt):
    """Хук: по номеру проходит луч и цепляется за одну цифру.

    Номер здесь ещё не объясняется — кадр показывает сам факт проверки.
    """
    d = ImageDraw.Draw(lay)
    x0, x1 = digit_x(0) - 26, digit_x(15) + 26
    p = clamp01((lt - 0.15) / 0.90)
    sx = x0 + (x1 - x0) * ease_out(p)
    items = []
    for i, ch in enumerate(CARD):
        passed = digit_x(i) < sx
        op = 0.95 if passed else 0.38
        items.append(gtext(ch, (digit_x(i), ROW_Y), DIG_SZ, op=op))
    if 0.01 < p < 0.99:
        d.line([(sx, ROW_Y - 52), (sx, ROW_Y + 52)], fill=WHITE + (215,), width=5)
    u = clamp01((lt - 1.08) / 0.22)
    if u > 0:
        mark_box(d, digit_x(9), ROW_Y, 34, 62, u, alpha=int(240 * u), width=4)
        cross_mark(d, digit_x(9), 1030, 26, clamp01((lt - 1.30) / 0.30), width=8)
    lay.alpha_composite(text_layer((W, H), items))


def g_card16(lay, lt):
    """«не просто случайный набор цифр»: номер набирается группами по четыре."""
    d = ImageDraw.Draw(lay)
    items = []
    for i, ch in enumerate(CARD):
        u = clamp01((lt - (i // 4) * 0.28 - (i % 4) * 0.05) / 0.22)
        if u > 0:
            items.append(gtext(ch, (digit_x(i), ROW_Y), DIG_SZ, op=ease_out(u)))
    for g in range(4):
        u = clamp01((lt - 1.55 - g * 0.10) / 0.30)
        if u <= 0:
            continue
        a, b = digit_x(g * 4) - 20, digit_x(g * 4 + 3) + 20
        d.line([(a, 985), (a + (b - a) * ease_out(u), 985)],
               fill=WHITE + (int(150 * u),), width=4)
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_check(lay, lt):
    """«в нём зашита математическая проверка»: все цифры сходятся в один узел."""
    d = ImageDraw.Draw(lay)
    node = (540, 1330)
    for i in range(16):
        u = ease_out(clamp01((lt - 0.25 - i * 0.04) / 0.42))
        if u <= 0:
            continue
        a = (digit_x(i), ROW_Y + 44)
        d.line([a, (a[0] + (node[0] - a[0]) * u, a[1] + (node[1] - a[1]) * u)],
               fill=WHITE + (int(105 * u),), width=3)
    v = clamp01((lt - 1.35) / 0.30)
    if v > 0:
        circ(d, node[0], node[1], 46 * ease_out(v), 1.0, width=6, alpha=int(235 * v))
        puls = 0.5 + 0.5 * math.sin((lt - 1.35) * 5.2)
        dot(d, node[0], node[1], 16 + 5 * puls, 140 + 90 * puls)
    items = []
    digits_row(items, alpha=0.9)
    lay.alpha_composite(text_layer((W, H), items))


def g_every2(lay, lt):
    """«начиная с конца каждую вторую цифру удваивают»: стрелка справа налево,
    кольца на каждой второй цифре, под ними — удвоенные значения."""
    d = ImageDraw.Draw(lay)
    p = clamp01((lt - 0.15) / 0.60)
    if p > 0.01:
        arrow(d, (digit_x(15) + 26, 800),
              (digit_x(15) + 26 - (ROW_W + 10) * ease_out(p), 800),
              alpha=200, width=5, head=20)
    items = []
    for i, ch in enumerate(CARD):
        marked = i % 2 == 0
        dimmed = clamp01((lt - 0.90) / 0.35)
        op = 1.0 if marked else 1.0 - 0.55 * dimmed
        items.append(gtext(ch, (digit_x(i), ROW_Y), DIG_SZ, op=op))
    for k, i in enumerate(DBL_IDX_R):
        u = clamp01((lt - 0.75 - k * 0.09) / 0.22)
        if u > 0:
            tick_under(d, digit_x(i), ROW_Y + 38, 30, u, alpha=int(235 * u))
    for k, i in enumerate(DBL_IDX_R):
        u = clamp01((lt - 2.05 - k * 0.055) / 0.20)
        if u <= 0:
            continue
        e = ease_out(u)
        d.line([(digit_x(i), ROW_Y + 54), (digit_x(i), ROW_Y + 54 + 34 * e)],
               fill=WHITE + (int(120 * u),), width=3)
        items.append(gtext(str(PARTS[i][1]), (digit_x(i), DBL_Y), 42, op=u))
    lay.alpha_composite(text_layer((W, H), items))


def g_gt9(lay, lt):
    """«если после удвоения получилось число больше 9»: такие значения окольцованы.

    Порог 9 в кадре не пишется — он звучит в речи (brand-kit §5).
    """
    d = ImageDraw.Draw(lay)
    items = []
    digits_row(items, alpha=0.34)
    for i in DBL_IDX:
        big = i in BIG_IDX
        dimmed = clamp01((lt - 0.90) / 0.35)
        op = 1.0 if big else 1.0 - 0.55 * dimmed
        items.append(gtext(str(PARTS[i][1]), (digit_x(i), DBL_Y), 42, op=op))
    for k, i in enumerate(BIG_IDX):
        u = clamp01((lt - 0.30 - k * 0.12) / 0.26)
        if u <= 0:
            continue
        puls = 0.5 + 0.5 * math.sin((lt - 0.30 - k * 0.12) * 4.6)
        tick_under(d, digit_x(i), DBL_Y + 36, 52, u,
                   alpha=int((190 + 55 * puls) * u), width=6)
    lay.alpha_composite(text_layer((W, H), items))


def g_minus9(lay, lt):
    """«из него вычитают девятку»: 12 − 9 = 3, и строка удвоенных обновляется.

    Субтитра на этом плане нет: графика пишет ровно то, что произносится.
    """
    items = []
    parts = [("12", 400, 0.05), ("−", 500, 0.30), ("9", 566, 0.30),
             ("=", 660, 0.62), ("3", 736, 0.62)]
    for txt, x, t0 in parts:
        u = clamp01((lt - t0) / 0.24)
        if u > 0:
            items.append(gtext(txt, (x, 850), 86, op=ease_out(u)))
    for k, i in enumerate(DBL_IDX):
        big = i in BIG_IDX
        if not big:
            items.append(gtext(str(PARTS[i][1]), (digit_x(i), DBL_Y), 42, op=0.55))
            continue
        j = BIG_IDX.index(i)
        sw = clamp01((lt - 0.95 - j * 0.15) / 0.18)
        if sw < 1.0:
            items.append(gtext(str(PARTS[i][1]), (digit_x(i), DBL_Y), 42,
                               op=1.0 - sw))
        if sw > 0:
            items.append(gtext(str(PARTS[i][2]), (digit_x(i), DBL_Y),
                               int(42 * (0.7 + 0.3 * sw)), op=sw))
    lay.alpha_composite(text_layer((W, H), items))


def pack_xy(n):
    """Компактная укладка n точек: 8 столбиков по 10, заполнение по столбцам."""
    c, r = divmod(n, 10)
    return 540 + (c - 3.5) * PACK_DX, GRID_TOP + r * GRID_DY


def spread_xy(n):
    """Разнесённые десятки: те же столбики, но с широким шагом."""
    c, r = divmod(n, 10)
    return 540 + (c - 3.5) * SPREAD_DX, SPREAD_TOP + r * GRID_DY


def g_sumrow(lay, lt):
    """«все 16 цифр с этими изменениями складывают»: из строки сыплются точки."""
    d = ImageDraw.Draw(lay)
    items = []
    for i in range(16):
        u = clamp01((lt - i * 0.035) / 0.20)
        if u > 0:
            items.append(gtext(str(PARTS[i][2]), (digit_x(i), ROW_Y), DIG_SZ,
                               op=ease_out(u)))
    for n in range(CARD_SUM):
        u = clamp01((lt - 0.60 - n * 0.018) / 0.16)
        if u <= 0:
            continue
        x, y = pack_xy(n)
        dot(d, x, y, DOT_R * ease_out(u), 225)
    lay.alpha_composite(text_layer((W, H), items))


def g_sum80(lay, lt):
    """Сумма всех 16 значений. Число 80 в речи не звучит — это добавление."""
    d = ImageDraw.Draw(lay)
    for n in range(CARD_SUM):
        dot(d, *pack_xy(n), DOT_R, 215)
    p, o = blue_pop(lt, 0.38)
    if p > 0:
        lay.alpha_composite(text_layer((W, H), [
            gtext(str(CARD_SUM), (540, 920), int(175 * (0.55 + 0.45 * p) * o),
                  blue=True),
        ]))


def g_div10(lay, lt):
    """«без остатка»: восемь полных десятков, лишнего не остаётся.

    Цифр в кадре нет — «10» и «остаток» звучат в речи, кадр показывает их точками.
    """
    d = ImageDraw.Draw(lay)
    for n in range(CARD_SUM):
        c = n // 10
        u = clamp01((lt - 0.05 - c * 0.05) / 0.22)
        if u <= 0:
            continue
        px, py = pack_xy(n)
        sx, sy = spread_xy(n)
        e = ease_out(u)
        dot(d, px + (sx - px) * e, py + (sy - py) * e, DOT_R, 225)
    for c in range(8):
        u = clamp01((lt - 0.30 - c * 0.05) / 0.24)
        if u <= 0:
            continue
        cx = 540 + (c - 3.5) * SPREAD_DX
        e = ease_out(u)
        d.line([(cx - 26 * e, 1400), (cx + 26 * e, 1400)],
               fill=WHITE + (int(190 * u),), width=4)
    check_mark(d, clamp01((lt - 0.62) / 0.36), (462, 806), (512, 864), (626, 744))


def g_swap(lay, lt):
    """«перепутали местами две соседние цифры»: первые две меняются местами."""
    d = ImageDraw.Draw(lay)
    items = []
    for i in range(2, 16):
        items.append(gtext(CARD[i], (digit_x(i), ROW_Y), DIG_SZ, op=0.92))
    u = ease_out(clamp01((lt - 0.35) / 0.80))
    xa, xb = digit_x(0), digit_x(1)
    for src, dst, lift, ch in ((xa, xb, -70, CARD[0]), (xb, xa, 70, CARD[1])):
        x = src + (dst - src) * u
        y = ROW_Y + lift * math.sin(math.pi * u)
        items.append(gtext(ch, (x, y), DIG_SZ, op=1.0))
    if 0.02 < u < 0.98:
        bez_arrow(d, (xa, ROW_Y - 44), ((xa + xb) / 2, ROW_Y - 116), (xb, ROW_Y - 44),
                  min(1.0, u * 1.3), alpha=170, width=4, head=14)
    v = clamp01((lt - 1.25) / 0.28)
    if v > 0:
        puls = 0.5 + 0.5 * math.sin((lt - 1.25) * 5.0)
        mark_box(d, (xa + xb) / 2, ROW_Y, 84, 62, v,
                 alpha=int((180 + 60 * puls) * v), width=4)
    lay.alpha_composite(text_layer((W, H), items))


def g_nodiv(lay, lt):
    """«сумма перестаёт делиться на 10»: семь полных десятков и две лишние точки.

    Лишняя пара — единственный лаймовый элемент ролика (brand-kit §4).
    """
    d = ImageDraw.Draw(lay)
    for n in range(CARD_SUM_BAD):
        c = n // 10
        u = clamp01((lt - 0.55 - c * 0.06) / 0.22)
        if u <= 0:
            continue
        x, y = spread_xy(n)
        left = c == 7
        dot(d, x, y, DOT_R * (1.25 if left else 1.0) * ease_out(u), 235,
            LIME if left else WHITE)
    for c in range(7):
        u = clamp01((lt - 0.80 - c * 0.05) / 0.24)
        if u <= 0:
            continue
        cx = 540 + (c - 3.5) * SPREAD_DX
        e = ease_out(u)
        d.line([(cx - 26 * e, 1400), (cx + 26 * e, 1400)],
               fill=WHITE + (int(190 * u),), width=4)
    v = clamp01((lt - 1.35) / 0.26)
    if v > 0:
        cx = 540 + 3.5 * SPREAD_DX
        circ(d, cx, SPREAD_TOP + 15, 46 * (0.7 + 0.3 * ease_out(v)), 1.0,
             width=5, alpha=int(230 * v), color=LIME)
    cross_mark(d, 540, 1480, 46, clamp01((lt - 1.70) / 0.34))
    p, o = blue_pop(lt - 0.05, 0.38)
    if p > 0:
        lay.alpha_composite(text_layer((W, H), [
            gtext(str(CARD_SUM_BAD), (540, 900), int(165 * (0.55 + 0.45 * p) * o),
                  blue=True),
        ]))


def g_barcode(lay, lt):
    """«но и в штрих-кодах»: штрих-код с его собственным рядом цифр."""
    d = ImageDraw.Draw(lay)
    rnd = random.Random(17)
    x = 250
    bars = []
    while x < 830:
        w = rnd.choice((4, 4, 7, 11))
        bars.append((x, w))
        x += w + rnd.choice((5, 7, 9))
    for k, (bx, bw) in enumerate(bars):
        u = clamp01((lt - 0.05 - k * 0.010) / 0.14)
        if u <= 0:
            continue
        h = 250 * ease_out(u)
        d.rectangle([bx, 1030 - h, bx + bw, 1030], fill=WHITE + (int(235 * u),))
    p = clamp01((lt - 0.55) / 0.42)
    if 0.01 < p < 0.99:
        sx = 250 + 580 * p
        d.line([(sx, 760), (sx, 1060)], fill=WHITE + (200,), width=5)
    items = []
    code = "4 601234 567893"
    for k, ch in enumerate(code):
        u = clamp01((lt - 0.45 - k * 0.012) / 0.16)
        if u > 0 and ch != " ":
            items.append(gtext(ch, (300 + k * 36, 1120), 40, op=ease_out(u)))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


GFX = {
    "scan": g_scan, "card16": g_card16, "check": g_check, "every2": g_every2,
    "gt9": g_gt9, "minus9": g_minus9, "sumrow": g_sumrow, "sum80": g_sum80,
    "div10": g_div10, "swap": g_swap, "nodiv": g_nodiv, "barcode": g_barcode,
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

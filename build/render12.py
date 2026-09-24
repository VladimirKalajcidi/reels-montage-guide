"""Сборка ролика 12 («формула Пика»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx12.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры — выше y600 (brand-kit §5).

Предметная часть ролика — узлы решётки, поэтому графика строится не «примерно
по клеточкам», а по реальным пересечениям сетки-подложки: NX0/NY0/STEP выведены
из style.grid_canvas. Кривизна сетки на этих планах уменьшена (wobble=0.35),
иначе линия уходит от номинального пересечения до ~14px и «вершина точно в узле»
перестаёт быть правдой.

Количество внутренних и граничных точек не вписано руками: lattice_interior и
lattice_boundary считают их из самого многоугольника, площадь — по формуле
площади Гаусса. Для основной фигуры выходит A=9, i=5, b=10, то есть 5 + 10/2 − 1 = 9.
"""
import math
import os
import subprocess
import sys
from math import gcd

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard12 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_pick.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/12/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GRID_CACHE = {}
GRID_WOBBLE = 0.35

# --- решётка: пересечения сетки-подложки в координатах холста ----------------
# style.grid_canvas рисует линию i по x = i*cell - cell/2 + (seed*13%40)/4,
# линию j по y = j*cell - cell/2 + (seed*7%40)/4, всё внутри карточки A.
STEP = 168
NX0 = CARD_A[0] + 1 * STEP - STEP * 0.5 + (7 * 13 % 40) / 4.0   # 191.75
NY0 = CARD_A[1] + 7 * STEP - STEP * 0.5 + (7 * 7 % 40) / 4.0    # 1332.25
NGX, NGY = 5, 4          # видимых узлов по горизонтали / вертикали


def npos(gx, gy):
    """Узел решётки (gx вправо, gy вверх) -> пиксели холста."""
    return (NX0 + STEP * gx, NY0 - STEP * gy)


# Левый столбец узлов стоит на x=191.75 — это первая линия сетки, попадающая внутрь
# карточки A, сдвинуть её нельзя. Значит любой маркер узла обязан уложиться в
# 191.75 − GX0 = 16.75px радиуса, иначе графика вылезает из зоны (проверено qa12).
R_MAX = 16


# --- многоугольники ролика (все вершины — в узлах) --------------------------
MAIN = [(0, 0), (4, 0), (4, 2), (2, 3), (0, 1)]
TRI = [(0, 0), (4, 0), (1, 3)]
STAR = [(2, 3), (3, 2), (4, 2), (3, 1), (4, 0), (2, 1), (0, 0), (1, 1), (0, 2), (1, 2)]
ANY = [(0, 0), (4, 0), (4, 1), (2, 1), (2, 3), (0, 3)]
# «кривой» многоугольник: ступеньки плюс диагональ, площадь получается дробной
WEIRD = [(0, 0), (4, 0), (4, 2), (3, 2), (3, 3), (1, 3), (1, 2), (2, 1), (0, 1)]


def poly_area(p):
    """Площадь по формуле Гаусса (шнуровка)."""
    s = 0
    for i in range(len(p)):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % len(p)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def lattice_boundary(p):
    """Целые точки на границе: на ребре (dx,dy) их ровно gcd(|dx|,|dy|)."""
    out = []
    for i in range(len(p)):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % len(p)]
        g = gcd(abs(x2 - x1), abs(y2 - y1))
        for k in range(g):
            out.append((x1 + (x2 - x1) * k // g, y1 + (y2 - y1) * k // g))
    return out


def _inside(x, y, p):
    """Чётно-нечётный тест. Точки границы сюда не подаются."""
    res = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            res = not res
        j = i
    return res


def lattice_interior(p):
    bnd = set(lattice_boundary(p))
    xs = [q[0] for q in p]
    ys = [q[1] for q in p]
    return [(x, y)
            for x in range(min(xs), max(xs) + 1)
            for y in range(min(ys), max(ys) + 1)
            if (x, y) not in bnd and _inside(x, y, p)]


MAIN_IN = lattice_interior(MAIN)      # 5
MAIN_BD = lattice_boundary(MAIN)      # 10
MAIN_AREA = poly_area(MAIN)           # 9.0


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
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_LANCZOS4)


UNSHARP = 0.35     # brand-kit §1: unsharp=5:5:0.30 — в тракте его не было вовсе


def unsharp(img, amount, radius=1.4):
    """Штатное подчёркивание из грейда brand-kit §1, реализованное на месте."""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    out = img.astype(np.float32) * (1 + amount) - blur.astype(np.float32) * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def source_card(frame, kind):
    """A-roll: hflip + кроп под карточку A + грейд (brand-kit §1).

    hflip оставлен штатным: на исходнике надпись на толстовке читается
    задом наперёд, значит дубль зеркальный (delivery-specs §6).
    """
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    if kind == "A2":
        w, h = 900, 1428
        x, y = (iw - w) // 2 + 8, 230
    else:
        w, h = 1000, 1585
        x, y = (iw - w) // 2 + 10, 125
    x = max(0, min(x, iw - w))
    y = max(0, min(y, ih - h))
    fr = fr[y:y + h, x:x + w]
    # LANCZOS4, а не INTER_AREA: кроп 1000x1585 ужимается до 870x1380, и area-фильтр
    # на таком мягком уменьшении срезает детализацию лица (замер дисперсии Лапласиана:
    # 18.7 у кропа -> 13.5 через INTER_AREA против 24.6 через LANCZOS4).
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_LANCZOS4)
    fr = unsharp(fr, UNSHARP)
    fr = cv2.convertScaleAbs(fr, alpha=1.06, beta=9)
    b, g, r = cv2.split(fr.astype(np.float32))
    r = np.clip(r * 1.04 + 3, 0, 255)
    b = np.clip(b * 0.96, 0, 255)
    fr = cv2.merge([b, g, r]).astype(np.uint8)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)
    fr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


STOCK_GAIN = 0.72      # притемнение вставки на 28% по экспозиции (brand-kit §6)


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


def gtext(text, xy, size=80, blue=False, serif=False, op=1.0, anchor="mm"):
    kind = "serif_it" if serif else "sans"
    return dict(text=text, font=font(kind, size), xy=xy, anchor=anchor,
                fill=BLUE if blue else WHITE,
                glow=BLUE_GLOW if blue else WHITE,
                glow_r=34 if blue else 15, glow_a=0.86 if blue else 0.48,
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
    """Быстрая версия style.text_layer: размывается только bbox строки."""
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


def poly(d, pts, p, color=WHITE, width=6, alpha=225, closed=True):
    """Контур с прогрессивной отрисовкой по периметру (p: 0..1)."""
    p = clamp01(p)
    if p <= 0:
        return
    pl = list(pts) + ([pts[0]] if closed else [])
    segs = [(pl[i], pl[i + 1]) for i in range(len(pl) - 1)]
    lens = [math.dist(a, b) for a, b in segs]
    tot = sum(lens) or 1.0
    left = tot * p
    for (a, b), L in zip(segs, lens):
        if left <= 0:
            break
        f = min(1.0, left / L) if L else 1.0
        d.line([a, (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)],
               fill=color + (alpha,), width=width)
        left -= L


def ppx(p):
    return [npos(*q) for q in p]


def dot(d, xy, r, alpha=220, fill=True, width=5):
    x, y = xy
    box = [x - r, y - r, x + r, y + r]
    if fill:
        d.ellipse(box, fill=WHITE + (int(alpha),))
    else:
        d.ellipse(box, outline=WHITE + (int(alpha),), width=width)


def grid_nodes(d, lt, t0=0.0, r=5, alpha=135, stag=0.010):
    """Мелкие узлы решётки — то, что в речи называется «узлами сетки»."""
    k = 0
    for gy in range(NGY):
        for gx in range(NGX):
            u = clamp01((lt - t0 - k * stag) / 0.18)
            k += 1
            if u <= 0:
                continue
            dot(d, npos(gx, gy), r * ease_out(u), alpha * u)


def area_fill(lay, pts, a):
    if a <= 0.004:
        return
    f = _layer()
    ImageDraw.Draw(f).polygon(pts, fill=WHITE + (int(54 * a),))
    lay.alpha_composite(f)


# --- графика планов -------------------------------------------------------

def g_shape(lay, lt):
    """«площадь любой фигуры на клетке»: фигура обводится и заливается."""
    area_fill(lay, ppx(MAIN), ease_out(clamp01((lt - 0.62) / 0.45)))
    d = ImageDraw.Draw(lay)
    grid_nodes(d, lt, 0.0)
    poly(d, ppx(MAIN), (lt - 0.12) / 0.55, width=8, alpha=240)


def g_nodes(lay, lt):
    """«вершины стоят точно в узлах сетки»: вершины садятся на пересечения."""
    d = ImageDraw.Draw(lay)
    grid_nodes(d, lt, 0.0, r=6, alpha=150)
    for k, p in enumerate(MAIN):
        u = clamp01((lt - 0.55 - k * 0.11) / 0.22)
        if u <= 0:
            continue
        e = ease_out(u)
        dot(d, npos(*p), R_MAX - 5 * e, 235)
    poly(d, ppx(MAIN), (lt - 1.18) / 0.95, width=8, alpha=235)


def _outline_shape(lay, lt, p, draw_dur=0.55, verts_at=0.10):
    d = ImageDraw.Draw(lay)
    grid_nodes(d, lt, 0.0, r=5, alpha=120, stag=0.006)
    for k, q in enumerate(p):
        u = clamp01((lt - verts_at - k * 0.03) / 0.16)
        if u > 0:
            dot(d, npos(*q), 11 * ease_out(u), 235)
    poly(d, ppx(p), (lt - verts_at - 0.06) / draw_dur, width=8, alpha=240)


def g_tri(lay, lt):
    _outline_shape(lay, lt, TRI, draw_dur=0.48)


def g_star(lay, lt):
    _outline_shape(lay, lt, STAR, draw_dur=0.80)


FORMULA_CARDS = [
    (300, [(-72, 62), (72, 62), (0, -58)], "a·h/2", 40),
    (540, [(-72, -46), (72, -46), (72, 46), (-72, 46)], "a·b", 40),
    (780, [(-76, 46), (76, 46), (40, -46), (-40, -46)], "(a+b)·h/2", 34),
]
FC_Y = 960          # центр фигурки
FC_TXT_Y = 1150     # строка формулы


def g_formulas(lay, lt):
    """«не нужно вспоминать отдельную формулу для каждой фигуры».

    Три школьные формулы площади, каждая привязана к своей фигуре, и все
    три перечёркиваются: дальше площадь считается одной формулой.
    """
    d = ImageDraw.Draw(lay)
    items = []
    for k, (cx, shape, txt, sz) in enumerate(FORMULA_CARDS):
        u = clamp01((lt - k * 0.12) / 0.30)
        if u <= 0:
            continue
        pts = [(cx + dx, FC_Y + dy) for dx, dy in shape]
        poly(d, pts, u, width=6, alpha=int(215 * u))
        v = clamp01((lt - 0.30 - k * 0.12) / 0.26)
        if v > 0:
            items.append(gtext(txt, (cx, FC_TXT_Y), sz, op=v))
    if items:
        lay.alpha_composite(text_layer((W, H), items))
    d = ImageDraw.Draw(lay)
    for k, (cx, _shape, txt, sz) in enumerate(FORMULA_CARDS):
        c = clamp01((lt - 1.15 - k * 0.13) / 0.28)
        if c <= 0:
            continue
        # зачёркивается только формула, фигура остаётся читаемой: крест на всю
        # карточку съедал и фигуру, и саму формулу — проверено на контакт-листе
        half = font("sans", sz).getlength(txt) / 2 + 12
        e = ease_out(c)
        d.line([(cx - half, FC_TXT_Y + 2), (cx - half + 2 * half * e, FC_TXT_Y + 2)],
               fill=WHITE + (235,), width=6)


def g_inner(lay, lt):
    """Первое число: узлы строго внутри фигуры — заполненные точки."""
    d = ImageDraw.Draw(lay)
    grid_nodes(d, lt, 0.0, r=5, alpha=120, stag=0.006)
    poly(d, ppx(MAIN), lt / 0.32, width=8, alpha=235)
    for k, p in enumerate(MAIN_IN):
        u = clamp01((lt - 0.38 - k * 0.13) / 0.20)
        if u > 0:
            dot(d, npos(*p), R_MAX * ease_out(u), 245)


def g_inner_n(lay, lt):
    """R5a: белое число внутренних точек. Слова несёт субтитр, цифру — графика."""
    p = pop(lt, 0.09)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext(str(len(MAIN_IN)), (GCX, 1120), int(190 * (0.6 + 0.4 * p))),
    ]))


def g_border(lay, lt):
    """Второе число: узлы на границе — кольца."""
    d = ImageDraw.Draw(lay)
    grid_nodes(d, lt, 0.0, r=5, alpha=120, stag=0.006)
    poly(d, ppx(MAIN), lt / 0.32, width=8, alpha=235)
    for k, p in enumerate(MAIN_BD):
        u = clamp01((lt - 0.34 - k * 0.10) / 0.20)
        if u > 0:
            dot(d, npos(*p), (R_MAX - 3) * ease_out(u), 245, fill=False, width=6)


def g_border_n(lay, lt):
    p = pop(lt, 0.09)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext(str(len(MAIN_BD)), (GCX, 1120), int(190 * (0.6 + 0.4 * p))),
    ]))


# --- строка формулы Пика ---------------------------------------------------
# Токены строки: ('t', текст) | ('dot', None) | ('ring', None).
# Место под все токены резервируется сразу, поэтому дописывание правой части
# не сдвигает уже показанную левую (та же логика, что в render11.formula_items).
F_TOKENS = [("t", "S"), ("t", " = "), ("dot", None), ("t", " + "),
            ("ring", None), ("t", "/2"), ("t", " − "), ("t", "1")]
F_Y = 1120
F_MAX_W = 690       # зона графики 175…905 минус запас под свечение


def _f_layout(base=96):
    """Кегль строки подбирается один раз под всю формулу целиком.

    Считать надо по полному набору токенов, а не по видимой части: иначе строка
    ужимается на f3, когда дописывается «− 1», и уже показанная левая часть прыгает.
    """
    size = base
    while size > 40:
        f = font("sans", size)
        r = 30 * size / base
        widths = [f.getlength(v) if k == "t" else 2 * r + 18 for k, v in F_TOKENS]
        if sum(widths) <= F_MAX_W:
            return size, r, widths
        size = int(size * 0.94)
    f = font("sans", size)
    r = 30 * size / base
    return size, r, [f.getlength(v) if k == "t" else 2 * r + 18 for k, v in F_TOKENS]


F_SIZE, F_R, F_W = _f_layout()


def formula_row(lay, alphas):
    """alphas — список прозрачностей по токенам F_TOKENS."""
    widths = F_W
    total = sum(widths)
    x = GCX - total / 2
    items = []
    circles = []
    for (kind, val), w, a in zip(F_TOKENS, widths, alphas):
        if a > 0.004:
            if kind == "t":
                items.append(dict(text=val, font=font("sans", F_SIZE), xy=(x, F_Y),
                                  anchor="lm", fill=WHITE, glow=WHITE,
                                  glow_r=15, glow_a=0.48, opacity=a))
            else:
                circles.append((kind, x + w / 2, a))
        x += w
    if items:
        lay.alpha_composite(text_layer((W, H), items))
    d = ImageDraw.Draw(lay)
    for kind, cx, a in circles:
        r = F_R * (0.7 + 0.3 * a)
        if kind == "dot":
            d.ellipse([cx - r, F_Y - r, cx + r, F_Y + r], fill=WHITE + (int(240 * a),))
        else:
            d.ellipse([cx - r, F_Y - r, cx + r, F_Y + r],
                      outline=WHITE + (int(240 * a),), width=7)


def g_f1(lay, lt):
    """«площадь равна числу внутренних точек» — формулу несёт графика."""
    a1 = ease_out(clamp01(lt / 0.36))
    a2 = ease_out(clamp01((lt - 0.45) / 0.40))
    formula_row(lay, [a1, a1, a2, 0, 0, 0, 0, 0])


def g_f2(lay, lt):
    """«плюс половина точек на границе»."""
    a = ease_out(clamp01((lt - 0.08) / 0.42))
    formula_row(lay, [1, 1, 1, a, a, a, 0, 0])


def g_f3(lay, lt):
    """«минус единица»."""
    a = ease_out(clamp01((lt - 0.04) / 0.36))
    formula_row(lay, [1, 1, 1, 1, 1, 1, a, a])


def g_result(lay, lt):
    """Подстановка чисел этой фигуры и синий итог (R5b)."""
    area = MAIN_AREA
    txt = f"{len(MAIN_IN)} + {len(MAIN_BD)}/2 − 1"
    a = ease_out(clamp01(lt / 0.34))
    lay.alpha_composite(text_layer((W, H), [gtext(txt, (GCX, 980), 84, op=a)]))
    p, o = blue_pop(lt - 0.72, 0.38)
    if p <= 0:
        return
    val = f"{area:.0f}" if float(area).is_integer() else f"{area:.1f}".replace(".", ",")
    lay.alpha_composite(text_layer((W, H), [
        gtext("=", (486, 1266), 84, op=p),
        gtext(val, (522, 1270), int(200 * (0.55 + 0.45 * p) * o), blue=True, anchor="lm"),
    ]))


def _lattice_plan(lay, lt, p, draw_dur):
    """Обвести фигуру и подсветить её узлы: внутренние точками, граничные кольцами."""
    d = ImageDraw.Draw(lay)
    grid_nodes(d, lt, 0.0, r=5, alpha=115, stag=0.006)
    poly(d, ppx(p), lt / draw_dur, width=8, alpha=240)
    bd = lattice_boundary(p)
    inn = lattice_interior(p)
    for k, q in enumerate(bd):
        u = clamp01((lt - draw_dur - 0.05 - k * 0.045) / 0.18)
        if u > 0:
            dot(d, npos(*q), (R_MAX - 3) * ease_out(u), 235, fill=False, width=5)
    for k, q in enumerate(inn):
        u = clamp01((lt - draw_dur - 0.05 - k * 0.10) / 0.18)
        if u > 0:
            dot(d, npos(*q), (R_MAX - 2) * ease_out(u), 245)


def g_any(lay, lt):
    _lattice_plan(lay, lt, ANY, 0.62)


def g_weird(lay, lt):
    _lattice_plan(lay, lt, WEIRD, 0.70)


GFX = {
    "shape": g_shape, "nodes": g_nodes, "tri": g_tri, "star": g_star,
    "formulas": g_formulas, "inner": g_inner, "inner_n": g_inner_n,
    "border": g_border, "border_n": g_border_n,
    "f1": g_f1, "f2": g_f2, "f3": g_f3, "result": g_result,
    "any": g_any, "weird": g_weird,
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
            GRID_CACHE[key] = grid_canvas(w, h, phase=(key / 10) * 0.7,
                                          wobble=GRID_WOBBLE)
        canvas.paste(GRID_CACHE[key], (x, y), rounded_mask(w, h, R_A))
    return canvas


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    # Кадры идут сырым RGB прямо в x264. Раньше здесь стоял cv2.VideoWriter с mp4v:
    # это MPEG-4 Part 2 на ~3 Мбит/с, и он съедал картинку ДО финального кодека —
    # замер по лицу: PSNR 26 dB к эталону, дисперсия Лапласиана 16.0 -> 11.1.
    # Финальный x264 потом просто переупаковывал уже испорченные кадры.
    wr = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "15", "-preset", "slow",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", VID],
        stdin=subprocess.PIPE)
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
        wr.stdin.write(canvas.convert("RGB").tobytes())
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}")

    wr.stdin.close()
    wr.wait()
    cap.release()
    if stock_cap is not None:
        stock_cap.release()
    print("готово:", VID)


if __name__ == "__main__":
    main()

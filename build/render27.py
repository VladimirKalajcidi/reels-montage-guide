"""Сборка ролика 27 («формула шнуровки»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx27.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп (ролик 16: подтяжка гаммы до
  ориентира brand-kit §6 дала вымытое лицо и рост шума);
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр;
* межстрочный шаг считается от кеглей соседних строк;
* кегль числа подбирается под ширину зоны, а не по шкале XL.

Один визуальный язык на весь ролик:
* точка на плоскости = вершина многоугольника;
* белая замкнутая ломаная = сам многоугольник;
* пара «x y» в таблице = координаты одной вершины, шестая строка —
  повторённая первая точка;
* диагональ ↘ = произведение, которое складывается в первую сумму;
* диагональ ↙ = произведение, которое складывается во вторую;
* синее число = ответ.
Слова несёт только субтитр, числа и геометрию — только графика.
Лайм #C3DB4E в этом ролике не используется: обе диагонали белые, но они
никогда не стоят в кадре одновременно, и путать их не с чем.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard27 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, STEP, NX, NY, OX, OY,
                          TCOL_X, TCOL_Y, TROW0, TDY, TSZ,
                          TPROD_R, TPROD_L, TSUM_Y, shot_at, slot_for)
from geom27 import POLY, POLY2, ROWS, D1, D2, S1, S2, DIFF, AREA
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/27/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GRID_CACHE = {}
GRID_WOBBLE = 0.30      # на планах с координатами кривизна сетки уменьшена:
                        # при полной линия уходит от узла до ~14px и «вершина
                        # ровно в узле» перестаёт быть правдой (ролик 12)

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (36 замеров по всему дублю: cx 347±6, линия глаз y 677±3, ширина лица 313±8).
# Кроп ставит глаза на 42% высоты карточки в обоих вариантах; высоту A1
# ограничивает сам кадр — при глазах на 42% выше 1039px кроп не берётся.
FRAMINGS = {
    "A1": dict(w=636, h=1009, x=29, y=253),
    "A2": dict(w=580, h=920, x=57, y=291),
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
    """A-roll: hflip + кроп под карточку A. Грейда нет — см. шапку модуля."""
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    f = FRAMINGS[kind]
    w, h = f["w"], f["h"]
    x = max(0, min(f["x"], iw - w))
    y = max(0, min(f["y"], ih - h))
    fr = fr[y:y + h, x:x + w]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_CUBIC)
    return cv_to_pil(fr)


STOCK_GAIN = 0.72      # притемнение вставки на 28% по экспозиции (brand-kit §6)


def stock_card(fr):
    fr = fit_bgr(fr, CARD_B[2], CARD_B[3])
    fr = np.clip(fr.astype(np.float32) * STOCK_GAIN, 0, 255)
    hsv = cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255)
    fr = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


def clamp01(v):
    return max(0.0, min(1.0, v))


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


def gtext(lay, text, xy, size, color=WHITE, gcol=None, ga=None, a=1.0, anchor="mm"):
    """Число или подпись графики со свечением. a — общая прозрачность."""
    if a <= 0.004:
        return
    blue = color == BLUE
    lay.alpha_composite(text_layer((W, H), [dict(
        text=text, font=font("sans", int(size)), xy=xy, anchor=anchor,
        fill=color, glow=gcol or (BLUE_GLOW if blue else color),
        glow_r=int(34 if blue else max(12, size * 0.18)),
        glow_a=(ga if ga is not None else (0.86 if blue else 0.48)),
        opacity=a)]))


# --- маска зоны графики ----------------------------------------------------
_ZMASK = None


def zone_mask():
    """Страховка от вылета графики за GFX_ZONE/GFX_X: за границами зоны
    не остаётся ни одного пикселя с ненулевой альфой."""
    global _ZMASK
    if _ZMASK is None:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([GX0 + 16, GY0 + 16, GX1 - 16, GY1 - 16], fill=255)
        _ZMASK = m.filter(ImageFilter.GaussianBlur(5))
    return _ZMASK


def _col(color, alpha):
    return tuple(color) + (max(0, min(255, int(round(255 * alpha)))),)


# --- координатная плоскость -------------------------------------------------

def px(x, y):
    return (OX + x * STEP, OY - y * STEP)


X_END = OX + (NX + 0.55) * STEP
Y_TOP = OY - (NY + 0.55) * STEP


def axes(d, p=1.0, a=1.0):
    """Оси с засечками. p — доля прорисовки (обе оси растут из начала)."""
    p, a = clamp01(p), clamp01(a)
    if p <= 0.004 or a <= 0.004:
        return
    col = _col(WHITE, 0.82 * a)
    xe = OX - 26 + (X_END - OX + 26) * p
    yt = OY + 26 - (OY + 26 - Y_TOP) * p
    d.line([(OX - 26, OY), (xe, OY)], fill=col, width=5)
    d.line([(OX, OY + 26), (OX, yt)], fill=col, width=5)
    if p > 0.985:
        d.polygon([(X_END + 18, OY), (X_END - 10, OY - 11), (X_END - 10, OY + 11)], fill=col)
        d.polygon([(OX, Y_TOP - 18), (OX - 11, Y_TOP + 10), (OX + 11, Y_TOP + 10)], fill=col)
    for i in range(1, NX + 1):
        if OX + i * STEP <= xe:
            d.line([(OX + i * STEP, OY - 9), (OX + i * STEP, OY + 9)], fill=col, width=4)
        if OY - i * STEP >= yt:
            d.line([(OX - 9, OY - i * STEP), (OX + 9, OY - i * STEP)], fill=col, width=4)


def axis_labels(lay, a=1.0):
    """Подписи осей. Единственные буквы на графике: в речи их нет, а без них
    координатная схема не читается (shot-recipes R10)."""
    gtext(lay, "x", (X_END - 10, OY + 48), 44, a=a)
    gtext(lay, "y", (OX + 46, Y_TOP + 10), 44, a=a)


def poly_outline(d, p, prog=1.0, a=1.0, width=7, color=WHITE):
    """Замкнутый контур, прорисованный на долю prog по периметру."""
    prog, a = clamp01(prog), clamp01(a)
    if prog <= 0.002 or a <= 0.004:
        return
    pts = [px(*q) for q in p] + [px(*p[0])]
    segs = list(zip(pts, pts[1:]))
    lens = [math.dist(s, e) for s, e in segs]
    left = sum(lens) * prog
    for (s, e), L in zip(segs, lens):
        if left <= 0:
            break
        f = min(1.0, left / L) if L else 1.0
        d.line([s, (s[0] + (e[0] - s[0]) * f, s[1] + (e[1] - s[1]) * f)],
               fill=_col(color, 0.94 * a), width=width)
        left -= L


def poly_fill(lay, p, a):
    if a <= 0.004:
        return
    f = _layer()
    ImageDraw.Draw(f).polygon([px(*q) for q in p], fill=_col(WHITE, 0.16 * a))
    lay.alpha_composite(f)


def verts(d, p, prog=None, a=1.0, r=12):
    for i, q in enumerate(p):
        u = 1.0 if prog is None else clamp01(prog[i])
        if u <= 0.004:
            continue
        c = px(*q)
        rr = r * (0.5 + 0.5 * ease_out(u))
        d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], fill=_col(WHITE, 0.96 * a * u))


def stagger(lt, n, t0=0.10, step=0.09, dur=0.30):
    return [clamp01((lt - t0 - i * step) / dur) for i in range(n)]


# Куда уходит подпись координат каждой вершины. Считано по месту: подпись
# ставится наружу от контура и целиком остаётся в зоне графики.
COORD_OFF = {0: (-72, 54), 1: (66, 54), 2: (20, 60), 3: (0, -62), 4: (-20, -60)}


def coord_labels(lay, p, prog, size=40):
    for i, q in enumerate(p):
        u = clamp01(prog[i])
        if u <= 0.004:
            continue
        cx, cy = px(*q)
        dx, dy = COORD_OFF[i]
        gtext(lay, f"{q[0]}; {q[1]}", (cx + dx, cy + dy), size, a=u)


# --- таблица координат ------------------------------------------------------

def table(lay, prog=None, a=1.0, dim_from=None):
    """Столбик координат. prog — прогресс появления по строкам."""
    for i in range(len(ROWS)):
        u = 1.0 if prog is None else clamp01(prog[i])
        if u <= 0.004:
            continue
        aa = a * u
        if dim_from is not None and i >= dim_from:
            aa *= 0.55
        yy = TROW0 + i * TDY
        gtext(lay, str(ROWS[i][0]), (TCOL_X, yy), TSZ, a=aa)
        gtext(lay, str(ROWS[i][1]), (TCOL_Y, yy), TSZ, a=aa)


def diagonals(d, kind, prog, a=1.0):
    """kind: 'r' — вниз-вправо (x_i · y_i+1), 'l' — вниз-влево (y_i · x_i+1)."""
    for i in range(len(POLY)):
        u = clamp01(prog[i])
        if u <= 0.004:
            continue
        y1, y2 = TROW0 + i * TDY, TROW0 + (i + 1) * TDY
        if kind == "r":
            s, e = (TCOL_X + 30, y1 + 20), (TCOL_Y - 30, y2 - 20)
        else:
            s, e = (TCOL_Y - 30, y1 + 20), (TCOL_X + 30, y2 - 20)
        d.line([s, (s[0] + (e[0] - s[0]) * u, s[1] + (e[1] - s[1]) * u)],
               fill=_col(WHITE, 0.66 * a), width=4)


def products(lay, vals, x, prog, a=1.0, size=52):
    for i, v in enumerate(vals):
        u = clamp01(prog[i])
        if u > 0.004:
            gtext(lay, str(v), (x, TROW0 + i * TDY + TDY / 2), size, a=a * u)


def sum_row(lay, d, value, x, u, a=1.0):
    """Черта и сумма под столбиком произведений (R5a: резкий поп)."""
    if u <= 0.004:
        return
    w = 74 * ease_out(clamp01(u * 1.6))
    d.line([(x - w, TSUM_Y - 66), (x + w, TSUM_Y - 66)], fill=_col(WHITE, 0.78 * a), width=4)
    gtext(lay, str(value), (x, TSUM_Y), int(104 * (0.6 + 0.4 * ease_out(u))), a=a)


# --- планы ------------------------------------------------------------------

def g_shape(lay, lt):
    """Хук: многоугольник на клетке — контур и заливка, без осей.

    Вершины и контур стартуют с нулевого кадра плана: пустая сетка сразу
    после реза читается как провал, и на контакт-листе это видно.
    """
    d = ImageDraw.Draw(lay)
    poly_fill(lay, POLY, ease_out(clamp01((lt - 0.45) / 0.45)))
    d = ImageDraw.Draw(lay)
    prog = stagger(lt, 5, t0=0.0, step=0.04, dur=0.14)
    verts(d, POLY, prog=prog)
    poly_outline(d, POLY, prog=clamp01((lt - 0.02) / 0.58))
    verts(d, POLY, prog=prog)


def g_hold(lay, lt):
    """«здесь координаты это»: плоскость с фигурой уже стоит, вершины
    подсвечиваются по кругу — план держит кадр, пока не пошли сами числа."""
    d = ImageDraw.Draw(lay)
    axes(d, a=0.75)
    poly_outline(d, POLY, a=0.8)
    verts(d, POLY, a=0.95)
    k = int(lt / 0.20) % len(POLY)
    c = px(*POLY[k])
    rr = 15 + 5 * math.sin(math.pi * ((lt / 0.20) % 1.0))
    d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], fill=_col(WHITE, 0.98))
    axis_labels(lay, a=0.75)


def g_plane(lay, lt):
    """«нужно ввести координатную плоскость»: оси выезжают под уже
    нарисованную фигуру, дальше проступают подписи осей."""
    d = ImageDraw.Draw(lay)
    axes(d, p=ease_out(clamp01(lt / 0.55)))
    poly_fill(lay, POLY, 0.75)
    d = ImageDraw.Draw(lay)
    poly_outline(d, POLY, a=0.85)
    verts(d, POLY)
    axis_labels(lay, a=ease_out(clamp01((lt - 0.60) / 0.35)))


# Каждая пара координат выходит ровно на своём слове (тайминги из words.json,
# отсчёт от начала плана `coords` на 11.209).
COORD_AT = [0.000, 0.663, 1.517, 2.505, 3.241]


def g_coords(lay, lt):
    """«здесь координаты это 1;1 5;1 6;4 3;6 и 1;4» — числа несёт только
    графика, субтитра на плане нет (brand-kit §5)."""
    d = ImageDraw.Draw(lay)
    axes(d, a=0.75)
    poly_outline(d, POLY, a=0.7)
    prog = [clamp01((lt - t) / 0.26) for t in COORD_AT]
    verts(d, POLY, a=0.9)
    for i, u in enumerate(prog):
        if u > 0.004:
            c = px(*POLY[i])
            rr = 15 + 5 * math.sin(math.pi * min(1.0, u))
            d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], fill=_col(WHITE, 0.98))
    axis_labels(lay, a=0.75)
    coord_labels(lay, POLY, prog)


def g_table(lay, lt):
    """«записываем координаты всех вершин друг под другом по порядку»:
    столбик из пяти пар."""
    table(lay, prog=stagger(lt, 5, t0=0.06, step=0.32, dur=0.24) + [0.0])


def g_repeat(lay, lt):
    """«а в конце ещё раз повторяем первую точку»: шестая строка —
    та же первая пара, она приезжает отдельно."""
    d = ImageDraw.Draw(lay)
    table(lay, prog=[1.0] * 5 + [clamp01((lt - 1.85) / 0.28)])
    u = ease_out(clamp01((lt - 1.10) / 0.65))
    if u > 0.004:
        # дуга от первой строки к шестой: видно, что это одна и та же точка
        x0, y0 = TCOL_X - 66, TROW0
        y1 = TROW0 + 5 * TDY
        steps = 40
        pts = [(x0 - 74 * math.sin(math.pi * (k / steps)),
                y0 + (y1 - y0) * (k / steps)) for k in range(steps + 1)]
        n = max(2, int(round(len(pts) * u)))
        d.line(pts[:n], fill=_col(WHITE, 0.72), width=5)
        if u > 0.985:                       # остриё приезжает в шестую строку
            d.polygon([(x0 + 2, y1 + 4), (x0 - 20, y1 - 14), (x0 - 2, y1 - 20)],
                      fill=_col(WHITE, 0.72))


def g_lace1(lay, lt):
    """«перемножаем координаты по диагонали в одну сторону и складываем»:
    диагонали ↘ и произведения справа. Суммы здесь ещё нет — она выходит
    отдельным планом, ровно на слове «получаем»."""
    d = ImageDraw.Draw(lay)
    table(lay)
    prog = stagger(lt, 5, t0=0.10, step=0.24, dur=0.20)
    diagonals(d, "r", prog)
    products(lay, D1, TPROD_R, [clamp01((p - 0.55) / 0.45) for p in prog])


def g_lace2(lay, lt):
    """«то же самое, только в другую сторону»: диагонали ↙, произведения слева."""
    d = ImageDraw.Draw(lay)
    table(lay)
    prog = stagger(lt, 5, t0=0.35, step=0.22, dur=0.18)
    diagonals(d, "l", prog)
    products(lay, D2, TPROD_L, [clamp01((p - 0.55) / 0.45) for p in prog])


def _sum_shot(lay, lt, kind, vals, x, total, num_at):
    """Столбик произведений сворачивается в сумму: таблица и диагонали
    приглушаются, число выходит резким попом (R5a)."""
    d = ImageDraw.Draw(lay)
    dim = 1.0 - 0.58 * ease_out(clamp01(lt / 0.30))
    table(lay, a=dim)
    diagonals(d, kind, [1.0] * 5, a=dim)
    products(lay, vals, x, [1.0] * 5)
    sum_row(lay, d, total, x, clamp01((lt - num_at) / 0.09))


def g_sum1(lay, lt):
    """«получаем 70» — число несёт графика, субтитра на плане нет."""
    _sum_shot(lay, lt, "r", D1, TPROD_R, S1, 1.148)


def g_sum2(lay, lt):
    """«получаем 33»."""
    _sum_shot(lay, lt, "l", D2, TPROD_L, S2, 0.561)


# Куда встаёт порядковый номер вершины на плане `order`.
ORDER_OFF = {0: (-58, 46), 1: (54, 46), 2: (18, 56), 3: (0, -56), 4: (-56, -18)}


def g_order(lay, lt):
    """«записать их по порядку вдоль границы»: точка обходит контур, и вершины
    нумеруются в том порядке, в каком их и записывают. Номера белые: синий
    в этом ролике зарезервирован за ответом."""
    d = ImageDraw.Draw(lay)
    axes(d, a=0.45)
    poly_outline(d, POLY, a=0.55, width=5)
    verts(d, POLY, a=0.75, r=10)
    pts = [px(*q) for q in POLY] + [px(*POLY[0])]
    lens = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    tot = sum(lens)
    walk = clamp01((lt - 0.14) / 1.75)
    left = tot * walk
    passed = 0
    for k, ((s, e), L) in enumerate(zip(zip(pts, pts[1:]), lens)):
        if left <= 0:
            break
        f = min(1.0, left / L)
        d.line([s, (s[0] + (e[0] - s[0]) * f, s[1] + (e[1] - s[1]) * f)],
               fill=_col(WHITE, 0.95), width=8)
        if f >= 1.0:
            passed = k + 1
        else:
            head = (s[0] + (e[0] - s[0]) * f, s[1] + (e[1] - s[1]) * f)
            d.ellipse([head[0] - 13, head[1] - 13, head[0] + 13, head[1] + 13],
                      fill=_col(WHITE, 1.0))
        left -= L
    for i in range(min(passed + 1, len(POLY))):
        cx, cy = px(*POLY[i])
        dx, dy = ORDER_OFF[i]
        gtext(lay, str(i + 1), (cx + dx, cy + dy), 52)


def _thread(offset_first):
    """Точки одной «шнуровки»: зигзаг между колонками x и y по строкам."""
    out = []
    for i in range(len(ROWS)):
        right = (i % 2 == 0) == offset_first
        out.append(((TCOL_Y - 30) if right else (TCOL_X + 30), TROW0 + i * TDY))
    return out


THREADS = (_thread(False), _thread(True))


def g_laces(lay, lt):
    """«диагональные линии между координатами напоминают шнурки»: обе
    диагонали разом, прорисованные как две сплошные нити. Это единственный
    план ролика, где они стоят в кадре одновременно — в этом и весь смысл."""
    d = ImageDraw.Draw(lay)
    table(lay, a=0.78)
    for k, pts in enumerate(THREADS):
        p = clamp01((lt - 0.12 - k * 0.22) / 1.55)
        if p <= 0.004:
            continue
        segs = list(zip(pts, pts[1:]))
        lens = [math.dist(a, b) for a, b in segs]
        left = sum(lens) * p
        for (s, e), L in zip(segs, lens):
            if left <= 0:
                break
            f = min(1.0, left / L)
            d.line([s, (s[0] + (e[0] - s[0]) * f, s[1] + (e[1] - s[1]) * f)],
                   fill=_col(WHITE, 0.72), width=5)
            left -= L


def g_diff(lay, lt):
    """«далее 70 вычитаем 33, берём модуль»: числа выходят на своих словах,
    порядок на экране тот же, что в речи. Модуль показан скобками, а не словом.
    Само 37 в речи не звучит — графика добавляет то, чего в словах нет."""
    gtext(lay, str(S1), (380, 1000), 140, a=ease_out(clamp01((lt - 0.291) / 0.24)))
    gtext(lay, "−", (540, 1000), 96, a=ease_out(clamp01((lt - 0.561) / 0.24)))
    gtext(lay, str(S2), (700, 1000), 140, a=ease_out(clamp01((lt - 1.412) / 0.24)))
    m = ease_out(clamp01((lt - 2.035) / 0.26))
    if m > 0.004:
        d = ImageDraw.Draw(lay)
        for bx in (256, 824):
            d.line([(bx, 1000 - 74), (bx, 1000 + 74)], fill=_col(WHITE, 0.8 * m), width=6)
    q = clamp01((lt - 2.15) / 0.09)
    if q > 0:
        gtext(lay, str(DIFF), (540, 1300), int(190 * (0.6 + 0.4 * ease_out(q))))


def g_result(lay, lt):
    """«делим пополам, получаем площадь 18,5»: деление и синий ответ (R5b).
    Синее число в ролике одно — это оно."""
    gtext(lay, f"{DIFF} : 2", (540, 960), 110, a=ease_out(clamp01(lt / 0.30)))
    p, o = blue_pop(lt - 1.028, 0.38)
    if p <= 0:
        return
    val = f"{AREA:.1f}".replace(".", ",")
    gtext(lay, val, (540, 1290), int(200 * (0.55 + 0.45 * p) * o), color=BLUE)


def g_any(lay, lt):
    """«работает для любого многоугольника»: невыпуклая фигура на той же
    плоскости — контур обходится, вершины садятся в узлы."""
    d = ImageDraw.Draw(lay)
    axes(d, a=0.55)
    poly_fill(lay, POLY2, ease_out(clamp01((lt - 0.62) / 0.45)) * 0.9)
    d = ImageDraw.Draw(lay)
    poly_outline(d, POLY2, prog=clamp01(lt / 0.85))
    verts(d, POLY2, prog=stagger(lt, len(POLY2), t0=0.02, step=0.04, dur=0.18), r=11)
    axis_labels(lay, a=0.55)


GFX = {
    "shape": g_shape, "plane": g_plane, "hold": g_hold,
    "coords": g_coords, "table": g_table,
    "repeat": g_repeat, "lace1": g_lace1, "sum1": g_sum1, "lace2": g_lace2,
    "sum2": g_sum2, "diff": g_diff, "result": g_result, "any": g_any,
    "order": g_order, "laces": g_laces,
}

# планы, у которых сетка-подложка привязана к координатной плоскости
ALIGNED = {"shape", "plane", "hold", "coords", "any", "order"}


def graphics_layer(kind, lt):
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    lay.putalpha(ImageChops.multiply(lay.split()[3], zone_mask()))
    return lay


# --- субтитры --------------------------------------------------------------

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
    """Блок = до 2 фраз в пределах одного плана, разрыв по паузе >0.30с.

    Вертикальный шаг внутри блока считается от кеглей соседних строк
    (brand-kit §5), поэтому строка 54pt не садится на соседнюю 42pt.
    """
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
        sizes = [max(sz for _, _, sz in CAPS[i][2]) for i in b]
        yoff, acc = [], 0.0
        for k in range(len(b)):
            if k:
                acc += max(60.0, (sizes[k - 1] + sizes[k]) * 0.70)
            yoff.append(acc)
        for pos, i in enumerate(b):
            info[i] = (bid, pos, len(b), end, yoff[pos])
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
        bid, pos, _, _, yoff = BLOCKS[idx]
        newer = sum(1 for j in active if j > idx and BLOCKS[j][0] == bid)
        dim = DIM[min(newer, 2)]
        bx, by = slot["x"], slot["y"] + yoff
        if slot["scatter"]:
            calm = bid % 3 == 0                      # каждый третий блок — спокойный
            bx += (0 if calm else [-46, 38, -22][min(pos, 2)])
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


# --- фон -------------------------------------------------------------------

def grid_for(kind, phase):
    """Сетка-подложка. На планах координатной плоскости её шаг равен шагу
    плоскости, а начало сдвинуто так, что узел (0;0) попадает точно
    на пересечение линий."""
    cw, ch = CARD_A[2], CARD_A[3]
    if kind in ALIGNED:
        big = grid_canvas(cw + 2 * STEP, ch + 2 * STEP, cell=STEP,
                          phase=phase, wobble=GRID_WOBBLE)
        cx = int(round(2 * STEP - STEP * 0.5 + (7 * 13 % 40) / 4.0 + CARD_A[0] - OX))
        cy = int(round(13 * STEP - STEP * 0.5 + (7 * 7 % 40) / 4.0 + CARD_A[1] - OY))
        return big.crop((cx, cy, cx + cw, cy + ch))
    return grid_canvas(cw, ch, phase=phase)


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
        key = (kind in ALIGNED, int(t * 10))
        if key not in GRID_CACHE:
            if len(GRID_CACHE) > 8:
                GRID_CACHE.clear()
            GRID_CACHE[key] = grid_for(kind, (key[1] / 10) * 0.7)
        canvas.paste(GRID_CACHE[key], (x, y), rounded_mask(w, h, R_A))
    return canvas


def compose(kind, prm, fr, t, t0, stock_reader=None):
    canvas = background(kind, prm, fr, t, stock_reader)
    gl = graphics_layer(kind, t - t0)
    if gl is not None:
        canvas.alpha_composite(gl)
    cl = caption_layer(t)
    if cl is not None:
        canvas.alpha_composite(cl)
    return canvas


class StockReader:
    """Сток конформится к 30 fps проекта по времени: клип идёт со своей
    скоростью, а не «кадр исходника на кадр проекта»."""

    def __init__(self):
        self.cap = None
        self.key = None
        self.last = None
        self.pos = -1.0

    def frame(self, clip, ss, rel):
        if self.key != clip:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(f"{STOCK_DIR}/stock_{clip}.mp4")
            self.key = clip
            self.pos = -1.0
            self.last = None
        want = ss + rel
        dur = (self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
               / max(1e-6, self.cap.get(cv2.CAP_PROP_FPS)))
        if dur > 0.2:
            want = want % dur
        if want < self.pos - 0.05:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, want * 1000)
            self.pos = want
            ok, fr = self.cap.read()
            if ok:
                self.last = fr
            return self.last
        guard = 0
        while self.pos < want and guard < 400:
            ok, fr = self.cap.read()
            if not ok:
                break
            self.last = fr
            self.pos = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            guard += 1
        return self.last

    def release(self):
        if self.cap is not None:
            self.cap.release()


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    nf = int(round(DUR * FPS))
    # кадры уходят в кодек одним проходом: сырой поток в ffmpeg
    enc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
        "-pix_fmt", "yuv420p", VID,
    ], stdin=subprocess.PIPE)
    stock = StockReader()

    last = None
    for fno in range(nf):
        ok, fr = cap.read()
        if ok:
            last = fr
        else:
            fr = last
        t = fno / FPS
        t0, t1, kind, prm = shot_at(t)
        canvas = compose(kind, prm, fr, t, t0,
                         lambda: stock.frame(prm["clip"], prm.get("ss", 0), t - t0))
        enc.stdin.write(pil_to_cv(canvas.convert("RGB")).tobytes())
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}", flush=True)

    enc.stdin.close()
    enc.wait()
    cap.release()
    stock.release()
    print("готово:", VID)


if __name__ == "__main__":
    main()

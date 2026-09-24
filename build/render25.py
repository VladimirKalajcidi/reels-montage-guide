"""Сборка ролика 25 («теорема Морли»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx25.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр;
* межстрочный шаг считается от кеглей соседних строк, а не берётся фиксированным.

Предметная графика — один визуальный язык на весь ролик, без единой надписи:
* белые отрезки = стороны исходного треугольника;
* тонкие линии от вершины до противоположной стороны = трисектрисы;
* дуга у вершины = сам угол, три дуги подряд = «на три равные части»
  (единственный лаймовый элемент ролика — у вершины C на плане `trisect`);
* точка = пересечение соседних трисектрис;
* синий залитый треугольник = результат, он же несёт своё 60°;
* засечки на сторонах = «стороны равны».
Слова несёт только субтитр, геометрию — только графика, поэтому продублировать
одно другим физически нечем.

Цифр на экране две, и обе поставлены так, что не спорят с речью:
`60°` добавляет то, чего в речи нет, а `1899` стоит на плане без субтитра.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard25 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, MAIN, ALT1, ALT2,
                          DEG_XY, DEG_SIZE, YEAR_XY, YEAR_SIZE,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/25/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (30 замеров по всему дублю: cx 392±9, линия глаз y 641±4, ширина лица 286±8).
# Кроп ставит глаза на 42.2% высоты карточки у A1 и на 42.3% у A2.
FRAMINGS = {
    "A1": dict(w=666, h=1056, x=54, y=196),
    "A2": dict(w=600, h=952, x=92, y=242),
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


STOCK_GAIN = 0.72      # притемнение вставки на 28% по экспозиции (brand-kit §6: 25-30%)


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


# --- маска зоны графики ----------------------------------------------------
_ZMASK = None


def zone_mask():
    """Страховка от вылета графики за GFX_ZONE/GFX_X.

    Врезка 16px при blur 5 гарантирует, что за границами зоны не остаётся
    ни одного пикселя с ненулевой альфой. Рабочее кадрирование делает сама
    геометрия: все три треугольника подобраны так, что помещаются целиком.
    """
    global _ZMASK
    if _ZMASK is None:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([GX0 + 16, GY0 + 16, GX1 - 16, GY1 - 16], fill=255)
        _ZMASK = m.filter(ImageFilter.GaussianBlur(5))
    return _ZMASK


# --- примитивы -------------------------------------------------------------

def _col(color, alpha):
    return tuple(color) + (max(0, min(255, int(round(255 * alpha)))),)


def seg(d, a, b, p=1.0, alpha=1.0, width=5, color=WHITE):
    """Отрезок, прорисованный на долю p от a к b."""
    p = clamp01(p)
    if p <= 0.002 or alpha <= 0.002:
        return
    d.line([a, (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)],
           fill=_col(color, alpha), width=width)


def dot(d, c, r=13, color=WHITE, alpha=1.0, p=1.0):
    if p <= 0.001 or alpha <= 0.002:
        return
    rr = r * (0.55 + 0.45 * p)
    d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], fill=_col(color, alpha))


def ring_dot(d, c, r=17, color=WHITE, alpha=1.0, width=4):
    if alpha <= 0.002:
        return
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r],
              outline=_col(color, alpha), width=width)


def arc_pts(V, u0, u1, r, n=64, lo=0.0, hi=1.0):
    """Дуга радиуса r при вершине V от направления u0 до u1 (кратчайшим поворотом).

    lo/hi — доля углового раствора: `arc_pts(..., lo=1/3, hi=2/3)` даёт ровно
    среднюю треть угла, по ней и рисуются «три равные части».
    """
    a0 = math.atan2(u0[1], u0[0])
    a1 = math.atan2(u1[1], u1[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    return [(V[0] + r * math.cos(a0 + da * (lo + (hi - lo) * i / n)),
             V[1] + r * math.sin(a0 + da * (lo + (hi - lo) * i / n)))
            for i in range(n + 1)]


def dashed_line(d, a, b, alpha=1.0, width=3, color=WHITE, dash=15, gap=11):
    if alpha <= 0.004:
        return
    L = math.dist(a, b)
    if L < 1:
        return
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    t, on = 0.0, True
    while t < L:
        e = min(t + (dash if on else gap), L)
        if on:
            d.line([(a[0] + ux * t, a[1] + uy * t), (a[0] + ux * e, a[1] + uy * e)],
                   fill=_col(color, alpha), width=width)
        t, on = e, not on


def draw_poly(d, pts, p=1.0, alpha=1.0, width=4, color=WHITE):
    p = clamp01(p)
    if p <= 0.002 or alpha <= 0.002:
        return
    k = max(2, int(round(len(pts) * p)))
    d.line(pts[:k], fill=_col(color, alpha), width=width, joint="curve")


def tick_mark(d, a, b, size=13, alpha=1.0, width=5, color=WHITE, p=1.0):
    """Засечка поперёк середины отрезка ab — «эта сторона равна тем»."""
    if p <= 0.01 or alpha <= 0.002:
        return
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    ang = math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 2
    dx, dy = math.cos(ang) * size * p, math.sin(ang) * size * p
    d.line([mx - dx, my - dy, mx + dx, my + dy], fill=_col(color, alpha), width=width)


def _glow_sub(sub, radius, color, strength):
    a = sub.split()[3]
    blur = a.filter(ImageFilter.GaussianBlur(radius))
    gl = Image.new("RGBA", sub.size, tuple(color) + (0,))
    gl.putalpha(blur.point(lambda v: int(v * strength)))
    out = Image.new("RGBA", sub.size, (0, 0, 0, 0))
    out.alpha_composite(gl)
    out.alpha_composite(sub)
    return out


def draw_number(lay, text, xy, size, color=WHITE, glow=None, glow_a=0.6,
                scale=1.0, reveal=1.0, alpha=1.0):
    """Крупное число со свечением (R5a/R5b).

    scale — масштаб появления, reveal — доля проявленных слева направо
    символов (набор по разрядам у синего числа).
    """
    if alpha <= 0.004 or scale <= 0.02 or reveal <= 0.001:
        return
    sz = max(10, int(round(size * scale)))
    f = font("sans", sz)
    gr = max(6, int(round(size * 0.20)))
    pad = gr * 3 + 8
    tw = int(math.ceil(f.getlength(text)))
    bb = f.getbbox(text)
    th = bb[3] - bb[1]
    sub = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(sub)
    d.text((pad, pad - bb[1]), text, font=f, fill=tuple(color) + (255,))
    if reveal < 0.999:
        nshow = max(1, int(math.ceil(len(text) * reveal)))
        cut = pad + f.getlength(text[:nshow]) + 2
        m = Image.new("L", sub.size, 0)
        ImageDraw.Draw(m).rectangle([0, 0, int(cut), sub.size[1]], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(1.5))
        sub.putalpha(Image.composite(sub.split()[3], Image.new("L", sub.size, 0), m))
    sub = _glow_sub(sub, gr, glow or color, glow_a)
    if alpha < 0.999:
        sub.putalpha(sub.split()[3].point(lambda v: int(v * alpha)))
    lay.alpha_composite(sub, (int(xy[0] - sub.size[0] / 2),
                              int(xy[1] - sub.size[1] / 2)))


# --- строительные блоки сцены ---------------------------------------------

def sides(d, sc, p=1.0, alpha=1.0, width=6, stroke=None):
    """Три стороны треугольника. stroke — список прогрессов по сторонам."""
    A, B, C = sc["tri"]
    pairs = ((A, B), (B, C), (C, A))
    for i, (a, b) in enumerate(pairs):
        pp = stroke[i] if stroke else p
        seg(d, a, b, p=pp, alpha=alpha, width=width)


def verts(d, sc, alpha=1.0, p=1.0, r=11):
    for v in sc["tri"]:
        dot(d, v, r=r, alpha=alpha, p=p)


def rays(d, sc, prog=None, alpha=0.55, width=3, color=WHITE):
    """Шесть трисектрис: от каждой вершины до противоположной стороны."""
    for i, key in enumerate("ABC"):
        V = sc["tri"]["ABC".index(key)]
        for j, foot in enumerate(sc["feet"][key]):
            k = i * 2 + j
            p = prog[k] if prog else 1.0
            a = alpha[k] if isinstance(alpha, (list, tuple)) else alpha
            seg(d, V, foot, p=p, alpha=a, width=width, color=color)


def morley_tri(d, sc, p=1.0, alpha=1.0, width=6, color=WHITE):
    MA, MB, MC = sc["M"]
    pts = (MA, MB, MC)
    for i in range(3):
        seg(d, pts[i], pts[(i + 1) % 3], p=p, alpha=alpha, width=width, color=color)


def morley_fill(lay, sc, alpha=1.0, glow=0.0):
    if alpha <= 0.004:
        return
    fill = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(fill).polygon(list(sc["M"]), fill=_col(BLUE, alpha))
    if glow > 0.004:
        fill = _glow_sub(fill, 26, BLUE_GLOW, glow)
    lay.alpha_composite(fill)


def stagger(lt, n, t0=0.10, step=0.09, dur=0.42):
    return [ease_out(clamp01((lt - t0 - i * step) / dur)) for i in range(n)]


# --- планы -----------------------------------------------------------------

def g_anyTri(lay, lt):
    """«возьмите любой треугольник, даже самый кривой»: три штриха и три вершины."""
    d = ImageDraw.Draw(lay)
    st = [ease_out(clamp01((lt - 0.02 - i * 0.40) / 0.40)) for i in range(3)]
    sides(d, MAIN, stroke=st, alpha=0.95, width=6)
    A, B, C = MAIN["tri"]
    for i, v in enumerate((A, B, C)):
        dot(d, v, r=12, alpha=0.95, p=pop(lt - i * 0.40, 0.16))


def g_teaser(lay, lt):
    """Хук: приём целиком за 2.3с — трисектрисы, три точки, синий результат."""
    d = ImageDraw.Draw(lay)
    sides(d, MAIN, alpha=0.55, width=5)
    verts(d, MAIN, alpha=0.55, r=9)
    rays(d, MAIN, prog=stagger(lt, 6, t0=0.16, step=0.07, dur=0.40), alpha=0.42, width=3)
    for i, m in enumerate(MAIN["M"]):
        dot(d, m, r=13, alpha=0.95, p=pop(lt - 1.02 - i * 0.06, 0.14))
    q = ease_out(clamp01((lt - 1.30) / 0.34))
    morley_tri(d, MAIN, p=1.0, alpha=0.95 * q, width=6, color=BLUE_GLOW)
    morley_fill(lay, MAIN, alpha=0.62 * q, glow=0.75 * q)


def g_trisect(lay, lt):
    """«каждый из трёх углов на три равные части»: сначала сами углы дугами,
    потом угол при C делится двумя линиями на три лаймовые дуги.

    Дуги долей нарисованы с угловым зазором между собой и вместе с двумя
    короткими линиями деления: без зазора три соседние дуги одного радиуса
    сливаются в одну и «три равные части» перестают читаться (проверено
    на первом прогоне контакт-листа).
    """
    d = ImageDraw.Draw(lay)
    A, B, C = MAIN["tri"]
    sides(d, MAIN, alpha=0.72, width=5)
    verts(d, MAIN, alpha=0.55, r=9)
    marks = ((A, B, C, 110), (B, C, A, 122), (C, A, B, 150))
    for i, (V, U, Wp, r) in enumerate(marks):
        p = ease_out(clamp01((lt - 0.10 - i * 0.15) / 0.36))
        a = 0.85 if i < 2 else 0.85 * (1.0 - ease_out(clamp01((lt - 0.84) / 0.26)))
        draw_poly(d, arc_pts(V, _u(V, U), _u(V, Wp), r), p=p, alpha=a, width=5)
    # деление угла при C — единственный лаймовый элемент ролика
    tC = MAIN["tris"]["C"]
    for k, u in enumerate(tC):
        p = ease_out(clamp01((lt - 0.78 - k * 0.08) / 0.26))
        seg(d, C, (C[0] + u[0] * 235, C[1] + u[1] * 235), p=p, alpha=0.90,
            width=5, color=LIME)
    g = 0.055                                   # угловой зазор между долями
    for k in range(3):
        p = ease_out(clamp01((lt - 0.92 - k * 0.09) / 0.28))
        draw_poly(d, arc_pts(C, _u(C, A), _u(C, B), 150,
                             lo=k / 3 + g, hi=(k + 1) / 3 - g),
                  p=p, alpha=0.95, width=7, color=LIME)


def _u(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    return (dx / L, dy / L)


def g_trisectAll(lay, lt):
    """«проведите по две линии с каждой вершины»: шесть трисектрис каскадом."""
    d = ImageDraw.Draw(lay)
    A, B, C = MAIN["tri"]
    sides(d, MAIN, alpha=0.68, width=5)
    verts(d, MAIN, alpha=0.55, r=9)
    for (V, U, Wp, r) in ((A, B, C, 110), (B, C, A, 122), (C, A, B, 132)):
        draw_poly(d, arc_pts(V, _u(V, U), _u(V, Wp), r), p=1.0, alpha=0.28, width=4)
    rays(d, MAIN, prog=stagger(lt, 6, t0=0.15, step=0.16, dur=0.45), alpha=0.88, width=4)


def g_adjacent(lay, lt):
    """«соседние линии соседних углов пересекаются попарно»:
    три пары загораются по очереди, каждая рождает свою точку."""
    d = ImageDraw.Draw(lay)
    sides(d, MAIN, alpha=0.55, width=5)
    rays(d, MAIN, alpha=0.20, width=3)
    for i, ((V1, u1), (V2, u2), M) in enumerate(MAIN["pairs"]):
        t0 = 0.02 + i * 0.72
        on = ease_out(clamp01((lt - t0) / 0.26))
        if on <= 0.004:
            continue
        live = lt < t0 + 0.72
        a = (0.35 + 0.65 * on) * (1.0 if live else 0.38)
        for (V, u) in ((V1, u1), (V2, u2)):
            far = _far_point(V, u, M)
            seg(d, V, far, p=1.0, alpha=a, width=6 if live else 4)
        pp = pop(lt - t0 - 0.34, 0.16)
        dot(d, M, r=14, alpha=0.95, p=pp)
        ring_dot(d, M, r=28, alpha=0.45 * pp * (1.0 if live else 0.5), width=4)


def _far_point(V, u, M):
    """Точка на луче V+t*u чуть за точкой M — трисектриса доводится до пересечения."""
    L = math.dist(V, M) * 1.06
    return (V[0] + u[0] * L, V[1] + u[1] * L)


def g_points(lay, lt):
    """«образуя три точки внутри треугольника»: остаются только три точки."""
    d = ImageDraw.Draw(lay)
    sides(d, MAIN, alpha=0.55, width=5)
    rays(d, MAIN, alpha=0.16, width=3)
    for i, m in enumerate(MAIN["M"]):
        pp = pop(lt - 0.12 - i * 0.10, 0.18)
        ph = 0.5 + 0.5 * math.sin(lt * 4.0 - i * 2.1)
        dot(d, m, r=15, alpha=1.0, p=pp)
        ring_dot(d, m, r=28 + 4 * ph, alpha=0.42 * pp, width=4)


def g_connect(lay, lt):
    """«соедините эти три точки»: три отрезка прорисовываются подряд."""
    d = ImageDraw.Draw(lay)
    sides(d, MAIN, alpha=0.42, width=5)
    rays(d, MAIN, alpha=0.12, width=3)
    MA, MB, MC = MAIN["M"]
    pts = (MA, MB, MC)
    for i in range(3):
        p = ease_out(clamp01((lt - 0.15 - i * 0.60) / 0.52))
        seg(d, pts[i], pts[(i + 1) % 3], p=p, alpha=0.95, width=6)
    for m in pts:
        dot(d, m, r=14, alpha=1.0)


def g_equilateral(lay, lt):
    """«идеально равносторонним»: контур обводится и заливается синим."""
    d = ImageDraw.Draw(lay)
    sides(d, MAIN, alpha=0.30, width=5)
    rays(d, MAIN, alpha=0.08, width=3)
    q = ease_out(clamp01((lt - 0.04) / 0.46))
    morley_tri(d, MAIN, p=q, alpha=0.95, width=6)
    f = ease_out(clamp01((lt - 0.99) / 0.36))
    if f > 0.004:
        morley_tri(d, MAIN, p=1.0, alpha=0.95 * f, width=6, color=BLUE_GLOW)
        morley_fill(lay, MAIN, alpha=0.66 * f, glow=0.80 * f)
    dd = ImageDraw.Draw(lay)
    for m in MAIN["M"]:
        dot(dd, m, r=13, color=BLUE_GLOW if f > 0.5 else WHITE, alpha=1.0)


def g_equalMarks(lay, lt):
    """«все стороны и все углы одинаковые»: засечки на сторонах, дуги в углах
    и синее 60° в пустом клине справа — число, которого в речи нет."""
    d = ImageDraw.Draw(lay)
    sides(d, MAIN, alpha=0.24, width=5)
    morley_fill(lay, MAIN, alpha=0.60, glow=0.55)
    d = ImageDraw.Draw(lay)
    MA, MB, MC = MAIN["M"]
    pts = (MA, MB, MC)
    morley_tri(d, MAIN, p=1.0, alpha=0.95, width=6, color=BLUE_GLOW)
    for i in range(3):
        a, b = pts[i], pts[(i + 1) % 3]
        tick_mark(d, a, b, size=15, alpha=0.95, width=5,
                  p=pop(lt - 0.05 - i * 0.07, 0.14))
        V = pts[i]
        u0 = _u(V, pts[(i + 1) % 3])
        u1 = _u(V, pts[(i + 2) % 3])
        draw_poly(d, arc_pts(V, u0, u1, 26), p=pop(lt - 0.20 - i * 0.07, 0.18),
                  alpha=0.90, width=4)
    # R5b: 0.55 -> 1.0 с оверщутом ~3%, разряды слева направо
    q = clamp01((lt - 0.45) / 0.38)
    if q > 0:
        # выноска от числа к углу треугольника: без неё 60° висит само по себе
        dashed_line(d, (662, 1004), (540, 1222), alpha=0.42 * q, width=3,
                    color=BLUE_GLOW)
        sc = 0.55 + 0.45 * ease_out(q) + 0.03 * math.sin(math.pi * min(1.0, q))
        draw_number(lay, "60°", DEG_XY, DEG_SIZE, color=BLUE, glow=BLUE_GLOW,
                    glow_a=0.85, scale=sc, reveal=clamp01((lt - 0.45) / 0.26))


def _alt_plan(lay, lt, sc):
    """«независимо от того, с какого треугольника начали»: другой исходный
    треугольник — тот же приём, тот же равносторонний результат."""
    d = ImageDraw.Draw(lay)
    # план длится ~1с и его смысл — «а теперь ДРУГОЙ треугольник»: фигура
    # приходит целиком с рампой прозрачности, а не дорисовывается. Штриховая
    # отрисовка на такой длине давала 2-3 кадра разомкнутой фигуры на склейке.
    sides(d, sc, alpha=0.80 * ease_out(clamp01(lt / 0.12)), width=6)
    verts(d, sc, alpha=0.60 * ease_out(clamp01(lt / 0.12)), r=9, p=1.0)
    rays(d, sc, prog=stagger(lt, 6, t0=0.10, step=0.045, dur=0.28), alpha=0.30, width=3)
    for i, m in enumerate(sc["M"]):
        dot(d, m, r=12, alpha=0.90, p=pop(lt - 0.36 - i * 0.04, 0.12))
    q = ease_out(clamp01((lt - 0.46) / 0.26))
    morley_tri(d, sc, p=1.0, alpha=0.95 * q, width=6, color=BLUE_GLOW)
    morley_fill(lay, sc, alpha=0.62 * q, glow=0.72 * q)
    d = ImageDraw.Draw(lay)
    MA, MB, MC = sc["M"]
    pts = (MA, MB, MC)
    for i in range(3):
        tick_mark(d, pts[i], pts[(i + 1) % 3], size=14, alpha=0.95, width=5,
                  p=pop(lt - 0.60 - i * 0.06, 0.14))


def g_anyStart1(lay, lt):
    _alt_plan(lay, lt, ALT1)


def g_anyStart2(lay, lt):
    _alt_plan(lay, lt, ALT2)


def g_year(lay, lt):
    """«доказана только в …»: год берёт на себя графика, субтитра на плане нет.

    Под числом прочерчивается ровная черта — акцент, а не шкала: шкала без
    подписей читалась бы как незакрытая диаграмма (первый прогон контакт-листа).
    """
    d = ImageDraw.Draw(lay)
    p = ease_out(clamp01((lt - 0.24) / 0.55))
    if p > 0.004:
        y, half = YEAR_XY[1] + 148, 214
        d.line([(YEAR_XY[0] - half * p, y), (YEAR_XY[0] + half * p, y)],
               fill=_col(WHITE, 0.50), width=4)
    # R5a: резкий поп за 3 кадра, 0.6 -> 1.0, без оверщута
    q = clamp01((lt - 0.08) / 0.10)
    if q > 0:
        draw_number(lay, "1899", YEAR_XY, YEAR_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


def g_recap(lay, lt):
    """«формулировка настолько простая»: весь приём одним быстрым проходом."""
    d = ImageDraw.Draw(lay)
    sides(d, MAIN, alpha=0.70, width=6,
          stroke=[ease_out(clamp01((lt - i * 0.025) / 0.07)) for i in range(3)])
    rays(d, MAIN, prog=stagger(lt, 6, t0=0.14, step=0.05, dur=0.26), alpha=0.30, width=3)
    for i, m in enumerate(MAIN["M"]):
        dot(d, m, r=12, alpha=0.90, p=pop(lt - 0.46 - i * 0.04, 0.12))
    q = ease_out(clamp01((lt - 0.56) / 0.24))
    morley_tri(d, MAIN, p=1.0, alpha=0.95 * q, width=6, color=BLUE_GLOW)
    morley_fill(lay, MAIN, alpha=0.62 * q, glow=0.72 * q)


GFX = {
    "anyTri": g_anyTri, "teaser": g_teaser, "trisect": g_trisect,
    "trisectAll": g_trisectAll, "adjacent": g_adjacent, "points": g_points,
    "connect": g_connect, "equilateral": g_equilateral,
    "equalMarks": g_equalMarks, "anyStart1": g_anyStart1,
    "anyStart2": g_anyStart2, "year": g_year, "recap": g_recap,
}


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
    """Сток конформится к 30 fps проекта по времени: клип идёт со своей скоростью,
    а не «кадр исходника на кадр проекта» (иначе 24/25 fps играют замедленно)."""

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
        if want < self.pos - 0.05:                    # петля клипа — перемотка в начало
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

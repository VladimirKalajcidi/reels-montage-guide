"""Сборка ролика 26 («задача коммивояжёра»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx26.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр;
* межстрочный шаг считается от кеглей соседних строк, а не берётся фиксированным;
* кегль числа подбирается под ширину зоны графики, а не по шкале XL.

Один визуальный язык на весь ролик, без единой надписи:
* точка = адрес, кольцо вокруг точки = старт (откуда выезжает курьер);
* белая замкнутая ломаная = один вариант маршрута;
* мелькающие ломаные = перебор вариантов;
* синяя ломаная = найденный маршрут (результат);
* лаймовая пара рёбер = самопересечение жадного маршрута, то самое
  «не идеальное» решение (единственный лаймовый элемент ролика);
* крупная цифра = сколько таких маршрутов бывает.
Слова несёт только субтитр, геометрию — только графика.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard26 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG,
                          ADDR5, ADDR6, ADDR10, ADDR20, ADDR50,
                          OPT6, LONG6, FLASH6, FLASH10, TOURS5,
                          NN10, GOOD10, CROSS10,
                          THUMB_COLS, THUMB_ROWS, THUMB_W, THUMB_H,
                          THUMB_X0, THUMB_Y0,
                          TEN_XY, TEN_SIZE, FACT_XY, FACT_SIZE,
                          FACT_FULL_XY, FACT_FULL_SIZE, HUGE_XY, HUGE_SIZE,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/26/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (33 замера по всему дублю: cx 402±7, линия глаз y 642±4, ширина лица 283±6).
# Кроп ставит глаза на 42% высоты карточки в обоих вариантах. Ширину A1
# ограничивает сам кадр: лицо стоит правее центра, симметрично шире 636px не взять.
FRAMINGS = {
    "A1": dict(w=636, h=1009, x=84, y=218),
    "A2": dict(w=580, h=920, x=112, y=256),
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
    геометрия: все точки лежат в поле PX/PY с запасом.
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


def draw_power(lay, base, exp, xy, size, color=BLUE, glow=BLUE_GLOW, glow_a=0.85,
               scale=1.0, reveal=1.0):
    """Степень: основание кеглем size, показатель кеглем 0.6*size с подъёмом.

    Отдельная функция нужна, потому что надстрочных цифр в SF Pro нет,
    а «10^62» текстом читается как опечатка.
    """
    if scale <= 0.02 or reveal <= 0.001:
        return
    sz = max(10, int(round(size * scale)))
    ez = max(8, int(round(size * 0.60 * scale)))
    fb, fe = font("sans", sz), font("sans", ez)
    gr = max(6, int(round(size * 0.20)))
    pad = gr * 3 + 8
    wb = fb.getlength(base)
    we = fe.getlength(exp)
    bb = fb.getbbox(base)
    th = bb[3] - bb[1]
    rise = int(round(th * 0.52))
    sub = Image.new("RGBA", (int(wb + we + 10) + 2 * pad, th + rise + 2 * pad),
                    (0, 0, 0, 0))
    d = ImageDraw.Draw(sub)
    d.text((pad, pad + rise - bb[1]), base, font=fb, fill=tuple(color) + (255,))
    eb = fe.getbbox(exp)
    d.text((pad + wb + 10, pad - eb[1]), exp, font=fe, fill=tuple(color) + (255,))
    if reveal < 0.999:
        cut = pad + (wb + we + 10) * reveal + 2
        m = Image.new("L", sub.size, 0)
        ImageDraw.Draw(m).rectangle([0, 0, int(cut), sub.size[1]], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(1.5))
        sub.putalpha(Image.composite(sub.split()[3], Image.new("L", sub.size, 0), m))
    sub = _glow_sub(sub, gr, glow, glow_a)
    lay.alpha_composite(sub, (int(xy[0] - sub.size[0] / 2),
                              int(xy[1] - sub.size[1] / 2)))


# --- адреса и маршруты -----------------------------------------------------

def squash(pts, y0, y1, x0=None, x1=None):
    """Вписать набор точек в прямоугольник, сохранив форму.

    Нужно на планах с числом: облако адресов уезжает в верхнюю часть зоны,
    цифра встаёт под ним, и они не спорят за одно место.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0 = 250 if x0 is None else x0
    x1 = 830 if x1 is None else x1
    sx = (x1 - x0) / max(1e-6, max(xs) - min(xs))
    sy = (y1 - y0) / max(1e-6, max(ys) - min(ys))
    s = min(sx, sy)
    cx = (x0 + x1) / 2 - s * (min(xs) + max(xs)) / 2
    cy = (y0 + y1) / 2 - s * (min(ys) + max(ys)) / 2
    return [(p[0] * s + cx, p[1] * s + cy) for p in pts]


def addrs(d, pts, alpha=1.0, r=12, prog=None, start=True, ring=22):
    """Адреса: точки; стартовая (нулевая) — с кольцом вокруг."""
    for i, p in enumerate(pts):
        pp = prog[i] if prog else 1.0
        dot(d, p, r=r, alpha=alpha, p=pp)
        if start and i == 0:
            ring_dot(d, p, r=ring, alpha=alpha * 0.85 * clamp01(pp * 1.4),
                     width=max(3, int(r * 0.28)))


def route(d, pts, order, p=1.0, alpha=1.0, width=5, color=WHITE):
    """Замкнутый маршрут, прорисованный на долю p (по рёбрам, а не по длине)."""
    p = clamp01(p)
    if p <= 0.002 or alpha <= 0.002:
        return
    n = len(order)
    done = p * n
    for i in range(n):
        k = clamp01(done - i)
        if k <= 0:
            break
        seg(d, pts[order[i]], pts[order[(i + 1) % n]], p=k, alpha=alpha,
            width=width, color=color)


def edge(d, pts, order, i, alpha=1.0, width=6, color=WHITE, p=1.0):
    n = len(order)
    seg(d, pts[order[i]], pts[order[(i + 1) % n]], p=p, alpha=alpha,
        width=width, color=color)


def stagger(lt, n, t0=0.10, step=0.09, dur=0.42):
    return [ease_out(clamp01((lt - t0 - i * step) / dur)) for i in range(n)]


# --- планы -----------------------------------------------------------------

def g_bruteFlash(lay, lt):
    """Хук «решить полным перебором»: адреса и мелькающие варианты маршрута."""
    d = ImageDraw.Draw(lay)
    addrs(d, ADDR6, alpha=0.95, prog=stagger(lt, 6, t0=0.02, step=0.05, dur=0.20))
    if lt < 0.34:
        return
    k = (lt - 0.34) / 0.105
    cur = int(k)
    if cur > 0:                                   # след предыдущего варианта
        route(d, ADDR6, FLASH6[(cur - 1) % len(FLASH6)], alpha=0.16, width=4)
    route(d, ADDR6, FLASH6[cur % len(FLASH6)], alpha=0.90, width=5)


def g_route1(lay, lt):
    """«выбрав при этом самый короткий маршрут»: сначала длинный вариант,
    потом поверх него — точный оптимум синим."""
    d = ImageDraw.Draw(lay)
    addrs(d, ADDR6, alpha=0.95)
    a_long = 0.90 if lt < 1.00 else 0.90 - 0.68 * ease_out(clamp01((lt - 1.00) / 0.30))
    route(d, ADDR6, LONG6, p=ease_out(clamp01(lt / 0.80)), alpha=a_long, width=5)
    q = ease_out(clamp01((lt - 1.02) / 0.62))
    route(d, ADDR6, OPT6, p=q, alpha=0.95, width=7, color=BLUE_GLOW)
    addrs(d, ADDR6, alpha=0.95)


def _thumb_pts(cell):
    """Пять адресов, вписанных в клетку миниатюры."""
    x, y = cell
    return squash(ADDR5, y + 24, y + THUMB_H - 24, x + 22, x + THUMB_W - 22)


_THUMB_CELLS = [(THUMB_X0 + c * THUMB_W, THUMB_Y0 + r * THUMB_H)
                for r in range(THUMB_ROWS) for c in range(THUMB_COLS)]
_THUMB_PTS = [_thumb_pts(c) for c in _THUMB_CELLS]


def g_five(lay, lt):
    """«для пяти адресов вариантов маршрута всего 12»: двенадцать миниатюр —
    все различные маршруты по пяти адресам. Цифры на экране нет: её роль
    играет сам пересчёт (brand-kit §5)."""
    d = ImageDraw.Draw(lay)
    for i, (cell, pts) in enumerate(zip(_THUMB_CELLS, _THUMB_PTS)):
        p = ease_out(clamp01((lt - 0.06 - i * 0.072) / 0.30))
        if p <= 0.004:
            continue
        route(d, pts, TOURS5[i], p=1.0, alpha=0.92 * p, width=3)
        addrs(d, pts, alpha=0.95 * p, r=6, ring=12, start=True)


def g_ten(lay, lt):
    """«для 10 адресов»: десять точек и перебор, который уже не уследить."""
    d = ImageDraw.Draw(lay)
    addrs(d, ADDR10, alpha=0.95, prog=stagger(lt, 10, t0=0.02, step=0.028, dur=0.18))
    if lt < 0.42:
        return
    k = int((lt - 0.42) / 0.085)
    route(d, ADDR10, FLASH10[(k - 1) % len(FLASH10)], alpha=0.14, width=3)
    route(d, ADDR10, FLASH10[k % len(FLASH10)], alpha=0.80, width=4)


_TEN_PTS = squash(ADDR10, 760, 1180)


def g_tenNum(lay, lt):
    """Точное число вариантов для 10 адресов: 9!/2 = 181 440.

    Облако адресов уходит вверх зоны и гаснет до 0.3 — цифра стоит под ним
    в чистом поле, а не поверх линий того же цвета.
    """
    d = ImageDraw.Draw(lay)
    a = 0.85 - 0.55 * ease_out(clamp01((lt - 0.55) / 0.40))
    k = int(lt / 0.085)
    route(d, _TEN_PTS, FLASH10[k % len(FLASH10)], alpha=0.24 * a / 0.85, width=3)
    addrs(d, _TEN_PTS, alpha=a, r=9, ring=17)
    # R5a: резкий поп за 3 кадра, 0.6 -> 1.0, без оверщута
    q = clamp01((lt - 0.916) / 0.10)
    if q > 0:
        y = TEN_XY[1] + 96
        w = 232 * ease_out(clamp01((lt - 0.98) / 0.45))
        d.line([(TEN_XY[0] - w, y), (TEN_XY[0] + w, y)], fill=_col(WHITE, 0.45), width=4)
        draw_number(lay, "181 440", TEN_XY, TEN_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


def g_twenty(lay, lt):
    """«для 20 адресов»: то же поле, но точек вдвое больше и варианты
    сливаются в мельтешение."""
    d = ImageDraw.Draw(lay)
    addrs(d, ADDR20, alpha=0.95, r=10, ring=19,
          prog=stagger(lt, 20, t0=0.02, step=0.016, dur=0.16))
    if lt < 0.40:
        return
    n = len(ADDR20)
    k = int((lt - 0.40) / 0.075)
    for j, al in ((k - 1, 0.10), (k, 0.62)):
        o = tuple([0] + [1 + ((i * 7 + j * 3) % (n - 1)) for i in range(n - 1)])
        seen, order = set(), []
        for i in o:                                # перестановка без повторов
            if i not in seen:
                seen.add(i)
                order.append(i)
        order += [i for i in range(n) if i not in seen]
        route(d, ADDR20, tuple(order), alpha=al, width=3)


_FIFTY_EDGES = [(i, j) for i in range(50) for j in range(i + 1, 50)]


def g_fifty(lay, lt):
    """«а для 50 адресов»: пятьдесят адресов и все связи между ними —
    клубок, в котором вариант уже не разглядеть."""
    d = ImageDraw.Draw(lay)
    addrs(d, ADDR50, alpha=0.95, r=8, ring=15,
          prog=stagger(lt, 50, t0=0.02, step=0.007, dur=0.14))
    p = ease_out(clamp01((lt - 0.42) / 1.10))
    if p <= 0.004:
        return
    shown = int(len(_FIFTY_EDGES) * p)
    for i, j in _FIFTY_EDGES[:shown]:
        d.line([ADDR50[i], ADDR50[j]], fill=_col(WHITE, 0.085), width=2)
    addrs(d, ADDR50, alpha=0.95, r=8, ring=15)


_FIFTY_TOP = squash(ADDR50, 760, 1130)


def g_huge(lay, lt):
    """Сколько маршрутов у 50 адресов: 49!/2 = 3 · 10^62.

    На экране только само число. Сравнения с числом атомов здесь нет
    намеренно: в речи оно есть, но оно неверно (3e62 против ~1e80),
    и экран его не повторяет — см. шапку storyboard26.
    """
    d = ImageDraw.Draw(lay)
    a = 0.55 - 0.33 * ease_out(clamp01((lt - 0.40) / 0.40))
    for i, j in _FIFTY_EDGES[::7]:
        d.line([_FIFTY_TOP[i], _FIFTY_TOP[j]], fill=_col(WHITE, 0.05 * a / 0.55), width=1)
    addrs(d, _FIFTY_TOP, alpha=a, r=6, ring=11)
    # R5b: 0.55 -> 1.0 с оверщутом ~3%, разряды слева направо
    q = clamp01((lt - 0.557) / 0.38)
    if q > 0:
        sc = 0.55 + 0.45 * ease_out(q) + 0.03 * math.sin(math.pi * min(1.0, q))
        draw_power(lay, "3 · 10", "62", HUGE_XY, HUGE_SIZE, scale=sc,
                   reveal=clamp01((lt - 0.557) / 0.30))


def g_factorial(lay, lt):
    """«это 19 факториал»: 19! синим и оно же полностью — белым.

    Полное число добавляет то, чего в речи нет, поэтому имеет право стоять
    на экране; субтитра на плане нет вовсе.
    """
    q = clamp01((lt - 0.124) / 0.38)
    if q > 0:
        sc = 0.55 + 0.45 * ease_out(q) + 0.03 * math.sin(math.pi * min(1.0, q))
        draw_number(lay, "19!", FACT_XY, FACT_SIZE, color=BLUE, glow=BLUE_GLOW,
                    glow_a=0.85, scale=sc, reveal=clamp01((lt - 0.124) / 0.26))
    r = clamp01((lt - 0.62) / 0.42)
    if r > 0:
        draw_number(lay, "121 645 100 408 832 000", FACT_FULL_XY, FACT_FULL_SIZE,
                    color=WHITE, glow=WHITE, glow_a=0.45, reveal=r)


def g_heuristic(lay, lt):
    """«хитрые приближённые алгоритмы находят не идеальные»: жадный маршрут
    строится ребро за ребром, и на нём остаётся самопересечение —
    те два ребра подсвечиваются лаймом (единственный лайм ролика)."""
    d = ImageDraw.Draw(lay)
    addrs(d, ADDR10, alpha=0.95)
    n = len(NN10)
    for i in range(n):
        p = ease_out(clamp01((lt - 0.10 - i * 0.135) / 0.20))
        edge(d, ADDR10, NN10, i, alpha=0.92, width=5, p=p)
    addrs(d, ADDR10, alpha=0.95)
    hi = ease_out(clamp01((lt - 1.62) / 0.30))
    if hi > 0.004:
        ph = 0.72 + 0.28 * math.sin((lt - 1.62) * 6.2)
        for i in set([k for pair in CROSS10 for k in pair]):
            edge(d, ADDR10, NN10, i, alpha=0.95 * hi * ph, width=7, color=LIME)


def g_goodRoute(lay, lt):
    """«а просто очень хороший маршрут»: то же поле после 2-opt —
    пересечение развязано, маршрут короче и он же результат (синий)."""
    d = ImageDraw.Draw(lay)
    addrs(d, ADDR10, alpha=0.95)
    fade = 1.0 - ease_out(clamp01((lt - 0.06) / 0.28))
    if fade > 0.004:
        route(d, ADDR10, NN10, alpha=0.55 * fade, width=4)
        for i in set([k for pair in CROSS10 for k in pair]):
            edge(d, ADDR10, NN10, i, alpha=0.85 * fade, width=6, color=LIME)
    q = ease_out(clamp01((lt - 0.24) / 0.62))
    route(d, ADDR10, GOOD10, p=q, alpha=0.95, width=7, color=BLUE_GLOW)
    addrs(d, ADDR10, alpha=0.95)


GFX = {
    "bruteFlash": g_bruteFlash, "route1": g_route1, "five": g_five,
    "ten": g_ten, "tenNum": g_tenNum, "twenty": g_twenty,
    "factorial": g_factorial, "fifty": g_fifty, "huge": g_huge,
    "heuristic": g_heuristic, "goodRoute": g_goodRoute,
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

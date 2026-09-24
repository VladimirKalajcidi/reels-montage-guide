"""Сборка ролика 23 («как телефон находит себя»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx23.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр;
* межстрочный шаг считается от кеглей соседних строк, а не берётся фиксированным.

Предметная графика — один визуальный язык на весь ролик, без единой надписи
и без единой цифры:
* иконка спутника = спутник; иконка телефона = приёмник;
* расходящиеся кольца = сигнал, который спутник передаёт во все стороны;
* циферблат = время (отправки — у спутника, с ошибкой — у телефона);
* отрезок с засечками = расстояние;
* окружность вокруг спутника = «вы где-то на этой сфере»;
* тонкий эллипс = окружность пересечения двух сфер, видимая почти с ребра;
* точка = кандидат на положение; синяя точка = единственный верный итог;
* лаймовый импульс = поправка часов (единственный лаймовый элемент ролика).
Поэтому субтитр и графика физически не могут написать одно и то же:
слова есть только в субтитре, геометрия — только в графике.

Каждая из трёх окружностей помещается в зону графики целиком: обрезанная
окружность читается как случайная дуга, а не как сфера. Конфигурация спутников
подобрана под это условие численно (см. шапку storyboard23). Маска зоны стоит
как страховка от вылета, а не как рабочий инструмент кадрирования.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard23 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, SAT, RAD, P_TRUE, P_ALT,
                          ELL_M, ELL_U, ELL_V, ELL_A, ELL_B, ell_point,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/23/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (30 замеров по всему дублю: cx 387±6, линия глаз y 634±4, ширина лица 286±7).
# Кроп ставит глаза на 42.0% высоты карточки у A1 и на 41.5% у A2.
FRAMINGS = {
    "A1": dict(w=666, h=1056, x=54, y=190),
    "A2": dict(w=600, h=952, x=87, y=239),
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
    """Мягкая подрезка графики по GFX_ZONE/GFX_X.

    Сфера вокруг спутника заведомо больше «доски»: показывается та её часть,
    где идёт пересечение. Край размывается, чтобы окружность не обрубалась
    линейкой. Врезка 16px при blur 5 гарантирует, что за границами зоны
    не остаётся ни одного пикселя с альфой выше нуля.
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


def circle_pts(c, r, n=280, start=-90.0):
    out = []
    for i in range(n + 1):
        a = math.radians(start + 360.0 * i / n)
        out.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    return out


def ellipse_pts(n=240, start=0.0):
    return [ell_point(start + 360.0 * i / n) for i in range(n + 1)]


def draw_poly(d, pts, p=1.0, alpha=1.0, width=4, color=WHITE):
    """Прогрессивная отрисовка ломаной: p — доля пройденного пути."""
    p = clamp01(p)
    if p <= 0.001 or alpha <= 0.002:
        return
    k = max(2, int(round(len(pts) * p)))
    d.line(pts[:k], fill=_col(color, alpha), width=width, joint="curve")


def draw_poly_dashed(d, pts, alpha=1.0, width=3, color=WHITE, dash=16, gap=12):
    if alpha <= 0.002:
        return
    acc, on, run = 0.0, True, [pts[0]]
    for a, b in zip(pts, pts[1:]):
        seg = math.dist(a, b)
        if seg <= 1e-6:
            continue
        acc += seg
        run.append(b)
        lim = dash if on else gap
        if acc >= lim:
            if on and len(run) > 1:
                d.line(run, fill=_col(color, alpha), width=width, joint="curve")
            run, acc, on = [b], 0.0, not on
    if on and len(run) > 1:
        d.line(run, fill=_col(color, alpha), width=width, joint="curve")


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


def satellite(d, c, s=1.25, alpha=1.0, p=1.0, color=WHITE):
    """Корпус + две панели + антенна. Полуширина ~51*s."""
    if p <= 0.001 or alpha <= 0.002:
        return
    a = alpha * p
    cx, cy = c
    bw, bh = 26 * s, 34 * s
    lw = max(2, int(round(5 * s)))
    d.rounded_rectangle([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                        radius=7 * s, outline=_col(color, a), width=lw)
    pw, ph = 30 * s, 18 * s
    for sgn in (-1, 1):
        x0 = cx + sgn * (bw / 2 + 9 * s)
        x1 = x0 + sgn * pw
        d.line([cx + sgn * bw / 2, cy, x0, cy], fill=_col(color, a), width=max(2, int(3 * s)))
        d.rounded_rectangle([min(x0, x1), cy - ph / 2, max(x0, x1), cy + ph / 2],
                            radius=3 * s, outline=_col(color, a), width=max(2, int(round(4 * s))))
    d.line([cx, cy - bh / 2, cx, cy - bh / 2 - 11 * s], fill=_col(color, a),
           width=max(2, int(3 * s)))


def phone(d, c, s=1.60, alpha=1.0, p=1.0, color=WHITE):
    if p <= 0.001 or alpha <= 0.002:
        return
    a = alpha * p
    cx, cy = c
    w, h = 46 * s, 82 * s
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                        radius=10 * s, outline=_col(color, a), width=max(2, int(5 * s)))
    d.line([cx - w / 2 + 9 * s, cy + h / 2 - 12 * s, cx + w / 2 - 9 * s, cy + h / 2 - 12 * s],
           fill=_col(color, a * 0.8), width=max(2, int(3 * s)))


def clock(d, c, r=80, turn=0.0, alpha=1.0, p=1.0, color=WHITE, width=5):
    """Циферблат без цифр: обод, четыре засечки, две стрелки."""
    if p <= 0.001 or alpha <= 0.002:
        return
    a = alpha * p
    cx, cy = c
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=_col(color, a), width=width)
    for k in range(4):
        ang = math.radians(90 * k)
        d.line([cx + math.cos(ang) * r * 0.80, cy + math.sin(ang) * r * 0.80,
                cx + math.cos(ang) * r * 0.94, cy + math.sin(ang) * r * 0.94],
               fill=_col(color, a * 0.85), width=max(2, width - 2))
    for ln, sp in ((0.50, 1.0), (0.72, 7.3)):
        ang = math.radians(-90 + 360 * turn * sp)
        d.line([cx, cy, cx + math.cos(ang) * r * ln, cy + math.sin(ang) * r * ln],
               fill=_col(color, a), width=max(2, width - 1))


def arrow(d, a, b, p=1.0, alpha=1.0, width=6, head=20, color=WHITE):
    if p <= 0.001 or alpha <= 0.002:
        return
    tip = (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)
    d.line([a, tip], fill=_col(color, alpha), width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (2.5, -2.5):
        d.line([tip, (tip[0] + head * math.cos(ang + s), tip[1] + head * math.sin(ang + s))],
               fill=_col(color, alpha), width=width)


def cross(d, c, size, p=1.0, alpha=1.0, width=10, color=WHITE):
    if p <= 0.001 or alpha <= 0.002:
        return
    s = size * (0.6 + 0.4 * p)
    d.line([c[0] - s, c[1] - s, c[0] + s, c[1] + s], fill=_col(color, alpha), width=width)
    d.line([c[0] - s, c[1] + s, c[0] + s, c[1] - s], fill=_col(color, alpha), width=width)


def tick_seg(d, a, b, p=1.0, alpha=1.0, width=5, tick=15, color=WHITE):
    """Отрезок с засечками на концах — «вот это расстояние»."""
    if p <= 0.001 or alpha <= 0.002:
        return
    tip = (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)
    d.line([a, tip], fill=_col(color, alpha), width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0]) + math.pi / 2
    dx, dy = math.cos(ang) * tick, math.sin(ang) * tick
    d.line([a[0] - dx, a[1] - dy, a[0] + dx, a[1] + dy], fill=_col(color, alpha), width=width)
    if p >= 0.99:
        d.line([b[0] - dx, b[1] - dy, b[0] + dx, b[1] + dy], fill=_col(color, alpha), width=width)


# --- планы -----------------------------------------------------------------

def _sphere(d, key, p=1.0, alpha=1.0, width=4, color=WHITE, start=-90.0):
    draw_poly(d, circle_pts(SAT[key], RAD[key], start=start), p=p, alpha=alpha,
              width=width, color=color)


def g_teaser(lay, lt):
    """Хук: три сферы сходятся в одну точку — обещание всего ролика."""
    d = ImageDraw.Draw(lay)
    for i, key in enumerate(("S1", "S2", "S3")):
        t0 = 0.16 + i * 0.22
        p = ease_out(clamp01((lt - t0) / 0.60))
        satellite(d, SAT[key], alpha=0.95, p=pop(lt - t0 + 0.10, 0.22))
        _sphere(d, key, p=p, alpha=0.80)
    dot(d, P_TRUE, r=16, alpha=1.0, p=pop(lt - 1.30, 0.14))
    ring_dot(d, P_TRUE, r=30, alpha=0.55 * pop(lt - 1.42, 0.20))


def g_notTrack(lay, lt):
    """«не отслеживают телефон напрямую»: связь спутник->телефон перечёркнута."""
    d = ImageDraw.Draw(lay)
    sat_c, ph_c = (540, 830), (540, 1420)
    satellite(d, sat_c, s=1.45, p=pop(lt, 0.22))
    phone(d, ph_c, s=1.75, p=pop(lt - 0.12, 0.22))
    arrow(d, (540, 900), (540, 1350), p=ease_out(clamp01((lt - 0.22) / 0.50)), alpha=0.80)
    cross(d, (540, 1125), 62, p=pop(lt - 0.88, 0.16), alpha=0.95, width=12)


def g_broadcast(lay, lt):
    """«постоянно передаёт»: кольца расходятся во все стороны, без адресата."""
    d = ImageDraw.Draw(lay)
    c = (540, 1080)
    satellite(d, c, s=1.45, p=pop(lt, 0.22))
    for i in range(3):
        t0 = 0.18 + i * 0.42
        q = clamp01((lt - t0) / 1.15)
        if q <= 0:
            continue
        r = 46 + 300 * ease_out(q)
        draw_poly(d, circle_pts(c, r), p=1.0, alpha=0.72 * (1 - q) ** 0.8, width=4)


def g_timestamp(lay, lt):
    """«сигнал с точным временем отправки»: кольцо + идущие часы у спутника."""
    d = ImageDraw.Draw(lay)
    c = (450, 1000)
    satellite(d, c, s=1.45, p=pop(lt, 0.22))
    for i in range(2):
        t0 = 0.10 + i * 0.55
        q = clamp01((lt - t0) / 1.25)
        if q <= 0:
            continue
        r = 46 + 224 * ease_out(q)
        draw_poly(d, circle_pts(c, r), p=1.0, alpha=0.66 * (1 - q) ** 0.8, width=4)
    clock(d, (782, 1330), r=96, turn=0.06 + lt * 0.16, p=pop(lt - 0.25, 0.26))


def g_delay(lay, lt):
    """«по крошечной задержке»: волна идёт от спутника к телефону."""
    d = ImageDraw.Draw(lay)
    sat_c, ph_c = (540, 800), (540, 1450)
    satellite(d, sat_c, s=1.45, p=pop(lt, 0.20))
    phone(d, ph_c, s=1.70, p=pop(lt - 0.08, 0.20))
    draw_poly_dashed(d, [(540, 870), (540, 1385)], alpha=0.30, width=3)
    q = clamp01((lt - 0.24) / 0.85)
    y = 870 + (1385 - 870) * q
    if q > 0:
        d.line([(540, 870), (540, y)], fill=_col(WHITE, 0.85), width=6)
        for k in range(3):                       # фронт волны
            w = 34 + k * 22
            d.arc([540 - w, y - w * 0.55, 540 + w, y + w * 0.55], 200, 340,
                  fill=_col(WHITE, 0.85 * (1 - k * 0.28)), width=5)
    if q >= 1.0:
        ring_dot(d, ph_c, r=54, alpha=0.55 * pop(lt - 1.12, 0.18), width=5)


def g_distance(lay, lt):
    """«вычисляет расстояние до спутника»: отрезок с засечками."""
    d = ImageDraw.Draw(lay)
    sat_c, ph_c = (540, 800), (540, 1450)
    satellite(d, sat_c, s=1.45, alpha=0.95)
    phone(d, ph_c, s=1.70, alpha=0.95)
    tick_seg(d, (540, 870), (540, 1385), p=ease_out(clamp01((lt - 0.14) / 0.55)),
             alpha=0.95, width=6, tick=26)


def g_sphereStart(lay, lt):
    """«одно расстояние до одного спутника»: радиус от спутника."""
    d = ImageDraw.Draw(lay)
    satellite(d, SAT["S1"], p=pop(lt, 0.20))
    tick_seg(d, SAT["S1"], P_TRUE, p=ease_out(clamp01((lt - 0.18) / 0.55)),
             alpha=0.92, width=5, tick=20)


def g_sphere1(lay, lt):
    """«вы где-то на поверхности сферы»: кандидаты по всей окружности."""
    d = ImageDraw.Draw(lay)
    satellite(d, SAT["S1"], alpha=0.95)
    tick_seg(d, SAT["S1"], P_TRUE, p=1.0, alpha=0.42, width=4, tick=18)
    _sphere(d, "S1", p=ease_out(clamp01((lt - 0.10) / 0.70)), alpha=0.92, width=5)
    if lt > 0.80:                                # кандидаты мерцают по кругу
        for i in range(12):
            a = math.radians(-90 + 30 * i)
            c = (SAT["S1"][0] + RAD["S1"] * math.cos(a),
                 SAT["S1"][1] + RAD["S1"] * math.sin(a))
            ph = 0.5 + 0.5 * math.sin((lt - 0.80) * 5.2 - i * 0.7)
            dot(d, c, r=10, alpha=0.30 + 0.55 * ph, p=pop(lt - 0.80 - i * 0.035, 0.16))


def _two_spheres(d, p2=1.0, alpha=0.92, width=5):
    satellite(d, SAT["S1"], alpha=0.95)
    _sphere(d, "S1", p=1.0, alpha=alpha, width=width)
    satellite(d, SAT["S2"], alpha=0.95, p=pop(p2 * 2, 0.35))
    _sphere(d, "S2", p=ease_out(clamp01(p2)), alpha=alpha, width=width)


def g_twoSats(lay, lt):
    """«2 спутника — это уже пересечение двух сфер»: приходит вторая сфера.

    Точек здесь ещё нет: двух сфер мало, кандидатом остаётся целая окружность,
    и она появится следующим планом. Поставить тут две точки — соврать.
    """
    d = ImageDraw.Draw(lay)
    _two_spheres(d, p2=clamp01((lt - 0.20) / 0.70))


def g_twoSpheres(lay, lt):
    """Обе сферы стоят, подсвечена линза их пересечения."""
    d = ImageDraw.Draw(lay)
    _two_spheres(d, p2=1.0)
    draw_poly(d, ellipse_pts(), p=1.0, alpha=0.30 * pop(lt - 0.10, 0.30), width=5)


def g_circlePos(lay, lt):
    """«что даёт окружность возможных положений»: эллипс — та самая окружность,
    видимая почти с ребра; её концы точно совпадают с точками пересечения."""
    d = ImageDraw.Draw(lay)
    _two_spheres(d, p2=1.0, alpha=0.34, width=4)
    draw_poly(d, ellipse_pts(), p=ease_out(clamp01((lt - 0.14) / 0.62)),
              alpha=0.98, width=6)
    if lt > 0.85:
        for i in range(8):
            pt = ell_point(360 * i / 8 + lt * 26)
            dot(d, pt, r=8, alpha=0.55, p=1.0)


def g_threeSats(lay, lt):
    """«три спутника»: третья сфера приходит и режет эту окружность."""
    d = ImageDraw.Draw(lay)
    _two_spheres(d, p2=1.0, alpha=0.30, width=4)
    draw_poly(d, ellipse_pts(), p=1.0, alpha=0.55, width=5)
    satellite(d, SAT["S3"], alpha=0.95, p=pop(lt, 0.24))
    _sphere(d, "S3", p=ease_out(clamp01((lt - 0.16) / 0.70)), alpha=0.92, width=5)


def g_twoPoints(lay, lt):
    """«до двух конкретных точек»: остаются ровно P_TRUE и Q."""
    d = ImageDraw.Draw(lay)
    _two_spheres(d, p2=1.0, alpha=0.24, width=4)
    satellite(d, SAT["S3"], alpha=0.70)
    _sphere(d, "S3", p=1.0, alpha=0.28, width=4)
    draw_poly(d, ellipse_pts(), p=1.0, alpha=0.34 * max(0.0, 1 - (lt - 0.30) / 0.55), width=5)
    for i, pt in enumerate((P_TRUE, P_ALT)):
        pp = pop(lt - 0.22 - i * 0.14, 0.18)
        dot(d, pt, r=15, alpha=1.0, p=pp)
        ring_dot(d, pt, r=31, alpha=0.50 * pp, width=5)


def _three_dim(d, a=0.20):
    for k in ("S1", "S2", "S3"):
        satellite(d, SAT[k], alpha=0.55)
        _sphere(d, k, p=1.0, alpha=a, width=4)


def g_fourthClock(lay, lt):
    """«устранить погрешность часов»: сдвинутые пунктирные сферы садятся
    на истинные; лаймовый импульс — сама поправка (единственный лайм ролика)."""
    d = ImageDraw.Draw(lay)
    _three_dim(d, a=0.22)
    satellite(d, SAT["S4"], alpha=0.95, p=pop(lt, 0.24))
    clock(d, (287, 1372), r=92, turn=0.04 + lt * 0.13, p=pop(lt - 0.18, 0.24),
          alpha=0.90, width=5)
    # ошибка часов = все радиусы смещены на одну и ту же величину
    fix = ease_out(clamp01((lt - 1.05) / 0.60))
    err = 42 * (1 - fix)
    if err > 1.0:
        for k in ("S1", "S2", "S3"):
            draw_poly_dashed(d, circle_pts(SAT[k], RAD[k] + err, n=200),
                             alpha=0.55, width=3)
    if 0.95 < lt < 2.05:                          # лаймовый импульс поправки
        q = clamp01((lt - 0.95) / 0.65)
        draw_poly(d, circle_pts(P_TRUE, 34 + 150 * ease_out(q)), p=1.0,
                  alpha=0.85 * (1 - q), width=6, color=LIME)
    for i, pt in enumerate((P_TRUE, P_ALT)):
        dot(d, pt, r=14, alpha=0.85, p=1.0)


def g_pickTwo(lay, lt):
    """«и выбрать из двух точек»: оба кандидата ещё равноправны."""
    d = ImageDraw.Draw(lay)
    _three_dim(d, a=0.20)
    satellite(d, SAT["S4"], alpha=0.55)
    for i, pt in enumerate((P_TRUE, P_ALT)):
        ph = 0.5 + 0.5 * math.sin(lt * 4.6 - i * math.pi)
        dot(d, pt, r=16, alpha=0.70 + 0.30 * ph, p=1.0)
        ring_dot(d, pt, r=32 + 5 * ph, alpha=0.42, width=5)


def g_pickOne(lay, lt):
    """«единственную верную»: Q гаснет, P_TRUE становится синим итогом."""
    d = ImageDraw.Draw(lay)
    _three_dim(d, a=0.18)
    satellite(d, SAT["S4"], alpha=0.50)
    fade = 1.0 - ease_out(clamp01((lt - 0.10) / 0.45))
    dot(d, P_ALT, r=16, alpha=0.75 * fade, p=1.0)
    if fade > 0.02:
        cross(d, P_ALT, 26, p=1.0, alpha=0.55 * fade, width=6)
    q = pop(lt - 0.50, 0.30)
    if q > 0:
        ring_dot(d, P_TRUE, r=34 + 74 * (1 - q), color=BLUE_GLOW, alpha=0.75 * q, width=6)
        dot(d, P_TRUE, r=15 + 8 * q, color=BLUE, alpha=1.0, p=1.0)
    else:
        dot(d, P_TRUE, r=15, alpha=0.85, p=1.0)


GFX = {
    "teaser": g_teaser, "notTrack": g_notTrack, "broadcast": g_broadcast,
    "timestamp": g_timestamp, "delay": g_delay, "distance": g_distance,
    "sphereStart": g_sphereStart, "sphere1": g_sphere1, "twoSats": g_twoSats,
    "twoSpheres": g_twoSpheres, "circlePos": g_circlePos, "threeSats": g_threeSats,
    "twoPoints": g_twoPoints, "fourthClock": g_fourthClock,
    "pickTwo": g_pickTwo, "pickOne": g_pickOne,
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
        while self.pos < want and guard < 200:
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

    for fno in range(nf):
        ok, fr = cap.read()
        if not ok:
            break
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

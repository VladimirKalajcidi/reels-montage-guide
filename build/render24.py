"""Сборка ролика 24 («регрессия к среднему»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx24.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр.

Предметная графика — один визуальный язык на весь ролик, без единой надписи:
* горизонтальная пунктирная линия поперёк кадра = среднее;
* точка = один человек / один результат, её высота над линией = отклонение;
* столбик от линии вверх = то же отклонение, разложенное на два сегмента:
  **сплошной** (нижний) = наследуемая часть, **пунктирный контур** (верхний) =
  случайная часть;
* стрелка к линии = движение к среднему; крест на стрелке = «не передаётся»;
* синий = результат следующего поколения / следующей попытки.
Поэтому субтитр и графика физически не могут написать одно и то же:
слова есть только в субтитре, величина — только в графике.

Межстрочный шаг считается по brand-kit §5 от кеглей соседних строк
(`max(60, (s1 + s2) * 0.70)`), а не берётся фиксированным.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard24 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, MEAN_Y, DEV_TALL, SEG_GENE, SEG_LUCK,
                          CHILD_LUCK, DEV_CHILD, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/24/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (35 замеров по всему дублю: cx 398±7, линия глаз y 640±6, ширина лица 282±5).
# Кроп ставит глаза на 42.0% высоты карточки у A1 и на 41.5% у A2.
# Ширину A1 ограничивает сам кадр: лицо стоит правее центра (cx 398 из 720),
# поэтому симметричный кроп шире 644px из исходника уже не вырезается.
FRAMINGS = {
    "A1": dict(w=644, h=1021, x=76, y=211),
    "A2": dict(w=580, h=920, x=108, y=258),
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

def arrow_head(d, tip, ang, alpha=215, width=7, head=22, color=WHITE):
    """Усы стрелки в точке tip. Направление приходит углом, а не второй точкой:
    на полностью выросшей стрелке точки совпадают и atan2(0, 0) отдал бы 0."""
    for s in (1, -1):
        d.line([tip, (tip[0] - head * math.cos(ang - s * 0.45),
                      tip[1] - head * math.sin(ang - s * 0.45))],
               fill=color + (int(alpha),), width=width)


def grow_arrow(d, a, b, p, alpha=215, width=7, head=22, color=WHITE):
    """Стрелка, вырастающая от a к b."""
    p = clamp01(p)
    if p <= 0.02:
        return
    tip = (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)
    d.line([a, tip], fill=color + (int(alpha),), width=width)
    if p > 0.75:
        d.line([tip, b], fill=color + (int(alpha),), width=width)
        arrow_head(d, b, math.atan2(b[1] - a[1], b[0] - a[0]),
                   alpha=int(alpha), width=width, head=head, color=color)


def cross(d, cx, cy, s, p=1.0, width=11, alpha=235, color=WHITE):
    p = clamp01(p)
    if p <= 0.02:
        return
    a = int(alpha * p)
    f1 = min(1.0, p / 0.5)
    d.line([(cx - s, cy - s), (cx - s + 2 * s * f1, cy - s + 2 * s * f1)],
           fill=color + (a,), width=width)
    if p > 0.5:
        f2 = (p - 0.5) / 0.5
        d.line([(cx + s, cy - s), (cx + s - 2 * s * f2, cy - s + 2 * s * f2)],
               fill=color + (a,), width=width)


def dashed_line(d, a, b, p=1.0, dash=18, gap=14, width=3, alpha=150, color=WHITE):
    if p <= 0:
        return
    ln = math.hypot(b[0] - a[0], b[1] - a[1])
    if ln < 1:
        return
    n = max(1, int(ln // (dash + gap)) + 1)
    lim = clamp01(p)
    for i in range(n):
        u0 = i * (dash + gap) / ln
        u1 = min(1.0, (i * (dash + gap) + dash) / ln)
        if u0 > lim:
            break
        u1 = min(u1, lim)
        d.line([(a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0),
                (a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)],
               fill=color + (int(alpha),), width=width)


# --- геометрия шкалы -------------------------------------------------------
# Всё внутри GFX_ZONE 700..1560 / GFX_X 175..905.
MEAN_X0, MEAN_X1 = 210, 870
BAR_W = 150
SOLO_X = 540                 # одиночный столбик
PAIR_L, PAIR_R = 380, 700    # родитель / ребёнок
DOT_R = 22
CLOUD_R = 16                 # точки облака стоят через 44px, крупнее их не сделать

# облако результатов: детерминированный список (x, отклонение от среднего)
CLOUD = (
    (240, 4), (284, 135), (328, 346), (372, -105), (416, 147),
    (460, -209), (504, -304), (548, -91), (592, 328), (636, 93),
    (680, 315), (724, -143), (768, -196), (812, -278), (856, 98),
)
EXTREMES = tuple(p for p in CLOUD if abs(p[1]) >= 300)   # 4 точки, их и стягиваем
PULL = 0.35                  # куда приходит следующее поколение: доля от отклонения

# «случайные колебания» одного и того же человека — тоже фиксированный ряд
WOBBLE = ((250, 158), (335, -116), (420, 100), (505, -147),
          (590, 126), (675, -84), (760, 137), (845, -110))


def mean_line(d, p=1.0, alpha=195):
    dashed_line(d, (MEAN_X0, MEAN_Y), (MEAN_X1, MEAN_Y), p=p,
                dash=24, gap=16, width=5, alpha=alpha)


def dot(d, x, dev, p=1.0, color=WHITE, alpha=1.0, r=DOT_R):
    p = clamp01(p)
    if p <= 0.02 or alpha <= 0.01:
        return
    rr = r * (0.55 + 0.45 * ease_out(p))
    y = MEAN_Y - dev
    d.ellipse([x - rr, y - rr, x + rr, y + rr],
              fill=color + (int(240 * alpha * min(1.0, p * 1.6)),))


def seg_solid(d, cx, dev_bottom, h, p=1.0, alpha=1.0, color=WHITE, w=BAR_W,
              flat_top=False):
    """Наследуемая часть: сплошной столбик, растёт снизу вверх.

    flat_top — верх без скругления: когда сверху встаёт пунктирный сегмент,
    скруглённые углы читаются как отдельная «пилюля», а не как низ столбика."""
    p = clamp01(p)
    if p <= 0.02 or alpha <= 0.01 or h <= 0:
        return
    y0 = MEAN_Y - dev_bottom
    hh = h * ease_out(p)
    if hh < 3:
        return
    fill = color + (int(228 * alpha),)
    r = min(20, w / 2)
    d.rounded_rectangle([cx - w / 2, y0 - hh, cx + w / 2, y0], radius=r, fill=fill)
    if flat_top and hh > r:
        d.rectangle([cx - w / 2, y0 - hh, cx + w / 2, y0 - hh + r], fill=fill)


def seg_dashed(d, cx, dev_bottom, h, p=1.0, alpha=1.0, color=WHITE, w=BAR_W):
    """Случайная часть: пунктирный контур той же ширины, растёт снизу вверх."""
    p = clamp01(p)
    if p <= 0.02 or alpha <= 0.01 or h <= 0:
        return
    y0 = MEAN_Y - dev_bottom
    hh = h * ease_out(p)
    if hh < 4:
        return
    y1 = y0 - hh
    a = int(210 * alpha)
    d.rectangle([cx - w / 2, y1, cx + w / 2, y0], fill=WHITE + (int(46 * alpha),))
    for x in (cx - w / 2, cx + w / 2):
        dashed_line(d, (x, y0), (x, y1), dash=15, gap=11, width=5, alpha=a)
    if p > 0.8:
        dashed_line(d, (cx - w / 2, y1), (cx + w / 2, y1), dash=13, gap=10,
                    width=5, alpha=a)


def bar(d, cx, gene=SEG_GENE, luck=SEG_LUCK, p_gene=1.0, p_luck=1.0,
        a_gene=1.0, a_luck=1.0, cap=None, p_cap=1.0, w=BAR_W):
    """Столбик целиком: сплошной сегмент + пунктирный над ним + точка-вершина."""
    seg_solid(d, cx, 0, gene, p=p_gene, alpha=a_gene, w=w, flat_top=p_luck > 0.05)
    seg_dashed(d, cx, gene, luck, p=p_luck, alpha=a_luck, w=w)
    if cap is not None:
        dot(d, cx, gene + luck, p=p_cap, color=cap)


def notch(d, cx, alpha=1.0, w=BAR_W):
    """Засечка на линии среднего — «здесь начинается столбик»."""
    d.line([(cx - w / 2 - 10, MEAN_Y), (cx + w / 2 + 10, MEAN_Y)],
           fill=WHITE + (int(150 * alpha),), width=5)


def brace(d, x, y_top, y_bot, p=1.0, alpha=1.0, arm=22):
    """Скобка слева от столбика: отмечает выделенный кусок отклонения."""
    p = clamp01(p)
    if p <= 0.02:
        return
    a = int(215 * alpha)
    yy = y_bot + (y_top - y_bot) * ease_out(p)
    d.line([(x, y_bot), (x, yy)], fill=WHITE + (a,), width=5)
    d.line([(x, y_bot), (x + arm, y_bot)], fill=WHITE + (a,), width=5)
    if p > 0.88:
        d.line([(x, y_top), (x + arm, y_top)], fill=WHITE + (a,), width=5)


# --- планы -----------------------------------------------------------------

def g_tallPair(lay, lt):
    """Хук: родитель высоко над средним, ребёнок заметно ниже, но ещё выше линии."""
    d = ImageDraw.Draw(lay)
    mean_line(d, p=clamp01(lt / 0.22))
    dot(d, PAIR_L, DEV_TALL, p=clamp01((lt - 0.20) / 0.22))
    dot(d, PAIR_R, DEV_CHILD, p=clamp01((lt - 0.42) / 0.22))
    dashed_line(d, (PAIR_L + 40, MEAN_Y - DEV_TALL), (846, MEAN_Y - DEV_TALL),
                p=clamp01((lt - 0.58) / 0.34), dash=17, gap=13, width=3, alpha=140)


def _cloud(d, lt, t0=0.20, stagger=0.052, dim=1.0, dim_extreme=1.0):
    for i, (x, dev) in enumerate(CLOUD):
        a = dim_extreme if (x, dev) in EXTREMES else dim
        dot(d, x, dev, p=clamp01((lt - t0 - i * stagger) / 0.14), alpha=a, r=CLOUD_R)


def g_statCloud(lay, lt):
    """«а в чистой статистике» — результаты рассыпаются вокруг линии среднего."""
    d = ImageDraw.Draw(lay)
    mean_line(d, p=clamp01(lt / 0.25))
    _cloud(d, lt)


def g_spread(lay, lt):
    """«статистический эффект» — то же облако, но крайние точки отмечены:
    видно, как далеко они стоят от линии."""
    d = ImageDraw.Draw(lay)
    mean_line(d, p=clamp01(lt / 0.15))
    _cloud(d, lt, t0=0.10, stagger=0.02, dim=0.55)
    for i, (x, dev) in enumerate(EXTREMES):
        p = ease_out(clamp01((lt - 0.72 - i * 0.08) / 0.40))
        if p <= 0.02:
            continue
        dashed_line(d, (x, MEAN_Y), (x, MEAN_Y - dev + (CLOUD_R + 8) * (1 if dev > 0 else -1)),
                    p=p, dash=14, gap=11, width=3, alpha=150)


def g_pullMean(lay, lt):
    """«регрессией к среднему» — крайние точки стягиваются к линии,
    следующее поколение приходит ближе к ней. Синий = результат."""
    d = ImageDraw.Draw(lay)
    mean_line(d)
    _cloud(d, lt, t0=0.0, stagger=0.0, dim=0.40)
    for i, (x, dev) in enumerate(EXTREMES):
        s = 1 if dev > 0 else -1
        tgt = dev * PULL
        a0 = (x, MEAN_Y - dev + s * (CLOUD_R + 8))
        a1 = (x, MEAN_Y - tgt - s * (CLOUD_R + 12))
        grow_arrow(d, a0, a1, ease_out(clamp01((lt - 0.30 - i * 0.10) / 0.45)),
                   width=6, head=19)
        dot(d, x, tgt, p=clamp01((lt - 0.82 - i * 0.10) / 0.16), color=BLUE, r=CLOUD_R)


def g_sumBar(lay, lt):
    """«складывается из наследственности и случайных факторов» — сначала
    сплошной сегмент, потом пунктирный поверх него."""
    d = ImageDraw.Draw(lay)
    mean_line(d, p=clamp01(lt / 0.25))
    notch(d, SOLO_X)
    bar(d, SOLO_X,
        p_gene=clamp01((lt - 0.15) / 0.65),
        p_luck=clamp01((lt - 1.36) / 0.59),
        cap=WHITE, p_cap=clamp01((lt - 2.05) / 0.16))


def g_tallBar(lay, lt):
    """«родитель оказался особенно высоким» — весь столбик и уровень его вершины."""
    d = ImageDraw.Draw(lay)
    mean_line(d)
    notch(d, SOLO_X)
    bar(d, SOLO_X,
        p_gene=clamp01((lt - 0.10) / 0.60),
        p_luck=clamp01((lt - 0.55) / 0.55),
        cap=WHITE, p_cap=clamp01((lt - 1.12) / 0.16))
    dashed_line(d, (250, MEAN_Y - DEV_TALL), (SOLO_X - BAR_W / 2 - 16, MEAN_Y - DEV_TALL),
                p=clamp01((lt - 1.30) / 0.40), dash=17, gap=13, width=3, alpha=145)
    dashed_line(d, (SOLO_X + BAR_W / 2 + 16, MEAN_Y - DEV_TALL), (846, MEAN_Y - DEV_TALL),
                p=clamp01((lt - 1.30) / 0.40), dash=17, gap=13, width=3, alpha=145)


def g_genesLit(lay, lt):
    """«сыграли роль и гены» — подсвечен нижний, наследуемый сегмент."""
    d = ImageDraw.Draw(lay)
    mean_line(d)
    notch(d, SOLO_X)
    bar(d, SOLO_X, a_luck=0.34, cap=WHITE)
    brace(d, SOLO_X - BAR_W / 2 - 26, MEAN_Y - SEG_GENE, MEAN_Y,
          p=ease_out(clamp01((lt - 0.20) / 0.50)))


def g_randomLit(lay, lt):
    """«и удачное стечение случайных факторов одновременно» — сначала подсвечен
    верхний, случайный сегмент, потом скобка раскрывается на весь столбик."""
    d = ImageDraw.Draw(lay)
    mean_line(d)
    notch(d, SOLO_X)
    both = clamp01((lt - 1.15) / 0.45)
    bar(d, SOLO_X, a_gene=0.30 + 0.70 * both, cap=WHITE)
    y_bot = (MEAN_Y - SEG_GENE) + SEG_GENE * ease_out(both)
    brace(d, SOLO_X - BAR_W / 2 - 26, MEAN_Y - DEV_TALL, y_bot,
          p=ease_out(clamp01((lt - 0.20) / 0.50)))


def _parent_full(d, cap=True):
    notch(d, PAIR_L)
    bar(d, PAIR_L, cap=WHITE if cap else None)


def g_inherit(lay, lt):
    """«ребёнку не передаются напрямую» — сплошной сегмент переходит к ребёнку,
    пунктирный не переходит: его стрелка перечёркнута."""
    d = ImageDraw.Draw(lay)
    mean_line(d)
    _parent_full(d)
    notch(d, PAIR_R)
    x0 = PAIR_L + BAR_W / 2 + 16
    x1 = PAIR_R - BAR_W / 2 - 16
    grow_arrow(d, (x0, MEAN_Y - SEG_GENE / 2), (x1, MEAN_Y - SEG_GENE / 2),
               ease_out(clamp01((lt - 0.25) / 0.50)), width=6, head=19)
    seg_solid(d, PAIR_R, 0, SEG_GENE, p=clamp01((lt - 0.55) / 0.50))
    y2 = MEAN_Y - SEG_GENE - SEG_LUCK / 2
    grow_arrow(d, (x0, y2), (x1, y2), ease_out(clamp01((lt - 1.20) / 0.45)),
               alpha=170, width=6, head=19)
    cross(d, (x0 + x1) / 2, y2, 34, p=ease_out(clamp01((lt - 1.70) / 0.30)), width=10)


def g_childRes(lay, lt):
    """«окажутся более средними» — случайная часть ребёнка мала, его вершина
    (синяя) стоит заметно ниже родительской."""
    d = ImageDraw.Draw(lay)
    mean_line(d)
    _parent_full(d)
    notch(d, PAIR_R)
    bar(d, PAIR_R, luck=CHILD_LUCK,
        p_luck=clamp01((lt - 0.15) / 0.60),
        cap=BLUE, p_cap=clamp01((lt - 0.90) / 0.18))
    top = MEAN_Y - DEV_TALL
    dashed_line(d, (PAIR_L + BAR_W / 2 + 16, top), (856, top),
                p=clamp01((lt - 1.30) / 0.45), dash=17, gap=13, width=3, alpha=145)
    grow_arrow(d, (822, top + 16), (822, MEAN_Y - DEV_CHILD - 16),
               ease_out(clamp01((lt - 1.85) / 0.45)), width=6, head=19)


def g_topDown(lay, lt):
    """«у отличника следующий результат в среднем чуть хуже»."""
    d = ImageDraw.Draw(lay)
    mean_line(d, p=clamp01(lt / 0.22))
    dot(d, PAIR_L, 346, p=clamp01((lt - 0.18) / 0.20))
    grow_arrow(d, (413, 823), (666, 973),
               ease_out(clamp01((lt - 0.45) / 0.50)), width=6, head=19)
    dot(d, PAIR_R, 157, p=clamp01((lt - 0.98) / 0.18), color=BLUE)


def g_bottomUp(lay, lt):
    """«а у отстающего чуть лучше» — та же стрелка, но снизу вверх."""
    d = ImageDraw.Draw(lay)
    mean_line(d, p=clamp01(lt / 0.22))
    dot(d, PAIR_L, -315, p=clamp01((lt - 0.18) / 0.20))
    grow_arrow(d, (414, 1447), (665, 1316),
               ease_out(clamp01((lt - 0.50) / 0.50)), width=6, head=19)
    dot(d, PAIR_R, -147, p=clamp01((lt - 1.05) / 0.18), color=BLUE)


def g_wobble(lay, lt):
    """«просто случайных колебаний» — один и тот же человек прыгает
    вокруг линии от попытки к попытке."""
    d = ImageDraw.Draw(lay)
    mean_line(d)
    prev = None
    for i, (x, dev) in enumerate(WOBBLE):
        p = clamp01((lt - 0.15 - i * 0.11) / 0.20)
        if p <= 0.02:
            break
        if prev is not None:
            q = clamp01((lt - 0.15 - i * 0.11 + 0.05) / 0.16)
            d.line([prev, (prev[0] + (x - prev[0]) * q,
                           prev[1] + (MEAN_Y - dev - prev[1]) * q)],
                   fill=WHITE + (120,), width=4)
        dot(d, x, dev, p=p, r=15)
        prev = (x, MEAN_Y - dev)


GFX = {
    "tallPair": g_tallPair, "statCloud": g_statCloud, "spread": g_spread,
    "pullMean": g_pullMean, "sumBar": g_sumBar, "tallBar": g_tallBar,
    "genesLit": g_genesLit, "randomLit": g_randomLit, "inherit": g_inherit,
    "childRes": g_childRes, "topDown": g_topDown, "bottomUp": g_bottomUp,
    "wobble": g_wobble,
}


def graphics_layer(kind, lt):
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
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
    (brand-kit §5), поэтому строка 60pt не садится на соседнюю 42pt.
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
            print(f"frame {fno}/{nf}")

    enc.stdin.close()
    enc.wait()
    cap.release()
    stock.release()
    print("готово:", VID)


if __name__ == "__main__":
    main()

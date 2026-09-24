"""Сборка ролика 22 («дилемма заключённых»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx22.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр.

Предметная графика — один визуальный язык на весь ролик, без единой надписи:
* фигура в комнате = участник; чёрточка на лице = «молчит»;
  стрелка к соседу = «сдаёт»;
* вертикальная полоса под фигурой = срок, её длина и есть «маленький/средний/
  ещё больше»; пустая засечка без полосы = свобода;
* два блока рядом с пунктирным разделителем = сравнение двух сценариев.
Поэтому субтитр и графика физически не могут написать одно и то же:
слова есть только в субтитре, длина — только в графике.

Межстрочный шаг считается по brand-kit §5 от кеглей соседних строк
(`max(60, (s1 + s2) * 0.70)`), а не берётся фиксированным: в ролике есть
блоки, где рядом стоят 44 и 62 pt.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard22 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, TERM_SHORT, TERM_MID, TERM_LONG,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/22/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (28 замеров по всему дублю: cx 395±10, линия глаз y 671±4, ширина лица 265).
# Кроп ставит глаза на 42.0% высоты карточки у A1 и на 41.5% у A2.
FRAMINGS = {
    "A1": dict(w=660, h=1047, x=60, y=231),
    "A2": dict(w=590, h=936, x=100, y=282),
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
    считать его по отрезку tip->конец нельзя — на полностью выросшей стрелке
    эти точки совпадают, atan2(0, 0) отдаёт 0, и голова разворачивается вправо
    независимо от реального направления."""
    for s in (1, -1):
        d.line([tip, (tip[0] - head * math.cos(ang - s * 0.45),
                      tip[1] - head * math.sin(ang - s * 0.45))],
               fill=color + (int(alpha),), width=width)


def arrow(d, a, b, alpha=215, width=7, head=22, color=WHITE):
    d.line([a, b], fill=color + (int(alpha),), width=width)
    arrow_head(d, b, math.atan2(b[1] - a[1], b[0] - a[0]),
               alpha=alpha, width=width, head=head, color=color)


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


def check_mark(d, p, a, b, c, width=13, alpha=238, color=WHITE):
    p = clamp01(p)
    if p <= 0:
        return
    f1 = min(1.0, p / 0.40)
    d.line([a, (a[0] + (b[0] - a[0]) * f1, a[1] + (b[1] - a[1]) * f1)],
           fill=color + (alpha,), width=width)
    if p > 0.40:
        f2 = (p - 0.40) / 0.60
        d.line([b, (b[0] + (c[0] - b[0]) * f2, b[1] + (c[1] - b[1]) * f2)],
               fill=color + (alpha,), width=width)


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


# --- геометрия сцены -------------------------------------------------------
# Крупная схема (две комнаты): всё внутри GFX_ZONE 700..1560 / GFX_X 175..905.
ROOM_L = (200, 716, 480, 1012)      # x0, y0, x1, y1
ROOM_R = (620, 716, 900, 1012)
CX_L, CX_R = 340, 760
HEAD_Y = 850
BAR_TOP = 1075
BAR_W = 96

# Схема сравнения двух сценариев: два блока и пунктирный разделитель.
CMP_S = 0.72
CMP_BLOCKS = (330, 750)             # центры блоков
CMP_DX = 100                        # фигуры блока: центр ± CMP_DX
CMP_HEAD_Y = 800
CMP_BAR_TOP = 940
CMP_BAR_W = 68

TERMS = dict(short=TERM_SHORT, mid=TERM_MID, long=TERM_LONG, free=0)


def figure(d, cx, cy, s=1.0, alpha=1.0, silent=False, silent_p=1.0, p=1.0):
    """Человек: голова + плечи, обводкой. silent — чёрточка вместо рта."""
    p = clamp01(p)
    if p <= 0.02 or alpha <= 0.01:
        return
    a = int(230 * alpha * p)
    r = 36 * s
    lw = max(2, int(round(6 * s)))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WHITE + (a,), width=lw)
    bw, bh = 62 * s, 128 * s
    d.arc([cx - bw, cy + 24 * s, cx + bw, cy + 24 * s + 2 * bh],
          180, 360, fill=WHITE + (a,), width=lw)
    if silent:
        sp = clamp01(silent_p)
        if sp > 0.02:
            mw = 17 * s * sp
            d.line([(cx - mw, cy + 15 * s), (cx + mw, cy + 15 * s)],
                   fill=WHITE + (int(a * sp),), width=max(2, int(round(7 * s))))


def term_bar(d, cx, top, length, w, p=1.0, alpha=1.0, color=WHITE, ghost=0):
    """Полоса срока: растёт вниз от засечки. ghost — пунктирное продолжение
    до длины, которая была бы при другом выборе."""
    a = int(200 * alpha)
    d.line([(cx - w / 2 - 12, top), (cx + w / 2 + 12, top)],
           fill=WHITE + (int(140 * alpha),), width=4)
    h = length * ease_out(clamp01(p))
    if h > 4:
        d.rounded_rectangle([cx - w / 2, top, cx + w / 2, top + h],
                            radius=min(22, w / 2), fill=color + (int(232 * alpha),))
    if ghost > length:
        gp = clamp01((p - 0.55) / 0.45)
        if gp > 0.02:
            y0 = top + length
            y1 = top + length + (ghost - length) * ease_out(gp)
            for x in (cx - w / 2, cx + w / 2):
                dashed_line(d, (x, y0), (x, y1), p=1.0, dash=14, gap=11,
                            width=3, alpha=int(150 * alpha))
            dashed_line(d, (cx - w / 2, y1), (cx + w / 2, y1), p=1.0, dash=12,
                        gap=9, width=3, alpha=int(150 * alpha))
    return a


def room(d, rect, alpha=1.0, p=1.0, gap=None):
    """Комната. gap — вертикальный интервал (y0, y1) на левой стене: открытая дверь."""
    p = clamp01(p)
    if p <= 0.02 or alpha <= 0.01:
        return
    x0, y0, x1, y1 = rect
    a = int(160 * alpha * p)
    d.rounded_rectangle([x0, y0, x1, y1], radius=26, outline=WHITE + (a,), width=5)
    if gap:
        d.line([(x0, gap[0]), (x0, gap[1])], fill=(0, 0, 0, 255), width=9)


# --- планы -----------------------------------------------------------------

def g_selfboth(lay, lt):
    """«это самый нелогичный выбор для двоих людей сразу» — у каждого выбор
    логичен (галка), а общий итог перечёркнут."""
    d = ImageDraw.Draw(lay)
    figure(d, 380, HEAD_Y, p=clamp01(lt / 0.40))
    figure(d, 700, HEAD_Y, p=clamp01((lt - 0.22) / 0.40))
    check_mark(d, ease_out(clamp01((lt - 0.45) / 0.36)),
               (340, 1096), (372, 1128), (432, 1052))
    check_mark(d, ease_out(clamp01((lt - 0.62) / 0.36)),
               (660, 1096), (692, 1128), (752, 1052))
    bp = ease_out(clamp01((lt - 1.05) / 0.40))
    if bp > 0.02:
        a = int(190 * bp)
        y = 1250
        d.line([(380, y), (380 + (700 - 380) * bp, y)], fill=WHITE + (a,), width=5)
        d.line([(380, y), (380, y - 34)], fill=WHITE + (a,), width=5)
        if bp > 0.9:
            d.line([(700, y), (700, y - 34)], fill=WHITE + (a,), width=5)
    cross(d, 540, 1382, 58, p=ease_out(clamp01((lt - 1.48) / 0.42)))


def _rooms_base(d, lt, left_alpha=1.0, right_alpha=1.0, left_gap=None,
                left_silent=False, right_silent=False, ls_p=1.0, rs_p=1.0,
                left_fig=1.0, right_fig=1.0, p=1.0):
    room(d, ROOM_L, alpha=left_alpha, p=p, gap=left_gap)
    room(d, ROOM_R, alpha=right_alpha, p=p)
    if left_fig > 0.01:
        figure(d, CX_L, HEAD_Y, alpha=left_alpha * left_fig,
               silent=left_silent, silent_p=ls_p)
    if right_fig > 0.01:
        figure(d, CX_R, HEAD_Y, alpha=right_alpha * right_fig,
               silent=right_silent, silent_p=rs_p)


def g_rooms(lay, lt):
    """«в разных комнатах без возможности договориться» — связь перечёркнута."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt, p=clamp01(lt / 0.40))
    lp = clamp01((lt - 0.32) / 0.40)
    dashed_line(d, (486, 870), (614, 870), p=lp, dash=15, gap=12, width=5, alpha=170)
    cross(d, 550, 870, 30, p=ease_out(clamp01((lt - 0.86) / 0.36)), width=8)


def g_bothSilent(lay, lt):
    """«если оба молчат, оба получают маленький срок»."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt, left_silent=True, right_silent=True,
                ls_p=clamp01((lt - 0.15) / 0.30), rs_p=clamp01((lt - 0.32) / 0.30))
    bp = clamp01((lt - 0.95) / 0.55)
    term_bar(d, CX_L, BAR_TOP, TERM_SHORT, BAR_W, p=bp)
    term_bar(d, CX_R, BAR_TOP, TERM_SHORT, BAR_W, p=clamp01((lt - 1.08) / 0.55))


def g_oneBetrays(lay, lt):
    """«если один сдаёт другого, а тот молчит» — стрелка от левого к правому."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt, right_silent=True, rs_p=clamp01((lt - 1.62) / 0.32))
    grow_arrow(d, (430, 772), (676, 772), ease_out(clamp01((lt - 0.55) / 0.42)))


DOOR_GAP = (812, 985)      # проём в левой стене, по росту вышедшей фигуры
FREE_CX, FREE_S = 232, 0.80


def _free_figure(d, walk=1.0):
    """Вышедший стоит в проёме и остаётся в кадре до конца эпизода: пустая
    засечка сама по себе читается как «кадр опустел», а не как «срока нет»."""
    w = clamp01(walk)
    figure(d, CX_L + (FREE_CX - CX_L) * w, HEAD_Y, s=1.0 + (FREE_S - 1.0) * w)


def g_traitorFree(lay, lt):
    """«предатель выходит на свободу» — стена открывается, он выходит в проём,
    засечка под его комнатой остаётся пустой."""
    d = ImageDraw.Draw(lay)
    walk = ease_out(clamp01((lt - 0.25) / 0.85))
    _rooms_base(d, lt, left_gap=DOOR_GAP, right_silent=True, left_fig=0.0)
    _free_figure(d, walk)
    grow_arrow(d, (322, 1030), (208, 1030), ease_out(clamp01((lt - 0.50) / 0.50)),
               width=7, head=22)
    term_bar(d, CX_L, BAR_TOP, 0, BAR_W)
    term_bar(d, CX_R, BAR_TOP, 0, BAR_W)


def g_silentMax(lay, lt):
    """«а молчавший получает ещё больше срок» — его полоса самая длинная,
    а вышедший так и стоит на свободе с пустой засечкой."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt, left_gap=DOOR_GAP, left_fig=0.0, right_silent=True)
    _free_figure(d)
    term_bar(d, CX_L, BAR_TOP, 0, BAR_W)
    term_bar(d, CX_R, BAR_TOP, TERM_LONG, BAR_W, p=clamp01((lt - 0.45) / 0.75))


def _both_arrows(d, p1, p2):
    grow_arrow(d, (430, 758), (676, 758), p1)
    grow_arrow(d, (670, 806), (424, 806), p2)


def g_bothBetray(lay, lt):
    """«если сдают друг друга оба» — две встречные стрелки."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt)
    _both_arrows(d, ease_out(clamp01((lt - 0.20) / 0.40)),
                 ease_out(clamp01((lt - 0.52) / 0.40)))


def g_bothMid(lay, lt):
    """«оба получают средний срок»."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt)
    _both_arrows(d, 1.0, 1.0)
    term_bar(d, CX_L, BAR_TOP, TERM_MID, BAR_W, p=clamp01((lt - 0.30) / 0.55))
    term_bar(d, CX_R, BAR_TOP, TERM_MID, BAR_W, p=clamp01((lt - 0.42) / 0.55))


# --- сравнение двух сценариев ----------------------------------------------

def cmp_block(d, cx, left, right, alpha=1.0, p=1.0, bar_p=1.0, colors=None):
    """Блок сценария: две фигуры и их сроки. left/right — dict(term, silent,
    betray, ghost).

    «Сдаёт» здесь помечается короткой стрелкой над самой фигурой в сторону
    соседа, а не линией через весь блок: двум встречным линиям в блоке шириной
    200px не разойтись, они сели бы одна на другую.
    Пустая засечка без полосы = срока нет (язык задан планом `traitorFree`).
    """
    colors = colors or (WHITE, WHITE)
    for i, (side, dx) in enumerate(((left, -CMP_DX), (right, CMP_DX))):
        fx = cx + dx
        figure(d, fx, CMP_HEAD_Y, s=CMP_S, alpha=alpha, p=p,
               silent=side.get("silent", False))
        if side.get("betray"):
            sgn = 1 if dx < 0 else -1
            grow_arrow(d, (fx - 26 * sgn, 742), (fx + 34 * sgn, 742),
                       clamp01(p * 1.4), alpha=int(210 * alpha), width=5, head=15)
        term_bar(d, fx, CMP_BAR_TOP, side.get("term", 0) * CMP_S, CMP_BAR_W,
                 p=bar_p, alpha=alpha, color=colors[i],
                 ghost=side.get("ghost", 0) * CMP_S)
        if side.get("good") and bar_p > 0.25:
            check_mark(d, ease_out(clamp01((bar_p - 0.25) / 0.45)),
                       (fx - 26, 1006), (fx - 4, 1028), (fx + 40, 972),
                       width=10, alpha=int(238 * alpha))


def cmp_divider(d, p=1.0, alpha=90):
    dashed_line(d, (540, 762), (540, 1252), p=p, dash=16, gap=14, width=3, alpha=alpha)


MID_B = dict(term=TERM_MID, betray=True)
SHORT_B = dict(term=TERM_SHORT, silent=True)


def g_ladder(lay, lt):
    """«но лучше, чем быть единственным молчавшим» — средний срок против
    самого длинного."""
    d = ImageDraw.Draw(lay)
    cmp_divider(d, p=clamp01(lt / 0.35))
    cmp_block(d, CMP_BLOCKS[0], MID_B, MID_B, alpha=0.55,
              p=clamp01(lt / 0.40), bar_p=clamp01((lt - 0.20) / 0.45))
    cmp_block(d, CMP_BLOCKS[1],
              dict(term=0, betray=True),
              dict(term=TERM_LONG, silent=True),
              p=clamp01((lt - 0.55) / 0.40), bar_p=clamp01((lt - 0.80) / 0.60))
    lp = clamp01((lt - 1.45) / 0.45)
    y = CMP_BAR_TOP + TERM_MID * CMP_S
    dashed_line(d, (196, y), (880, y), p=lp, dash=16, gap=13, width=3, alpha=155)


def g_domA(lay, lt):
    """«выгоднее сдать другого» — считаем от твоего лица: ты сдаёшь."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt, right_alpha=0.40, right_fig=1.0)
    grow_arrow(d, (430, 772), (676, 772), ease_out(clamp01((lt - 0.18) / 0.42)))


def _branch_blocks(d, lt, p_left, p_right, bar_l=0.0, bar_r=0.0,
                   al=1.0, ar=1.0, ghost=0):
    """Слева ты сдаёшь, а он молчит: у тебя срока нет, у него самый длинный.
    Справа сдают оба: у обоих средний, пунктир под твоей полосой показывает,
    каким он был бы, если бы ты молчал."""
    cmp_block(d, CMP_BLOCKS[0], dict(term=0, betray=True, good=al > 0.5),
              dict(term=TERM_LONG, silent=True), alpha=al, p=p_left, bar_p=bar_l)
    cmp_block(d, CMP_BLOCKS[1], dict(term=TERM_MID, betray=True, ghost=ghost),
              dict(term=TERM_MID, betray=True),
              alpha=ar, p=p_right, bar_p=bar_r)


def g_branch(lay, lt):
    """«независимо от того, что сделает он» — два случая рядом: он молчит / он сдаёт."""
    d = ImageDraw.Draw(lay)
    cmp_divider(d, p=clamp01(lt / 0.35))
    _branch_blocks(d, lt, clamp01((lt - 0.15) / 0.40), clamp01((lt - 0.75) / 0.40))


def g_branchFree(lay, lt):
    """«это либо освобождает вас» — в случае «он молчит» твоего срока нет."""
    d = ImageDraw.Draw(lay)
    cmp_divider(d)
    _branch_blocks(d, lt, 1.0, 1.0, bar_l=clamp01((lt - 0.20) / 0.55), ar=0.34)


def g_branchMid(lay, lt):
    """«либо хотя бы даёт шанс» — в случае «он сдаёт» у тебя средний срок,
    пунктир показывает, каким он был бы, если бы ты молчал."""
    d = ImageDraw.Draw(lay)
    cmp_divider(d)
    _branch_blocks(d, lt, 1.0, 1.0, bar_l=1.0, al=0.34,
                   bar_r=clamp01((lt - 0.18) / 0.55), ghost=TERM_LONG)


def g_eqArrows(lay, lt):
    """«оба выбирают сдать друг друга»."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt)
    _both_arrows(d, ease_out(clamp01((lt - 0.10) / 0.38)),
                 ease_out(clamp01((lt - 0.38) / 0.38)))


def g_eqResult(lay, lt):
    """«и оба получают средний срок» — это и есть результат, он синий."""
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt)
    _both_arrows(d, 1.0, 1.0)
    p = clamp01((lt - 0.40) / 0.45)
    term_bar(d, CX_L, BAR_TOP, TERM_MID, BAR_W, p=p, color=BLUE)
    term_bar(d, CX_R, BAR_TOP, TERM_MID, BAR_W, p=clamp01((lt - 0.52) / 0.45),
             color=BLUE)


def g_verdict(lay, lt):
    """«хуже, чем если бы оба просто промолчали» — синий итог против молчания."""
    d = ImageDraw.Draw(lay)
    cmp_divider(d)
    cmp_block(d, CMP_BLOCKS[0], MID_B, MID_B, colors=(BLUE, BLUE))
    cmp_block(d, CMP_BLOCKS[1], SHORT_B, SHORT_B,
              p=clamp01((lt - 0.35) / 0.40), bar_p=clamp01((lt - 0.55) / 0.55))


def _delta_bracket(d, p):
    """Скобка на разнице: от уровня «оба молчали» до дна синих полос."""
    p = clamp01(p)
    if p <= 0.02:
        return
    y0 = CMP_BAR_TOP + TERM_SHORT * CMP_S
    y1 = CMP_BAR_TOP + TERM_MID * CMP_S
    x = CMP_BLOCKS[0] - CMP_DX - CMP_BAR_W / 2 - 14
    a = int(210 * p)
    d.line([(x, y0), (x, y0 + (y1 - y0) * p)], fill=WHITE + (a,), width=5)
    d.line([(x, y0), (x + 22, y0)], fill=WHITE + (a,), width=5)
    if p > 0.9:
        d.line([(x, y1), (x + 22, y1)], fill=WHITE + (a,), width=5)
    dashed_line(d, (x, y0), (880, y0), p=p, dash=16, gap=13, width=3, alpha=150)


def g_together(lay, lt):
    """«а вместе они пришли к результату хуже» — разница отмечена скобкой."""
    d = ImageDraw.Draw(lay)
    cmp_divider(d)
    cmp_block(d, CMP_BLOCKS[0], MID_B, MID_B, colors=(BLUE, BLUE))
    cmp_block(d, CMP_BLOCKS[1], SHORT_B, SHORT_B)
    _delta_bracket(d, ease_out(clamp01((lt - 0.45) / 0.50)))


def g_couldHave(lay, lt):
    """«чем могли бы получить» — молчание подсвечено, итог погашен."""
    d = ImageDraw.Draw(lay)
    cmp_divider(d, alpha=60)
    cmp_block(d, CMP_BLOCKS[0], MID_B, MID_B, alpha=0.30, colors=(BLUE, BLUE))
    cmp_block(d, CMP_BLOCKS[1], SHORT_B, SHORT_B)
    check_mark(d, ease_out(clamp01((lt - 0.30) / 0.40)),
               (692, 1252), (720, 1280), (786, 1206))


def g_deal(lay, lt):
    """«если бы просто договорились» — связь между комнатами восстановлена.

    Единственный лаймовый элемент ролика (brand-kit §4).
    """
    d = ImageDraw.Draw(lay)
    _rooms_base(d, lt, left_silent=True, right_silent=True)
    lp = ease_out(clamp01((lt - 0.22) / 0.40))
    if lp > 0.02:
        a = int(235 * lp)
        d.line([(486, 870), (486 + (614 - 486) * lp, 870)],
               fill=LIME + (a,), width=10)
        if lp > 0.85:
            for x in (486, 614):
                d.ellipse([x - 11, 859, x + 11, 881], fill=LIME + (a,))
    bp = clamp01((lt - 0.85) / 0.55)
    term_bar(d, CX_L, BAR_TOP, TERM_SHORT, BAR_W, p=bp)
    term_bar(d, CX_R, BAR_TOP, TERM_SHORT, BAR_W, p=clamp01((lt - 0.97) / 0.55))


GFX = {
    "selfboth": g_selfboth, "rooms": g_rooms, "bothSilent": g_bothSilent,
    "oneBetrays": g_oneBetrays, "traitorFree": g_traitorFree,
    "silentMax": g_silentMax, "bothBetray": g_bothBetray, "bothMid": g_bothMid,
    "ladder": g_ladder, "domA": g_domA, "branch": g_branch,
    "branchFree": g_branchFree, "branchMid": g_branchMid,
    "eqArrows": g_eqArrows, "eqResult": g_eqResult, "verdict": g_verdict,
    "together": g_together, "couldHave": g_couldHave, "deal": g_deal,
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
    (brand-kit §5), поэтому строка 62pt не садится на соседнюю 44pt.
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

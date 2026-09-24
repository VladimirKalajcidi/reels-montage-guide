"""Сборка ролика 31 («парадокс инспекции»).

Слойность: фон (лицо / сетка / сток) -> графика -> субтитры. Аудио добавляет
sfx31.py, исходная речь не режется.

Механику несёт своя графика: линия времени с засечками-автобусами, где ширина
промежутка физически равна его длительности. Сток — только там, где в речи назван
реальный объект (автобус, остановка, поезд), и лежит в videos/31/stock
(assets-manifest §2).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* межстрочный шаг считается от кеглей соседних строк, а не берётся фиксированным;
* кегль числа подбирается под ширину зоны графики, а не по шкале XL.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard31 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, SHORT_MIN, LONG_MIN, RATIO,
                          SCHED_MEAN, NAIVE_WAIT, SEEN_MEAN, INTERVALS,
                          SPAN_MIN, UNIFORM_N, DROP_MIN, ROW_PATTERNS, NUM_AT,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (28 замеров по всему дублю: cx 401±9, линия глаз y 701±5, ширина лица 273±7).
# Кроп ставит глаза на 42% высоты карточки — тот же приём, что в роликах 22-30.
FRAMINGS = {
    "A1": dict(w=630, h=1000, x=86, y=280),
}


def cv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def source_card(frame, kind):
    """A-roll: hflip + кроп под карточку A. Грейда нет."""
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    f = FRAMINGS["A1"]
    w, h = f["w"], f["h"]
    x = max(0, min(f["x"], iw - w))
    y = max(0, min(f["y"], ih - h))
    fr = fr[y:y + h, x:x + w]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_CUBIC)
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


# --- маска зоны графики -----------------------------------------------------
_ZMASK = None


def zone_mask():
    global _ZMASK
    if _ZMASK is None:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([GX0 + 16, GY0 + 16, GX1 - 16, GY1 - 16], fill=255)
        _ZMASK = m.filter(ImageFilter.GaussianBlur(5))
    return _ZMASK


def _col(color, alpha):
    return tuple(color) + (max(0, min(255, int(round(255 * alpha)))),)


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


def stagger(lt, n, t0=0.10, step=0.09, dur=0.42):
    return [ease_out(clamp01((lt - t0 - i * step) / dur)) for i in range(n)]


def pop(lt, at, dur=0.10):
    """R5a: белое число резким попом, 3 кадра, 0.6 -> 1.0 без оверщута."""
    return ease_out(clamp01((lt - at) / dur))


# ============================================================================
# ГЕОМЕТРИЯ ЛИНИИ ВРЕМЕНИ. Зона графики: x 175...905, y 700...1560.
# Жёсткие пиксели держим внутри x 197...883, y 722...1538 (маска зоны + запас).
# ============================================================================
AX_Y = 1040            # линия времени
TICK_H = 38            # полувысота засечки-автобуса
BAND_Y = 1112          # полоса промежутка (под осью), ширина полосы = время
BAND_H = 26
NUM_Y = 1270           # крупные числа
DROP_TOP = 870         # старт стрелки прихода пассажира
DROP_TIP = AX_Y - TICK_H - 8

# композиция L — вся линия целиком, 90 минут на 640px
L_X0, L_X1 = 220, 860
PPM_L = (L_X1 - L_X0) / SPAN_MIN                     # 7.111 px/мин

# композиция P — два промежутка рядом, 24 px/мин
PPM_P = 24
P_SHORT_X0 = 260
P_SHORT_X1 = P_SHORT_X0 + SHORT_MIN * PPM_P          # 308
P_LONG_X0 = 360
P_LONG_X1 = P_LONG_X0 + LONG_MIN * PPM_P             # 840
P_SHORT_CX = (P_SHORT_X0 + P_SHORT_X1) / 2           # 284
P_LONG_CX = (P_LONG_X0 + P_LONG_X1) / 2              # 600

# композиция G — один промежуток на весь кадр, 50 px/мин
G_X0 = 290
G_X1 = G_X0 + SCHED_MEAN * 50                        # 790
G_MID = (G_X0 + G_X1) / 2                            # 540

# столбики «средний интервал»: та же логика — длина равна времени
BAR_X0 = 230
PPM_BAR = 22
BAR_SEEN_Y, BAR_SCHED_Y = 1010, 1200
BAR_H = 80
BAR_SEEN_X1 = BAR_X0 + SEEN_MEAN * PPM_BAR           # 626
BAR_SCHED_X1 = BAR_X0 + SCHED_MEAN * PPM_BAR         # 450
BAR_SEEN_NUM_X, BAR_SCHED_NUM_X = 762, 587

NUM_BIG = 150          # кегль крупного числа под зону 730px
NUM_BAR = 110


def _axis(d, x0, x1, y, prog=1.0, alpha=0.70):
    if prog <= 0.004:
        return
    d.line([(x0, y), (x0 + (x1 - x0) * prog, y)], fill=_col(WHITE, alpha), width=5)


def _tick(d, x, y, prog=1.0, h=TICK_H, alpha=1.0, color=WHITE, width=8):
    if prog <= 0.004:
        return
    hh = h * ease_out(prog)
    d.line([(x, y - hh), (x, y + hh)], fill=_col(color, alpha), width=width)


def _band(d, x0, x1, prog=1.0, color=WHITE, alpha=0.85, y=BAND_Y, h=BAND_H):
    if prog <= 0.004:
        return
    xe = x0 + (x1 - x0) * ease_out(prog)
    if xe - x0 < 3:
        xe = x0 + 3
    d.rounded_rectangle([x0, y - h / 2, xe, y + h / 2], radius=h / 2,
                        fill=_col(color, alpha))


def _drop(d, x, prog=1.0, color=WHITE, alpha=0.95, top=DROP_TOP, tip=DROP_TIP):
    """Стрелка сверху: момент, когда пассажир пришёл на остановку."""
    if prog <= 0.004:
        return
    p = ease_out(prog)
    y0 = top + (1 - p) * 46
    yt = tip
    d.line([(x, y0), (x, yt)], fill=_col(color, alpha * p), width=7)
    for sx in (-1, 1):
        d.line([(x, yt), (x + sx * 16, yt - 24)], fill=_col(color, alpha * p), width=7)


def _bounds(seq):
    """Границы промежутков в минутах: [0, i0, i0+i1, ...]."""
    out, acc = [0.0], 0.0
    for v in seq:
        acc += v
        out.append(acc)
    return out


L_BOUNDS = _bounds(INTERVALS)
L_X = [L_X0 + m * PPM_L for m in L_BOUNDS]
LONG_IDX = [i for i, v in enumerate(INTERVALS) if v == LONG_MIN]
UNI_X = [L_X0 + i * (L_X1 - L_X0) / UNIFORM_N for i in range(UNIFORM_N + 1)]
UNI_MARK = UNIFORM_N // 2          # какой промежуток подсвечиваем на schedTicks


def _drop_x(m):
    return L_X0 + m * PPM_L


def _in_long(m):
    for i in LONG_IDX:
        if L_BOUNDS[i] <= m < L_BOUNDS[i + 1]:
            return True
    return False


# --- планы ------------------------------------------------------------------

def g_schedTicks(lay, lt):
    """«в среднем раз в 10 минут»: линия времени с равными промежутками по
    10 минут. Полоса под одним промежутком + белое 10 (R5a) на слове «10»."""
    d = ImageDraw.Draw(lay)
    _axis(d, L_X0, L_X1, AX_Y, ease_out(clamp01(lt / 0.24)))
    prog = stagger(lt, UNIFORM_N + 1, t0=0.10, step=0.032, dur=0.22)
    for i, x in enumerate(UNI_X):
        _tick(d, x, AX_Y, prog[i])
    for i in range(UNIFORM_N):
        _band(d, UNI_X[i] + 4, UNI_X[i + 1] - 4, prog[i + 1], alpha=0.26)
    _band(d, UNI_X[UNI_MARK], UNI_X[UNI_MARK + 1],
          ease_out(clamp01((lt - 0.46) / 0.24)))
    q = pop(lt, NUM_AT["schedTicks"])
    if q > 0:
        cx = (UNI_X[UNI_MARK] + UNI_X[UNI_MARK + 1]) / 2
        draw_number(lay, str(SCHED_MEAN), (cx, NUM_Y), NUM_BIG, color=WHITE,
                    glow=WHITE, glow_a=0.55, scale=0.6 + 0.4 * q)


def g_naiveWait(lay, lt):
    """«это ещё не значит, что вы будете ждать в среднем 5 минут»: тот же
    десятиминутный промежуток крупно. Пассажир приходит ровно посередине —
    это и есть наивная картинка, — и остаток до автобуса помечен белым 5."""
    d = ImageDraw.Draw(lay)
    _axis(d, G_X0, G_X1, AX_Y, ease_out(clamp01(lt / 0.22)))
    p0 = stagger(lt, 2, t0=0.06, step=0.10, dur=0.24)
    _tick(d, G_X0, AX_Y, p0[0])
    _tick(d, G_X1, AX_Y, p0[1])
    _band(d, G_X0, G_X1, ease_out(clamp01((lt - 0.20) / 0.28)), alpha=0.26)
    _drop(d, G_MID, ease_out(clamp01((lt - 1.035) / 0.30)))
    _band(d, G_MID, G_X1, ease_out(clamp01((lt - 1.60) / 0.32)))
    q = pop(lt, NUM_AT["naiveWait"])
    if q > 0:
        draw_number(lay, str(NAIVE_WAIT), ((G_MID + G_X1) / 2, NUM_Y), NUM_BIG,
                    color=WHITE, glow=WHITE, glow_a=0.55, scale=0.6 + 0.4 * q)


def _pair(d, lt, short_prog=1.0, long_prog=0.0, long_color=WHITE,
          axis_prog=1.0, short_ticks=None):
    """Композиция P: короткий и длинный промежутки в честном масштабе 24 px/мин.

    short_ticks — общий прогресс обеих засечек короткого промежутка вместо
    штатного каскада. Нужен там, где композиция вводится сразу под ревил числа
    и на стаггер 0.08с времени нет (gapShortNum)."""
    _axis(d, L_X0, L_X1, AX_Y, axis_prog, alpha=0.42)
    if short_prog > 0.004:
        if short_ticks is not None:
            sp = [short_ticks, short_ticks]
        else:
            sp = stagger(lt, 2, t0=0.06, step=0.08, dur=0.22) if short_prog < 1 else [1.0, 1.0]
        _tick(d, P_SHORT_X0, AX_Y, sp[0])
        _tick(d, P_SHORT_X1, AX_Y, sp[1])
        _band(d, P_SHORT_X0, P_SHORT_X1, short_prog)
    if long_prog > 0.004:
        _tick(d, P_LONG_X0, AX_Y, min(1.0, long_prog * 3))
        _tick(d, P_LONG_X1, AX_Y, min(1.0, long_prog * 3))
        _band(d, P_LONG_X0, P_LONG_X1, long_prog, color=long_color,
              alpha=0.90 if long_color is BLUE else 0.85)


def g_gapShortNum(lay, lt):
    """«...через 2 минуты»: два автобуса подряд — промежуток в две минуты,
    48px при масштабе 24 px/мин, — и белое 2 (R5a) под ним. Композиция
    вводится прямо на этом плане (до него на экране был автобус), поэтому
    засечки и полоса собираются за 0.22с, до ревила числа. Субтитра нет."""
    d = ImageDraw.Draw(lay)
    _pair(d, lt, short_prog=ease_out(clamp01((lt - 0.02) / 0.20)),
          axis_prog=ease_out(clamp01(lt / 0.14)),
          short_ticks=ease_out(clamp01(lt / 0.16)))
    q = pop(lt, NUM_AT["gapShortNum"])
    if q > 0:
        draw_number(lay, str(SHORT_MIN), (P_SHORT_CX, NUM_Y), NUM_BIG,
                    color=WHITE, glow=WHITE, glow_a=0.55, scale=0.6 + 0.4 * q)


def g_gapLong(lay, lt):
    """«а иногда через 20 минут»: справа вырастает промежуток ровно в десять
    раз шире. Двойка остаётся на месте — сравнение видно без слов."""
    d = ImageDraw.Draw(lay)
    _pair(d, lt, long_prog=ease_out(clamp01((lt - 0.04) / 0.42)))
    draw_number(lay, str(SHORT_MIN), (P_SHORT_CX, NUM_Y), NUM_BIG, color=WHITE,
                glow=WHITE, glow_a=0.55)
    q = pop(lt, NUM_AT["gapLong"])
    if q > 0:
        draw_number(lay, str(LONG_MIN), (P_LONG_CX, NUM_Y), NUM_BIG, color=WHITE,
                    glow=WHITE, glow_a=0.55, scale=0.6 + 0.4 * q)


SWING_T = 1.10


def g_pairBoth(lay, lt):
    """«в какой промежуток вы скорее попадёте — в двухминутный или 20 минут»:
    оба промежутка с числами, стрелка прихода качается между ними. Субтитров
    на плане нет: оба числа уже написаны графикой."""
    d = ImageDraw.Draw(lay)
    _pair(d, lt, long_prog=1.0)
    draw_number(lay, str(SHORT_MIN), (P_SHORT_CX, NUM_Y), NUM_BIG, color=WHITE,
                glow=WHITE, glow_a=0.55)
    draw_number(lay, str(LONG_MIN), (P_LONG_CX, NUM_Y), NUM_BIG, color=WHITE,
                glow=WHITE, glow_a=0.55)
    s = 0.5 - 0.5 * math.cos(2 * math.pi * lt / SWING_T)
    x = P_SHORT_CX + (P_LONG_CX - P_SHORT_CX) * s
    _drop(d, x, ease_out(clamp01((lt - 0.10) / 0.28)))


def g_pairAnswer(lay, lt):
    """«ответ — 20-минутный»: длинный промежуток заливается синим, стрелка
    встаёт над ним. Само слово «20-минутный» на экран не выносится."""
    d = ImageDraw.Draw(lay)
    q = ease_out(clamp01((lt - 0.08) / 0.30))
    _pair(d, lt, long_prog=1.0, long_color=BLUE if q > 0.5 else WHITE)
    draw_number(lay, str(SHORT_MIN), (P_SHORT_CX, NUM_Y), NUM_BIG, color=WHITE,
                glow=WHITE, glow_a=0.55)
    draw_number(lay, str(LONG_MIN), (P_LONG_CX, NUM_Y), NUM_BIG, color=WHITE,
                glow=WHITE, glow_a=0.55)
    _drop(d, P_LONG_CX, 1.0, color=BLUE_GLOW)


DIVS = RATIO - 1          # 9 делений режут длинный промежуток на 10 коротких


def g_ratioSetup(lay, lt):
    """«просто потому что он занимает...»: длинный промежуток делится ровно
    на десять коротких — это и есть проверяемая механика множителя."""
    d = ImageDraw.Draw(lay)
    _pair(d, lt, long_prog=1.0, long_color=BLUE)
    prog = stagger(lt, DIVS, t0=0.10, step=0.055, dur=0.20)
    for k in range(DIVS):
        x = P_LONG_X0 + (k + 1) * SHORT_MIN * PPM_P
        if prog[k] > 0.004:
            hh = (BAND_H / 2 + 9) * ease_out(prog[k])
            d.line([(x, BAND_Y - hh), (x, BAND_Y + hh)],
                   fill=_col(WHITE, 0.95 * prog[k]), width=4)


def g_ratioNum(lay, lt):
    """«...в 10 раз больше времени»: синее × 10 (R5b) набором по разрядам."""
    d = ImageDraw.Draw(lay)
    _pair(d, lt, long_prog=1.0, long_color=BLUE)
    for k in range(DIVS):
        x = P_LONG_X0 + (k + 1) * SHORT_MIN * PPM_P
        hh = BAND_H / 2 + 9
        d.line([(x, BAND_Y - hh), (x, BAND_Y + hh)], fill=_col(WHITE, 0.95), width=5)
    r = clamp01((lt - NUM_AT["ratioNum"]) / 0.38)
    if r > 0:
        sc = 0.55 + 0.45 * ease_out(r) + 0.03 * math.sin(math.pi * min(1.0, r))
        draw_number(lay, f"× {RATIO}", (540, NUM_Y), NUM_BIG, color=BLUE,
                    glow=BLUE_GLOW, glow_a=0.85, scale=sc,
                    reveal=clamp01((lt - NUM_AT["ratioNum"]) / 0.30))


def _timeline(d, lt, tick_prog=None):
    _axis(d, L_X0, L_X1, AX_Y, 1.0 if tick_prog is None else tick_prog)
    prog = ([1.0] * len(L_X) if tick_prog is None
            else stagger(lt, len(L_X), t0=0.06, step=0.026, dur=0.20))
    for i in range(len(INTERVALS)):
        _band(d, L_X[i] + 3, L_X[i + 1] - 3, prog[i + 1], alpha=0.26)
    for i, x in enumerate(L_X):
        _tick(d, x, AX_Y, prog[i])


def _drops(d, lt, n_prev, n_now, t0=0.06, step=0.16, blue=False):
    for i in range(n_now):
        if i < n_prev:
            p = 1.0
        else:
            p = ease_out(clamp01((lt - t0 - (i - n_prev) * step) / 0.24))
        if p <= 0.004:
            continue
        m = DROP_MIN[i]
        col = BLUE_GLOW if (blue and _in_long(m)) else WHITE
        _drop(d, _drop_x(m), p, color=col, top=DROP_TOP - (i % 3) * 30)


def g_drops(lay, lt):
    """«поэтому длинные интервалы...»: та же линия времени, но промежутки
    разные — 5 коротких по 2 минуты и 4 длинных по 20. Пассажир приходит
    в случайные моменты; первые три метки уже падают в длинные промежутки."""
    d = ImageDraw.Draw(lay)
    _timeline(d, lt, tick_prog=ease_out(clamp01(lt / 0.22)))
    _drops(d, lt, 0, 3, t0=0.34, step=0.20)


def g_dropsMany(lay, lt):
    """«...встречаются в вашем личном опыте»: меток становится девять —
    ровно столько, сколько нужно, чтобы доля 80/90 читалась глазом."""
    d = ImageDraw.Draw(lay)
    _timeline(d, lt)
    _drops(d, lt, 3, len(DROP_MIN), t0=0.04, step=0.13)


def g_dropsTally(lay, lt):
    """«гораздо чаще, чем короткие»: длинные промежутки заливаются синим,
    и метки в них — тоже. Белая остаётся одна из девяти."""
    d = ImageDraw.Draw(lay)
    _timeline(d, lt)
    prog = stagger(lt, len(LONG_IDX), t0=0.04, step=0.10, dur=0.26)
    for k, i in enumerate(LONG_IDX):
        _band(d, L_X[i], L_X[i + 1], prog[k], color=BLUE, alpha=0.92)
    _drops(d, lt, len(DROP_MIN), len(DROP_MIN), blue=lt > 0.30)


def g_seenBar(lay, lt):
    """«средний интервал, который видит случайный пассажир, оказывается
    длиннее»: полоса длиной 18 минут в том же масштабе «длина = время».
    18 = E[L²]/E[L] при интервалах 2 и 20 с долей коротких 5/9."""
    d = ImageDraw.Draw(lay)
    d.line([(BAR_X0, BAR_SEEN_Y - 96), (BAR_X0, BAR_SCHED_Y + 96)],
           fill=_col(WHITE, 0.55), width=5)
    p = ease_out(clamp01((lt - 0.06) / 0.42))
    _band(d, BAR_X0, BAR_SEEN_X1, p, color=BLUE, alpha=0.92,
          y=BAR_SEEN_Y, h=BAR_H)
    q = clamp01((lt - NUM_AT["seenBar"]) / 0.38)
    if q > 0:
        sc = 0.55 + 0.45 * ease_out(q) + 0.03 * math.sin(math.pi * min(1.0, q))
        draw_number(lay, str(SEEN_MEAN), (BAR_SEEN_NUM_X, BAR_SEEN_Y), NUM_BAR,
                    color=BLUE, glow=BLUE_GLOW, glow_a=0.85, scale=sc,
                    reveal=clamp01((lt - NUM_AT["seenBar"]) / 0.30))


def g_schedBar(lay, lt):
    """«...среднего интервала между автобусами вообще»: под синей полосой
    белая — 10 минут расписания. Обе от одного левого края, разница длин
    и есть парадокс."""
    d = ImageDraw.Draw(lay)
    d.line([(BAR_X0, BAR_SEEN_Y - 96), (BAR_X0, BAR_SCHED_Y + 96)],
           fill=_col(WHITE, 0.55), width=5)
    _band(d, BAR_X0, BAR_SEEN_X1, 1.0, color=BLUE, alpha=0.92,
          y=BAR_SEEN_Y, h=BAR_H)
    draw_number(lay, str(SEEN_MEAN), (BAR_SEEN_NUM_X, BAR_SEEN_Y), NUM_BAR,
                color=BLUE, glow=BLUE_GLOW, glow_a=0.85)
    p = ease_out(clamp01((lt - 0.04) / 0.36))
    _band(d, BAR_X0, BAR_SCHED_X1, p, color=WHITE, alpha=0.85,
          y=BAR_SCHED_Y, h=BAR_H)
    q = pop(lt, NUM_AT["schedBar"], dur=0.12)
    if q > 0:
        draw_number(lay, str(SCHED_MEAN), (BAR_SCHED_NUM_X, BAR_SCHED_Y), NUM_BAR,
                    color=WHITE, glow=WHITE, glow_a=0.55, scale=0.6 + 0.4 * q)


# --- «тот же эффект в других процессах» ------------------------------------
ROW_Y = [800 + i * 130 for i in range(len(ROW_PATTERNS))]
ROW_X0, ROW_X1 = 250, 830
ROW_TICK = 21
ROW_TOTAL = sum(ROW_PATTERNS[0])
PPM_ROW = (ROW_X1 - ROW_X0) / ROW_TOTAL


def _row(d, idx, prog):
    if prog <= 0.004:
        return
    pat = ROW_PATTERNS[idx]
    y = ROW_Y[idx]
    b = _bounds(pat)
    _axis(d, ROW_X0, ROW_X1, y, prog, alpha=0.55)
    for m in b:
        _tick(d, ROW_X0 + m * PPM_ROW, y, prog, h=ROW_TICK, alpha=0.95, width=5)
    k = max(range(len(pat)), key=lambda i: pat[i])
    mid = (b[k] + b[k + 1]) / 2
    x = ROW_X0 + mid * PPM_ROW
    p = ease_out(clamp01((prog - 0.45) / 0.55))
    if p > 0.004:
        d.polygon([(x, y - ROW_TICK - 4), (x - 17, y - ROW_TICK - 32),
                   (x + 17, y - ROW_TICK - 32)], fill=_col(BLUE_GLOW, 0.95 * p))


def g_manyOthers(lay, lt):
    """«и множеством других случайных процессов»: шесть независимых линий
    времени других процессов, и метка на каждой снова падает в её самый
    длинный промежуток — один и тот же исход повторяется шесть раз."""
    d = ImageDraw.Draw(lay)
    prog = stagger(lt, len(ROW_PATTERNS), t0=0.04, step=0.13, dur=0.34)
    for i in range(len(ROW_PATTERNS)):
        _row(d, i, prog[i])


def g_word_paradox(lt):
    """R6: крупное слово ПАРАДОКС поверх карточки A, брендовая позиция y≈440."""
    text = "ПАРАДОКС"
    target_w = 800
    f = font("sans", 150)
    while f.getlength(text) > target_w and f.size > 40:
        f = font("sans", f.size - 4)
    a = ease_out(clamp01(lt / 0.30)) * 0.62
    lay = _layer()
    d = ImageDraw.Draw(lay)
    d.text((540, 440), text, font=f, fill=WHITE + (int(255 * a),), anchor="mm")
    blur = lay.split()[3].filter(ImageFilter.GaussianBlur(24))
    gl = Image.new("RGBA", lay.size, WHITE + (0,))
    gl.putalpha(blur.point(lambda v: int(v * 0.35)))
    out = Image.new("RGBA", lay.size, (0, 0, 0, 0))
    out.alpha_composite(gl)
    out.alpha_composite(lay)
    return out


GFX = {
    "schedTicks": g_schedTicks, "naiveWait": g_naiveWait,
    "gapShortNum": g_gapShortNum, "gapLong": g_gapLong,
    "pairBoth": g_pairBoth, "pairAnswer": g_pairAnswer,
    "ratioSetup": g_ratioSetup, "ratioNum": g_ratioNum,
    "drops": g_drops, "dropsMany": g_dropsMany, "dropsTally": g_dropsTally,
    "seenBar": g_seenBar, "schedBar": g_schedBar,
    "manyOthers": g_manyOthers,
}


def graphics_layer(kind, lt):
    if kind == "A1word":
        return g_word_paradox(lt)
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    lay.putalpha(ImageChops.multiply(lay.split()[3], zone_mask()))
    return lay


# --- субтитры ----------------------------------------------------------------

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
            calm = bid % 3 == 0
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


STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/31/stock"
STOCK_GAIN = 0.72      # притемнение вставки на 28% по экспозиции (brand-kit §6: 25-30%)


def stock_card(fr, cx=0.5, cy=0.5):
    fr = fit_bgr(fr, CARD_B[2], CARD_B[3], cx, cy)
    fr = np.clip(fr.astype(np.float32) * STOCK_GAIN, 0, 255)
    hsv = cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255)
    fr = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


def fit_bgr(fr, w, h, cx=0.5, cy=0.5):
    """Кроп до пропорции карточки. cx/cy — куда смещать окно кропа."""
    ih, iw = fr.shape[:2]
    target = w / h
    if iw / ih > target:
        nw = int(ih * target)
        x = int(round((iw - nw) * cx))
        x = max(0, min(iw - nw, x))
        fr = fr[:, x:x + nw]
    else:
        nh = int(iw / target)
        y = int(round((ih - nh) * cy))
        y = max(0, min(ih - nh, y))
        fr = fr[y:y + nh, :]
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)


class StockReader:
    """Сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр."""

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


def background(kind, prm, fr, t, stock_reader=None):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        card = source_card(fr, kind)
        canvas.paste(card, (CARD_A[0], CARD_A[1]), rounded_mask(CARD_A[2], CARD_A[3], R_A))
    elif kind == "stock":
        card = stock_card(stock_reader(), prm.get("cx", 0.5), prm.get("cy", 0.5))
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


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    nf = int(round(DUR * FPS))
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

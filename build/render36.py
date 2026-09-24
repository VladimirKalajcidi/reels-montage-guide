"""Сборка ролика 36 («парадокс мальчика и девочки»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры. Аудио добавляет
sfx36.py, исходная речь не режется.

Материал: одна литеральная стоковая вставка (школьный класс, mixkit 35954,
скачан в `videos/36/stock/`) на «если ты учил в школе», всё остальное — своя
графика на сетке. Тема комбинаторная: живого видео, показывающего «четыре
равновероятных исхода», не существует, а случайная перебивка тут ломает смысл
(assets-manifest §2).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* межстрочный шаг считается от кеглей соседних строк;
* кегль числа подбирается под ширину зоны графики, а не по шкале XL;
* графика рисуется в отдельном буфере размером с зону (730x860) при ss=2 —
  свечение на границе зоны обрезается самим буфером.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard36 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, OUTCOMES, ANSWER_ATLEAST,
                          ANSWER_ELDER, ANSWER_NAIVE, BIG_WORD,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (29 замеров по всему дублю: cx 372+-8, линия глаз y 683+-5, ширина лица 264+-6).
# Кроп 630x1000 ставит глаза на 42% высоты карточки — то же окно, что в
# роликах 22-34.
FRAMINGS = {
    "A1": dict(w=630, h=1000, x=57, y=264),
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


def num_size(text, xy, cap):
    """Кегль числа — от ширины зоны, а не по шкале XL.

    Габарит с учётом свечения (pad = 0.20*size*3 + 8) должен целиком лечь
    в зону графики: иначе glow вылезает за y 700...1560 / x 175...905
    и qa36.gfx_in_zone это ловит.
    """
    s = cap
    while s > 24:
        f = font("sans", s)
        pad = max(6, int(round(s * 0.20))) * 3 + 8
        bb = f.getbbox(text)
        hw = f.getlength(text) / 2 + pad
        hh = (bb[3] - bb[1]) / 2 + pad
        if (xy[0] - hw >= GX0 and xy[0] + hw <= GX1
                and xy[1] - hh >= GY0 and xy[1] + hh <= GY1):
            return s
        s -= 4
    return s


def stagger(lt, n, t0=0.10, step=0.09, dur=0.42):
    return [ease_out(clamp01((lt - t0 - i * step) / dur)) for i in range(n)]


# =============================================================================
# Сцена задачи — квадрат исходов
# =============================================================================
# Буфер графики совпадает с зоной из brand-kit §5 (y 700...1560, x 175...905),
# поэтому свечение обрезается границей зоны, а не вылезает на субтитры.
ZX0, ZY0, ZW, ZH, SS = GX0, GY0, GX1 - GX0, GY1 - GY0, 2

# --- квадрат исходов ---------------------------------------------------------
GRID_CX, GRID_CY = 540, 1130
COL_DX, ROW_DY = 178, 148
CELL_W, CELL_H, CELL_R = 320, 250, 26
CELLS = [(GRID_CX - COL_DX, GRID_CY - ROW_DY), (GRID_CX + COL_DX, GRID_CY - ROW_DY),
         (GRID_CX - COL_DX, GRID_CY + ROW_DY), (GRID_CX + COL_DX, GRID_CY + ROW_DY)]
ELDER_SC, YOUNG_SC = 4.0, 3.2      # старший крупнее младшего — это и есть порядок
FIG_DX = 72                        # старший слева, младший справа
CELL_FEET = 88                     # линия ног относительно центра ячейки

# --- план family / atLeast / ask ---------------------------------------------
FAM_FEET = 1346
FATHER_X, FATHER_SC = 300, 9.5
CH1_X, CH1_SC = 590, 6.8           # старший
CH2_X, CH2_SC = 790, 5.4           # младший
BIGQ_XY, BIGQ_SZ = (690, 950), 150

# --- планы naive1 / naive2 ---------------------------------------------------
NV_FEET = 1346
NV_BOY_X, NV_BOY_SC = 300, 9.5
NV_UNK_X = 640                     # второй ребёнок, пока неизвестен
NV_OPT_X, NV_OPT_SC = 770, 5.6
NV_OPT_TOP_C, NV_OPT_BOT_C = 985, 1285   # центры двух вариантов
NV_FORK = (420, 1135)
NV_ARROW_X = 665

# --- план half1 --------------------------------------------------------------
H1_CELLS = [(362, 960), (718, 960)]
H1_NUM_XY = (540, 1330)

NUM_XY = (540, GRID_CY)


class Z:
    """Два буфера — под белое свечение и под синее; координаты холста."""

    def __init__(self):
        self.ims = {"w": Image.new("RGBA", (ZW * SS, ZH * SS), (0, 0, 0, 0)),
                    "b": Image.new("RGBA", (ZW * SS, ZH * SS), (0, 0, 0, 0))}
        self.ds = {k: ImageDraw.Draw(v) for k, v in self.ims.items()}
        self.k = "w"

    def use(self, k):
        self.k = k
        return self

    @property
    def d(self):
        return self.ds[self.k]

    def p(self, x, y):
        return ((x - ZX0) * SS, (y - ZY0) * SS)

    def line(self, pts, col, w):
        self.d.line([self.p(*q) for q in pts], fill=col,
                    width=max(1, int(round(w * SS))), joint="curve")

    def circ(self, c, r, fill=None, outline=None, w=1):
        x, y = self.p(*c)
        rr = r * SS
        self.d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill,
                       outline=outline, width=max(1, int(round(w * SS))))

    def rrect(self, c, ww, hh, rad, fill=None, outline=None, w=1):
        x, y = self.p(*c)
        self.d.rounded_rectangle([x - ww * SS / 2, y - hh * SS / 2,
                                  x + ww * SS / 2, y + hh * SS / 2],
                                 radius=max(0, rad) * SS, fill=fill, outline=outline,
                                 width=max(1, int(round(w * SS))))

    def poly(self, pts, fill):
        self.d.polygon([self.p(*q) for q in pts], fill=fill)

    def text(self, c, txt, size, col, anchor="mm", kind="sans"):
        f = font(kind, max(8, int(round(size * SS))))
        self.d.text(self.p(*c), txt, font=f, fill=col, anchor=anchor)

    def out(self):
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for k, glow in (("w", WHITE), ("b", BLUE_GLOW)):
            im = self.ims[k].resize((ZW, ZH), Image.LANCZOS)
            if im.getbbox() is None:
                continue
            comp = _glow_sub(im, 11, glow, 0.36 if k == "w" else 0.82)
            lay.alpha_composite(comp, (ZX0, ZY0))
        return lay


def _mix(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# --- элементы сцены ----------------------------------------------------------

# Пропорции знака-пиктограммы: голова = 25% высоты фигуры. Первая версия
# (голова 7sc при теле 24sc) читалась снеговиком, а не человеком.
FIG_H = 45.4        # полная высота фигуры в единицах sc
FIG_W = 20.0        # ширина тела в единицах sc
FIG_TOP = 22.4      # верх фигуры относительно опорной точки, в единицах sc
FIG_BOT = 23.0      # низ фигуры относительно опорной точки, в единицах sc


def person(z, c, sc=1.0, a=1.0, blue=False, girl=False):
    """Пиктограмма человека: голова-круг отдельно от тела.

    Прямоугольное тело = мальчик, треугольное (юбка) = девочка.
    Голова стоит над телом с зазором (~4.6*sc): слитая с телом голова
    читалась снеговиком, а не человеком.
    """
    if a <= 0.004 or sc <= 0.02:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    x, y = c
    z.circ((x, y - 17 * sc), 5.4 * sc, fill=col)
    if girl:
        z.poly([(x, y - 7 * sc), (x - 15 * sc, y + FIG_BOT * sc),
                (x + 15 * sc, y + FIG_BOT * sc)], col)
    else:
        z.rrect((x, y + 8 * sc), FIG_W * sc, 30 * sc, 8 * sc, fill=col)


_QM = {}


def q_metrics():
    """Реальные чернила «?» при кегле 100, от базовой линии (anchor ms)."""
    if "bb" not in _QM:
        d = ImageDraw.Draw(Image.new("L", (1, 1)))
        _QM["bb"] = d.textbbox((0, 0), "?", font=font("sans", 100), anchor="ms")
    return _QM["bb"]


def qfigure(z, x, feet, sc=1.0, a=1.0, blue=False, part=0.72):
    """Ребёнок, пол которого ещё не известен: знак «?» ростом с фигурку,
    стоящий на той же линии ног. Полая фигурка-мальчик тут не годится —
    силуэт всё равно читался бы как мальчик, а это и есть предмет задачи."""
    if a <= 0.004 or sc <= 0.02:
        return
    bb = q_metrics()
    h100 = bb[3] - bb[1]
    target = FIG_H * sc * part
    size = max(10, target * 100.0 / h100)
    base = feet - bb[3] * size / 100.0
    z.use("b" if blue else "w")
    f = font("sans", max(8, int(round(size * SS))))
    z.d.text(z.p(x, base), "?", font=f,
             fill=_col(BLUE_GLOW if blue else WHITE, a), anchor="ms")


def feet_y(feet, sc):
    """Опорная точка фигурки такая, чтобы её низ встал на линию ног."""
    return feet - FIG_BOT * sc


def fig_center(feet, sc):
    """Вертикальный центр фигуры, стоящей ногами на feet."""
    return feet - (FIG_TOP + FIG_BOT) * sc / 2


def ring(z, c, sc=1.0, a=1.0, prog=1.0, blue=False):
    """Скруглённая рамка вокруг фигурки — «вот про кого нам сказали».
    Габарит считается от самой фигуры, а не задаётся отдельным числом."""
    if a <= 0.004 or prog <= 0.02:
        return
    z.use("b" if blue else "w")
    k = 0.72 + 0.28 * prog
    z.rrect(c, (FIG_W * sc + 34) * k, (FIG_H * sc + 18) * k, 14 * sc,
            outline=_col(BLUE_GLOW if blue else WHITE, a * prog), w=5)


def qmark(z, xy, size, a=1.0, blue=False, scale=1.0):
    if a <= 0.004 or scale <= 0.02:
        return
    z.use("b" if blue else "w")
    z.text(xy, "?", size * scale, _col(BLUE_GLOW if blue else WHITE, a))


def cross(z, c, r, a=1.0, w=9, blue=False, prog=1.0):
    if a <= 0.004 or r <= 1 or prog <= 0.02:
        return
    x, y = c
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    rr = r * min(1.0, prog * 1.4)
    z.line([(x - rr, y - rr), (x + rr, y + rr)], col, w)
    if prog > 0.5:
        k = min(1.0, (prog - 0.5) * 2.4)
        z.line([(x + rr, y - rr), (x + rr - 2 * rr * k, y - rr + 2 * rr * k)], col, w)


def arrow(z, p0, p1, prog=1.0, a=1.0, blue=False, w=6, head=22):
    """Прямая стрелка p0 -> p1, отрисовывается по прогрессу."""
    if a <= 0.004 or prog <= 0.02:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    q1 = _mix(p0, p1, prog)
    z.line([p0, q1], col, w)
    if prog > 0.88:
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        for s in (-1, 1):
            a2 = ang + s * 2.5
            z.line([q1, (q1[0] + head * math.cos(a2), q1[1] + head * math.sin(a2))],
                   col, w)


def pair(z, c, out, a=1.0, blue=False, sc=1.0):
    """Пара «старший слева, младший справа» внутри ячейки."""
    cx, cy = c
    feet = cy + CELL_FEET * sc
    e, y = ELDER_SC * sc, YOUNG_SC * sc
    person(z, (cx - FIG_DX * sc, feet_y(feet, e)), sc=e, a=a, blue=blue,
           girl=out[0] == "G")
    person(z, (cx + FIG_DX * sc, feet_y(feet, y)), sc=y, a=a, blue=blue,
           girl=out[1] == "G")


def cell(z, c, out, a=1.0, blue=False, fill=1.0, xx=0.0, mark=0.0, sc=1.0):
    """Ячейка исхода: рамка + пара фигурок. xx — крест «исход отпал»,
    mark — рамка на старшем («про него нам и сказали»)."""
    if a <= 0.004:
        return
    cx, cy = c
    z.use("b" if blue else "w")
    z.rrect(c, CELL_W * sc, CELL_H * sc, CELL_R * sc,
            outline=_col(BLUE_GLOW if blue else WHITE, a * 0.85), w=3)
    if fill > 0.004:
        pair(z, c, out, a=a * fill, blue=blue, sc=sc)
    if mark > 0.02:
        feet = cy + CELL_FEET * sc
        ring(z, (cx - FIG_DX * sc, fig_center(feet, ELDER_SC * sc)),
             sc=ELDER_SC * sc, a=a, prog=mark)
    if xx > 0.02:
        cross(z, c, 78 * sc, a=a, w=9, prog=xx)


def grid_cells(z, states, sc=1.0, cells=None):
    """states: список dict-параметров на каждую ячейку (или None — не рисовать)."""
    cs = cells or CELLS
    for i, st in enumerate(states):
        if st is None:
            continue
        cell(z, cs[i], OUTCOMES[i], sc=sc, **st)


# --- планы -------------------------------------------------------------------

def g_family(lay, lt):
    """«у мужчины двое детей»: отец и двое детей. Дети — полые фигуры со
    знаком «?»: пол пока не известен ни у одного из них.
    Крупность задаёт порядок: кто крупнее, тот старший. Дальше весь ролик
    старший стоит слева и крупнее — другого кода в ролике нет."""
    z = Z()
    s = stagger(lt, 3, t0=0.05, step=0.20, dur=0.32)
    person(z, (FATHER_X, feet_y(FAM_FEET, FATHER_SC)), sc=FATHER_SC, a=s[0])
    qfigure(z, CH1_X, FAM_FEET, sc=CH1_SC, a=s[1])
    qfigure(z, CH2_X, FAM_FEET, sc=CH2_SC, a=s[2])
    lay.alpha_composite(z.out())


def g_at_least(lay, lt):
    """«хотя бы один из них мальчик»: мальчик перескакивает между старшим
    и младшим — известно, что один из двоих мальчик, но не известно который.
    Ни один слот не закрепляется, иначе ролик соврал бы уже здесь."""
    z = Z()
    person(z, (FATHER_X, feet_y(FAM_FEET, FATHER_SC)), sc=FATHER_SC, a=1.0)
    # перескок мгновенный, как обычный рез. Кроссфейд «?» и фигурки пробовали:
    # на середине оба слоя по 50% и слипаются в грязное пятно.
    k = 1 if int(lt / 0.42) % 2 else 0
    for x, sc, w in ((CH1_X, CH1_SC, 1 - k), (CH2_X, CH2_SC, k)):
        y = feet_y(FAM_FEET, sc)
        if w:
            person(z, (x, y), sc=sc, a=1.0, girl=False)
            ring(z, (x, fig_center(FAM_FEET, sc)), sc=sc, a=1.0, prog=1.0)
        else:
            qfigure(z, x, FAM_FEET, sc=sc, a=1.0)
    lay.alpha_composite(z.out())


def g_ask(lay, lt):
    """«какова вероятность, что оба ребёнка мальчики»: над парой встаёт
    большой «?», оба ребёнка проявляются мальчиками — это и есть вопрос."""
    z = Z()
    q = ease_out(clamp01((lt - 0.05) / 0.30))
    person(z, (FATHER_X, feet_y(FAM_FEET, FATHER_SC)), sc=FATHER_SC, a=0.45)
    # «?» гаснет раньше, чем приходит фигурка: два слоя не наезжают друг
    # на друга на полупрозрачности
    for x, sc, t0 in ((CH1_X, CH1_SC, 0.97), (CH2_X, CH2_SC, 1.25)):
        y = feet_y(FAM_FEET, sc)
        gone = clamp01((lt - t0) / 0.12)
        b = ease_out(clamp01((lt - t0 - 0.14) / 0.22))
        if gone < 0.999:
            qfigure(z, x, FAM_FEET, sc=sc, a=1.0 - gone)
        if b > 0.02:
            person(z, (x, y), sc=sc, a=b, girl=False)
            ring(z, (x, fig_center(FAM_FEET, sc)), sc=sc, a=1.0, prog=b)
    qmark(z, BIGQ_XY, BIGQ_SZ, a=q, scale=0.55 + 0.45 * q)
    lay.alpha_composite(z.out())


def g_naive1(lay, lt):
    """Ложная интуиция: «один уже мальчик, значит второй». Порядок детей
    здесь пропадает — фигурки одного размера, отца нет. Именно потеря
    порядка и есть ошибка, которую ролик потом разбирает."""
    z = Z()
    a = ease_out(clamp01(lt / 0.22))
    q = ease_out(clamp01((lt - 0.26) / 0.28))
    person(z, (NV_BOY_X, feet_y(NV_FEET, NV_BOY_SC)), sc=NV_BOY_SC, a=a)
    qfigure(z, NV_UNK_X, NV_FEET, sc=NV_BOY_SC, a=q)
    lay.alpha_composite(z.out())


def g_naive2(lay, lt):
    """«второй либо мальчик, либо девочка»: «?» уходит, вместо него вилка
    на два варианта. Кто именно в вариантах — несёт субтитр, на графике
    подписей нет."""
    z = Z()
    fade = 1.0 - ease_out(clamp01(lt / 0.22))
    stem = ease_out(clamp01((lt - 0.12) / 0.22))
    a1 = ease_out(clamp01((lt - 0.30) / 0.30))
    a2 = ease_out(clamp01((lt - 0.52) / 0.30))
    person(z, (NV_BOY_X, feet_y(NV_FEET, NV_BOY_SC)), sc=NV_BOY_SC, a=1.0)
    qfigure(z, NV_UNK_X, NV_FEET, sc=NV_BOY_SC, a=fade)
    fx, fy = NV_FORK
    z.use("w")
    z.line([(fx, fy), (fx + 100 * stem, fy)], _col(WHITE, stem), 6)
    top_c = NV_OPT_TOP_C + (FIG_TOP - FIG_BOT) * NV_OPT_SC / 2
    bot_c = NV_OPT_BOT_C + (FIG_TOP - FIG_BOT) * NV_OPT_SC / 2
    arrow(z, (fx + 100, fy), (NV_ARROW_X, NV_OPT_TOP_C + 30), prog=a1, a=a1)
    arrow(z, (fx + 100, fy), (NV_ARROW_X, NV_OPT_BOT_C - 30), prog=a2, a=a2)
    person(z, (NV_OPT_X, top_c), sc=NV_OPT_SC, a=a1, girl=False)
    person(z, (NV_OPT_X, bot_c), sc=NV_OPT_SC, a=a2, girl=True)
    lay.alpha_composite(z.out())


def g_half1(lay, lt):
    """«получается 1/2»: два варианта складываются в две пары, одна из них —
    искомая. Белое число резким попом (R5a) — это промежуточный, неверный
    ответ, поэтому оно белое, а не синее."""
    z = Z()
    c1 = ease_out(clamp01(lt / 0.26))
    c2 = ease_out(clamp01((lt - 0.16) / 0.26))
    cell(z, H1_CELLS[0], OUTCOMES[0], a=c1, fill=c1)
    cell(z, H1_CELLS[1], OUTCOMES[1], a=c2, fill=c2)
    if c1 > 0.9:                                 # искомая пара — жирнее обводка
        z.use("w")
        z.rrect(H1_CELLS[0], CELL_W, CELL_H, CELL_R,
                outline=_col(WHITE, 0.95), w=7)
    lay.alpha_composite(z.out())
    q = clamp01((lt - 0.576) / 0.10)             # 3 кадра, резкий поп (R5a)
    if q > 0:
        txt = f"{ANSWER_NAIVE[0]}/{ANSWER_NAIVE[1]}"
        draw_number(lay, txt, H1_NUM_XY, num_size(txt, H1_NUM_XY, 150),
                    color=WHITE, glow=WHITE, glow_a=0.55,
                    scale=0.6 + 0.4 * ease_out(q))


def g_grid(lay, lt):
    """«возможны 4 равновероятных варианта»: четыре пустые ячейки каскадом.
    Четвёрку несёт сама сетка ячеек, числа на экране нет."""
    z = Z()
    s = stagger(lt, 4, t0=0.05, step=0.11, dur=0.28)
    grid_cells(z, [dict(a=v, fill=0.0) for v in s])
    lay.alpha_composite(z.out())


def _fill_states(done, extra=None):
    st = []
    for i in range(4):
        d = dict(a=1.0, fill=done[i])
        if extra and i in extra:
            d.update(extra[i])
        st.append(d)
    return st


def g_fill1(lay, lt):
    """«мальчик-мальчик, мальчик-девочка»: первые две пары встают в ячейки
    ровно на своих словах. Перечисление несёт графика, субтитра здесь нет."""
    z = Z()
    f0 = ease_out(clamp01((lt - 0.05) / 0.26))
    f1 = ease_out(clamp01((lt - 0.75) / 0.26))
    grid_cells(z, _fill_states([f0, f1, 0.0, 0.0]))
    lay.alpha_composite(z.out())


def g_fill2(lay, lt):
    """«девочка-мальчик и девочка-девочка»: две оставшиеся пары."""
    z = Z()
    f2 = ease_out(clamp01(lt / 0.26))
    f3 = ease_out(clamp01((lt - 0.98) / 0.26))
    grid_cells(z, _fill_states([1.0, 1.0, f2, f3]))
    lay.alpha_composite(z.out())


def g_drop_dd(lay, lt):
    """«последний вариант нам точно не подходит»: белый крест на «девочка-
    девочка» — единственном исходе, где мальчика нет вообще."""
    z = Z()
    xx = ease_out(clamp01((lt - 0.55) / 0.32))
    dim = 1.0 - 0.55 * ease_out(clamp01((lt - 0.95) / 0.50))
    grid_cells(z, _fill_states([1.0, 1.0, 1.0, dim],
                               {3: dict(a=dim, xx=xx)}))
    lay.alpha_composite(z.out())


def g_three(lay, lt):
    """«остаются три варианта»: отпавший исход догорает до призрака,
    три оставшихся разгораются. Счёт несут ячейки, числа на экране нет."""
    z = Z()
    k = ease_out(clamp01(lt / 0.42))
    ghost = 0.45 - 0.33 * k
    up = 0.93 + 0.07 * math.sin(lt * 4.2)
    grid_cells(z, _fill_states([1.0, 1.0, 1.0, ghost],
                               {0: dict(a=up), 1: dict(a=up), 2: dict(a=up),
                                3: dict(a=ghost, xx=1.0 - k)}))
    lay.alpha_composite(z.out())


def g_one_of_three(lay, lt):
    """«и только в одном из них оба ребёнка мальчики»: ячейка-победитель
    загорается синим, два других исхода приглушаются. Синее = ответ."""
    z = Z()
    b = ease_out(clamp01(lt / 0.40))
    dim = 0.48 - 0.10 * math.sin(lt * 2.1)
    pulse = 0.88 + 0.12 * math.sin(lt * 2.6)
    grid_cells(z, _fill_states([0.0, 1.0, 1.0, 0.0],
                               {0: dict(a=0.0), 1: dict(a=dim), 2: dict(a=dim),
                                3: dict(a=0.10, fill=0.10)}))
    if b > 0.02:
        cell(z, CELLS[0], OUTCOMES[0], a=b * pulse, blue=True, fill=b)
    lay.alpha_composite(z.out())


def g_third_num(lay, lt):
    """«вероятность равна 1/3» — синий ревил (R5b): масштаб 0.55 -> 1.0
    с микро-оверщутом, разряды слева направо. Ячеек в кадре нет: число
    не должно лежать на объекте своего же цвета."""
    p = clamp01(lt / 0.38)
    e = ease_out(p)
    sc = 0.55 + 0.45 * e + 0.03 * math.sin(math.pi * min(1.0, p / 0.85))
    txt = f"{ANSWER_ATLEAST[0]}/{ANSWER_ATLEAST[1]}"
    draw_number(lay, txt, NUM_XY, num_size(txt, NUM_XY, 240),
                color=BLUE, glow=BLUE_GLOW, glow_a=0.85,
                scale=sc, reveal=clamp01(lt / 0.30))


def g_elder(lay, lt):
    """«если бы сказали, что старший мальчик»: условие меняется, поэтому
    все четыре исхода возвращаются в игру, а синее кольцо встаёт на левой
    (старшей) фигурке — теперь известно именно про неё."""
    z = Z()
    back = ease_out(clamp01(lt / 0.30))
    mk = ease_out(clamp01((lt - 0.34) / 0.34))
    grid_cells(z, _fill_states([1.0, 1.0, 1.0, back],
                               {0: dict(mark=mk), 1: dict(mark=mk),
                                2: dict(mark=mk), 3: dict(a=back, mark=mk * back)}))
    lay.alpha_composite(z.out())


def g_elder_two(lay, lt):
    """«тогда бы осталось только два варианта»: исходы, где старший —
    девочка, гаснут под крестом. Две ячейки в кадре и есть ответ «два»."""
    z = Z()
    xx = ease_out(clamp01((lt - 0.12) / 0.34))
    dim = 1.0 - 0.72 * ease_out(clamp01((lt - 0.52) / 0.55))
    grid_cells(z, _fill_states([1.0, 1.0, dim, dim],
                               {0: dict(mark=1.0), 1: dict(mark=1.0),
                                2: dict(a=dim, xx=xx, mark=dim),
                                3: dict(a=dim, xx=xx, mark=dim)}))
    lay.alpha_composite(z.out())


def g_half_blue(lay, lt):
    """«и ответ был бы действительно 1/2» — второй синий ревил (R5b).
    Пара ячеек уходит раньше, чем приходит число: синее число не ложится
    на синее кольцо."""
    z = Z()
    fade = 1.0 - ease_out(clamp01((lt - 0.15) / 0.55))
    if fade > 0.02:
        grid_cells(z, _fill_states([fade, fade, 0.0, 0.0],
                                   {0: dict(a=fade, mark=fade),
                                    1: dict(a=fade, mark=fade),
                                    2: dict(a=0.0), 3: dict(a=0.0)}))
    lay.alpha_composite(z.out())
    p = clamp01((lt - 0.999) / 0.38)
    if p > 0:
        e = ease_out(p)
        sc = 0.55 + 0.45 * e + 0.03 * math.sin(math.pi * min(1.0, p / 0.85))
        txt = f"{ANSWER_ELDER[0]}/{ANSWER_ELDER[1]}"
        draw_number(lay, txt, NUM_XY, num_size(txt, NUM_XY, 240),
                    color=BLUE, glow=BLUE_GLOW, glow_a=0.85,
                    scale=sc, reveal=clamp01((lt - 0.999) / 0.30))


def g_big_word(lt):
    """R6: крупное слово поверх карточки A, брендовая позиция y≈440."""
    text = BIG_WORD
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
    "family": g_family, "atLeast": g_at_least, "ask": g_ask,
    "naive1": g_naive1, "naive2": g_naive2, "half1": g_half1,
    "grid": g_grid, "fill1": g_fill1, "fill2": g_fill2,
    "dropDD": g_drop_dd, "three": g_three, "oneOfThree": g_one_of_three,
    "thirdNum": g_third_num, "elder": g_elder, "elderTwo": g_elder_two,
    "halfBlue": g_half_blue,
}


def graphics_layer(kind, lt):
    if kind == "A1word":
        return g_big_word(lt)
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
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


# --- сток --------------------------------------------------------------------

STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/36/stock"
STOCK_GAIN = 0.72      # притемнение вставки на 28% по экспозиции (brand-kit §6)


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


def stock_card(fr, cx=0.5, cy=0.5):
    fr = fit_bgr(fr, CARD_B[2], CARD_B[3], cx, cy)
    fr = np.clip(fr.astype(np.float32) * STOCK_GAIN, 0, 255)
    hsv = cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255)
    fr = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


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
        # Длинный скачок — только seek'ом. Догонять чтением подряд нельзя:
        # guard=400 кадров не доставал до ss>13.3с, и первый кадр плана
        # показывал чужой момент клипа (было 13.31с вместо 16.60с) — один
        # кадр-вспышка на склейке, глазами в контакт-листе не ловится.
        if want < self.pos - 0.05 or want > self.pos + 1.0:
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


_STOCK = StockReader()


def background(kind, prm, fr, t, t0):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        card = source_card(fr, kind)
        canvas.paste(card, (CARD_A[0], CARD_A[1]), rounded_mask(CARD_A[2], CARD_A[3], R_A))
    elif kind == "stock":
        sf = _STOCK.frame(prm["clip"], prm.get("ss", 0.0), t - t0)
        card = stock_card(sf, prm.get("cx", 0.5), prm.get("cy", 0.5))
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


def compose(kind, prm, fr, t, t0):
    canvas = background(kind, prm, fr, t, t0)
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

    last = None
    for fno in range(nf):
        ok, fr = cap.read()
        if ok:
            last = fr
        else:
            fr = last
        t = fno / FPS
        t0, t1, kind, prm = shot_at(t)
        canvas = compose(kind, prm, fr, t, t0)
        enc.stdin.write(pil_to_cv(canvas.convert("RGB")).tobytes())
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}", flush=True)

    enc.stdin.close()
    enc.wait()
    cap.release()
    _STOCK.release()
    print("готово:", VID)


if __name__ == "__main__":
    main()

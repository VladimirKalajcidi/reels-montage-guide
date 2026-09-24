"""Сборка ролика 37 («парадокс трёх карт»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры. Аудио добавляет
sfx37.py, исходная речь не режется.

Материал: две литеральные стоковые вставки с настоящими картами
(`videos/37/stock/`), всё остальное — своя графика на сетке. Первая вставка
стоит ДО первой пиктограммы: пока не показали настоящую карту с красными
бубнами и чёрными пиками, схема из прямоугольников не опознаётся.

Задача цветная, а палитра бренда — чёрное/белое/синее. Красный на экран не
выносится: сторона кодируется мастью (♥ / ♠), то есть формой, и дублируется
яркостью половины (белая / чёрная). См. storyboard37.

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* межстрочный шаг считается от кеглей соседних строк;
* кегль числа подбирается под ширину зоны графики, а не по шкале XL;
* графика рисуется в отдельном буфере размером с зону (730x860) при ss=2 —
  свечение на границе зоны обрезается самим буфером;
* смена состояния половины («?» -> красная) делается резом, а не кроссфейдом;
* ридер стока встаёт на нужное время seek'ом, а не чтением подряд.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard37 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, CARDS, RR, BB, RB, ANSWER,
                          ANSWER_NAIVE, BIG_WORD, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (29 замеров по всему дублю: cx 378+-6, линия глаз y 684+-6, ширина лица
# 264+-6 — то же, что у ролика 36). Кроп 630x1000 ставит глаза на 42% высоты
# карточки — то же окно, что в роликах 22-36.
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
    и qa37.gfx_in_zone это ловит.
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
# Сцена задачи — карты и их стороны
# =============================================================================
# Буфер графики совпадает с зоной из brand-kit §5 (y 700...1560, x 175...905),
# поэтому свечение обрезается границей зоны, а не вылезает на субтитры.
ZX0, ZY0, ZW, ZH, SS = GX0, GY0, GX1 - GX0, GY1 - GY0, 2

# --- ряд карт «постановка задачи» (cards3 / card1 / card2 / card3a / card3b)
SET_XS = (306, 540, 774)
SET_Y = 1130
SET_W, SET_H = 210, 340

# --- одна карта на столе (lay / askBottom / naiveHalf)
BIG_C = (540, 1010)
BIG_W, BIG_H = 318, 500
WHITE_NUM_XY = (540, 1420)

# --- два кандидата ложной интуиции (twoCards)
TWO_XS = (400, 680)
TWO_Y = 1090
TWO_W, TWO_H = 272, 392

# --- диаграмма «карта -> её красные стороны»
DIA_XS = (320, 540, 760)
DIA_CARD_Y, DIA_CW, DIA_CH = 1340, 186, 268
DIA_TILE_Y, DIA_TW, DIA_TH = 866, 186, 134

# --- сравнение «две стороны против одной» (twiceMore / waysRed)
CMP_CARD_Y, CMP_CW, CMP_CH = 1340, 176, 254
CMP_CARD_XS = (357, 742)
CMP_TILE_Y, CMP_TW, CMP_TH = 866, 132, 98
CMP_TILE_XS = ((270, 444), (742,))

BLUE_NUM_XY = (540, 1130)

# какие плитки из каких карт: (индекс карты, индекс стороны 0=верх, 1=низ)
RED_SIDES = [(RR, 0), (RR, 1), (RB, 0)]

# Красный вне палитры brand-kit §4 и добавлен по прямому решению автора:
# в этом ролике цвет стороны — предмет задачи, и «красная сторона» рисуется
# красной, а не белой. Масть ♥/♠ при этом остаётся: она держит смысл в
# миниатюре и на приглушённых планах, где заливка гаснет по альфе.
RED = (226, 48, 48)
RED_GLOW = (255, 107, 107)


class Z:
    """Три буфера — под белое свечение, под красное и под синее; координаты
    холста. Порядок наложения: красное (заливки сторон) -> белое (каркас карт,
    линии, чёрные стороны) -> синее (ответ), поэтому белая обводка всегда
    поверх красной заливки, а не тонет в её свечении."""

    def __init__(self):
        self.ims = {"w": Image.new("RGBA", (ZW * SS, ZH * SS), (0, 0, 0, 0)),
                    "r": Image.new("RGBA", (ZW * SS, ZH * SS), (0, 0, 0, 0)),
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

    def rrect(self, c, ww, hh, rad, fill=None, outline=None, w=1):
        x, y = self.p(*c)
        self.d.rounded_rectangle([x - ww * SS / 2, y - hh * SS / 2,
                                  x + ww * SS / 2, y + hh * SS / 2],
                                 radius=max(0, rad) * SS, fill=fill, outline=outline,
                                 width=max(1, int(round(w * SS))))

    def rbox(self, box, rad, fill=None, square=None, outline=None, w=1):
        """Скруглённый прямоугольник по координатам холста; square='top'/'bottom'
        закрашивает соответствующие углы обратно в прямые — так собирается
        половина карты, у которой скруглены только внешние углы."""
        x0, y0 = self.p(box[0], box[1])
        x1, y1 = self.p(box[2], box[3])
        r = max(0.0, rad) * SS
        self.d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill)
        if square == "bottom":
            self.d.rectangle([x0, y1 - r, x1, y1], fill=fill)
        elif square == "top":
            self.d.rectangle([x0, y0, x1, y0 + r], fill=fill)
        if outline is not None:
            self.d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=None,
                                     outline=outline, width=max(1, int(round(w * SS))))

    def glyph(self, c, target_h, ch, col):
        """Глиф, отцентрованный по РЕАЛЬНЫМ чернилам, а не по em-квадрату."""
        b0 = font("sans", 100).getbbox(ch)
        h100 = max(1, b0[3] - b0[1])
        size = max(8, int(round(target_h * 100.0 / h100 * SS)))
        f = font("sans", size)
        b = f.getbbox(ch)
        px, py = self.p(*c)
        self.d.text((px - (b[0] + b[2]) / 2, py - (b[1] + b[3]) / 2), ch, font=f, fill=col)

    def out(self):
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for k, glow, strength in (("r", RED_GLOW, 0.55), ("w", WHITE, 0.36),
                                  ("b", BLUE_GLOW, 0.82)):
            im = self.ims[k].resize((ZW, ZH), Image.LANCZOS)
            if im.getbbox() is None:
                continue
            comp = _glow_sub(im, 11, glow, strength)
            lay.alpha_composite(comp, (ZX0, ZY0))
        return lay


def _mix(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# --- элементы сцены ----------------------------------------------------------
# Половина карты: белая с чёрным ♥ = красная сторона, чёрная с белым ♠ =
# чёрная сторона, «?» = сторона ещё не известна. Красного в палитре нет
# (brand-kit §4), поэтому цвет кодируется мастью и яркостью половины.
HEART, SPADE = "♥", "♠"
INSET = 5           # поле между заливкой половины и обводкой карты
PIP_K = 0.52        # высота масти от высоты половины


def _side_fill(z, box, kind, a, blue, square, pip_h):
    """Заливка одной половины карты + масть по центру.

    Красная сторона заливается красным, чёрная — чёрным: цвет стороны и есть
    предмет задачи. Масть остаётся поверх заливки — она читается там, где
    заливка приглушена по альфе. Синий (ответ) перебивает обе заливки, но
    масть под ним не меняется, поэтому сторона не теряет опознания.
    """
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    if kind == "R":
        if blue:
            # масть на синей заливке остаётся БЕЛОЙ, как на красной: иначе при
            # подсветке ответа ♥ перекрашивается из белого в чёрный и мигает
            z.use("b")
            z.rbox(box, 12, fill=_col(BLUE, a), square=square)
            z.glyph((cx, cy), pip_h, HEART, _col(WHITE, a))
        else:
            z.use("r")
            z.rbox(box, 12, fill=_col(RED, a), square=square)
            z.glyph((cx, cy), pip_h, HEART, _col(WHITE, a))
    elif kind == "B":
        z.use("b" if blue else "w")
        bright = BLUE_GLOW if blue else WHITE
        # рамка обязательна: рядом с красной заливкой чистая чёрная половина
        # читается дырой в кадре, а не стороной карты
        z.rbox(box, 12, fill=_col(BLACK, a), square=square,
               outline=_col(bright, a * 0.85), w=3)
        z.glyph((cx, cy), pip_h, SPADE, _col(bright, a))
    elif kind == "?":
        z.use("b" if blue else "w")
        z.glyph((cx, cy), pip_h * 1.06, "?", _col(BLUE_GLOW if blue else WHITE, a))


def card(z, c, w, h, top, bot, a=1.0, blue=False, top_a=1.0, bot_a=1.0,
         outline_a=1.0):
    """Карта: скруглённый прямоугольник, разделённый чертой на верхнюю и
    нижнюю сторону. top/bot: None | 'R' | 'B' | '?'."""
    if a <= 0.004 or w <= 2:
        return
    bright = BLUE_GLOW if blue else WHITE
    x, y = c
    x0, y0, x1, y1 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    rad = w * 0.10
    pip_h = (h / 2 - 2 * INSET) * PIP_K
    _side_fill(z, (x0 + INSET, y0 + INSET, x1 - INSET, y - 1.5), top,
               a * top_a, blue, "bottom", pip_h)
    _side_fill(z, (x0 + INSET, y + 1.5, x1 - INSET, y1 - INSET), bot,
               a * bot_a, blue, "top", pip_h)
    z.use("b" if blue else "w")
    z.rrect(c, w, h, rad, outline=_col(bright, a * outline_a), w=4)
    z.line([(x0 + INSET, y), (x1 - INSET, y)], _col(bright, a * 0.9), 3)


def tile(z, c, w, h, kind, a=1.0, blue=False, k=1.0):
    """Одна сторона, вынутая из карты: «вот такую сторону мы могли увидеть»."""
    if a <= 0.004 or k <= 0.02:
        return
    bright = BLUE_GLOW if blue else WHITE
    w, h = w * k, h * k
    x, y = c
    box = (x - w / 2 + INSET, y - h / 2 + INSET, x + w / 2 - INSET, y + h / 2 - INSET)
    _side_fill(z, box, kind, a, blue, None, (h - 2 * INSET) * PIP_K)
    z.use("b" if blue else "w")
    z.rrect(c, w, h, w * 0.10, outline=_col(bright, a), w=4)


def link(z, p0, p1, prog=1.0, a=1.0, blue=False, w=4):
    """Линия «эта плитка вынута из этой карты»."""
    if a <= 0.004 or prog <= 0.02:
        return
    z.use("b" if blue else "w")
    z.line([p0, _mix(p0, p1, prog)], _col(BLUE_GLOW if blue else WHITE, a), w)


def arrow(z, p0, p1, prog=1.0, a=1.0, blue=False, w=5, head=20):
    if a <= 0.004 or prog <= 0.02:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    q1 = _mix(p0, p1, prog)
    z.line([p0, q1], col, w)
    if prog > 0.9:
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        for s in (-1, 1):
            a2 = ang + s * 2.5
            z.line([q1, (q1[0] + head * math.cos(a2), q1[1] + head * math.sin(a2))],
                   col, w)


def _dia_cards(z, alphas, blue=(False, False, False), halves=None):
    """Нижний ряд диаграммы: три карты задачи."""
    for i in range(3):
        ha = (1.0, 1.0) if halves is None else halves[i]
        card(z, (DIA_XS[i], DIA_CARD_Y), DIA_CW, DIA_CH, CARDS[i][0], CARDS[i][1],
             a=alphas[i], blue=blue[i], top_a=ha[0], bot_a=ha[1])


def _dia_link(z, i, prog=1.0, a=1.0, blue=False):
    """Линия от карты-источника плитки i к самой плитке."""
    ci = RED_SIDES[i][0]
    link(z, (DIA_XS[ci], DIA_CARD_Y - DIA_CH / 2),
         (DIA_XS[i], DIA_TILE_Y + DIA_TH / 2), prog=prog, a=a, blue=blue)


# --- планы -------------------------------------------------------------------

def g_cards3(lay, lt):
    """«итак, перед вами три карты»: три карты рубашкой к нам — стороны ещё
    не названы, поэтому обе половины пустые. Счёт несут сами карточки."""
    z = Z()
    s = stagger(lt, 3, t0=0.05, step=0.17, dur=0.30)
    for i in range(3):
        card(z, (SET_XS[i], SET_Y), SET_W, SET_H, None, None, a=s[i])
    lay.alpha_composite(z.out())


def _setup_row(z, states, focus):
    for i in range(3):
        a = 1.0 if i == focus else 0.70
        card(z, (SET_XS[i], SET_Y), SET_W, SET_H, states[i][0], states[i][1],
             a=a, top_a=states[i][2], bot_a=states[i][3])


def g_card1(lay, lt):
    """«первая красная с обеих сторон»: у первой карты загораются обе
    половины — белая с ♥ сверху и такая же снизу."""
    z = Z()
    t1 = ease_out(clamp01((lt - 0.05) / 0.26))
    t2 = ease_out(clamp01((lt - 0.55) / 0.26))
    _setup_row(z, [("R", "R", t1, t2), (None, None, 0, 0), (None, None, 0, 0)], 0)
    lay.alpha_composite(z.out())


def g_card2(lay, lt):
    """«вторая чёрная с обеих сторон»: чёрная половина — чёрная заливка
    с белым ♠, она гасит клетку сетки под собой."""
    z = Z()
    t1 = ease_out(clamp01((lt - 0.05) / 0.26))
    t2 = ease_out(clamp01((lt - 0.60) / 0.26))
    _setup_row(z, [("R", "R", 1, 1), ("B", "B", t1, t2), (None, None, 0, 0)], 1)
    lay.alpha_composite(z.out())


def g_card3a(lay, lt):
    """«а третья красная с одной стороны»: у третьей карты появляется только
    верхняя половина."""
    z = Z()
    t1 = ease_out(clamp01((lt - 0.10) / 0.28))
    _setup_row(z, [("R", "R", 1, 1), ("B", "B", 1, 1), ("R", "B", t1, 0)], 2)
    lay.alpha_composite(z.out())


def g_card3b(lay, lt):
    """«и чёрная с другой стороны»: у третьей карты дорисовывается нижняя,
    чёрная половина."""
    z = Z()
    t2 = ease_out(clamp01((lt - 0.05) / 0.28))
    _setup_row(z, [("R", "R", 1, 1), ("B", "B", 1, 1), ("R", "B", 1, t2)], 2)
    lay.alpha_composite(z.out())


LAY_RED_AT = 0.945          # t=15.245, слово «красную»


def _lay_card(z, lt, blue=False):
    a = ease_out(clamp01(lt / 0.24))
    q = ease_out(clamp01((lt - 0.10) / 0.22))
    if lt < LAY_RED_AT:
        # «?» и красная сторона не смешиваются на полупрозрачности: смена
        # состояния — резом, как обычный монтажный рез
        card(z, BIG_C, BIG_W, BIG_H, "?", "?", a=a, top_a=q, bot_a=q, blue=blue)
    else:
        r = ease_out(clamp01((lt - LAY_RED_AT) / 0.16))
        card(z, BIG_C, BIG_W, BIG_H, "R", "?", a=1.0, top_a=r, bot_a=1.0, blue=blue)


def g_lay(lay, lt):
    """«вы кладёте карту и видите сверху красную сторону»: карта на столе,
    верхняя половина открывается красной, нижняя остаётся под знаком «?»."""
    z = Z()
    _lay_card(z, lt)
    lay.alpha_composite(z.out())


def g_ask_bottom(lay, lt):
    """«что снизу будет тоже красный цвет?»: нижняя половина под вопросом,
    вокруг неё пульсирует рамка. Подсказки, что там окажется, на кадре нет."""
    z = Z()
    card(z, BIG_C, BIG_W, BIG_H, "R", "?", a=1.0)
    ring = ease_out(clamp01((lt - 0.12) / 0.30))
    if ring > 0.02:
        puls = 0.72 + 0.28 * (0.5 + 0.5 * math.sin(lt * 5.0))
        z.use("w")
        z.rrect((BIG_C[0], BIG_C[1] + BIG_H / 4 + 1),
                BIG_W - 2 * INSET - 6, BIG_H / 2 - 2 * INSET - 6, 14,
                outline=_col(WHITE, ring * puls), w=5)
    lay.alpha_composite(z.out())


NAIVE_NUM_AT = 0.927        # t=20.894, слово «1»


def g_naive_half(lay, lt):
    """«кажется очевидным ответ 1/2»: та же карта, под ней белое число
    резким попом (R5a). Ответ неверный, поэтому белое, а не синее."""
    z = Z()
    card(z, BIG_C, BIG_W, BIG_H, "R", "?", a=1.0)
    lay.alpha_composite(z.out())
    q = clamp01((lt - NAIVE_NUM_AT) / 0.10)          # 3 кадра, резкий поп
    if q > 0:
        txt = f"{ANSWER_NAIVE[0]}/{ANSWER_NAIVE[1]}"
        draw_number(lay, txt, WHITE_NUM_XY, num_size(txt, WHITE_NUM_XY, 130),
                    color=WHITE, glow=WHITE, glow_a=0.55,
                    scale=0.6 + 0.4 * ease_out(q))


def g_two_cards(lay, lt):
    """«либо красно-красная карта, либо красно-чёрная»: два кандидата, у
    обоих сверху красное. Ложная интуиция считает равновероятными их, а не
    стороны — отсюда и берётся 1/2."""
    z = Z()
    a1 = ease_out(clamp01((lt - 0.05) / 0.26))
    a2 = ease_out(clamp01((lt - 0.32) / 0.26))
    card(z, (TWO_XS[0], TWO_Y), TWO_W, TWO_H, "R", "R", a=a1)
    card(z, (TWO_XS[1], TWO_Y), TWO_W, TWO_H, "R", "B", a=a2)
    lay.alpha_composite(z.out())


def g_sides(lay, lt):
    """«посчитаем не карты, а красные стороны, которые могли увидеть»:
    три карты задачи, красные половины разгораются по очереди, чёрные гаснут."""
    z = Z()
    s = stagger(lt, 3, t0=0.10, step=0.30, dur=0.30)
    dim = 1.0 - 0.66 * ease_out(clamp01((lt - 0.20) / 0.45))
    halves = [(0.45 + 0.55 * s[0], 0.45 + 0.55 * s[1]),
              (dim, dim),
              (0.45 + 0.55 * s[2], dim)]
    _dia_cards(z, [1.0, 0.55, 1.0], halves=halves)
    lay.alpha_composite(z.out())


def g_sides_rr(lay, lt):
    """«у красно-красной карты их две»: из первой карты вверх выходят две
    линии и две плитки — две красные стороны, которые она даёт."""
    z = Z()
    _dia_cards(z, [1.0, 0.42, 0.46],
               halves=[(1.0, 1.0), (0.48, 0.48), (1.0, 0.48)])
    for i, t0 in ((0, 0.20), (1, 0.95)):
        p = ease_out(clamp01((lt - t0) / 0.34))
        _dia_link(z, i, prog=p, a=p)
        tile(z, (DIA_XS[i], DIA_TILE_Y), DIA_TW, DIA_TH, "R",
             a=ease_out(clamp01((lt - t0 - 0.20) / 0.26)))
    lay.alpha_composite(z.out())


def g_sides_rb(lay, lt):
    """«у красно-чёрной только одна»: из третьей карты выходит одна линия
    и одна плитка."""
    z = Z()
    _dia_cards(z, [0.44, 0.42, 1.0],
               halves=[(0.48, 0.48), (0.48, 0.48), (1.0, 1.0)])
    for i in (0, 1):
        _dia_link(z, i, prog=1.0, a=0.40)
        tile(z, (DIA_XS[i], DIA_TILE_Y), DIA_TW, DIA_TH, "R", a=0.70)
    p = ease_out(clamp01((lt - 0.45) / 0.34))
    _dia_link(z, 2, prog=p, a=p)
    tile(z, (DIA_XS[2], DIA_TILE_Y), DIA_TW, DIA_TH, "R",
         a=ease_out(clamp01((lt - 0.65) / 0.26)))
    lay.alpha_composite(z.out())


def g_three_sides(lay, lt):
    """«получается три возможных красных стороны»: карты уходят в фон, три
    плитки пересчитываются по очереди коротким импульсом. Числа на экране
    нет — счёт несут сами плитки."""
    z = Z()
    back = 1.0 - 0.60 * ease_out(clamp01(lt / 0.40))
    _dia_cards(z, [back, back * 0.6, back],
               halves=[(1.0, 1.0), (0.5, 0.5), (1.0, 0.5)])
    for i in range(3):
        _dia_link(z, i, prog=1.0, a=back * 0.8)
        p = clamp01((lt - 0.15 - i * 0.45) / 0.30)
        k = 1.0 + 0.07 * math.sin(math.pi * p) if 0 < p < 1 else 1.0
        tile(z, (DIA_XS[i], DIA_TILE_Y), DIA_TW, DIA_TH, "R", a=1.0, k=k)
    lay.alpha_composite(z.out())


BLUE_AT = 0.360             # t=39.060, слова «и две из них»


def g_two_of_three(lay, lt):
    """«и две из них принадлежат красно-красной карте»: две плитки, их линии
    и сама карта загораются синим. Синее = ответ."""
    z = Z()
    b = ease_out(clamp01((lt - BLUE_AT) / 0.34))
    puls = 0.90 + 0.10 * math.sin(lt * 2.6)
    _dia_cards(z, [0.42 * (1 - b), 0.28, 0.42],
               halves=[(1.0, 1.0), (0.55, 0.55), (1.0, 0.55)])
    if b > 0.02:
        card(z, (DIA_XS[RR], DIA_CARD_Y), DIA_CW, DIA_CH, "R", "R",
             a=b * puls, blue=True)
    for i in range(3):
        blue = i in (0, 1)
        _dia_link(z, i, prog=1.0, a=(b if blue else 0.40), blue=blue and b > 0.02)
        if blue and b <= 0.02:
            _dia_link(z, i, prog=1.0, a=0.40)
        tile(z, (DIA_XS[i], DIA_TILE_Y), DIA_TW, DIA_TH, "R",
             a=(1.0 - b) if blue else 0.62)
        if blue and b > 0.02:
            tile(z, (DIA_XS[i], DIA_TILE_Y), DIA_TW, DIA_TH, "R",
                 a=b * puls, blue=True)
    lay.alpha_composite(z.out())


def g_bottom_red(lay, lt):
    """«что снизу тоже красный цвет»: под каждой плиткой — вторая сторона
    ТОЙ ЖЕ карты. У двух синих плиток снизу опять ♥, у третьей ♠."""
    z = Z()
    gone = ease_out(clamp01(lt / 0.28))
    puls = 0.90 + 0.10 * math.sin(lt * 2.6)
    for i in range(3):
        blue = i in (0, 1)
        _dia_link(z, i, prog=1.0, a=0.40 * (1 - gone), blue=False)
        tile(z, (DIA_XS[i], DIA_TILE_Y), DIA_TW, DIA_TH, "R",
             a=(puls if blue else 0.72), blue=blue)
        p = ease_out(clamp01((lt - 0.15 - i * 0.22) / 0.30))
        arrow(z, (DIA_XS[i], DIA_TILE_Y + DIA_TH / 2 + 14),
              (DIA_XS[i], DIA_CARD_Y - DIA_TH / 2 - 16), prog=p,
              a=p, blue=blue)
        ci, si = RED_SIDES[i]
        other = CARDS[ci][1 - si]
        ta = ease_out(clamp01((lt - 0.32 - i * 0.22) / 0.26))
        tile(z, (DIA_XS[i], DIA_CARD_Y), DIA_TW, DIA_TH, other,
             a=ta * (puls if blue else 1.0), blue=blue)
    lay.alpha_composite(z.out())


def g_two_thirds(lay, lt):
    """«равна 2/3» — синий ревил (R5b): масштаб 0.55 -> 1.0 с микро-оверщутом,
    разряды слева направо. Плиток в кадре нет: число не должно лежать на
    объекте своего же цвета."""
    p = clamp01((lt - 0.434) / 0.38)
    if p <= 0:
        return
    e = ease_out(p)
    sc = 0.55 + 0.45 * e + 0.03 * math.sin(math.pi * min(1.0, p / 0.85))
    txt = f"{ANSWER[0]}/{ANSWER[1]}"
    draw_number(lay, txt, BLUE_NUM_XY, num_size(txt, BLUE_NUM_XY, 240),
                color=BLUE, glow=BLUE_GLOW, glow_a=0.85,
                scale=sc, reveal=clamp01((lt - 0.434) / 0.30))


def _cmp_scene(z, blue_left=0.0):
    """Слева красно-красная карта с двумя плитками, справа красно-чёрная
    с одной: два способа увидеть красное против одного."""
    puls = 0.92 + 0.08 * math.sin(blue_left * 3.0)
    card(z, (CMP_CARD_XS[0], CMP_CARD_Y), CMP_CW, CMP_CH, "R", "R",
         a=1.0 - blue_left)
    if blue_left > 0.02:
        card(z, (CMP_CARD_XS[0], CMP_CARD_Y), CMP_CW, CMP_CH, "R", "R",
             a=blue_left * puls, blue=True)
    card(z, (CMP_CARD_XS[1], CMP_CARD_Y), CMP_CW, CMP_CH, "R", "B", a=1.0)


def g_twice_more(lay, lt):
    """«красно-красная карта даёт в два раза больше способов»: слева две
    плитки, справа одна. Числа на экране нет — его несёт сравнение."""
    z = Z()
    _cmp_scene(z)
    for k, x in enumerate(CMP_TILE_XS[0]):
        p = ease_out(clamp01((lt - 0.15 - k * 0.28) / 0.30))
        link(z, (CMP_CARD_XS[0], CMP_CARD_Y - CMP_CH / 2),
             (x, CMP_TILE_Y + CMP_TH / 2), prog=p, a=p)
        tile(z, (x, CMP_TILE_Y), CMP_TW, CMP_TH, "R",
             a=ease_out(clamp01((lt - 0.30 - k * 0.28) / 0.26)))
    p = ease_out(clamp01((lt - 0.95) / 0.30))
    link(z, (CMP_CARD_XS[1], CMP_CARD_Y - CMP_CH / 2),
         (CMP_TILE_XS[1][0], CMP_TILE_Y + CMP_TH / 2), prog=p, a=p)
    tile(z, (CMP_TILE_XS[1][0], CMP_TILE_Y), CMP_TW, CMP_TH, "R",
         a=ease_out(clamp01((lt - 1.10) / 0.26)))
    lay.alpha_composite(z.out())


def g_ways_red(lay, lt):
    """«способов увидеть красную сторону»: две левые плитки и их карта
    загораются синим — это и есть двойка из ответа 2/3."""
    z = Z()
    b = ease_out(clamp01((lt - 0.20) / 0.34))
    puls = 0.90 + 0.10 * math.sin(lt * 2.6)
    _cmp_scene(z, blue_left=b)
    for x in CMP_TILE_XS[0]:
        link(z, (CMP_CARD_XS[0], CMP_CARD_Y - CMP_CH / 2),
             (x, CMP_TILE_Y + CMP_TH / 2), prog=1.0, a=max(1.0 - b, 0.0))
        if b > 0.02:
            link(z, (CMP_CARD_XS[0], CMP_CARD_Y - CMP_CH / 2),
                 (x, CMP_TILE_Y + CMP_TH / 2), prog=1.0, a=b, blue=True)
        tile(z, (x, CMP_TILE_Y), CMP_TW, CMP_TH, "R", a=max(1.0 - b, 0.0))
        if b > 0.02:
            tile(z, (x, CMP_TILE_Y), CMP_TW, CMP_TH, "R", a=b * puls, blue=True)
    link(z, (CMP_CARD_XS[1], CMP_CARD_Y - CMP_CH / 2),
         (CMP_TILE_XS[1][0], CMP_TILE_Y + CMP_TH / 2), prog=1.0, a=0.75)
    tile(z, (CMP_TILE_XS[1][0], CMP_TILE_Y), CMP_TW, CMP_TH, "R", a=0.75)
    lay.alpha_composite(z.out())


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
    "cards3": g_cards3, "card1": g_card1, "card2": g_card2,
    "card3a": g_card3a, "card3b": g_card3b,
    "lay": g_lay, "askBottom": g_ask_bottom, "naiveHalf": g_naive_half,
    "twoCards": g_two_cards,
    "sides": g_sides, "sidesRR": g_sides_rr, "sidesRB": g_sides_rb,
    "threeSides": g_three_sides, "twoOfThree": g_two_of_three,
    "bottomRed": g_bottom_red, "twoThirds": g_two_thirds,
    "twiceMore": g_twice_more, "waysRed": g_ways_red,
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

STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/37/stock"
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
        # Длинный скачок — только seek'ом (регрессия ролика 36): догонять
        # чтением подряд нельзя, иначе первый кадр плана показывает чужой
        # момент клипа — один кадр-вспышка ровно на склейке.
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
        """Закрыть клип И сбросить состояние.

        Без сброса `key` следующий вызов frame() с тем же клипом считает, что
        нужный файл уже открыт, и дёргает закрытый VideoCapture: наружу уходит
        последний удачно прочитанный кадр, один и тот же на весь план.
        Ловится тем, что покадровая разница по плану становится ровно 0.00.
        """
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.key = None
        self.last = None
        self.pos = -1.0


_STOCK = StockReader()


def background(kind, prm, fr, t, t0):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        card_img = source_card(fr, kind)
        canvas.paste(card_img, (CARD_A[0], CARD_A[1]),
                     rounded_mask(CARD_A[2], CARD_A[3], R_A))
    elif kind == "stock":
        sf = _STOCK.frame(prm["clip"], prm.get("ss", 0.0), t - t0)
        card_img = stock_card(sf, prm.get("cx", 0.5), prm.get("cy", 0.5))
        canvas.paste(card_img, (CARD_B[0], CARD_B[1]),
                     rounded_mask(CARD_B[2], CARD_B[3], R_B))
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

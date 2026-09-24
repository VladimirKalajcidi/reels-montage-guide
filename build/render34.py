"""Сборка ролика 34 («парадокс крокодила»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры. Аудио добавляет
sfx34.py, исходная речь не режется.

Материал: одна литеральная стоковая вставка (крокодил с раскрытой пастью,
mixkit 27364, скачан в `videos/34/stock/`) на слове «крокодил», всё остальное —
своя графика на сетке. Тема логическая: живого видео, буквально показывающего
«предсказание отца делает обещание невыполнимым», не существует, а случайная
перебивка тут ломает смысл (assets-manifest §2).

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
from storyboard34 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, CENTURIES, YEARS,
                          BRANCH_A, BRANCH_B, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (29 замеров по всему дублю: cx 377+-8, линия глаз y 684+-6, ширина лица 269+-7).
# Кроп 630x1000 ставит глаза на 42% высоты карточки — то же окно, что в
# роликах 22-33.
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


def stagger(lt, n, t0=0.10, step=0.09, dur=0.42):
    return [ease_out(clamp01((lt - t0 - i * step) / dur)) for i in range(n)]


# =============================================================================
# Сцена задачи — единственный графический объект ролика
# =============================================================================
# Буфер графики совпадает с зоной из brand-kit §5 (y 700...1560, x 175...905),
# поэтому свечение обрезается границей зоны, а не вылезает на субтитры.
ZX0, ZY0, ZW, ZH, SS = GX0, GY0, GX1 - GX0, GY1 - GY0, 2

# --- геометрия сцены ---------------------------------------------------------
# Крокодил слева, отец справа, дуга возврата между ними, облачко предсказания
# над отцом. Композиция не меняется ни на одном плане ролика.
JAW_H_UP = (285, 1272)           # угол пасти, внутренняя кромка верхней челюсти
JAW_H_LO = (285, 1277)           # то же для нижней
JAW_T_UP = (505, 1170)           # кончик верхней челюсти (внутренняя кромка)
JAW_T_LO = (505, 1340)           # кончик нижней челюсти
JAW_BACK = 200                   # где кончается туловище слева
JAW_EYE = (252, 1268)          # глаз позади угла пасти, на голове
JAW_NOSE = (476, 1176)         # ноздря у кончика верхней челюсти

CHILD_IN_JAW = (452, 1258)       # ребёнок внутри пасти, между челюстями
CHILD_SC = 1.9
FATHER_C = (805, 1270)
FATHER_SC = 4.2
CHILD_AT_FATHER = (706, 1300)

ARC_A = (526, 1230)              # дуга возврата: от пасти к отцу
ARC_B = (752, 1235)
ARC_BOW = -0.45

BUBBLE_C = (688, 990)            # облачко предсказания над отцом
BUBBLE_W, BUBBLE_H = 340, 200
BADGE_C = (806, 928)             # вердикт предсказания, верхний угол облачка
SYM_C = (676, 998)               # символ возврата внутри облачка
SYM_SC = 0.62

# --- план loop ---------------------------------------------------------------
PANEL_L = (352, 1140)
PANEL_R = (728, 1140)
PANEL_W, PANEL_H = 300, 250

# --- план ages ---------------------------------------------------------------
TICK_X0, TICK_X1 = 240, 840
TICK_Y = 1010
TICK_H = 72
YEARS_XY = (540, 1270)
YEARS_SIZE = 155


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

def person(z, c, sc=1.0, a=1.0, blue=False):
    """Фигурка человека: круг-голова + скруглённое тело. sc=1 -> ~37px."""
    if a <= 0.004 or sc <= 0.02:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    x, y = c
    z.circ((x, y - 9 * sc), 7 * sc, fill=col)
    z.rrect((x, y + 9 * sc), 21 * sc, 24 * sc, 9 * sc, fill=col)


def _teeth(z, p0, p1, n, side, col, ln=16, w=6.5):
    """Зубы вдоль кромки челюсти: n треугольников вершиной внутрь пасти."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L * side, dx / L * side
    for i in range(n):
        t = 0.20 + 0.72 * i / max(1, n - 1)
        bx, by = x0 + dx * t, y0 + dy * t
        z.poly([(bx - dx / L * w, by - dy / L * w),
                (bx + dx / L * w, by + dy / L * w),
                (bx + nx * ln, by + ny * ln)], col)


def jaws(z, a=1.0, blue=False):
    """Крокодил в профиль: туловище с гребнем, раскрытая пасть с зубами, глаз.

    Челюсти — сужающиеся к кончику полосы (не линии постоянной толщины):
    именно клин делает силуэт узнаваемым.
    """
    if a <= 0.004:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    hu, hl, tu, tl = JAW_H_UP, JAW_H_LO, JAW_T_UP, JAW_T_LO

    # верхняя челюсть: внутренняя кромка hu->tu, внешняя выше на 30 -> 15
    z.poly([(hu[0], hu[1] - 30), (tu[0], tu[1] - 15), tu, hu], col)
    # нижняя челюсть
    z.poly([hl, tl, (tl[0], tl[1] + 14), (hl[0], hl[1] + 28)], col)
    # кончики скругляем
    z.circ(((tu[0] + tu[0]) / 2, tu[1] - 7), 8, fill=col)
    z.circ((tl[0], tl[1] + 7), 7, fill=col)

    # туловище позади угла пасти + зубчатый гребень спины + лапа
    z.poly([(JAW_BACK, 1318), (JAW_BACK + 10, 1268), (JAW_BACK + 50, 1246),
            (hu[0] + 6, hu[1] - 30), (hl[0] + 6, hl[1] + 30),
            (JAW_BACK + 26, 1330)], col)
    for i in range(5):
        bx = JAW_BACK + 14 + i * 19
        by = 1272 - (bx - JAW_BACK - 14) * 0.30
        z.poly([(bx - 8, by), (bx + 8, by), (bx + 1, by - 19)], col)
    z.rrect((JAW_BACK + 46, 1340), 16, 44, 7, fill=col)

    _teeth(z, hu, tu, 6, +1, col)
    _teeth(z, hl, tl, 6, -1, col)
    z.circ(JAW_EYE, 13, fill=col)
    z.circ(JAW_EYE, 5.5, fill=_col(BLACK, a))
    z.circ(JAW_NOSE, 5, fill=_col(BLACK, a))


def arc_pts(p0, p1, bow=0.22, n=36, prog=1.0, start=0.0):
    """Квадратичная дуга от p0 к p1 с прогибом bow (доля длины)."""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    cx, cy = mx - dy / L * bow * L, my + dx / L * bow * L
    out = []
    for i in range(n + 1):
        t = start + (prog - start) * (i / n)
        u = 1 - t
        out.append((u * u * x0 + 2 * u * t * cx + t * t * x1,
                    u * u * y0 + 2 * u * t * cy + t * t * y1))
    return out


def arc_point(p0, p1, bow, t):
    return arc_pts(p0, p1, bow=bow, n=1, prog=t, start=t)[0]


def arrow(z, p0, p1, bow, prog=1.0, a=1.0, blue=False, w=8, head=26):
    """Дуга со стрелкой на конце — «вернуть ребёнка»."""
    if a <= 0.004 or prog <= 0.02:
        return
    pts = arc_pts(p0, p1, bow=bow, prog=prog)
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    z.line(pts, col, w)
    if prog > 0.88:
        q0, q1 = pts[-2], pts[-1]
        ang = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
        for s in (-1, 1):
            a2 = ang + s * 2.5
            z.line([q1, (q1[0] + head * math.cos(a2), q1[1] + head * math.sin(a2))],
                   col, w)


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


def check(z, c, r, a=1.0, w=9, blue=False, prog=1.0):
    """Галочка: короткое плечо вниз-вправо, длинное вверх-вправо."""
    if a <= 0.004 or prog <= 0.02:
        return
    x, y = c
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    p0 = (x - r, y + r * 0.05)
    p1 = (x - r * 0.30, y + r * 0.72)
    p2 = (x + r, y - r * 0.78)
    if prog < 0.45:
        k = prog / 0.45
        z.line([p0, _mix(p0, p1, k)], col, w)
    else:
        k = min(1.0, (prog - 0.45) / 0.55)
        z.line([p0, p1], col, w)
        z.line([p1, _mix(p1, p2, k)], col, w)


def return_symbol(z, c, sc=1.0, a=1.0, blue=False, prog=1.0, ride=None,
                  child=True):
    """Символ возврата: дуга со стрелкой + фигурка ребёнка на ней.

    Тот же символ живёт и в сцене (дуга ARC_A -> ARC_B), и внутри облачка
    предсказания, и в панелях петли — поэтому «предсказание совпало с
    реальностью» читается как одинаковая картинка в двух местах.
    """
    if a <= 0.004:
        return
    x, y = c
    p0 = (x - 118 * sc, y + 16 * sc)
    p1 = (x + 118 * sc, y + 28 * sc)
    arrow(z, p0, p1, ARC_BOW, prog=prog, a=a, blue=blue,
          w=max(3.5, 8 * sc), head=max(12, 26 * sc))
    if child and prog > 0.35:
        t = 0.5 if ride is None else ride
        q = arc_point(p0, p1, ARC_BOW, min(t, prog))
        person(z, (q[0], q[1] - 26 * sc), sc=1.5 * sc, a=a, blue=blue)


def bubble(z, a=1.0, blue=False, w=4):
    """Облачко предсказания над отцом, с хвостиком к его голове."""
    if a <= 0.004:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    z.rrect(BUBBLE_C, BUBBLE_W, BUBBLE_H, 36, outline=col, w=w)
    bx, by = BUBBLE_C
    y0 = by + BUBBLE_H / 2
    z.line([(bx + 60, y0 - 3), (bx + 118, y0 + 88)], col, w)
    z.line([(bx + 112, y0 - 3), (bx + 118, y0 + 88)], col, w)


def scene(z, a=1.0, jaw_a=None, child_at=None, child_a=1.0, father_a=None,
          bubble_a=0.0, bubble_content=None, bubble_cross=0.0,
          bubble_cross_blue=False, sym_ride=None,
          badge=None, badge_prog=1.0, badge_blue=True,
          arc_prog=0.0, arc_blue=False, arc_cross=0.0, arc_cross_blue=False):
    """Полная сцена: пасть с ребёнком, отец, облачко предсказания, дуга возврата.

    child_at: 'jaw' | 'father' | float 0..1 (положение на дуге возврата).
    bubble_content: None | '?' | 'symbol'
    badge: None | 'check' | 'cross'
    """
    jaws(z, a=a if jaw_a is None else jaw_a)
    person(z, FATHER_C, sc=FATHER_SC, a=a if father_a is None else father_a)

    if arc_prog > 0.02:
        arrow(z, ARC_A, ARC_B, ARC_BOW, prog=arc_prog, a=a, blue=arc_blue)
    if arc_cross > 0.02:
        c = arc_point(ARC_A, ARC_B, ARC_BOW, 0.5)
        cross(z, (c[0], c[1] - 4), 38, a=a, blue=arc_cross_blue, prog=arc_cross)

    if child_at == "jaw":
        person(z, CHILD_IN_JAW, sc=CHILD_SC, a=child_a)
    elif child_at == "father":
        person(z, CHILD_AT_FATHER, sc=CHILD_SC, a=child_a)
    elif isinstance(child_at, float):
        q = arc_point(ARC_A, ARC_B, ARC_BOW, clamp01(child_at))
        person(z, (q[0], q[1] - 40), sc=CHILD_SC, a=child_a)

    if bubble_a > 0.004:
        bubble(z, a=bubble_a)
        if bubble_content == "?":
            z.use("w")
            z.text((BUBBLE_C[0], BUBBLE_C[1] - 6), "?", 104, _col(WHITE, bubble_a))
        elif bubble_content == "symbol":
            return_symbol(z, SYM_C, sc=SYM_SC, a=bubble_a, prog=1.0,
                          ride=sym_ride)
            if bubble_cross > 0.02:
                cross(z, (SYM_C[0], SYM_C[1] + 4), 58, a=bubble_a, w=8,
                      blue=bubble_cross_blue, prog=bubble_cross)
        if badge == "check":
            check(z, BADGE_C, 28, a=bubble_a, w=9, blue=badge_blue,
                  prog=badge_prog)
        elif badge == "cross":
            cross(z, BADGE_C, 26, a=bubble_a, w=9, blue=badge_blue,
                  prog=badge_prog)


# --- планы -------------------------------------------------------------------

def g_ages(lay, lt):
    """«существует больше 2000 лет»: двадцать засечек = двадцать веков,
    потом белое 2000 резким попом (R5a). Число несёт график, не субтитр."""
    z = Z()
    pitch = (TICK_X1 - TICK_X0) / (CENTURIES - 1)
    for i in range(CENTURIES):
        p = ease_out(clamp01((lt - 0.05 - i * 0.017) / 0.20))
        if p <= 0.02:
            continue
        x = TICK_X0 + i * pitch
        h = TICK_H * (0.35 + 0.65 * p)
        z.use("w")
        z.line([(x, TICK_Y - h / 2), (x, TICK_Y + h / 2)], _col(WHITE, p), 7)
    lay.alpha_composite(z.out())
    q = clamp01((lt - 0.55) / 0.10)                    # 3 кадра, резкий поп (R5a)
    if q > 0:
        draw_number(lay, str(YEARS), YEARS_XY, YEARS_SIZE, color=WHITE,
                    glow=WHITE, glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


def g_rule(lay, lt):
    """«если ты правильно угадаешь ... верну» — сцена собирается, над отцом
    встаёт облачко с вопросом, потом галочка и дуга возврата: вот условие."""
    z = Z()
    s = stagger(lt, 3, t0=0.02, step=0.16, dur=0.30)
    b = ease_out(clamp01((lt - 0.80) / 0.28))
    badge = ease_out(clamp01((lt - 1.16) / 0.26))
    arc = ease_out(clamp01((lt - 1.44) / 0.40))
    scene(z, a=1.0, jaw_a=s[0], child_a=s[1], father_a=s[2], child_at="jaw",
          bubble_a=b, bubble_content="?",
          badge="check" if badge > 0.02 else None, badge_prog=badge,
          badge_blue=False, arc_prog=arc)
    lay.alpha_composite(z.out())


def g_promise(lay, lt):
    """«я его верну»: фигурка ребёнка едет по дуге из пасти к отцу."""
    z = Z()
    a = ease_out(clamp01(lt / 0.14))
    ride = ease_out(clamp01((lt - 0.22) / 0.62))
    at = "jaw" if ride <= 0.02 else ("father" if ride >= 0.995 else float(ride))
    scene(z, a=a, child_at=at, bubble_a=a, bubble_content="?",
          badge="check", badge_prog=1.0, badge_blue=False, arc_prog=1.0)
    lay.alpha_composite(z.out())


def g_predict(lay, lt):
    """«его не вернёшь»: в облачке проявляется тот же символ возврата
    и перечёркивается — это и есть предсказание отца."""
    z = Z()
    a = ease_out(clamp01(lt / 0.14))
    sym = ease_out(clamp01((lt - 0.16) / 0.26))
    xx = ease_out(clamp01((lt - 0.52) / 0.30))
    scene(z, a=a, child_at="jaw", bubble_a=a,
          bubble_content="symbol" if sym > 0.05 else None, bubble_cross=xx)
    lay.alpha_composite(z.out())


def g_case_a1(lay, lt):
    """Ветка A: «если крокодил не возвращает ребёнка» — дуга возврата
    перечёркнута, ребёнок остаётся в пасти."""
    z = Z()
    a = ease_out(clamp01(lt / 0.14))
    arc = ease_out(clamp01((lt - 0.18) / 0.34))
    xx = ease_out(clamp01((lt - 0.66) / 0.30))
    scene(z, a=a, child_at="jaw", bubble_a=a, bubble_content="symbol",
          bubble_cross=1.0, arc_prog=arc, arc_cross=xx)
    lay.alpha_composite(z.out())


def g_case_a2(lay, lt):
    """«отец угадал правильно»: в облачке и в сцене одна и та же картинка —
    перечёркнутый возврат. Оба креста загораются синим, синяя галочка = вывод."""
    z = Z()
    badge = ease_out(clamp01((lt - 0.30) / 0.32))
    blue = lt > 0.16
    pulse = 1.0 if not blue else (0.80 + 0.20 * math.sin((lt - 0.16) * 7.0))
    scene(z, a=1.0, child_at="jaw", bubble_a=1.0, bubble_content="symbol",
          bubble_cross=pulse if blue else 1.0, bubble_cross_blue=blue,
          arc_prog=1.0, arc_cross=pulse if blue else 1.0, arc_cross_blue=blue,
          badge="check" if badge > 0.02 else None, badge_prog=badge,
          badge_blue=True)
    lay.alpha_composite(z.out())


def g_must_return(lay, lt):
    """«по своему обещанию крокодил обязан его вернуть»: крест с дуги
    снимается, дуга становится синей, ребёнок едет к отцу — вывод."""
    z = Z()
    fade = 1.0 - ease_out(clamp01((lt - 0.10) / 0.24))
    ride = ease_out(clamp01((lt - 0.46) / 0.70))
    at = "jaw" if ride <= 0.02 else ("father" if ride >= 0.995 else float(ride))
    blue = lt > 0.34
    scene(z, a=1.0, child_at=at, bubble_a=1.0, bubble_content="symbol",
          bubble_cross=1.0, badge="check", badge_prog=1.0, badge_blue=True,
          arc_prog=1.0, arc_blue=blue, arc_cross=fade)
    lay.alpha_composite(z.out())


def g_case_b1(lay, lt):
    """Ветка B: «но если крокодил возвращает ребёнка» — белая посылка,
    ребёнок едет к отцу, у предсказания пока нет вердикта."""
    z = Z()
    a = ease_out(clamp01(lt / 0.14))
    ride = ease_out(clamp01((lt - 0.55) / 0.85))
    at = "jaw" if ride <= 0.02 else ("father" if ride >= 0.995 else float(ride))
    scene(z, a=a, child_at=at, bubble_a=a, bubble_content="symbol",
          bubble_cross=1.0, arc_prog=1.0)
    lay.alpha_composite(z.out())


def g_case_b2(lay, lt):
    """«предсказание отца было неправильным»: синий крест на облачке."""
    z = Z()
    badge = ease_out(clamp01((lt - 0.22) / 0.30))
    dim = 1.0 - 0.35 * ease_out(clamp01((lt - 0.30) / 0.40))
    scene(z, a=1.0, child_at="father", bubble_a=dim, bubble_content="symbol",
          bubble_cross=1.0, arc_prog=1.0,
          badge="cross" if badge > 0.02 else None, badge_prog=badge,
          badge_blue=True)
    lay.alpha_composite(z.out())


def g_case_b3(lay, lt):
    """«условия для возвращения не выполнены»: синий крест ложится на дугу,
    ребёнок возвращается в пасть — посылка ветки B отрицает сама себя."""
    z = Z()
    xx = ease_out(clamp01((lt - 0.20) / 0.32))
    back = ease_out(clamp01((lt - 0.86) / 0.62))
    pos = 1.0 - back
    at = "father" if back <= 0.02 else ("jaw" if back >= 0.995 else float(pos))
    scene(z, a=1.0, child_at=at, bubble_a=1.0, bubble_content="symbol",
          bubble_cross=1.0, badge="cross", badge_prog=1.0, badge_blue=True,
          arc_prog=1.0, arc_cross=xx, arc_cross_blue=True)
    lay.alpha_composite(z.out())


# --- петля -------------------------------------------------------------------

def panel(z, c, crossed, badge, a=1.0, sym_prog=1.0, badge_prog=1.0):
    """Панель состояния: символ возврата (перечёркнутый или нет) + вердикт."""
    if a <= 0.004:
        return
    z.use("w")
    z.rrect(c, PANEL_W, PANEL_H, 26, outline=_col(WHITE, a * 0.75), w=3)
    cx, cy = c
    return_symbol(z, (cx, cy - 34), sc=0.62, a=a, prog=sym_prog)
    if crossed and sym_prog > 0.6:
        cross(z, (cx, cy - 26), 58, a=a, w=8, prog=min(1.0, (sym_prog - 0.6) / 0.4))
    if badge == "check":
        check(z, (cx, cy + 74), 28, a=a, w=9, blue=True, prog=badge_prog)
    else:
        cross(z, (cx, cy + 74), 26, a=a, w=9, blue=True, prog=badge_prog)


def loop_arcs(z, top_prog=1.0, bot_prog=1.0, a=1.0, blue=True, head=None):
    """Две дуги между панелями: A->B сверху, B->A снизу. Замкнутый цикл."""
    top0 = (PANEL_L[0], PANEL_L[1] - PANEL_H / 2)
    top1 = (PANEL_R[0], PANEL_R[1] - PANEL_H / 2)
    bot0 = (PANEL_R[0], PANEL_R[1] + PANEL_H / 2)
    bot1 = (PANEL_L[0], PANEL_L[1] + PANEL_H / 2)
    arrow(z, top0, top1, -0.30, prog=top_prog, a=a, blue=blue, w=6)
    arrow(z, bot0, bot1, -0.30, prog=bot_prog, a=a, blue=blue, w=6)
    if head is not None:
        # бегущая точка по замкнутому маршруту: верх слева направо, низ обратно
        u = head % 1.0
        if u < 0.5:
            q = arc_point(top0, top1, -0.30, u * 2)
        else:
            q = arc_point(bot0, bot1, -0.30, (u - 0.5) * 2)
        z.use("b")
        z.circ(q, 11, fill=_col(BLUE_GLOW, a))


def g_loop(lay, lt):
    """«решение крокодила влияет на истинность предсказания»: два состояния
    и две стрелки между ними. Каждое состояние ведёт в другое — цикл."""
    z = Z()
    pl = ease_out(clamp01(lt / 0.30))
    pr = ease_out(clamp01((lt - 0.34) / 0.30))
    top = ease_out(clamp01((lt - 0.90) / 0.42))
    bot = ease_out(clamp01((lt - 1.42) / 0.42))
    panel(z, PANEL_L, crossed=True, badge="check", a=pl, sym_prog=pl,
          badge_prog=ease_out(clamp01((lt - 0.22) / 0.26)))
    panel(z, PANEL_R, crossed=False, badge="cross", a=pr, sym_prog=pr,
          badge_prog=ease_out(clamp01((lt - 0.56) / 0.26)))
    head = None if lt < 1.90 else (lt - 1.90) * 0.55
    loop_arcs(z, top_prog=top, bot_prog=bot, head=head)
    lay.alpha_composite(z.out())


def g_loop_spin(lay, lt):
    """«от которого зависит само решение»: содержимое панелей гаснет,
    остаются два узла и бесконечно бегущая по кольцу точка."""
    z = Z()
    fade = 1.0 - ease_out(clamp01(lt / 0.34))
    panel(z, PANEL_L, crossed=True, badge="check", a=fade)
    panel(z, PANEL_R, crossed=False, badge="cross", a=fade)
    node = ease_out(clamp01((lt - 0.10) / 0.30))
    z.use("b")
    for c in (PANEL_L, PANEL_R):
        z.circ(c, 30 * node, outline=_col(BLUE_GLOW, node), w=6)
    loop_arcs(z, head=0.6 + lt * 0.75, a=1.0)
    lay.alpha_composite(z.out())


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
    "ages": g_ages, "rule": g_rule, "promise": g_promise, "predict": g_predict,
    "caseA1": g_case_a1, "caseA2": g_case_a2, "mustReturn": g_must_return,
    "caseB1": g_case_b1, "caseB2": g_case_b2, "caseB3": g_case_b3,
    "loop": g_loop, "loopSpin": g_loop_spin,
}


def graphics_layer(kind, lt):
    if kind == "A1word":
        return g_word_paradox(lt)
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

STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/34/stock"
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

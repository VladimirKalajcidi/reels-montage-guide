"""Сборка ролика 35 («парадокс Рассела»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры. Аудио добавляет
sfx35.py, исходная речь не режется.

Материал: одна литеральная стоковая вставка (парикмахер за работой, mixkit
43222, скачан в `videos/35/stock/`) — она стоит ПЕРЕД пиктограммой парикмахера,
чтобы схему опознавали сразу. Всё остальное — своя графика на сетке: живого
видео, буквально показывающего «множество, содержащее само себя», не бывает.

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* межстрочный шаг считается от кеглей соседних строк;
* графика рисуется в отдельном буфере размером с зону (730x860) при ss=2 —
  свечение на границе зоны обрезается самим буфером;
* окружности и петли ищутся численно, чтобы целиком влезать в зону.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard35 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, BRANCH_A, BRANCH_B, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (29 замеров: cx 374+-9, линия глаз y 687+-6, ширина лица 271+-6) — то же
# окно, что в роликах 22-34: кроп 630x1000 ставит глаза на 42% высоты карточки.
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


def stagger(lt, n, t0=0.10, step=0.09, dur=0.42):
    return [ease_out(clamp01((lt - t0 - i * step) / dur)) for i in range(n)]


# =============================================================================
# Сцена задачи — буфер совпадает с зоной графики (brand-kit §5)
# =============================================================================
ZX0, ZY0, ZW, ZH, SS = GX0, GY0, GX1 - GX0, GY1 - GY0, 2

# --- геометрия --------------------------------------------------------------
# коробка = множество; точки внутри = элементы; копия коробки внутри = «содержит
# сам себя». Композиция не меняется от плана к плану.
SB_W, SB_H, SB_R = 170, 135, 20          # обычная коробка-множество

R_C = (540, 1150)                         # множество Рассела
R_W, R_H, R_RAD = 680, 520, 40
ROW1 = [(320, 1010), (540, 1010), (760, 1010)]   # коробки внутри R
ROW2 = [(340, 1295), (740, 1295)]                # самосодержащие (план boxes)
ROW2_W, ROW2_H = 190, 150

SLOT_C = (540, 1275)                      # место под копию самого R
SLOT_W, SLOT_H, SLOT_RAD = 300, 170, 26

RULE_C = (300, 800)                       # условие: самосодержащие не входят
RULE_W, RULE_H = 160, 128
BADGE_C = (790, 800)                      # вердикт по R (синий)

# --- план loop --------------------------------------------------------------
# рамок у состояний нет: две коробки R с вердиктами и две дуги между ними —
# так на каждое состояние остаётся вдвое больше места, чем в панели с рамкой.
STATE_L = (355, 1120)
STATE_R = (725, 1120)
STATE_W, STATE_H = 300, 200
STATE_BADGE_DY = 210
ARC_TOP_Y, ARC_BOT_Y = 1020, 1400

# --- планы barber -----------------------------------------------------------
BARBER_C = (300, 1150)
BARBER_SC = 4.2
CLIENTS = [(700, 900), (700, 1150), (700, 1400)]
CLIENT_SC = 3.0
SELF_SHAVER = 1                            # средний клиент бреется сам

# --- планы prop -------------------------------------------------------------
OBJ_Y = 790
OBJ_X = [300, 396, 492, 588, 684, 780]
OBJ_KIND = ["o", "t", "o", "t", "o", "t"]  # круги и треугольники вперемешку
BAR_C = (540, 920)
BAR_W, BAR_H = 480, 90
RES_C = (540, 1250)
RES_W, RES_H = 460, 330

# --- план axioms ------------------------------------------------------------
AX_Y = [1400, 1320, 1240]
AX_W, AX_H = 520, 44
AX_BOX_C = (540, 1128)
AX_BOX_W, AX_BOX_H = 300, 170
AX_CHECK_C = (540, 870)


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


# --- элементарные знаки ------------------------------------------------------

def sbox(z, c, w, h, rad, content=None, a=1.0, blue=False, lw=5,
         inner_a=1.0, scale=1.0):
    """Коробка-множество. content: None | 'dots' | 'self' | 'R' | 'q'.

    'self' — копия этой же коробки внутри неё: «множество содержит само себя».
    'R'    — три коробки внутри: содержимое множества Рассела.
    """
    if a <= 0.004 or scale <= 0.02:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    w, h = w * scale, h * scale
    z.rrect(c, w, h, rad * scale, outline=col, w=lw)
    if content is None or inner_a <= 0.004:
        return
    ic = _col(BLUE_GLOW if blue else WHITE, a * inner_a)
    x, y = c
    if content == "dots":
        for dx in (-0.22, 0.0, 0.22):
            z.circ((x + dx * w, y), 0.055 * w, fill=ic)
    elif content == "self":
        z.rrect(c, w * 0.48, h * 0.46, rad * 0.55 * scale, outline=ic, w=max(3, lw - 1))
    elif content == "R":
        for dx in (-0.31, 0.0, 0.31):
            z.rrect((x + dx * w, y), w * 0.24, h * 0.32, rad * 0.42 * scale,
                    outline=ic, w=max(3, lw - 1))
    elif content == "q":
        z.text((x, y - h * 0.04), "?", h * 0.62, ic)


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


def _head(z, q0, q1, col, w, head):
    ang = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
    for s in (-1, 1):
        a2 = ang + s * 2.5
        z.line([q1, (q1[0] + head * math.cos(a2), q1[1] + head * math.sin(a2))],
               col, w)


def arrow(z, p0, p1, bow, prog=1.0, a=1.0, blue=False, w=8, head=26):
    if a <= 0.004 or prog <= 0.02:
        return
    pts = arc_pts(p0, p1, bow=bow, prog=prog)
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    z.line(pts, col, w)
    if prog > 0.88:
        _head(z, pts[-2], pts[-1], col, w, head)


def self_loop(z, c, r, prog=1.0, a=1.0, blue=False, w=7, head=22, gap=0.9):
    """Петля «сам на себя»: почти замкнутое кольцо вокруг точки со стрелкой.

    Кольцо рисуется от -105 градусов по часовой на gap*360, чтобы начало и
    конец не сходились и стрелка читалась.
    """
    if a <= 0.004 or prog <= 0.02:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    a0 = math.radians(-105)
    span = math.radians(360 * gap) * min(1.0, prog)
    n = 48
    pts = [(c[0] + r * math.cos(a0 + span * i / n),
            c[1] + r * math.sin(a0 + span * i / n)) for i in range(n + 1)]
    z.line(pts, col, w)
    if prog > 0.9:
        _head(z, pts[-2], pts[-1], col, w, head)


def person(z, c, sc=1.0, a=1.0, blue=False):
    """Фигурка человека: круг-голова + скруглённое тело. sc=1 -> ~37px."""
    if a <= 0.004 or sc <= 0.02:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    x, y = c
    z.circ((x, y - 9 * sc), 7 * sc, fill=col)
    z.rrect((x, y + 9 * sc), 21 * sc, 24 * sc, 9 * sc, fill=col)


# --- сцена множества Рассела -------------------------------------------------

def russell_scene(z, a=1.0, row1=(1.0, 1.0, 1.0), box_a=1.0, box_sc=1.0,
                  slot="none", slot_a=1.0, slot_blue=False, slot_cross=0.0,
                  slot_cross_blue=True, rule_a=0.0, rule_cross=0.0,
                  badge=None, badge_prog=1.0, r_blue=False):
    """Коробка R с тремя обычными множествами внутри и местом под копию себя.

    slot: 'none' | 'empty' | 'q' | 'R'
    badge: None | 'check' | 'cross'  — вердикт по R, всегда синий (вывод)
    """
    if box_a > 0.004:
        sbox(z, R_C, R_W, R_H, R_RAD, a=a * box_a, blue=r_blue, lw=6,
             scale=box_sc)
    for i, c in enumerate(ROW1):
        sbox(z, c, SB_W, SB_H, SB_R, content="dots", a=a * row1[i])

    if slot != "none" and slot_a > 0.004:
        if slot == "empty":
            sbox(z, SLOT_C, SLOT_W, SLOT_H, SLOT_RAD, a=a * slot_a * 0.62, lw=4)
        elif slot == "q":
            sbox(z, SLOT_C, SLOT_W, SLOT_H, SLOT_RAD, content="q",
                 a=a * slot_a * 0.62, lw=4, inner_a=1.6)
        elif slot == "R":
            sbox(z, SLOT_C, SLOT_W, SLOT_H, SLOT_RAD, content="R",
                 a=a * slot_a, blue=slot_blue, lw=5)
        if slot_cross > 0.02:
            cross(z, SLOT_C, 96, a=a * slot_a, w=8, blue=slot_cross_blue,
                  prog=slot_cross)

    if rule_a > 0.004:
        sbox(z, RULE_C, RULE_W, RULE_H, 20, content="self", a=a * rule_a, lw=5)
        if rule_cross > 0.02:
            cross(z, RULE_C, 68, a=a * rule_a, w=6, prog=rule_cross)

    if badge == "check":
        check(z, BADGE_C, 42, a=a, w=10, blue=True, prog=badge_prog)
    elif badge == "cross":
        cross(z, BADGE_C, 40, a=a, w=10, blue=True, prog=badge_prog)


# --- планы -------------------------------------------------------------------

def g_boxes(lay, lt):
    """«множество всех множеств, которые не содержат сами себя»: пять коробок.

    Внизу три обычных (точки-элементы), вверху две с копией себя внутри —
    видно, чем они отличаются, ещё до всякого отбора."""
    z = Z()
    s = stagger(lt, 5, t0=0.05, step=0.11, dur=0.32)
    for i, c in enumerate(ROW1):
        sbox(z, c, SB_W, SB_H, SB_R, content="dots", a=s[i],
             inner_a=ease_out(clamp01((lt - 0.25 - i * 0.11) / 0.26)))
    for j, c in enumerate(ROW2):
        p = s[3 + j]
        sbox(z, c, ROW2_W, ROW2_H, 22, content="self", a=p,
             inner_a=ease_out(clamp01((lt - 0.62 - j * 0.11) / 0.28)))
    lay.alpha_composite(z.out())


def g_boxes_a(lay, lt):
    """Первая половина плана `boxes` для длинных дублей хука: каскадом встают
    три обычные коробки-множества. Тот же объект и те же позиции, что в
    основном ролике, — просто пятёрка разнесена на два плана."""
    z = Z()
    s = stagger(lt, 3, t0=0.05, step=0.13, dur=0.34)
    for i, c in enumerate(ROW1):
        sbox(z, c, SB_W, SB_H, SB_R, content="dots", a=s[i],
             inner_a=ease_out(clamp01((lt - 0.28 - i * 0.13) / 0.28)))
    lay.alpha_composite(z.out())


def g_boxes_b(lay, lt):
    """Вторая половина: три обычные уже стоят, каскадом появляются две
    коробки с копией себя внутри — под «которые не содержат сами себя»,
    ровно как в основном ролике."""
    z = Z()
    for c in ROW1:
        sbox(z, c, SB_W, SB_H, SB_R, content="dots")
    s = stagger(lt, 2, t0=0.08, step=0.16, dur=0.34)
    for j, c in enumerate(ROW2):
        sbox(z, c, ROW2_W, ROW2_H, 22, content="self", a=s[j],
             inner_a=ease_out(clamp01((lt - 0.46 - j * 0.16) / 0.30)))
    lay.alpha_composite(z.out())


def g_russell(lay, lt):
    """«множество Рассела»: самосодержащие вычёркиваются и уходят, вокруг
    оставшихся собирается большая коробка R с пустым местом под копию себя."""
    z = Z()
    xx = ease_out(clamp01((lt - 0.10) / 0.28))
    fade = 1.0 - ease_out(clamp01((lt - 0.52) / 0.30))
    box = ease_out(clamp01((lt - 0.60) / 0.42))
    slot = ease_out(clamp01((lt - 1.02) / 0.30))
    for j, c in enumerate(ROW2):
        sbox(z, c, ROW2_W, ROW2_H, 22, content="self", a=fade)
        cross(z, c, 80, a=fade, w=7, prog=xx)
    russell_scene(z, box_a=box, box_sc=0.94 + 0.06 * box,
                  slot="empty" if slot > 0.02 else "none", slot_a=slot)
    lay.alpha_composite(z.out())


def g_ask(lay, lt):
    """«содержит ли оно само себя»: в пустом месте внутри R встаёт вопрос."""
    z = Z()
    q = ease_out(clamp01((lt - 0.15) / 0.30))
    russell_scene(z, slot="q" if q > 0.05 else "empty", slot_a=max(q, 0.25))
    lay.alpha_composite(z.out())


def g_case_a1(lay, lt):
    """Ветка A, посылка «если содержит»: на месте вопроса проявляется копия
    самого R — коробка с тем же содержимым."""
    z = Z()
    p = ease_out(clamp01((lt - 0.20) / 0.34))
    russell_scene(z, slot="R" if p > 0.05 else "empty", slot_a=max(p, 0.30))
    lay.alpha_composite(z.out())


def g_rule(lay, lt):
    """«туда должны входить только те, которые себя не содержат»: слева
    встаёт условие отбора — коробка с копией себя, перечёркнутая."""
    z = Z()
    r = ease_out(clamp01((lt - 0.05) / 0.30))
    xx = ease_out(clamp01((lt - 0.45) / 0.34))
    russell_scene(z, slot="R", rule_a=r, rule_cross=xx)
    lay.alpha_composite(z.out())


def g_case_a2(lay, lt):
    """Вывод ветки A: R с копией себя внутри — ровно то, что запрещено
    условием. Копия перечёркивается синим, справа синий крест-вердикт."""
    z = Z()
    xx = ease_out(clamp01((lt - 0.18) / 0.30))
    badge = ease_out(clamp01((lt - 0.52) / 0.30))
    pulse = 0.80 + 0.20 * math.sin(max(0.0, lt - 0.18) * 7.0)
    russell_scene(z, slot="R", rule_a=1.0, rule_cross=pulse,
                  slot_cross=xx, slot_cross_blue=True,
                  badge="cross" if badge > 0.02 else None, badge_prog=badge)
    lay.alpha_composite(z.out())


def g_case_a3(lay, lt):
    """«значит оно не содержит себя»: копия и крест уходят, место пустеет,
    контур R загорается синим — это вывод, а не предположение."""
    z = Z()
    fade = 1.0 - ease_out(clamp01((lt - 0.30) / 0.42))
    blue = lt > 0.72
    russell_scene(z, slot="R" if fade > 0.02 else "empty", slot_a=max(fade, 0.45),
                  slot_blue=True, slot_cross=fade, rule_a=1.0, rule_cross=1.0,
                  badge="cross", badge_prog=1.0, r_blue=blue)
    lay.alpha_composite(z.out())


def g_case_b1(lay, lt):
    """Ветка B: R без копии себя — точно такая же коробка, как те три внутри.
    Значит R подходит под собственное условие: синяя галочка."""
    z = Z()
    badge = ease_out(clamp01((lt - 0.70) / 0.32))
    pulse = 0.55 + 0.45 * abs(math.sin(lt * 2.2))
    russell_scene(z, slot="empty", slot_a=0.45, rule_a=1.0, rule_cross=pulse,
                  badge="check" if badge > 0.02 else None, badge_prog=badge)
    lay.alpha_composite(z.out())


def g_case_b2(lay, lt):
    """«и должно содержать себя»: копия R возвращается на место — уже синим,
    как вынужденный вывод. Посылка ветки B отрицает сама себя."""
    z = Z()
    p = ease_out(clamp01((lt - 0.15) / 0.36))
    russell_scene(z, slot="R" if p > 0.05 else "empty", slot_a=max(p, 0.45),
                  slot_blue=True, rule_a=1.0, rule_cross=1.0,
                  badge="check", badge_prog=1.0)
    lay.alpha_composite(z.out())


# --- петля -------------------------------------------------------------------

def state(z, c, has_copy, badge, a=1.0, prog=1.0, badge_prog=1.0):
    """Состояние: коробка R с копией себя внутри или без неё + синий вердикт."""
    if a <= 0.004:
        return
    cx, cy = c
    sbox(z, c, STATE_W, STATE_H, 28, a=a * prog, lw=5)
    for dx in (-85, 0, 85):
        sbox(z, (cx + dx, cy - 45), 66, 46, 10, a=a * prog, lw=3)
    if has_copy:
        sbox(z, (cx, cy + 45), 120, 54, 14, content="R", a=a * prog, lw=3)
    else:
        sbox(z, (cx, cy + 45), 120, 54, 14, a=a * prog * 0.45, lw=3)
    if badge == "check":
        check(z, (cx, cy + STATE_BADGE_DY), 34, a=a, w=9, blue=True, prog=badge_prog)
    else:
        cross(z, (cx, cy + STATE_BADGE_DY), 32, a=a, w=9, blue=True, prog=badge_prog)


def loop_arcs(z, top_prog=1.0, bot_prog=1.0, a=1.0, head=None):
    """Две дуги между состояниями: A->B сверху, B->A снизу. Замкнутый цикл."""
    top0 = (STATE_L[0], ARC_TOP_Y)
    top1 = (STATE_R[0], ARC_TOP_Y)
    bot0 = (STATE_R[0], ARC_BOT_Y)
    bot1 = (STATE_L[0], ARC_BOT_Y)
    arrow(z, top0, top1, -0.30, prog=top_prog, a=a, blue=True, w=6)
    arrow(z, bot0, bot1, -0.30, prog=bot_prog, a=a, blue=True, w=6)
    if head is not None:
        u = head % 1.0
        if u < 0.5:
            q = arc_point(top0, top1, -0.30, u * 2)
        else:
            q = arc_point(bot0, bot1, -0.30, (u - 0.5) * 2)
        z.use("b")
        z.circ(q, 11, fill=_col(BLUE_GLOW, a))


def g_loop(lay, lt):
    """«противоречия при любом ответе»: слева R с копией себя (вердикт — крест),
    справа R без копии (вердикт — галочка). Каждый вывод ведёт в другую
    посылку, поэтому дуги замыкаются в цикл, по которому бежит точка."""
    z = Z()
    pl = ease_out(clamp01(lt / 0.30))
    pr = ease_out(clamp01((lt - 0.30) / 0.30))
    top = ease_out(clamp01((lt - 0.78) / 0.36))
    bot = ease_out(clamp01((lt - 1.14) / 0.36))
    state(z, STATE_L, True, "cross", a=pl, prog=pl,
          badge_prog=ease_out(clamp01((lt - 0.20) / 0.24)))
    state(z, STATE_R, False, "check", a=pr, prog=pr,
          badge_prog=ease_out(clamp01((lt - 0.50) / 0.24)))
    head = None if lt < 1.50 else (lt - 1.50) * 0.62
    loop_arcs(z, top_prog=top, bot_prog=bot, head=head)
    lay.alpha_composite(z.out())


# --- бытовая версия ----------------------------------------------------------

def barber_scene(z, arrows=(0.0, 0.0, 0.0), self_prog=0.0, ban_prog=0.0,
                 q_prog=0.0, a=1.0, people=(1.0, 1.0, 1.0), barber_a=1.0):
    """Парикмахер слева, три человека справа. Стрелка = «бреет его»,
    петля = «бреется сам». Тот же язык стрелок, что в сцене множеств."""
    person(z, BARBER_C, sc=BARBER_SC, a=a * barber_a)
    for i, c in enumerate(CLIENTS):
        person(z, c, sc=CLIENT_SC, a=a * people[i])
    for i, c in enumerate(CLIENTS):
        if arrows[i] > 0.02:
            arrow(z, (BARBER_C[0] + 58, BARBER_C[1] - 20), (c[0] - 52, c[1]),
                  0.16, prog=arrows[i], a=a, w=6, head=20)
    if self_prog > 0.02:
        c = CLIENTS[SELF_SHAVER]
        self_loop(z, (c[0] + 12, c[1] - 6), 78, prog=self_prog, a=a, w=7)
    if ban_prog > 0.02:
        c = CLIENTS[SELF_SHAVER]
        mid = arc_point((BARBER_C[0] + 58, BARBER_C[1] - 20), (c[0] - 52, c[1]),
                        0.16, 0.5)
        cross(z, mid, 34, a=a, w=8, prog=ban_prog)
    if q_prog > 0.02:
        self_loop(z, BARBER_C, 92, prog=min(1.0, q_prog * 1.5), a=a, w=8,
                  blue=True)
        z.use("b")
        z.text((BARBER_C[0], 985), "?", 96,
               _col(BLUE_GLOW, a * ease_out(clamp01((q_prog - 0.45) / 0.40))))


def g_barber1(lay, lt):
    """«бреет всех людей»: стрелки от парикмахера ко всем троим."""
    z = Z()
    s = stagger(lt, 4, t0=0.02, step=0.10, dur=0.28)
    ar = stagger(lt, 3, t0=0.30, step=0.11, dur=0.30)
    barber_scene(z, arrows=ar, barber_a=s[0], people=(s[1], s[2], s[3]))
    lay.alpha_composite(z.out())


def g_barber2(lay, lt):
    """«и только тех, которые не бреются сами»: у среднего появляется петля
    «сам себя», и стрелка от парикмахера к нему перечёркивается."""
    z = Z()
    sp = ease_out(clamp01((lt - 0.10) / 0.42))
    ban = ease_out(clamp01((lt - 0.66) / 0.30))
    barber_scene(z, arrows=(1.0, 1.0, 1.0), self_prog=sp, ban_prog=ban)
    lay.alpha_composite(z.out())


def g_barber3(lay, lt):
    """«должен ли парикмахер брить самого себя»: та же петля, но на нём,
    и знак вопроса — ровно тот же вопрос, что был к R."""
    z = Z()
    q = ease_out(clamp01((lt - 0.20) / 0.50))
    barber_scene(z, arrows=(0.75, 0.75, 0.75), self_prog=1.0, ban_prog=1.0,
                 q_prog=q)
    lay.alpha_composite(z.out())


# --- свойство и основания ----------------------------------------------------

def obj_glyph(z, c, kind, r, a=1.0, blue=False, fill=True):
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    x, y = c
    if kind == "o":
        z.circ(c, r, fill=col if fill else None,
               outline=None if fill else col, w=4)
    elif kind == "t":
        z.poly([(x, y - r), (x + r * 0.93, y + r * 0.7), (x - r * 0.93, y + r * 0.7)], col)
    elif kind == "s":
        z.rrect(c, r * 1.7, r * 1.7, r * 0.3, fill=col)


def prop_scene(z, glyph, obj_a=1.0, bar_a=1.0, res_a=1.0, filled=0.0,
               cross_prog=0.0, a=1.0):
    """Свойство в фильтре -> коробка, собранная из подходящих объектов.

    glyph: 'o' | 't' | 'self' — последнее и есть «не содержит сам себя».
    """
    for i, x in enumerate(OBJ_X):
        obj_glyph(z, (x, OBJ_Y), OBJ_KIND[i], 22, a=a * obj_a * 0.85)
    if bar_a > 0.004:
        z.use("w")
        z.rrect(BAR_C, BAR_W, BAR_H, 26, outline=_col(WHITE, a * bar_a), w=5)
        if glyph == "self":
            sbox(z, BAR_C, 96, 62, 12, content="self", a=a * bar_a, lw=4)
        else:
            obj_glyph(z, BAR_C, glyph, 26, a=a * bar_a)
    if res_a > 0.004:
        z.use("w")
        z.rrect(RES_C, RES_W, RES_H, 34, outline=_col(WHITE, a * res_a), w=6)
        n = 3
        for i in range(n):
            p = clamp01((filled - i * 0.22) / 0.30)
            if p <= 0.02:
                continue
            cx = RES_C[0] + (i - 1) * 130
            if glyph == "self":
                sbox(z, (cx, RES_C[1]), 108, 84, 14, content="self",
                     a=a * res_a * p, lw=4)
            else:
                obj_glyph(z, (cx, RES_C[1]), glyph, 34, a=a * res_a * p)
    if cross_prog > 0.02:
        cross(z, RES_C, 150, a=a, w=12, blue=True, prog=cross_prog)


def g_prop1(lay, lt):
    """«нельзя просто объявлять множество объектов»: свойство в фильтре,
    под ним собирается коробка из подходящих объектов."""
    z = Z()
    ob = ease_out(clamp01(lt / 0.28))
    bar = ease_out(clamp01((lt - 0.22) / 0.30))
    res = ease_out(clamp01((lt - 0.56) / 0.30))
    prop_scene(z, "o", obj_a=ob, bar_a=bar, res_a=res,
               filled=ease_out(clamp01((lt - 0.86) / 0.62)) * 1.5)
    lay.alpha_composite(z.out())


def g_prop2(lay, lt):
    """«удовлетворяющих любому придуманному свойству»: свойство меняется —
    и на третьем, «не содержит сам себя», коробка получает синий крест:
    такое свойство множества не задаёт."""
    z = Z()
    if lt < 0.72:
        g, k = "t", lt
    elif lt < 1.30:
        g, k = "o", lt - 0.72
    else:
        g, k = "self", lt - 1.30
    xx = 0.0 if g != "self" else ease_out(clamp01((k - 0.55) / 0.34))
    prop_scene(z, g, filled=ease_out(clamp01(k / 0.42)) * 1.5, cross_prog=xx)
    lay.alpha_composite(z.out())


def g_axioms(lay, lt):
    """«гораздо строже построить основания»: под множество подводят несколько
    явных правил, и только на них оно стоит."""
    z = Z()
    s = stagger(lt, 3, t0=0.05, step=0.16, dur=0.30)
    for i, y in enumerate(AX_Y):
        if s[i] <= 0.02:
            continue
        z.use("w")
        z.rrect((540, y), AX_W * (0.55 + 0.45 * s[i]), AX_H, AX_H / 2,
                outline=_col(WHITE, s[i]), w=5)
    drop = ease_out(clamp01((lt - 0.86) / 0.44))
    if drop > 0.02:
        y = AX_BOX_C[1] - 120 * (1 - drop)
        sbox(z, (AX_BOX_C[0], y), AX_BOX_W, AX_BOX_H, 26, content="R",
             a=drop, lw=5)
    ck = ease_out(clamp01((lt - 1.52) / 0.32))
    if ck > 0.02:
        check(z, AX_CHECK_C, 46, a=1.0, w=10, blue=True, prog=ck)
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
    "boxes": g_boxes, "boxesA": g_boxes_a, "boxesB": g_boxes_b,
    "russell": g_russell, "ask": g_ask, "caseA1": g_case_a1,
    "rule": g_rule, "caseA2": g_case_a2, "caseA3": g_case_a3,
    "caseB1": g_case_b1, "caseB2": g_case_b2, "loop": g_loop,
    "barber1": g_barber1, "barber2": g_barber2, "barber3": g_barber3,
    "prop1": g_prop1, "prop2": g_prop2, "axioms": g_axioms,
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

STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/35/stock"
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
        # ключ тоже сбрасывается: иначе следующий frame() решит, что клип уже
        # открыт, чтение с закрытого cap провалится и вернётся застрявший кадр
        self.cap = None
        self.key = None
        self.last = None
        self.pos = -1.0


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

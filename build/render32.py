"""Сборка ролика 32 («100 заключённых и шляпы»).

Слойность: фон (лицо / сетка) -> графика -> субтитры. Аудио добавляет
sfx32.py, исходная речь не режется. Графика — только своя: тема целиком
механическая (кто что видит и что из этого вычисляет), живого видео,
буквально показывающего шеренгу заключённых в шляпах, не существует,
а нелитеральная перебивка тут ломает смысл (assets-manifest §2).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* межстрочный шаг считается от кеглей соседних строк, а не берётся фиксированным;
* кегль числа подбирается под ширину зоны графики, а не по шкале XL;
* графика рисуется в отдельном буфере размером с зону (730x860) при ss=2 —
  свечение на границе зоны обрезается самим буфером, поэтому «графика вылезла
  за зону» физически невозможно.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard32 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, N, LAST, OBS, NEXT, HATS,
                          BLACKS_AHEAD, HAT_ORDER, PRISONERS, SAVED,
                          NUM_XY, NUM_SIZE, BLUE_XY, BLUE_SIZE,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (29 замеров по всему дублю: cx 390+-5, линия глаз y 648+-5, ширина лица 283+-6).
# Кроп 630x1000 ставит глаза на 42% высоты карточки — тот же приём и тот же
# размер окна, что в роликах 22-30.
FRAMINGS = {
    "A1": dict(w=630, h=1000, x=75, y=228),
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
# Ряд заключённых — единственный графический объект ролика
# =============================================================================
# Буфер графики совпадает с зоной из brand-kit §5 (y 700...1560, x 175...905),
# поэтому свечение обрезается границей зоны, а не вылезает на субтитры.
ZX0, ZY0, ZW, ZH, SS = GX0, GY0, GX1 - GX0, GY1 - GY0, 2

PITCH = 76
ROW_CX = 540
HEAD_Y = 1140
HEAD_R = 16
BODY_Y, BODY_W, BODY_H = 1192, 48, 76    # «плечи» — купол шире головы
BRIM_Y, BRIM_W, BRIM_T = 1121, 50, 7
HAT_W, HAT_H = 38, 32
CROWN_CY = BRIM_Y - HAT_H / 2 - 3          # 1102
PX = [ROW_CX + (i - (N - 1) / 2.0) * PITCH for i in range(N)]

MARK_Y, MARK_R = 1040, 13                  # маркер «эта шляпа сосчитана/вычислена»
HEAD_RING_R = 30
FLOW_Y = 983                               # стрелка/импульс над шляпами
PAIR_TOP, PAIR_SAG = MARK_Y, -76           # дуги «в пару» над маркерами
BUBBLE_C = (PX[N - 1] - 62, 883)
BUBBLE_W, BUBBLE_H = 184, 128


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
                                 radius=rad * SS, fill=fill, outline=outline,
                                 width=max(1, int(round(w * SS))))

    def poly(self, pts, fill):
        self.d.polygon([self.p(*q) for q in pts], fill=fill)

    def out(self):
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for k, glow in (("w", WHITE), ("b", BLUE_GLOW)):
            im = self.ims[k].resize((ZW, ZH), Image.LANCZOS)
            if im.getbbox() is None:
                continue
            comp = _glow_sub(im, 11, glow, 0.36 if k == "w" else 0.82)
            lay.alpha_composite(comp, (ZX0, ZY0))
        return lay


def _rrect_pts(cx, cy, w, h, rad, seg=7):
    """Точки контура скруглённого прямоугольника — для пунктира."""
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    r = min(rad, w / 2, h / 2)
    pts = []
    for (ccx, ccy, a0) in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0),
                           (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for s in range(seg + 1):
            a = math.radians(a0 + 90 * s / seg)
            pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    pts.append(pts[0])
    return pts


def dash_path(z, pts, col, width, dash=10, gap=8):
    acc, on = 0.0, True
    for a, b in zip(pts, pts[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        if d < 1e-6:
            continue
        t = 0.0
        while t < d:
            room = (dash if on else gap) - acc
            step = min(room, d - t)
            p0 = (a[0] + (b[0] - a[0]) * (t / d), a[1] + (b[1] - a[1]) * (t / d))
            t += step
            p1 = (a[0] + (b[0] - a[0]) * (t / d), a[1] + (b[1] - a[1]) * (t / d))
            if on:
                z.line([p0, p1], col, width)
            acc += step
            if acc >= (dash if on else gap) - 1e-6:
                acc, on = 0.0, not on


def person(z, i, a=1.0, sc=1.0):
    if a <= 0.004:
        return
    x = PX[i]
    col = _col(WHITE, a)
    z.use("w")
    z.circ((x, HEAD_Y), HEAD_R * sc, fill=col)
    z.rrect((x, BODY_Y), BODY_W * sc, BODY_H * sc, 24 * sc, fill=col)


def hat_at(z, x, brim_y, kind, a=1.0, sc=1.0):
    """kind: 'w' белая | 'b' чёрная (контур) | 'unk' пунктир (цвет не известен)."""
    if a <= 0.004 or sc <= 0.02:
        return
    z.use("w")
    col = _col(WHITE, a)
    crown_cy = brim_y - (HAT_H / 2 + 3) * sc
    cw, ch = HAT_W * sc, HAT_H * sc
    if kind == "unk":
        dash_path(z, _rrect_pts(x, crown_cy, cw, ch, 9 * sc),
                  _col(WHITE, a * 0.8), 4, dash=11 * sc, gap=8 * sc)
        dash_path(z, [(x - BRIM_W * sc / 2, brim_y), (x + BRIM_W * sc / 2, brim_y)],
                  _col(WHITE, a * 0.8), 6, dash=11 * sc, gap=8 * sc)
        return
    if kind == "w":
        z.rrect((x, crown_cy), cw, ch, 9 * sc, fill=col)
    else:
        z.rrect((x, crown_cy), cw, ch, 9 * sc, fill=_col(BLACK, a), outline=col, w=5)
    z.rrect((x, brim_y), BRIM_W * sc, BRIM_T * sc, 3, fill=col)


def hat(z, i, kind=None, a=1.0, sc=1.0):
    hat_at(z, PX[i], BRIM_Y, kind or ("b" if HATS[i] else "w"), a, sc)


def ring(z, c, r, a=1.0, blue=False, w=5):
    if a <= 0.004 or r <= 1:
        return
    z.use("b" if blue else "w")
    z.circ(c, r, outline=_col(BLUE_GLOW if blue else WHITE, a), w=w)


def hat_ring(z, i, a=1.0, blue=False, scale=1.0):
    """Маркер над шляпой: сосчитана (белый) / вычислена (синий)."""
    if a <= 0.004 or scale <= 0.02:
        return
    z.use("b" if blue else "w")
    z.circ((PX[i], MARK_Y), MARK_R * scale, fill=_col(BLUE if blue else WHITE, a))


def pointer(z, i, a=1.0, blue=False):
    """Указатель под фигурой: «речь сейчас про этого человека»."""
    if a <= 0.004:
        return
    x = PX[i]
    y = BODY_Y + BODY_H / 2 + 16
    h = 24 * a
    z.use("b" if blue else "w")
    z.poly([(x, y), (x - 15, y + h), (x + 15, y + h)],
           _col(BLUE_GLOW if blue else WHITE, 0.95))


def slash(z, c, r, a=1.0, w=9):
    """Одна диагональ поверх стрелки — «так делать не надо»."""
    if a <= 0.004 or r <= 1:
        return
    x, y = c
    z.use("w")
    z.line([(x - r * 0.72, y + r), (x + r * 0.72, y - r)], _col(WHITE, a), w)


def cross(z, c, r, a=1.0, w=8):
    if a <= 0.004 or r <= 1:
        return
    x, y = c
    z.use("w")
    z.line([(x - r, y - r), (x + r, y + r)], _col(WHITE, a), w)
    z.line([(x + r, y - r), (x - r, y + r)], _col(WHITE, a), w)


def arrow_left(z, x_from, x_to, y, a=1.0, blue=False, w=7):
    if a <= 0.004 or abs(x_to - x_from) < 6:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    z.line([(x_from, y), (x_to, y)], col, w)
    for sy in (-1, 1):
        z.line([(x_to, y), (x_to + 20, y + sy * 16)], col, w)


def arrow_down(z, x, y_from, y_to, a=1.0, w=7):
    if a <= 0.004 or y_to - y_from < 6:
        return
    z.use("w")
    col = _col(WHITE, a)
    z.line([(x, y_from), (x, y_to)], col, w)
    for sx in (-1, 1):
        z.line([(x, y_to), (x + sx * 15, y_to - 18)], col, w)


def row_base(z, alpha=1.0, dim=(), hats_kind=None, skip_hat=()):
    """Ряд целиком. dim — индексы, которые притушены; hats_kind — переопределение."""
    for i in range(N):
        a = alpha * (0.24 if i in dim else 1.0)
        person(z, i, a=a)
        if i in skip_hat:
            continue
        hat(z, i, (hats_kind or {}).get(i), a=a)


def curve_pts(x0, x1, y_top, sag, prog=1.0, n=28):
    out = []
    for s in range(n + 1):
        t = (s / n) * prog
        out.append((x0 + (x1 - x0) * t, y_top + sag * math.sin(math.pi * t)))
    return out


# --- планы -------------------------------------------------------------------

def g_row(lay, lt):
    """«стоящих в ряд»: шеренга собирается слева направо, потом белое 100.
    Шляп ещё нет — их надевают на следующем плане."""
    z = Z()
    prog = stagger(lt, N, t0=0.02, step=0.035, dur=0.20)
    for i in range(N):
        person(z, i, a=prog[i], sc=0.55 + 0.45 * prog[i])
    lay.alpha_composite(z.out())
    q = clamp01((lt - 0.62) / 0.10)                      # 3 кадра, резкий поп (R5a)
    if q > 0:
        draw_number(lay, str(PRISONERS), NUM_XY, NUM_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


def g_hats(lay, lt):
    """«чёрную или белую шляпу»: шляпы падают на головы в разнобой —
    порядок фиксирован (HAT_ORDER), но читается как случайный."""
    z = Z()
    for i in range(N):
        person(z, i)
    for k, i in enumerate(HAT_ORDER):
        p = ease_out(clamp01((lt - 0.05 - k * 0.085) / 0.22))
        hat(z, i, a=p, sc=0.45 + 0.55 * p)
    lay.alpha_composite(z.out())


def g_sees(lay, lt):
    """«каждый видит шляпы людей перед собой»: один человек в кольце, те,
    кто стоит за ним, притушены, шляпы впереди по очереди помечаются."""
    z = Z()
    row_base(z, dim=set(range(OBS + 1, N)))
    pointer(z, OBS, a=ease_out(clamp01(lt / 0.20)))
    q = ease_out(clamp01((lt - 0.10) / 0.38))
    arrow_left(z, PX[OBS] - 40, PX[OBS] - 40 - (PX[OBS] - PX[0] - 34) * q, FLOW_Y, a=q)
    prog = stagger(lt, OBS, t0=0.50, step=0.13, dur=0.22)
    for k, i in enumerate(reversed(range(OBS))):
        hat_ring(z, i, a=0.95 * prog[k], scale=0.45 + 0.55 * prog[k])
    lay.alpha_composite(z.out())


def g_own_unknown(lay, lt):
    """«но не видит свою»: та же композиция, но собственная шляпа стала
    пунктирной, а маркер над ней — крестом. Это про знание, не про цвет:
    зритель цвет по-прежнему видит у всех остальных."""
    z = Z()
    row_base(z, dim=set(range(OBS + 1, N)), hats_kind={OBS: "unk"})
    for i in range(OBS):
        hat_ring(z, i, a=0.7)
    pointer(z, OBS)
    c = ease_out(clamp01((lt - 0.20) / 0.24))
    cross(z, (PX[OBS], MARK_Y), 22 * c, a=0.98, w=8)
    lay.alpha_composite(z.out())


def bubble(z, kind, a=1.0, sc=1.0):
    """Реплика над крайним справа: в облачке — шляпа, которую он называет."""
    if a <= 0.004 or sc <= 0.02:
        return
    cx, cy = BUBBLE_C
    z.use("w")
    z.rrect((cx, cy), BUBBLE_W * sc, BUBBLE_H * sc, 30 * sc,
            outline=_col(WHITE, 0.9 * a), w=5)
    z.poly([(cx + 30 * sc, cy + BUBBLE_H * sc / 2 - 4),
            (cx + 72 * sc, cy + BUBBLE_H * sc / 2 - 4),
            (cx + 44 * sc, cy + BUBBLE_H * sc / 2 + 46 * sc)], _col(WHITE, 0.9 * a))
    hat_at(z, cx, cy + 30 * sc, kind, a=a, sc=0.92 * sc)


def g_call_out(lay, lt):
    """«должен назвать цвет своей шляпы»: крайний справа отвечает первым —
    в облачке шляпа переключается между белой и чёрной: цвет надо назвать."""
    z = Z()
    row_base(z, dim=set(range(0, LAST)), hats_kind={LAST: "unk"})
    pointer(z, LAST, a=ease_out(clamp01(lt / 0.20)))
    p = ease_out(clamp01((lt - 0.12) / 0.24))
    bubble(z, "w" if int(lt / 0.26) % 2 == 0 else "b", a=p, sc=0.6 + 0.4 * p)
    lay.alpha_composite(z.out())


def g_wrong(lay, lt):
    """«ошибка означает проигрыш»: он назвал белую, шляпа чёрная — крест,
    и весь ряд гаснет. Проигрывает группа, а не он один."""
    z = Z()
    fade = 1.0 - 0.74 * ease_out(clamp01((lt - 1.05) / 0.30))
    row_base(z, alpha=fade)
    bubble(z, "w", a=fade * ease_out(clamp01(lt / 0.20)))
    c = ease_out(clamp01((lt - 0.62) / 0.16))
    cross(z, (PX[LAST], CROWN_CY + 4), 32 * c, a=0.98, w=9)
    lay.alpha_composite(z.out())


def g_plan(lay, lt):
    """«она должна быть такой»: канал связи — единственный ответ крайнего
    справа проходит над всей шеренгой до самого первого."""
    z = Z()
    row_base(z)
    q = ease_out(clamp01((lt - 0.06) / 0.60))
    arrow_left(z, PX[LAST] - 40, PX[LAST] - 40 - (PX[LAST] - PX[0] - 34) * q, FLOW_Y, a=1.0)
    p = ((lt - 0.10) / 0.85) % 1.0 if lt > 0.10 else 0.0
    if 0 < p < 1:
        x = PX[LAST] - 40 - (PX[LAST] - PX[0] - 34) * p
        z.use("b")
        z.circ((x, FLOW_Y), 15, fill=_col(BLUE, 1.0))
    lay.alpha_composite(z.out())


def g_not_guess(lay, lt):
    """«не для угадывания собственной шляпы»: стрелка от его ответа к своей
    же шляпе перечёркнута — свой ответ он тратит не на себя."""
    z = Z()
    row_base(z, dim=set(range(0, LAST)), hats_kind={LAST: "unk"})
    pointer(z, LAST)
    q = ease_out(clamp01((lt - 0.08) / 0.30))
    arrow_down(z, PX[LAST], MARK_Y - 140, MARK_Y - 140 + 150 * q, a=0.9, w=8)
    c = ease_out(clamp01((lt - 0.52) / 0.18))
    slash(z, (PX[LAST], MARK_Y - 65), 46 * c, a=0.98, w=11)
    lay.alpha_composite(z.out())


def g_send_bit(lay, lt):
    """«а чтобы передать всем остальным один бит информации»: один синий
    импульс уходит влево и по дороге зажигает каждого."""
    z = Z()
    p = clamp01((lt - 0.18) / 1.55)
    x = PX[LAST] - 40 - (PX[LAST] - PX[0] - 34) * ease_out(p)
    lit = {i for i in range(LAST) if PX[i] >= x - 10}
    row_base(z, dim={i for i in range(LAST) if i not in lit})
    arrow_left(z, PX[LAST] - 40, x, FLOW_Y, a=0.9)
    if 0 < p < 1.0:
        z.use("b")
        z.circ((x, FLOW_Y), 16, fill=_col(BLUE, 1.0))
    for i in lit:
        hat_ring(z, i, a=0.9, blue=True)
    lay.alpha_composite(z.out())


def g_parity_a(lay, lt):
    """«например он сообщает чётное или нечётное»: из тех, кого он видит,
    помечаются только чёрные шляпы — их и считают."""
    z = Z()
    row_base(z, dim={LAST})
    prog = stagger(lt, len(BLACKS_AHEAD), t0=0.35, step=0.52, dur=0.30)
    for k, i in enumerate(BLACKS_AHEAD):
        hat_ring(z, i, a=prog[k], scale=0.4 + 0.6 * prog[k])
    lay.alpha_composite(z.out())


def g_parity_b(lay, lt):
    """«количество чёрных шляп он видит впереди»: помеченные чёрные разбиваются
    на пары дугами, одна остаётся без пары — это и есть «нечётно», один бит."""
    z = Z()
    row_base(z, dim={LAST})
    for i in BLACKS_AHEAD:
        hat_ring(z, i, a=0.95)
    pairs = [(BLACKS_AHEAD[k], BLACKS_AHEAD[k + 1])
             for k in range(0, len(BLACKS_AHEAD) - 1, 2)]
    for k, (a, b) in enumerate(pairs):
        p = ease_out(clamp01((lt - 0.15 - k * 0.40) / 0.55))
        if p <= 0.02:
            continue
        z.use("w")
        z.line(curve_pts(PX[a], PX[b], PAIR_TOP, PAIR_SAG, prog=p), _col(WHITE, 0.9), 7)
    if len(BLACKS_AHEAD) % 2:
        odd = BLACKS_AHEAD[-1]
        q = ease_out(clamp01((lt - 0.15 - len(pairs) * 0.40 - 0.20) / 0.36))
        if q > 0.02:
            pulse = 0.5 + 0.5 * math.sin((lt - 1.0) * 3.4)
            hat_ring(z, odd, a=1.0, blue=True, scale=1.5 + 0.3 * (1 - q))
            ring(z, (PX[odd], MARK_Y), 34 + 8 * (1 - q) + 4 * pulse,
                 a=(0.55 + 0.35 * pulse) * q, blue=True, w=5)
    lay.alpha_composite(z.out())


def g_deduce_a(lay, lt):
    """«используя этот ответ и видимые шляпы»: второй отвечающий получает
    синий бит сверху и видит те же чёрные шляпы, кроме одной."""
    z = Z()
    seen_blacks = [i for i in BLACKS_AHEAD if i < NEXT]
    row_base(z, dim={LAST}, hats_kind={NEXT: "unk"})
    pointer(z, NEXT, a=ease_out(clamp01(lt / 0.20)))
    q = ease_out(clamp01((lt - 0.10) / 0.26))
    if q > 0.02:
        z.use("b")
        z.circ((PX[NEXT], FLOW_Y), 17 * q, fill=_col(BLUE, 1.0))
        z.use("b")
        z.line([(PX[NEXT], FLOW_Y + 22), (PX[NEXT], FLOW_Y + 22 + 22 * q)],
               _col(BLUE_GLOW, 0.9), 5)
    prog = stagger(lt, len(seen_blacks), t0=0.36, step=0.18, dur=0.24)
    for k, i in enumerate(reversed(seen_blacks)):
        hat_ring(z, i, a=0.95 * prog[k], scale=0.45 + 0.55 * prog[k])
    lay.alpha_composite(z.out())


def g_deduce_b(lay, lt):
    """«уже может вычислить цвет своей»: пунктир на его шляпе схлопывается
    в настоящий цвет (белая — по расчёту чётности), маркер синий = вычислено."""
    z = Z()
    q = ease_out(clamp01((lt - 0.24) / 0.28))
    row_base(z, dim={LAST}, skip_hat={NEXT})
    for i in [b for b in BLACKS_AHEAD if b < NEXT]:
        hat_ring(z, i, a=0.55)
    if q < 0.999:
        hat(z, NEXT, "unk", a=1.0 - q)
    if q > 0.02:
        hat(z, NEXT, a=q, sc=0.55 + 0.45 * q)
        hat_ring(z, NEXT, a=q, blue=True, scale=1.5)
    lay.alpha_composite(z.out())


def g_cascade(lay, lt):
    """«и каждый следующий делает то же самое»: синие маркеры вычисленных
    шляп бегут справа налево — от второго отвечающего до самого первого."""
    z = Z()
    row_base(z, dim={LAST})
    order = list(range(NEXT, -1, -1))
    prog = stagger(lt, len(order), t0=0.06, step=0.20, dur=0.22)
    for k, i in enumerate(order):
        if prog[k] > 0.02:
            hat_ring(z, i, a=prog[k], blue=True, scale=(0.6 + 0.9 * prog[k]))
    lay.alpha_composite(z.out())


def g_first_fails(lay, lt):
    """«в итоге первый отвечающий может ошибиться»: его шляпа мигает между
    белой и чёрной — свой цвет он не выводит ниоткуда, это монетка."""
    z = Z()
    row_base(z, dim=set(range(LAST)), skip_hat={LAST})
    for i in range(LAST):
        hat_ring(z, i, a=0.34, blue=True)
    pointer(z, LAST)
    flip = int(lt / 0.18) % 2 == 0
    hat(z, LAST, "w" if flip else "b", a=0.95)
    q = ease_out(clamp01(lt / 0.24))
    if q > 0.02:
        dash_path(z, _rrect_pts(PX[LAST], CROWN_CY + 2, 66, 58, 19),
                  _col(WHITE, 0.8 * q), 4, dash=12, gap=9)
    lay.alpha_composite(z.out())


def g_ninety_nine(lay, lt):
    """«зато остальные 99 точно определяют свои цвета»: все, кроме крайнего
    справа, помечены синим; число несёт текст (R5b)."""
    z = Z()
    row_base(z, dim={LAST})
    prog = stagger(lt, LAST, t0=0.02, step=0.075, dur=0.22)
    for k, i in enumerate(range(LAST - 1, -1, -1)):
        if prog[k] > 0.02:
            hat_ring(z, i, a=prog[k], blue=True, scale=0.6 + 0.9 * prog[k])
    lay.alpha_composite(z.out())
    r = clamp01((lt - 1.10) / 0.38)
    if r > 0:
        sc = 0.55 + 0.45 * ease_out(r) + 0.03 * math.sin(math.pi * min(1.0, r))
        draw_number(lay, str(SAVED), BLUE_XY, BLUE_SIZE, color=BLUE, glow=BLUE_GLOW,
                    glow_a=0.85, scale=sc, reveal=clamp01((lt - 1.10) / 0.30))


def g_group(lay, lt):
    """«всей группе»: крайний справа остался с неизвестной шляпой, а его
    один бит волной уходит по всей шеренге — достаётся каждому."""
    z = Z()
    p = clamp01((lt - 0.08) / 1.20)
    x = PX[LAST] - 40 - (PX[LAST] - PX[0] - 34) * ease_out(p)
    lit = [i for i in range(LAST) if PX[i] >= x - 10]
    row_base(z, dim={LAST} | {i for i in range(LAST) if i not in lit},
             hats_kind={LAST: "unk"})
    arrow_left(z, PX[LAST] - 40, x, FLOW_Y, a=0.9)
    for i in lit:
        hat_ring(z, i, a=1.0, blue=True, scale=1.5)
    lay.alpha_composite(z.out())


def g_word_information(lt):
    """R6: крупное слово ИНФОРМАЦИЯ поверх карточки A, брендовая позиция y≈440."""
    text = "ИНФОРМАЦИЯ"
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
    "row": g_row, "hats": g_hats, "sees": g_sees, "ownUnknown": g_own_unknown,
    "callOut": g_call_out, "wrong": g_wrong, "plan": g_plan,
    "notGuess": g_not_guess, "sendBit": g_send_bit,
    "parityA": g_parity_a, "parityB": g_parity_b,
    "deduceA": g_deduce_a, "deduceB": g_deduce_b, "cascade": g_cascade,
    "firstFails": g_first_fails, "ninetyNine": g_ninety_nine, "group": g_group,
}


def graphics_layer(kind, lt):
    if kind == "A1word":
        return g_word_information(lt)
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


def background(kind, prm, fr, t):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        card = source_card(fr, kind)
        canvas.paste(card, (CARD_A[0], CARD_A[1]), rounded_mask(CARD_A[2], CARD_A[3], R_A))
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
    canvas = background(kind, prm, fr, t)
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
    print("готово:", VID)


if __name__ == "__main__":
    main()

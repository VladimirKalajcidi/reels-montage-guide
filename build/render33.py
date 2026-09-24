"""Сборка ролика 33 («100 заключённых и 100 коробок»).

Слойность: фон (лицо / сетка) -> графика -> субтитры. Аудио добавляет
sfx33.py, исходная речь не режется. Графика — только своя: тема целиком
механическая (кто какую коробку открывает и что из этого следует), живого
видео, буквально показывающего сто пронумерованных коробок и переходы по
циклу перестановки, не существует, а нелитеральная перебивка тут ломает
смысл (assets-manifest §2).

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
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard33 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, COLS, ROWS, CELLS, HALF,
                          PRISONERS, LIMIT, PCT, ACTOR, PERM, CHAIN, OWN_BOX,
                          LONG_CYCLE, BLUE_CELLS, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (29 замеров по всему дублю: cx 372+-7, линия глаз y 684+-5, ширина лица 273+-6).
# Кроп 630x1000 ставит глаза на 42% высоты карточки — тот же приём и тот же
# размер окна, что в роликах 22-32; сдвинуто только окно, под другую посадку
# головы в этом дубле.
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
# Решётка 10x10 — единственный графический объект ролика
# =============================================================================
# Буфер графики совпадает с зоной из brand-kit §5 (y 700...1560, x 175...905),
# поэтому свечение обрезается границей зоны, а не вылезает на субтитры.
ZX0, ZY0, ZW, ZH, SS = GX0, GY0, GX1 - GX0, GY1 - GY0, 2

BOX = 44
PITCH = 60
LAT_CX, LAT_CY = 540, 1040
SPAN = (COLS - 1) * PITCH + BOX
LX0 = LAT_CX - SPAN / 2 + BOX / 2          # центр первой колонки
LY0 = LAT_CY - SPAN / 2 + BOX / 2          # центр первого ряда
LAT_TOP = LAT_CY - SPAN / 2                # 748
LAT_BOT = LAT_CY + SPAN / 2                # 1332

ACTOR_Y = 1420                             # фигурка заключённого под решёткой
NUM_XY = (540, 1445)                       # белые числа-ревилы (100, 50)
NUM_SIZE = 140
BLUE_XY = (540, 1085)                      # синий итог 31%
BLUE_SIZE = 175
FRAC_XY = (540, 1080)                      # дробь 1/2^100

FOCUS = 2.30                               # во сколько раз крупнее коробка в фокусе

_rng = np.random.default_rng(33)
RAND_HALF = sorted(_rng.choice(np.arange(1, CELLS + 1), HALF, replace=False).tolist())
RAND_ORDER = _rng.permutation(np.arange(1, CELLS + 1)).tolist()
RANK = {n: i for i, n in enumerate(RAND_ORDER)}   # порядок каскада, без index() в кадре


SLIP_VALS = None


def slip_values():
    global SLIP_VALS
    if SLIP_VALS is None:
        v = np.random.default_rng(331).permutation(np.arange(1, CELLS + 1)).tolist()
        for box, val in PERM.items():
            j = v.index(val)
            k = box - 1
            v[j], v[k] = v[k], v[j]
        SLIP_VALS = v
    return SLIP_VALS


def cell_c(n):
    """Центр клетки коробки n: нумерация по чтению, слева направо сверху вниз."""
    r, c = (n - 1) // COLS, (n - 1) % COLS
    return LX0 + c * PITCH, LY0 + r * PITCH


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

    def fit_text(self, c, txt, size, max_w, col, anchor="mm", kind="sans"):
        """Тот же текст, но кегль ужимается, если строка шире max_w."""
        sz = max(8, int(round(size * SS)))
        lim = max_w * SS
        f = font(kind, sz)
        while f.getlength(txt) > lim and sz > 8:
            sz -= 2
            f = font(kind, sz)
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


# --- элементы ----------------------------------------------------------------

def box_at(z, c, a=1.0, sc=1.0, state="closed", blue=False, fill=0.0,
           slip=None, label=None, mask_bg=False):
    """Одна коробка. state: closed | open. slip — число на бумажке внутри.
    mask_bg — залить нутро чёрным: под крупной коробкой не должны просвечивать
    притушенные соседи решётки."""
    if a <= 0.004 or sc <= 0.02:
        return
    key = "b" if blue else "w"
    base = BLUE_GLOW if blue else WHITE
    z.use(key)
    s = BOX * sc
    col = _col(base, a)
    if mask_bg:
        z.rrect(c, s, s, 9 * sc, fill=_col(BLACK, 1.0))
    if fill > 0.004:
        z.rrect(c, s, s, 9 * sc, fill=_col(base, a * fill))
    z.rrect(c, s, s, 9 * sc, outline=col, w=3 * max(1.0, sc * 0.75))
    cx, cy = c
    if state == "closed":
        # линия крышки внутри верхнего края
        y = cy - s / 2 + 12 * sc
        z.line([(cx - s / 2 + 5 * sc, y), (cx + s / 2 - 5 * sc, y)],
               _col(base, a * 0.62), 2.5 * sc)
    elif label is None:
        # откинутая крышка — наклонная планка над коробкой. Если у коробки есть
        # номер, планку не рисуем: она перечёркивает подпись и та перестаёт
        # читаться как «номер этой коробки».
        z.line([(cx - s / 2 + 3 * sc, cy - s / 2 - 2 * sc),
                (cx + s / 2 - 2 * sc, cy - s / 2 - 15 * sc)],
               _col(base, a * 0.85), 3 * sc)
    if slip is not None:
        sw, sh = s * 0.78, s * 0.52
        z.rrect((cx, cy + s * 0.07), sw, sh, 3 * sc, fill=_col(WHITE, a))
        if slip != "":
            # кегль числа — от ширины бумажки, а не от её высоты: «100» шире «7»
            z.fit_text((cx, cy + s * 0.07), str(slip), sh * 0.80, sw * 0.84,
                       _col(BLACK, a))
    if label is not None:
        z.text((cx, cy - s / 2 - 13 * sc), str(label), 17 * sc,
               _col(base, a * 0.95), anchor="ms")


def person_at(z, c, a=1.0, sc=1.0, blue=False):
    if a <= 0.004 or sc <= 0.02:
        return
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    x, y = c
    z.circ((x, y - 9 * sc), 7 * sc, fill=col)
    z.rrect((x, y + 9 * sc), 21 * sc, 24 * sc, 9 * sc, fill=col)


def lattice(z, a=1.0, state="closed", skip=(), slips=False, blue_cells=(),
            fills=None, prog=None):
    """prog — список прогрессов появления по номерам клеток (индекс n-1)."""
    """Все 100 клеток. prog — список прогрессов появления по номерам клеток."""
    for n in range(1, CELLS + 1):
        if n in skip:
            continue
        p = 1.0 if prog is None else prog[n - 1]
        if p <= 0.02:
            continue
        f = 0.0 if fills is None else fills.get(n, 0.0)
        box_at(z, cell_c(n), a=a * p, sc=0.45 + 0.55 * p, state=state,
               blue=n in blue_cells, fill=f,
               slip="" if slips else None)


def pointer(z, c, a=1.0, blue=False, size=13):
    if a <= 0.004:
        return
    x, y = c
    z.use("b" if blue else "w")
    z.poly([(x, y), (x - size, y + size * 1.5), (x + size, y + size * 1.5)],
           _col(BLUE_GLOW if blue else WHITE, 0.95 * a))


def cross(z, c, r, a=1.0, w=8, blue=False):
    if a <= 0.004 or r <= 1:
        return
    x, y = c
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, a)
    z.line([(x - r, y - r), (x + r, y + r)], col, w)
    z.line([(x + r, y - r), (x - r, y + r)], col, w)


def arc_pts(p0, p1, bow=0.22, n=30, prog=1.0):
    """Дуга от p0 к p1 с прогибом bow (доля длины) — квадратичная кривая."""
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) or 1.0
    cx, cy = mx - dy / L * bow * L, my + dx / L * bow * L
    out = []
    for i in range(n + 1):
        t = (i / n) * prog
        u = 1 - t
        out.append((u * u * x0 + 2 * u * t * cx + t * t * x1,
                    u * u * y0 + 2 * u * t * cy + t * t * y1))
    return out


def arc_arrow(z, a_cell, b_cell, prog=1.0, alpha=1.0, blue=False, w=6, bow=0.22):
    """Стрелка от коробки к коробке: от края до края, с наконечником в конце."""
    if alpha <= 0.004 or prog <= 0.02:
        return
    ca, cb = cell_c(a_cell), cell_c(b_cell)
    r = BOX * FOCUS * 0.62
    dx, dy = cb[0] - ca[0], cb[1] - ca[1]
    L = math.hypot(dx, dy) or 1.0
    p0 = (ca[0] + dx / L * r, ca[1] + dy / L * r)
    p1 = (cb[0] - dx / L * r, cb[1] - dy / L * r)
    pts = arc_pts(p0, p1, bow=bow, prog=prog)
    z.use("b" if blue else "w")
    col = _col(BLUE_GLOW if blue else WHITE, alpha)
    z.line(pts, col, w)
    if prog > 0.92:
        q0, q1 = pts[-2], pts[-1]
        ang = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
        for s in (-1, 1):
            a2 = ang + s * 2.5
            z.line([q1, (q1[0] + 20 * math.cos(a2), q1[1] + 20 * math.sin(a2))], col, w)


def actor(z, a=1.0, num_a=1.0, blue=False):
    """Фигурка заключённого №45 под решёткой + его номер."""
    person_at(z, (498, ACTOR_Y), a=a, sc=1.5, blue=blue)
    if num_a > 0.004:
        z.use("b" if blue else "w")
        z.text((540, ACTOR_Y + 1), str(ACTOR), 52,
               _col(BLUE_GLOW if blue else WHITE, num_a), anchor="lm")


# --- планы -------------------------------------------------------------------

def g_people(lay, lt):
    """«итак есть 100 заключённых»: сто фигурок занимают узлы решётки,
    потом белое 100 резким попом (R5a). Количество показано, а не подписано."""
    z = Z()
    for n in range(1, CELLS + 1):
        p = ease_out(clamp01((lt - 0.04 - RANK[n] * 0.0055) / 0.18))
        person_at(z, cell_c(n), a=p, sc=0.45 + 0.55 * p)
    lay.alpha_composite(z.out())
    q = clamp01((lt - 1.60) / 0.10)                     # 3 кадра, резкий поп (R5a)
    if q > 0:
        draw_number(lay, str(PRISONERS), NUM_XY, NUM_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


def g_boxes(lay, lt):
    """«и 100 закрытых коробок»: те же узлы занимают закрытые коробки."""
    z = Z()
    prog = [ease_out(clamp01((lt - 0.03 - RANK[n] * 0.0060) / 0.20))
            for n in range(1, CELLS + 1)]
    lattice(z, prog=prog)
    lay.alpha_composite(z.out())


def g_slips(lay, lt):
    """«внутри случайно разложены бумажки»: крышки откидываются, в каждой
    коробке появляется белая бумажка."""
    z = Z()
    for n in range(1, CELLS + 1):
        p = ease_out(clamp01((lt - 0.02 - RANK[n] * 0.0090) / 0.20))
        box_at(z, cell_c(n), state="open" if p > 0.55 else "closed",
               slip="" if p > 0.55 else None, a=1.0)
    lay.alpha_composite(z.out())


def g_numbers(lay, lt):
    """«с числами от 1 до 100»: на бумажках проступают числа. Расклад —
    зафиксированная перестановка `slip_values()`, та самая, по которой дальше
    идёт цепочка. В конце плана крупно всплывают две коробки — с бумажкой 1
    и с бумажкой 100: это и есть «от 1 до 100». Числа несёт график целиком —
    субтитра на этом плане нет."""
    z = Z()
    vals = slip_values()
    for n in range(1, CELLS + 1):
        p = ease_out(clamp01((lt - 0.02 - RANK[n] * 0.0105) / 0.18))
        box_at(z, cell_c(n), state="open",
               slip=(vals[n - 1] if p > 0.5 else ""), a=1.0)
    q = ease_out(clamp01((lt - 1.05) / 0.26))
    if q > 0.02:
        for val in (1, PRISONERS):
            n = vals.index(val) + 1
            box_at(z, cell_c(n), a=1.0, sc=1.0 + (FOCUS - 1.0) * q,
                   state="open", slip=val, mask_bg=True)
    lay.alpha_composite(z.out())


def g_one_per(lay, lt):
    """«по одной в каждой коробке»: одна коробка крупно — внутри ровно
    одна бумажка с одним числом."""
    z = Z()
    lattice(z, a=0.17, state="open", slips=True)
    p = ease_out(clamp01((lt - 0.06) / 0.30))
    if p > 0.02:
        s = 5.4 * p
        z.use("w")
        box_at(z, (540, 1075), a=1.0, sc=s, state="open",
               slip=(slip_values()[16] if p > 0.6 else ""), mask_bg=True)
    lay.alpha_composite(z.out())


def g_find_own(lay, lt):
    """«найти бумажку со своим номером»: у заключённого номер 45, и ровно
    такая бумажка лежит в коробке 23 — эту клетку план и подсвечивает."""
    z = Z()
    lattice(z, a=0.20, skip={OWN_BOX})
    actor(z, a=ease_out(clamp01(lt / 0.24)), num_a=ease_out(clamp01((lt - 0.16) / 0.26)))
    p = ease_out(clamp01((lt - 0.55) / 0.32))
    if p > 0.02:
        box_at(z, cell_c(OWN_BOX), a=1.0, sc=FOCUS * (0.5 + 0.5 * p),
               state="open" if p > 0.6 else "closed", blue=True,
               slip=ACTOR if p > 0.6 else "", mask_bg=True)
        pulse = 0.5 + 0.5 * math.sin(lt * 5.2)
        cx, cy = cell_c(OWN_BOX)
        z.use("b")
        z.circ((cx, cy), BOX * FOCUS * 0.86 + 5 * pulse,
               outline=_col(BLUE_GLOW, (0.30 + 0.30 * pulse) * p), w=4)
    lay.alpha_composite(z.out())


def g_fifty(lay, lt):
    """«открыв максимум 50 коробок»: ровно половина клеток подсвечивается,
    число несёт белый ревил (R5a)."""
    z = Z()
    lattice(z, a=0.30)
    fills = {}
    for i, n in enumerate(RAND_HALF):
        p = ease_out(clamp01((lt - 0.04 - i * 0.0125) / 0.20))
        if p > 0.02:
            box_at(z, cell_c(n), a=p, sc=1.0, state="closed", fill=0.34 * p)
            fills[n] = p
    lay.alpha_composite(z.out())
    q = clamp01((lt - 1.10) / 0.10)
    if q > 0:
        draw_number(lay, str(LIMIT), NUM_XY, NUM_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


def g_all_saved(lay, lt):
    """«все спасены»: все сто коробок гаснут синим — результат для всей группы."""
    z = Z()
    lattice(z, a=0.24)
    for n in range(1, CELLS + 1):
        p = ease_out(clamp01((lt - 0.02 - RANK[n] * 0.0068) / 0.20))
        if p > 0.02:
            box_at(z, cell_c(n), a=p, sc=1.0, state="closed", blue=True, fill=0.52 * p)
    lay.alpha_composite(z.out())


def g_random_pick(lay, lt):
    """«выбирает коробки случайно»: крышки откидываются вразнобой, без всякого
    порядка — ни одна бумажка не помечена как своя."""
    z = Z()
    lattice(z, a=0.26)
    for i, n in enumerate(RAND_ORDER[:HALF]):
        p = clamp01((lt - 0.02 - i * 0.0210) / 0.10)
        if p > 0.4:
            box_at(z, cell_c(n), a=0.95, sc=1.0, state="open", slip="")
    lay.alpha_composite(z.out())


def g_zero_chance(lay, lt):
    """«практически нулевой»: решётка уходит, остаётся сама вероятность —
    одна вторая в сотой степени (формула как графика на сетке, R10).
    Каждый выбирает 50 коробок из 100, то есть находит свою с шансом 1/2,
    и все сто должны попасть одновременно."""
    z = Z()
    fade = 1.0 - ease_out(clamp01(lt / 0.45))
    if fade > 0.02:
        lattice(z, a=0.55 * fade, state="open", slips=True)
    lay.alpha_composite(z.out())
    p = ease_out(clamp01((lt - 0.50) / 0.34))
    if p <= 0.02:
        return
    fx, fy = FRAC_XY
    sc = 0.62 + 0.38 * p
    zf = Z()
    zf.use("w")
    zf.text((fx, fy - 92 * sc), "1", 132 * sc, _col(WHITE, p), anchor="mm")
    zf.line([(fx - 168 * sc, fy - 6 * sc), (fx + 168 * sc, fy - 6 * sc)],
            _col(WHITE, p), 9 * sc)
    zf.text((fx - 46 * sc, fy + 92 * sc), "2", 132 * sc, _col(WHITE, p), anchor="mm")
    q = ease_out(clamp01((lt - 0.86) / 0.26))
    if q > 0.02:
        zf.text((fx + 16 * sc, fy + 34 * sc), str(PRISONERS), 68 * sc,
                _col(WHITE, q), anchor="lm")
    lay.alpha_composite(zf.out())


def _focus_chain(z, upto, lt, opened, slips_shown):
    """Общая композиция цепочки: решётка притушена, коробки цепочки крупно."""
    lattice(z, a=0.18, skip=set(CHAIN[:upto]))
    for i, n in enumerate(CHAIN[:upto]):
        box_at(z, cell_c(n), a=1.0, sc=FOCUS,
               state="open" if i in opened else "closed",
               slip=(PERM[n] if i in slips_shown else ("" if i in opened else None)),
               label=n, blue=(n == OWN_BOX and 2 in slips_shown), mask_bg=True)


def g_open_own(lay, lt):
    """«каждый сначала открывает коробку со своим номером»: его номер 45 —
    и он идёт в клетку 45; номер под решёткой и подпись на коробке
    загораются вместе, это и есть связка «свой номер»."""
    z = Z()
    lattice(z, a=0.20, skip={CHAIN[0]})
    a = ease_out(clamp01(lt / 0.24))
    na = ease_out(clamp01((lt - 0.30) / 0.26))
    actor(z, a=a, num_a=na)
    p = ease_out(clamp01((lt - 0.30) / 0.30))
    if p > 0.02:
        cx, cy = cell_c(CHAIN[0])
        opened = lt > 1.85
        box_at(z, cell_c(CHAIN[0]), a=1.0, sc=FOCUS * (0.5 + 0.5 * p),
               state="open" if opened else "closed",
               slip="" if opened else None,
               label=CHAIN[0] if na > 0.1 else None, mask_bg=True)
        pointer(z, (cx, cy + BOX * FOCUS * 0.5 + 12), a=p)
    lay.alpha_composite(z.out())


def g_inside_other(lay, lt):
    """«если внутри другое число»: в коробке 45 лежит бумажка 78 —
    не его номер."""
    z = Z()
    _focus_chain(z, 1, lt, opened={0}, slips_shown=set())
    actor(z, a=0.75, num_a=0.75)
    p = ease_out(clamp01((lt - 0.14) / 0.24))
    if p > 0.02:
        n = CHAIN[0]
        box_at(z, cell_c(n), a=1.0, sc=FOCUS, state="open",
               slip=PERM[n], label=n, mask_bg=True)
    lay.alpha_composite(z.out())


def g_go_next(lay, lt):
    """«он открывает коробку уже с этим номером»: стрелка из коробки 45
    ведёт ровно в клетку с подписью 78, и та открывается."""
    z = Z()
    _focus_chain(z, 1, lt, opened={0}, slips_shown={0})
    actor(z, a=0.55, num_a=0.55)
    q = ease_out(clamp01((lt - 0.10) / 0.52))
    arc_arrow(z, CHAIN[0], CHAIN[1], prog=q, alpha=0.95)
    p = ease_out(clamp01((lt - 0.62) / 0.30))
    if p > 0.02:
        opened = lt > 1.45
        box_at(z, cell_c(CHAIN[1]), a=1.0, sc=FOCUS * (0.55 + 0.45 * p),
               state="open" if opened else "closed",
               slip=(PERM[CHAIN[1]] if lt > 1.75 else ("" if opened else None)),
               label=CHAIN[1], mask_bg=True)
    lay.alpha_composite(z.out())


def g_chain(lay, lt):
    """«и продолжает переходить так от числа к числу»: второй переход,
    и в коробке 23 лежит его собственный номер 45 — бумажка синяя."""
    z = Z()
    lattice(z, a=0.18, skip=set(CHAIN))
    box_at(z, cell_c(CHAIN[0]), a=1.0, sc=FOCUS, state="open",
           slip=PERM[CHAIN[0]], label=CHAIN[0], mask_bg=True)
    box_at(z, cell_c(CHAIN[1]), a=1.0, sc=FOCUS, state="open",
           slip=PERM[CHAIN[1]], label=CHAIN[1], mask_bg=True)
    arc_arrow(z, CHAIN[0], CHAIN[1], prog=1.0, alpha=0.55)
    actor(z, a=0.55, num_a=0.55)
    q = ease_out(clamp01((lt - 0.20) / 0.55))
    arc_arrow(z, CHAIN[1], CHAIN[2], prog=q, alpha=0.95)
    p = ease_out(clamp01((lt - 0.80) / 0.30))
    if p > 0.02:
        found = lt > 1.65
        box_at(z, cell_c(CHAIN[2]), a=1.0, sc=FOCUS * (0.55 + 0.45 * p),
               state="open" if lt > 1.35 else "closed",
               slip=(ACTOR if found else ("" if lt > 1.35 else None)),
               label=CHAIN[2], blue=found, mask_bg=True)
        if found:
            pulse = 0.5 + 0.5 * math.sin((lt - 1.65) * 6.0)
            cx, cy = cell_c(CHAIN[2])
            z.use("b")
            z.circ((cx, cy), BOX * FOCUS * 0.86 + 5 * pulse,
                   outline=_col(BLUE_GLOW, 0.30 + 0.32 * pulse), w=4)
    lay.alpha_composite(z.out())


def g_cycle(lay, lt):
    """«по одному из циклов случайной перестановки»: последняя стрелка
    возвращается в стартовую коробку — путь замкнулся."""
    z = Z()
    lattice(z, a=0.16, skip=set(CHAIN))
    for i, n in enumerate(CHAIN):
        box_at(z, cell_c(n), a=1.0, sc=FOCUS, state="open",
               slip=PERM[n], label=n, blue=True, mask_bg=True)
    arc_arrow(z, CHAIN[0], CHAIN[1], prog=1.0, alpha=0.85, blue=True)
    arc_arrow(z, CHAIN[1], CHAIN[2], prog=1.0, alpha=0.85, blue=True)
    q = ease_out(clamp01((lt - 0.12) / 0.60))
    # замыкающая дуга уходит в другую сторону — иначе она ложится на путь туда
    arc_arrow(z, CHAIN[2], CHAIN[0], prog=q, alpha=0.95, blue=True, bow=-0.26)
    actor(z, a=0.5, num_a=0.5, blue=True)
    lay.alpha_composite(z.out())


def g_cycle_len(lay, lt):
    """«нет цикла длиннее 50»: клетки заливаются по порядку чтения, поэтому
    граница пятого ряда — это ровно 50 коробок, а не «примерно половина».
    Показанный цикл занимает 62 коробки: заливка переходит за линию, и
    лишний кусок перечёркнут — такая раздача проиграна."""
    z = Z()
    lattice(z, a=0.26)
    filled = int(clamp01((lt - 0.06) / 1.15) * LONG_CYCLE)
    for n in range(1, min(filled, HALF) + 1):
        box_at(z, cell_c(n), a=1.0, sc=1.0, state="closed", fill=0.30)
    y = LY0 + (HALF // COLS - 1) * PITCH + PITCH / 2          # 1040 — граница 50-й
    ql = ease_out(clamp01((lt - 0.62) / 0.24))
    if ql > 0.02:
        x0 = LAT_CX - SPAN / 2 - 14
        z.line([(x0, y), (x0 + (SPAN + 28) * ql, y)], _col(WHITE, 0.98), 6)
    for n in range(HALF + 1, filled + 1):
        box_at(z, cell_c(n), a=1.0, sc=1.0, state="closed", fill=0.88)
    lay.alpha_composite(z.out())
    if ql > 0.9:
        draw_number(lay, str(LIMIT), NUM_XY, NUM_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.55, scale=1.0)
    if filled >= LONG_CYCLE:
        z2 = Z()
        c = ease_out(clamp01((lt - 1.28) / 0.22))
        cross(z2, (LAT_CX, LY0 + 5.6 * PITCH), 62 * c, a=0.95, w=11)
        lay.alpha_composite(z2.out())


def g_result(lay, lt):
    """«примерно 31%»: решётка уходит, остаётся синий итог (R5b).
    Субтитра на плане нет — число несёт график."""
    z = Z()
    fade = 1.0 - ease_out(clamp01(lt / 0.40))
    if fade > 0.02:
        lattice(z, a=0.5 * fade)
        lay.alpha_composite(z.out())
    r = clamp01((lt - 0.50) / 0.40)
    if r > 0:
        sc = 0.55 + 0.45 * ease_out(r) + 0.03 * math.sin(math.pi * min(1.0, r))
        draw_number(lay, f"{PCT}%", BLUE_XY, BLUE_SIZE, color=BLUE, glow=BLUE_GLOW,
                    glow_a=0.85, scale=sc, reveal=clamp01((lt - 0.50) / 0.32))


def g_payoff(lay, lt):
    """«примерно один шанс из трёх»: из ста коробок синими становятся 31 —
    почти ровно треть решётки. Числа на плане нет, его несёт сама доля."""
    z = Z()
    lattice(z, a=0.30)
    prog = stagger(lt, BLUE_CELLS, t0=0.06, step=0.045, dur=0.24)
    for i in range(BLUE_CELLS):
        if prog[i] > 0.02:
            box_at(z, cell_c(i + 1), a=prog[i], sc=1.0, state="closed",
                   blue=True, fill=0.55 * prog[i])
    lay.alpha_composite(z.out())


def g_word_strategy(lt):
    """R6: крупное слово СТРАТЕГИЯ поверх карточки A, брендовая позиция y≈440."""
    text = "СТРАТЕГИЯ"
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
    "people": g_people, "boxes": g_boxes, "slips": g_slips,
    "numbers": g_numbers, "onePer": g_one_per, "findOwn": g_find_own,
    "fifty": g_fifty, "allSaved": g_all_saved, "randomPick": g_random_pick,
    "zeroChance": g_zero_chance, "openOwn": g_open_own,
    "insideOther": g_inside_other, "goNext": g_go_next, "chain": g_chain,
    "cycle": g_cycle, "cycleLen": g_cycle_len, "result": g_result,
    "payoff": g_payoff,
}


def graphics_layer(kind, lt):
    if kind == "A1word":
        return g_word_strategy(lt)
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

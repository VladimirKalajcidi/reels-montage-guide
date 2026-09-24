"""Сборка ролика 19 («QR-код и код Рида-Соломона»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx19.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Отличия, унаследованные из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v).

QR на экране настоящий (segno, версия 2, уровень H). Панель рисуется белой,
модули — чёрными: инвертированный QR штатный декодер не читает, проверено.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard19 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, QR_N, TOTAL_CW, DATA_CW, ECC_CW,
                          RECOVER_PCT, qr_matrix, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_qr.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/19/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо
# стабильно (cx~378, линия глаз y~651, d~289). Кроп ставит глаза
# на 43% высоты карточки у A1 и на 41% у A2.
FRAMINGS = {
    "A1": dict(w=700, h=1110, x=20, y=170),
    "A2": dict(w=620, h=983, x=68, y=248),
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


def blue_pop(lt, dur=0.38):
    """R5b: 0.55 -> 1.0 с микро-оверщутом ~3%."""
    if lt < 0:
        return 0.0, 1.0
    p = clamp01(lt / dur)
    return ease_out(p), 1.0 + 0.03 * math.sin(p * math.pi)


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

def arrow(d, a, b, alpha=215, width=7, head=22, color=WHITE):
    d.line([a, b], fill=color + (int(alpha),), width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (1, -1):
        d.line([b, (b[0] - head * math.cos(ang - s * 0.45),
                    b[1] - head * math.sin(ang - s * 0.45))],
               fill=color + (int(alpha),), width=width)


def check_mark(d, p, a, b, c, width=14, alpha=238, color=WHITE):
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


def dashed_rect(d, box, p=1.0, dash=16, gap=12, width=3, alpha=120, color=WHITE):
    """Пунктирная рамка — «место, которое сейчас заполнится»."""
    if p <= 0:
        return
    x0, y0, x1, y1 = box
    a = int(alpha * clamp01(p))
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1),
                             (x1, y1, x0, y1), (x0, y1, x0, y0)):
        ln = math.hypot(bx - ax, by - ay)
        n = max(1, int(ln // (dash + gap)))
        for i in range(n):
            u0 = i * (dash + gap) / ln
            u1 = min(1.0, (i * (dash + gap) + dash) / ln)
            d.line([(ax + (bx - ax) * u0, ay + (by - ay) * u0),
                    (ax + (bx - ax) * u1, ay + (by - ay) * u1)],
                   fill=color + (a,), width=width)


# --- геометрия QR ----------------------------------------------------------

QM = qr_matrix()                      # 25x25, True = тёмный модуль

MOD = 18                              # сторона модуля, px
QUIET = 3 * MOD                       # тихая зона 3 модуля
PANEL = QR_N * MOD + 2 * QUIET        # 558
PX0 = 540 - PANEL // 2                # 261
PY0 = 781                             # панель 781..1339
QX0, QY0 = PX0 + QUIET, PY0 + QUIET   # левый верхний модуль

MOD_S = 14                            # уменьшенная панель для плана pct30
PANEL_S = QR_N * MOD_S + 2 * 3 * MOD_S
PSX0 = 540 - PANEL_S // 2
PSY0 = 730

PANEL_FILL = 236                      # не чистый 255: мягче на чёрном, декодер читает


def _mod_rect(r, c, mod=MOD, x0=None, y0=None):
    x0 = QX0 if x0 is None else x0
    y0 = QY0 if y0 is None else y0
    return [x0 + c * mod, y0 + r * mod, x0 + (c + 1) * mod - 1, y0 + (r + 1) * mod - 1]


def qr_panel(d, hide=None, alpha=1.0, mod=MOD, px0=None, py0=None, appear=None):
    """Белая панель + чёрные модули. hide — множество (r,c), которые не рисуем."""
    if alpha <= 0.01:
        return
    px0 = PX0 if px0 is None else px0
    py0 = PSY0 if py0 is None and mod == MOD_S else (PY0 if py0 is None else py0)
    side = QR_N * mod + 2 * 3 * mod
    a = int(255 * alpha)
    v = int(PANEL_FILL * alpha)
    d.rounded_rectangle([px0, py0, px0 + side - 1, py0 + side - 1], radius=16,
                        fill=(v, v, v, a))
    x0, y0 = px0 + 3 * mod, py0 + 3 * mod
    hide = hide or set()
    for r in range(QR_N):
        for c in range(QR_N):
            if not QM[r][c] or (r, c) in hide:
                continue
            if appear is not None and appear(r, c) <= 0.02:
                continue
            d.rectangle(_mod_rect(r, c, mod, x0, y0), fill=(0, 0, 0, a))


def qr_hole(d, cells, mod=MOD, px0=None, py0=None, grow=1.0, color=BLACK, alpha=255):
    """Дырка в панели: закрашиваем область цветом фона — читается как утраченный кусок."""
    if grow <= 0.01 or not cells:
        return
    px0 = PX0 if px0 is None else px0
    py0 = (PSY0 if mod == MOD_S else PY0) if py0 is None else py0
    x0, y0 = px0 + 3 * mod, py0 + 3 * mod
    n = max(1, int(round(len(cells) * clamp01(grow))))
    for (r, c) in cells[:n]:
        d.rectangle(_mod_rect(r, c, mod, x0, y0), fill=color + (alpha,))


# --- зоны повреждений (детерминированные) ----------------------------------

def _protected():
    """Служебные структуры символа: три поисковых узора с разделителями,
    узор выравнивания и синхродорожки. Их не повреждаем — без них декодер
    не находит символ вообще, и «код всё равно считается» перестаёт быть правдой."""
    out = set()
    for (r0, c0) in ((0, 0), (0, QR_N - 8), (QR_N - 8, 0)):
        for r in range(r0, r0 + 8):
            for c in range(c0, c0 + 8):
                if 0 <= r < QR_N and 0 <= c < QR_N:
                    out.add((r, c))
    for r in range(16, 21):                       # узор выравнивания версии 2
        for c in range(16, 21):
            out.add((r, c))
    for i in range(QR_N):                         # синхродорожки
        out.add((6, i))
        out.add((i, 6))
    return out


PROTECTED = _protected()


def _keep(cells):
    seen, out = set(), []
    for rc in cells:
        if rc in PROTECTED or rc in seen:
            continue
        seen.add(rc)
        out.append(rc)
    return out


def _blob(cr, cc, rad, jitter=0.35, seed=3):
    """Кляксообразная область модулей вокруг (cr, cc)."""
    out = []
    for r in range(QR_N):
        for c in range(QR_N):
            dr, dc = r - cr, c - cc
            dist = math.hypot(dr, dc)
            wob = math.sin(math.atan2(dr, dc) * 3 + seed) * jitter * rad
            if dist <= rad + wob:
                out.append((dist, r, c))
    out.sort()
    return [(r, c) for _, r, c in out]


def _scratch(r0, c0, r1, c1, thick=1):
    out = []
    n = max(abs(r1 - r0), abs(c1 - c0)) * 3 + 1
    for i in range(n + 1):
        u = i / n
        r = int(round(r0 + (r1 - r0) * u))
        c = int(round(c0 + (c1 - c0) * u))
        for dr in range(-thick + 1, thick):
            rr = r + dr
            if 0 <= rr < QR_N and 0 <= c < QR_N:
                out.append((rr, c))
    return out


LOGO_CELLS = _keep([(r, c) for r in range(9, 16) for c in range(9, 16)])   # 7x7 в центре
TEAR_CELLS = _keep(_blob(20, 13, 3.6, seed=1.1))                           # рваный кусок
SCRATCH_A = _keep(_scratch(2, 10, 15, 3, thick=2))
SCRATCH_B = _keep(_scratch(9, 22, 22, 17, thick=1))

LOST_CELLS = _keep(TEAR_CELLS + SCRATCH_A + SCRATCH_B + LOGO_CELLS)
WORN_CELLS = _keep(TEAR_CELLS + SCRATCH_A + SCRATCH_B)

# ровно RECOVER_PCT процентов модулей символа — для плана pct30
_ALL_CELLS = [(r, c) for r in range(QR_N) for c in range(QR_N)]
_PCT_ORDER = sorted(_ALL_CELLS, key=lambda rc: (
    math.hypot(rc[0] - 6.5, rc[1] - 18.5) + 0.6 * math.sin(rc[0] * 1.7 + rc[1] * 0.9)))
PCT_CELLS = _PCT_ORDER[:int(round(QR_N * QR_N * RECOVER_PCT / 100))]


def logo_badge(d, p, mod=MOD, px0=None, py0=None):
    """Чёрная скруглённая плашка с белым кругом — «какой-нибудь логотип», без текста."""
    if p <= 0.01:
        return
    px0 = PX0 if px0 is None else px0
    py0 = (PSY0 if mod == MOD_S else PY0) if py0 is None else py0
    x0, y0 = px0 + 3 * mod, py0 + 3 * mod
    cx = x0 + 12.5 * mod
    cy = y0 + 12.5 * mod
    s = 3.5 * mod * (0.6 + 0.4 * ease_out(clamp01(p)))
    d.rounded_rectangle([cx - s, cy - s, cx + s, cy + s], radius=int(mod * 0.9),
                        fill=(0, 0, 0, 255))
    rr = s * 0.42
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=WHITE + (255,))


def sweep(d, p, mod=MOD, px0=None, py0=None):
    """Луч сканера сверху вниз по панели."""
    if p <= 0 or p >= 1:
        return
    px0 = PX0 if px0 is None else px0
    py0 = (PSY0 if mod == MOD_S else PY0) if py0 is None else py0
    side = QR_N * mod + 2 * 3 * mod
    y = py0 + side * p
    for k, a in ((0, 215), (6, 95), (12, 42)):
        for s in (1, -1):
            yy = y + s * k
            if py0 <= yy <= py0 + side:
                d.line([(px0 + 8, yy), (px0 + side - 8, yy)], fill=WHITE + (a,), width=4)


# --- геометрия блоков кодовых слов -----------------------------------------

CELL, CGAP = 52, 9
CPITCH = CELL + CGAP
CCOLS = 8
CBX0 = 330                                   # блок 330..809
CDY0 = 880                                   # строки данных 880..993
CEY0 = 1075                                  # строки резерва 1075..1310
DATA_ROWS = (DATA_CW + CCOLS - 1) // CCOLS   # 2
ECC_ROWS = (ECC_CW + CCOLS - 1) // CCOLS     # 4


def cell_xy(i, y0, cols=CCOLS):
    return CBX0 + (i % cols) * CPITCH, y0 + (i // cols) * CPITCH


def draw_cell(d, x, y, p=1.0, lime=False, fill=False, size=CELL):
    p = clamp01(p)
    if p <= 0.02:
        return
    e = 0.55 + 0.45 * ease_out(p)
    s = size * e / 2
    cx, cy = x + size / 2, y + size / 2
    col = LIME if lime else WHITE
    a = int(235 * p)
    if fill:
        d.rounded_rectangle([cx - s, cy - s, cx + s, cy + s], radius=10, fill=col + (a,))
    else:
        d.rounded_rectangle([cx - s, cy - s, cx + s, cy + s], radius=10,
                            outline=col + (a,), width=5)


def stagger(i, lt, step=0.055, dur=0.26):
    return clamp01((lt - i * step) / dur)


# --- графика планов ---------------------------------------------------------

def g_qrlogo(lay, lt):
    d = ImageDraw.Draw(lay)
    qr_panel(d, alpha=ease_out(clamp01(lt / 0.22)),
             appear=lambda r, c: pop(lt - (r + c) * 0.006, 0.18))
    logo_badge(d, pop(lt - 0.42, 0.30))


def g_qrok(lay, lt):
    d = ImageDraw.Draw(lay)
    qr_panel(d)
    logo_badge(d, 1.0)
    sweep(d, clamp01(lt / 0.85))
    if lt > 0.95:
        check_mark(d, ease_out(clamp01((lt - 0.95) / 0.35)),
                   (487, 1424), (523, 1462), (596, 1386))


def g_qrbuild(lay, lt):
    d = ImageDraw.Draw(lay)
    qr_panel(d, alpha=ease_out(clamp01(lt / 0.30)),
             appear=lambda r, c: pop(lt - (r + c) * 0.011, 0.16))


def g_qrcells(lay, lt):
    """QR распадается на 44 кодовых слова: 16 данных + 28 резерва."""
    d = ImageDraw.Draw(lay)
    fade = 1.0 - ease_out(clamp01((lt - 0.20) / 0.45))
    qr_panel(d, alpha=fade)
    for i in range(DATA_CW):
        x, y = cell_xy(i, CDY0)
        draw_cell(d, x, y, stagger(i, lt - 0.65, 0.030), fill=True)
    for i in range(ECC_CW):
        x, y = cell_xy(i, CEY0)
        draw_cell(d, x, y, stagger(i, lt - 1.05, 0.030), lime=True)


def g_rsdata(lay, lt):
    d = ImageDraw.Draw(lay)
    items = []
    for i in range(DATA_CW):
        x, y = cell_xy(i, CDY0)
        draw_cell(d, x, y, stagger(i, lt, 0.030), fill=True)
    p = pop(lt - 0.55, 0.10)
    if p > 0:
        items.append(gtext(str(DATA_CW), (250, 936), int(72 * (0.6 + 0.4 * p)), op=p))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_rsmethod(lay, lt):
    d = ImageDraw.Draw(lay)
    for i in range(DATA_CW):
        x, y = cell_xy(i, CDY0)
        draw_cell(d, x, y, 1.0, fill=True)
    items = [gtext(str(DATA_CW), (250, 936), 72)]
    p = ease_out(clamp01(lt / 0.35))
    if p > 0.05:
        arrow(d, (540, 1003), (540, 1003 + 52 * p), alpha=int(230 * p), width=6, head=18)
    dashed_rect(d, (CBX0 - 12, CEY0 - 12,
                    CBX0 + CCOLS * CPITCH - CGAP + 12,
                    CEY0 + ECC_ROWS * CPITCH - CGAP + 12),
                p=ease_out(clamp01((lt - 0.30) / 0.40)),
                alpha=int(165 + 55 * math.sin(lt * 5)), width=4, color=LIME)
    lay.alpha_composite(text_layer((W, H), items))


def g_rsappend(lay, lt):
    d = ImageDraw.Draw(lay)
    for i in range(DATA_CW):
        x, y = cell_xy(i, CDY0)
        draw_cell(d, x, y, 1.0, fill=True)
    for i in range(ECC_CW):
        x, y = cell_xy(i, CEY0)
        draw_cell(d, x, y, stagger(i, lt - 0.10, 0.032), lime=True)
    items = [gtext(str(DATA_CW), (250, 936), 72)]
    p = pop(lt - 1.15, 0.10)
    if p > 0:
        items.append(gtext(str(ECC_CW), (250, 1192), int(72 * (0.6 + 0.4 * p)),
                           op=p, color=LIME))
    lay.alpha_composite(text_layer((W, H), items))


def g_rsback(lay, lt):
    """44 кодовых слова схлопываются внутрь и гаснут, следом проявляется символ.

    Кубики уходят волной и теряют непрозрачность быстрее, чем сходятся к центру:
    иначе на середине анимации они наваливаются друг на друга и кадр
    превращается в серое пятно — проверено на кадре 22.61.
    """
    d = ImageDraw.Draw(lay)
    cx, cy = 540, PY0 + PANEL // 2
    for i in range(DATA_CW + ECC_CW):
        lime = i >= DATA_CW
        j = i - DATA_CW if lime else i
        x, y = cell_xy(j, CEY0 if lime else CDY0)
        p = ease_out(clamp01((lt - 0.05 - i * 0.006) / 0.45))
        if p >= 0.99:
            continue
        x += (cx - CELL / 2 - x) * p
        y += (cy - CELL / 2 - y) * p
        draw_cell(d, x, y, (1.0 - p) ** 1.6, lime=lime, fill=not lime,
                  size=CELL * (1 - 0.8 * p))
    qr_panel(d, alpha=ease_out(clamp01((lt - 0.34) / 0.30)))


def g_damage1(lay, lt):
    d = ImageDraw.Draw(lay)
    qr_panel(d)
    qr_hole(d, TEAR_CELLS, grow=ease_out(clamp01(lt / 0.55)))


def g_damage2(lay, lt):
    d = ImageDraw.Draw(lay)
    qr_panel(d)
    qr_hole(d, TEAR_CELLS, grow=1.0)
    qr_hole(d, SCRATCH_A, grow=ease_out(clamp01(lt / 0.40)))
    qr_hole(d, SCRATCH_B, grow=ease_out(clamp01((lt - 0.30) / 0.40)))
    logo_badge(d, pop(lt - 0.85, 0.28))


def g_restore(lay, lt):
    """Утраченные модули возвращаются: сначала лаймовой заливкой (резерв
    потрачен), затем нормальным чёрным по белому."""
    d = ImageDraw.Draw(lay)
    lost = LOST_CELLS
    qr_panel(d, hide=set(lost))
    qr_hole(d, lost, grow=1.0)
    logo_badge(d, 1.0 - ease_out(clamp01(lt / 0.35)))
    n = len(lost)
    for i, (r, c) in enumerate(lost):
        p = stagger(i, lt - 0.15, 1.15 / max(1, n), 0.22)
        if p <= 0.02:
            continue
        box = _mod_rect(r, c)
        settle = clamp01((lt - 0.15 - i * (1.15 / max(1, n)) - 0.75) / 0.40)
        d.rectangle(box, fill=(PANEL_FILL, PANEL_FILL, PANEL_FILL, int(255 * p)))
        if QM[r][c]:
            col = tuple(int(LIME[k] * (1 - settle)) for k in range(3))
            d.rectangle(box, fill=col + (int(255 * p),))


def g_pct30(lay, lt):
    """Предел уровня H: символ читается, пока повреждено не больше 30% модулей."""
    d = ImageDraw.Draw(lay)
    qr_panel(d, mod=MOD_S, px0=PSX0, py0=PSY0)
    qr_hole(d, PCT_CELLS, mod=MOD_S, px0=PSX0, py0=PSY0,
            grow=ease_out(clamp01(lt / 0.60)))
    p, sc = blue_pop(lt - 0.26)
    if p > 0.02:
        items = []
        base = 168
        items.append(gtext(f"{RECOVER_PCT} %", (540, 1330), int(base * (0.55 + 0.45 * p) * sc),
                           blue=True, op=p))
        lay.alpha_composite(text_layer((W, H), items))


def g_redund2(lay, lt):
    """Возврат к смыслу: внутри символа лежит 28 блоков резерва."""
    d = ImageDraw.Draw(lay)
    qr_panel(d)
    n, cs, gp = ECC_CW, 14, 3
    tot = n * (cs + gp) - gp
    x0 = 540 - tot // 2
    for i in range(n):
        p = stagger(i, lt - 0.20, 0.022, 0.20)
        if p <= 0.02:
            continue
        puls = 0.75 + 0.25 * math.sin(lt * 4.2 + i * 0.35)
        x = x0 + i * (cs + gp)
        d.rounded_rectangle([x, 1400, x + cs, 1400 + cs * 2], radius=4,
                            fill=LIME + (int(230 * p * puls),))


def g_qrworn(lay, lt):
    """Потрёпанный символ без логотипа: рваный кусок + две царапины, 12.5% модулей.
    Ровно в таком виде кадр читается декодером — проверяется в qa19.qr_decodes."""
    d = ImageDraw.Draw(lay)
    qr_panel(d)
    qr_hole(d, WORN_CELLS, grow=ease_out(clamp01(lt / 0.75)))


def g_qrok2(lay, lt):
    d = ImageDraw.Draw(lay)
    qr_panel(d)
    qr_hole(d, WORN_CELLS, grow=1.0)
    sweep(d, clamp01(lt / 1.00))
    if lt > 1.10:
        check_mark(d, ease_out(clamp01((lt - 1.10) / 0.35)),
                   (487, 1424), (523, 1462), (596, 1386))


GFX = {
    "qrlogo": g_qrlogo, "qrok": g_qrok, "qrbuild": g_qrbuild, "qrcells": g_qrcells,
    "rsdata": g_rsdata, "rsmethod": g_rsmethod, "rsappend": g_rsappend,
    "rsback": g_rsback, "damage1": g_damage1, "damage2": g_damage2,
    "restore": g_restore, "pct30": g_pct30, "redund2": g_redund2,
    "qrworn": g_qrworn, "qrok2": g_qrok2,
}


def graphics_layer(kind, lt):
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    return lay


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
    """Блок = до 2 фраз в пределах одного плана, разрыв по паузе >0.30с."""
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
        for pos, i in enumerate(b):
            info[i] = (bid, pos, len(b), end)
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
        bid, pos, _, _ = BLOCKS[idx]
        newer = sum(1 for j in active if j > idx and BLOCKS[j][0] == bid)
        dim = DIM[min(newer, 2)]
        bx, by = slot["x"], slot["y"] + pos * slot["step"]
        if slot["scatter"]:
            calm = bid % 3 == 0                      # каждый третий блок — спокойный
            bx += (0 if calm else [-48, 40, -22][min(pos, 2)])
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


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    nf = int(round(DUR * FPS))
    # кадры уходят в ffmpeg сырыми: один проход кодека вместо mp4v + x264
    enc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
        "-pix_fmt", "yuv420p", VID,
    ], stdin=subprocess.PIPE)
    stock_cap, stock_key = None, None

    for fno in range(nf):
        ok, fr = cap.read()
        if not ok:
            break
        t = fno / FPS
        t0, t1, kind, prm = shot_at(t)

        def read_stock():
            nonlocal stock_cap, stock_key
            key = (t0, prm["clip"])
            if key != stock_key:
                if stock_cap is not None:
                    stock_cap.release()
                stock_cap = cv2.VideoCapture(f"{STOCK_DIR}/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
                stock_key = key
            ok_s, sfr = stock_cap.read()
            if not ok_s:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok_s, sfr = stock_cap.read()
            return sfr

        canvas = compose(kind, prm, fr, t, t0, read_stock)
        enc.stdin.write(pil_to_cv(canvas.convert("RGB")).tobytes())
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}")

    enc.stdin.close()
    enc.wait()
    cap.release()
    if stock_cap is not None:
        stock_cap.release()
    print("готово:", VID)


if __name__ == "__main__":
    main()

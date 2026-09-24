"""Сборка ролика 20 («эффект приманки — попкорн в кино»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx20.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр
  (клипы идут 24/25/29.97 — последовательное чтение дало бы замедление).

Предметная графика — три стакана попкорна (маленький/средний/большой):
трапеция-обводка, заполненная зёрнами, и ценник под каждым. Все размеры
и позиции считаются от одной базовой линии, поэтому пропорция стаканов
на экране совпадает с той, о которой идёт речь.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard20 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, PRICE_S, PRICE_M, PRICE_L, DELTA,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/20/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо
# стабильно (cx~381, линия глаз y~671, d~269). Кроп ставит глаза
# на 43% высоты карточки у A1 и на 41.5% у A2.
FRAMINGS = {
    "A1": dict(w=668, h=1060, x=47, y=215),
    "A2": dict(w=601, h=954, x=81, y=275),
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


def dashed_rect(d, box, p=1.0, dash=16, gap=12, width=3, alpha=120, color=WHITE):
    """Пунктирная рамка — «этот вариант здесь для сравнения, а не для покупки»."""
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


def dashed_line(d, a, b, p=1.0, dash=18, gap=14, width=3, alpha=150, color=WHITE):
    if p <= 0:
        return
    ln = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(ln // (dash + gap)))
    lim = clamp01(p)
    al = int(alpha)
    for i in range(n):
        u0 = i * (dash + gap) / ln
        u1 = min(1.0, (i * (dash + gap) + dash) / ln)
        if u0 > lim:
            break
        u1 = min(u1, lim)
        d.line([(a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0),
                (a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)],
               fill=color + (al,), width=width)


def cross(d, cx, cy, s, p=1.0, width=11, alpha=235, color=WHITE):
    """Крест «этот вариант не покупают»."""
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


# --- геометрия стаканов попкорна -------------------------------------------

BASE_Y = 1350          # общее дно всех стаканов
PRICE_Y = 1412         # базовая линия ценника под стаканом
PRICE_SZ = 52

CUPS = {
    "S": dict(cx=275, h=218, tw=130, bw=88, heap=42, price=PRICE_S),
    "M": dict(cx=525, h=300, tw=170, bw=115, heap=52, price=PRICE_M),
    "L": dict(cx=770, h=395, tw=198, bw=136, heap=62, price=PRICE_L),
}
ORDER = ["S", "M", "L"]


def _half_w(key, y):
    """Полуширина стакана на высоте y (canvas-координата)."""
    c = CUPS[key]
    top = BASE_Y - c["h"]
    if y >= BASE_Y:
        return c["bw"] / 2
    if y >= top:
        u = (BASE_Y - y) / c["h"]
        return c["bw"] / 2 + (c["tw"] / 2 - c["bw"] / 2) * u
    v = (top - y) / c["heap"]
    if v >= 1.0:
        return 0.0
    return (c["tw"] / 2) * math.sqrt(max(0.0, 1 - v * v))


_KERNELS = {}


def kernels(key):
    """Зёрна попкорна: детерминированная упаковка внутри стакана и горкой над ним.

    Координаты относительные (dx от центра стакана, абсолютный y).
    Порядок — снизу вверх, чтобы каскад заполнял стакан как настоящий.
    """
    if key in _KERNELS:
        return _KERNELS[key]
    c = CUPS[key]
    rng = np.random.default_rng(17 + ORDER.index(key))
    pitch = 25
    out = []
    y = BASE_Y - 16
    row = 0
    while y > BASE_Y - c["h"] - c["heap"] + 6:
        hw = _half_w(key, y)
        r = 11 + int(rng.integers(0, 3))
        usable = hw - r - 2
        if usable > 0:
            n = max(1, int((2 * usable) // pitch) + 1)
            span = (n - 1) * pitch
            x0 = -span / 2 + (pitch / 2 if row % 2 else 0) * 0.5
            for i in range(n):
                dx = x0 + i * pitch + float(rng.integers(-3, 4))
                if abs(dx) + r > hw:
                    continue
                out.append((dx, y + float(rng.integers(-3, 4)), r))
        y -= pitch - 3
        row += 1
    _KERNELS[key] = out
    return out


def draw_cup(d, key, cx=None, p=1.0, alpha=1.0, lime_y=None):
    """Стакан: трапеция-обводка + зёрна. p — прогресс появления (каскадом)."""
    p = clamp01(p)
    if p <= 0.01 or alpha <= 0.01:
        return
    c = CUPS[key]
    cx = c["cx"] if cx is None else cx
    top = BASE_Y - c["h"]
    a_line = int(235 * alpha * min(1.0, p / 0.30))
    pts = [(cx - c["bw"] / 2, BASE_Y), (cx + c["bw"] / 2, BASE_Y),
           (cx + c["tw"] / 2, top), (cx - c["tw"] / 2, top)]
    d.line(pts[:2] + [pts[2]], fill=WHITE + (a_line,), width=6, joint="curve")
    d.line([pts[0], pts[3]], fill=WHITE + (a_line,), width=6)
    d.line([pts[3], pts[2]], fill=WHITE + (a_line,), width=6)
    hw14 = _half_w(key, top + 16)
    d.line([(cx - hw14, top + 16), (cx + hw14, top + 16)],
           fill=WHITE + (int(a_line * 0.65),), width=4)

    ks = kernels(key)
    n = len(ks)
    for i, (dx, y, r) in enumerate(ks):
        kp = clamp01((p - 0.14 - 0.72 * i / max(1, n)) / 0.18)
        if kp <= 0.02:
            continue
        col = LIME if (lime_y is not None and y < lime_y) else WHITE
        rr = r * (0.55 + 0.45 * ease_out(kp))
        d.ellipse([cx + dx - rr, y - rr, cx + dx + rr, y + rr],
                  fill=col + (int(232 * alpha * kp),))


def price_item(key, cx=None, p=1.0, alpha=1.0, blue=False, size=None, y=None):
    c = CUPS[key]
    cx = c["cx"] if cx is None else cx
    size = PRICE_SZ if size is None else size
    sz = int(size * (0.62 + 0.38 * ease_out(clamp01(p))))
    # синий ценник стоит на своём месте под стаканом, поэтому свечение у него
    # уже штатных 34px: иначе оно вылезает за правую границу зоны графики
    return gtext(f"{c['price']} ₽", (cx, PRICE_Y if y is None else y), sz,
                 blue=blue, op=alpha * clamp01(p) ** 0.6,
                 glow_r=26 if blue else None)


# --- графика планов ---------------------------------------------------------

def _cups_and_prices(lay, d, bright, ps=None, prices=(), price_alpha=None,
                     cxs=None, lime_y=None, blue=None, price_p=None):
    """Общий каркас: три (или меньше) стакана + ценники под ними."""
    items = []
    ps = ps or {}
    cxs = cxs or {}
    price_alpha = price_alpha or {}
    price_p = price_p or {}
    for key in ORDER:
        if key not in bright:
            continue
        draw_cup(d, key, cx=cxs.get(key), p=ps.get(key, 1.0),
                 alpha=bright[key], lime_y=lime_y if key == "L" else None)
    for key in prices:
        items.append(price_item(key, cx=cxs.get(key), p=price_p.get(key, 1.0),
                                alpha=price_alpha.get(key, bright.get(key, 1.0)),
                                blue=(key == blue)))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_cups3(lay, lt):
    """«три размера попкорна» — стаканы набираются зёрнами один за другим."""
    d = ImageDraw.Draw(lay)
    ps = {"S": clamp01(lt / 0.62), "M": clamp01((lt - 0.30) / 0.62),
          "L": clamp01((lt - 0.60) / 0.62)}
    _cups_and_prices(lay, d, {k: 1.0 for k in ORDER}, ps=ps)


def g_cupspick(lay, lt):
    """«маленький, средний и большой» — подсветка идёт за речью."""
    d = ImageDraw.Draw(lay)
    # окна относительно начала плана 7.52: S 0.00-0.28, M 0.28-0.70, L 0.70-..
    if lt < 0.28:
        cur = "S"
    elif lt < 0.70:
        cur = "M"
    else:
        cur = "L"
    bright = {k: (1.0 if k == cur else 0.26) for k in ORDER}
    _cups_and_prices(lay, d, bright)


def g_priceS(lay, lt):
    d = ImageDraw.Draw(lay)
    bright = {"S": 1.0, "M": 0.26, "L": 0.26}
    _cups_and_prices(lay, d, bright, prices=["S"],
                     price_p={"S": pop(lt - 1.20, 0.10)},
                     price_alpha={"S": 1.0})


def g_priceM(lay, lt):
    d = ImageDraw.Draw(lay)
    bright = {"S": 0.30, "M": 1.0, "L": 0.26}
    _cups_and_prices(lay, d, bright, prices=["S", "M"],
                     price_p={"M": pop(lt - 0.26, 0.10)},
                     price_alpha={"S": 0.45, "M": 1.0})


def g_priceL(lay, lt):
    d = ImageDraw.Draw(lay)
    bright = {"S": 0.30, "M": 0.34, "L": 1.0}
    _cups_and_prices(lay, d, bright, prices=["S", "M", "L"],
                     price_p={"L": pop(lt - 0.66, 0.10)},
                     price_alpha={"S": 0.45, "M": 0.50, "L": 1.0})


DELTA_Y = 900          # ряд ценников на плане delta
DELTA_SZ = 58
DELTA_CX = {"S": 285, "M": 530, "L": 790}


def g_delta(lay, lt):
    """«разница между средним и большим всего 50 рублей» — только ценники.

    Стаканы уходят: сравниваются цены, а не объёмы, и крупному синему числу
    нужна пустая зона под ним (brand-kit §4 — результат не кладём на объект).
    """
    d = ImageDraw.Draw(lay)
    items = []
    for key in ORDER:
        al = 0.28 if key == "S" else 1.0
        items.append(price_item(key, cx=DELTA_CX[key], alpha=al,
                                size=DELTA_SZ, y=DELTA_Y))
    bp = ease_out(clamp01((lt - 0.78) / 0.34))
    if bp > 0.02:
        a = int(210 * bp)
        y = 1010
        d.line([(455, y), (455 + (865 - 455) * bp, y)], fill=WHITE + (a,), width=4)
        d.line([(455, y), (455, y - 30)], fill=WHITE + (a,), width=4)
        if bp > 0.9:
            d.line([(865, y), (865, y - 30)], fill=WHITE + (a,), width=4)
    p, sc = blue_pop(lt - 1.64)
    if p > 0.02:
        items.append(gtext(f"+{DELTA} ₽", (540, 1200),
                           int(170 * (0.55 + 0.45 * p) * sc), blue=True, op=p))
    lay.alpha_composite(text_layer((W, H), items))


VOL_CX = {"M": 380, "L": 720}


def g_volume(lay, lt):
    """«но заметно больше попкорна в большом» — уровень среднего переносится
    на большой; всё, что выше него, и есть лишний объём (единственный лайм)."""
    d = ImageDraw.Draw(lay)
    lp = ease_out(clamp01((lt - 0.86) / 0.40))
    m_rim = BASE_Y - CUPS["M"]["h"]
    lime_y = m_rim if lp > 0.55 else None
    _cups_and_prices(lay, d, {"M": 1.0, "L": 1.0}, prices=["M", "L"],
                     cxs=VOL_CX, lime_y=lime_y)
    if lp > 0.02:
        dashed_line(d, (300, m_rim), (860, m_rim), p=lp, alpha=170)


def g_decoyM(lay, lt):
    """«средний не для того, чтобы его покупали» — средний гаснет и перечёркнут."""
    d = ImageDraw.Draw(lay)
    fade = 1.0 - 0.72 * ease_out(clamp01(lt / 0.45))
    bright = {"S": 0.34, "M": fade, "L": 0.34}
    _cups_and_prices(lay, d, bright, prices=["S", "M", "L"],
                     price_alpha={"S": 0.42, "M": fade, "L": 0.42})
    cross(d, CUPS["M"]["cx"], 1190, 62, p=ease_out(clamp01((lt - 0.48) / 0.40)))


def _hook_arrow(d, p):
    """Стрелка «средний толкает выбор к большому», в пустой полосе над стаканами."""
    if p <= 0.02:
        return
    a = int(225 * p)
    d.line([(525, 975), (525, 850)], fill=WHITE + (a,), width=6)
    d.line([(525, 850), (525 + (770 - 525) * p, 850)], fill=WHITE + (a,), width=6)
    if p > 0.85:
        arrow(d, (770, 850), (770, 884), alpha=a, width=6, head=20)


def g_decoyWhy(lay, lt):
    """«он существует, чтобы на его фоне» — средний обведён как точка отсчёта."""
    d = ImageDraw.Draw(lay)
    _cups_and_prices(lay, d, {"S": 0.34, "M": 0.95, "L": 0.60},
                     prices=["S", "M", "L"],
                     price_alpha={"S": 0.42, "M": 0.95, "L": 0.60})
    dashed_rect(d, (430, 978, 622, 1450),
                p=ease_out(clamp01((lt - 0.32) / 0.40)),
                alpha=int(150 + 45 * math.sin(lt * 5)), width=4)
    _hook_arrow(d, ease_out(clamp01((lt - 0.90) / 0.45)))


def g_dealL(lay, lt):
    """«большой выглядел выгодной сделкой» — его цена загорается синим (R5b)."""
    d = ImageDraw.Draw(lay)
    p, _ = blue_pop(lt - 0.58)
    _cups_and_prices(lay, d, {"S": 0.24, "M": 0.24, "L": 1.0},
                     prices=["S", "M"], price_alpha={"S": 0.34, "M": 0.34})
    items = [price_item("L", p=1.0, alpha=1.0, blue=p > 0.02,
                        size=int(PRICE_SZ * (1.0 + 0.22 * p)))]
    lay.alpha_composite(text_layer((W, H), items))
    check_mark(d, ease_out(clamp01((lt - 1.00) / 0.40)),
               (722, 843), (752, 873), (822, 803))


def g_noM(lay, lt):
    """«без среднего варианта» — приманку убирают."""
    d = ImageDraw.Draw(lay)
    fade = 1.0 - ease_out(clamp01(lt / 0.40))
    bright = {"S": 1.0, "L": 1.0}
    if fade > 0.02:
        bright["M"] = fade
    _cups_and_prices(lay, d, bright, prices=["S", "M", "L"],
                     price_alpha={"S": 1.0, "M": fade, "L": 1.0})


def g_pickS(lay, lt):
    """«все бы взяли маленький»."""
    d = ImageDraw.Draw(lay)
    _cups_and_prices(lay, d, {"S": 1.0, "L": 0.30}, prices=["S", "L"],
                     price_alpha={"S": 1.0, "L": 0.38})
    check_mark(d, ease_out(clamp01((lt - 0.42) / 0.40)),
               (232, 1030), (262, 1060), (332, 990))


def g_backM(lay, lt):
    """«рядом с невыгодным средним» — приманка возвращается на своё место."""
    d = ImageDraw.Draw(lay)
    p = ease_out(clamp01((lt - 0.05) / 0.34))
    _cups_and_prices(lay, d, {"S": 0.52, "M": 0.95 * p, "L": 0.52},
                     prices=["S", "M", "L"],
                     price_alpha={"S": 0.55, "M": 0.95 * p, "L": 0.55},
                     price_p={"M": p})
    dashed_rect(d, (430, 978, 622, 1450), p=ease_out(clamp01((lt - 0.40) / 0.40)),
                alpha=int(150 + 45 * math.sin(lt * 5)), width=4)


def g_pickL(lay, lt):
    """«почти все берут большой» — выбор уезжает к большому."""
    d = ImageDraw.Draw(lay)
    _cups_and_prices(lay, d, {"S": 0.24, "M": 0.24, "L": 1.0},
                     prices=["S", "M", "L"],
                     price_alpha={"S": 0.32, "M": 0.32, "L": 1.0})
    p = ease_out(clamp01((lt - 0.42) / 0.45))
    if p > 0.02:
        arrow(d, (420, 900), (420 + (700 - 420) * p, 900),
              alpha=int(225 * p), width=7, head=22)
    check_mark(d, ease_out(clamp01((lt - 0.72) / 0.40)),
               (722, 843), (752, 873), (822, 803))


def g_baitframe(lay, lt):
    """«это называют эффектом приманки» — механизм целиком, без подписей."""
    d = ImageDraw.Draw(lay)
    _cups_and_prices(lay, d, {"S": 0.42, "M": 0.90, "L": 0.90},
                     prices=["S", "M", "L"],
                     price_alpha={"S": 0.48, "M": 0.90, "L": 0.90})
    dashed_rect(d, (430, 978, 622, 1450), p=ease_out(clamp01((lt - 0.48) / 0.40)),
                alpha=int(150 + 50 * math.sin(lt * 5)), width=4)
    _hook_arrow(d, ease_out(clamp01((lt - 0.90) / 0.50)))


GFX = {
    "cups3": g_cups3, "cupspick": g_cupspick, "priceS": g_priceS,
    "priceM": g_priceM, "priceL": g_priceL, "delta": g_delta,
    "volume": g_volume, "decoyM": g_decoyM, "decoyWhy": g_decoyWhy,
    "dealL": g_dealL, "noM": g_noM, "pickS": g_pickS, "backM": g_backM,
    "pickL": g_pickL, "baitframe": g_baitframe,
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

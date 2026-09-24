"""Сборка ролика 14 («гипотеза Кеплера — плотнейшая упаковка шаров»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx14.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Вся предметная графика построена на одной честной решётке: расстояние между
центрами касающихся шаров ровно 2R, вертикальный шаг гексагонального слоя
R·√3 — поэтому шар верхнего слоя действительно лежит в ложбине между двумя
нижними, а пустоты между тремя шарами — настоящие криволинейные треугольники
(shot-recipes R10: «геометрия должна быть проверяемой»).
"""
import math
import os
import random
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard14 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_kepler.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/14/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GRID_CACHE = {}

# крупность A-roll: детектор даёт лицо x~281 y~655 d~435 на 1080x1920,
# после hflip центр лица x=582, линия глаз y=838. Кропы поставлены так,
# чтобы лицо село ровно по центру карточки, а глаза — на 42-43% её высоты.
FRAMINGS = {
    "A1": dict(w=1000, h=1586, x=80, y=172),
    "A2": dict(w=880, h=1396, x=142, y=238),
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
    """A-roll: hflip + кроп под карточку A + грейд (brand-kit §1).

    hflip безусловный: надпись на футболке в исходнике читается зеркально —
    проверено по кадру (crop 420x220+180+1330 на 25с), дубль снят фронталкой.
    """
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    f = FRAMINGS[kind]
    w, h = f["w"], f["h"]
    x = max(0, min(f["x"], iw - w))
    y = max(0, min(f["y"], ih - h))
    fr = fr[y:y + h, x:x + w]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_AREA)
    fr = cv2.convertScaleAbs(fr, alpha=1.06, beta=9)
    b, g, r = cv2.split(fr.astype(np.float32))
    r = np.clip(r * 1.04 + 3, 0, 255)
    b = np.clip(b * 0.96, 0, 255)
    fr = cv2.merge([b, g, r]).astype(np.uint8)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)
    fr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


STOCK_GAIN = 0.72      # притемнение вставки на 28% по экспозиции (brand-kit §6: 25-30%)


def stock_card(fr):
    """Вставка в карточке B: притемнение умножением + лёгкое обесцвечивание."""
    fr = fit_bgr(fr, CARD_B[2], CARD_B[3])
    fr = np.clip(fr.astype(np.float32) * STOCK_GAIN, 0, 255)
    hsv = cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255)
    fr = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


def clamp01(v):
    return max(0.0, min(1.0, v))


def gtext(text, xy, size=80, blue=False, serif=False, op=1.0, anchor="mm"):
    kind = "serif_it" if serif else "sans"
    return dict(text=text, font=font(kind, size), xy=xy, anchor=anchor,
                fill=BLUE if blue else WHITE,
                glow=BLUE_GLOW if blue else WHITE,
                glow_r=34 if blue else 15, glow_a=0.86 if blue else 0.48,
                opacity=op)


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

def circ(d, cx, cy, r, p=1.0, width=7, alpha=230, color=WHITE, start=-90):
    p = clamp01(p)
    if p <= 0.001:
        return
    if p >= 0.999:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color + (int(alpha),), width=width)
    else:
        d.arc([cx - r, cy - r, cx + r, cy + r], start, start + 360 * p,
              fill=color + (int(alpha),), width=width)


def ball(d, cx, cy, r, p=1.0, width=7, alpha=230):
    """Шар: окружность + короткий блик-дуга внутри, чтобы круг читался объёмным."""
    p = clamp01(p)
    if p <= 0.001:
        return
    circ(d, cx, cy, r, p, width, alpha)
    if p > 0.80 and r >= 40:
        rr = r * 0.63
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], 186, 250,
              fill=WHITE + (int(alpha * 0.42),), width=max(3, width - 3))


def arrow(d, a, b, alpha=215, width=7, head=24, color=WHITE):
    d.line([a, b], fill=color + (int(alpha),), width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (1, -1):
        d.line([b, (b[0] - head * math.cos(ang - s * 0.45),
                    b[1] - head * math.sin(ang - s * 0.45))],
               fill=color + (int(alpha),), width=width)


def bez_arrow(d, a, c, b, p=1.0, alpha=215, width=6, color=WHITE, head=20):
    p = clamp01(p)
    if p <= 0.01:
        return
    n = 28
    pts = []
    for i in range(n + 1):
        u = (i / n) * p
        x = (1 - u) ** 2 * a[0] + 2 * (1 - u) * u * c[0] + u ** 2 * b[0]
        y = (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * c[1] + u ** 2 * b[1]
        pts.append((x, y))
    d.line(pts, fill=color + (int(alpha),), width=width, joint="curve")
    if p > 0.92:
        ang = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
        for s in (1, -1):
            d.line([pts[-1], (pts[-1][0] - head * math.cos(ang - s * 0.45),
                              pts[-1][1] - head * math.sin(ang - s * 0.45))],
                   fill=color + (int(alpha),), width=width)


def check(d, a, b, c, p=1.0, width=12, alpha=238):
    """Галка тремя точками, рисуется прогрессивно."""
    p = clamp01(p)
    if p <= 0:
        return
    f1 = min(1.0, p / 0.40)
    d.line([a, (a[0] + (b[0] - a[0]) * f1, a[1] + (b[1] - a[1]) * f1)],
           fill=WHITE + (int(alpha),), width=width)
    if p > 0.40:
        f2 = (p - 0.40) / 0.60
        d.line([b, (b[0] + (c[0] - b[0]) * f2, b[1] + (c[1] - b[1]) * f2)],
               fill=WHITE + (int(alpha),), width=width)


def bar(d, x0, x1, y, alpha=225, width=6, tick=18):
    d.line([(x0, y), (x1, y)], fill=WHITE + (int(alpha),), width=width)
    d.line([(x0, y - tick), (x0, y + tick)], fill=WHITE + (int(alpha),), width=width)
    d.line([(x1, y - tick), (x1, y + tick)], fill=WHITE + (int(alpha),), width=width)


def gap_patch(lay, tri, r, alpha):
    """Пустота между тремя касающимися шарами — настоящий криволинейный треугольник:
    треугольник по центрам минус три диска. Именно она и есть смысл кадра."""
    if alpha <= 1:
        return
    xs = [p[0] for p in tri]
    ys = [p[1] for p in tri]
    x0, y0 = int(min(xs)) - 1, int(min(ys)) - 1
    x1, y1 = int(max(xs)) + 2, int(max(ys)) + 2
    w, h = x1 - x0, y1 - y0
    m = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(m)
    md.polygon([(p[0] - x0, p[1] - y0) for p in tri], fill=255)
    rr = r + 4                       # +4: не залезать под штрих окружности
    for p in tri:
        md.ellipse([p[0] - x0 - rr, p[1] - y0 - rr, p[0] - x0 + rr, p[1] - y0 + rr], fill=0)
    patch = Image.new("RGBA", (w, h), WHITE + (0,))
    patch.putalpha(m.point(lambda v: int(v * alpha / 255)))
    lay.alpha_composite(patch, (x0, y0))


def hex_rows(r, x0, y0, counts):
    """Гексагональные слои: шаг по x = 2r, по y = r·√3, каждый слой сдвинут на r."""
    dy = r * math.sqrt(3)
    rows = []
    for i, n in enumerate(counts):
        off = r if i % 2 else 0
        rows.append([(x0 + off + k * 2 * r, y0 + i * dy) for k in range(n)])
    return rows


def rand_pack(box, r, seed, tries=4000):
    """Детерминированная случайная («хитрая») укладка: шары не пересекаются."""
    x0, y0, x1, y1 = box
    rnd = random.Random(seed)
    pts = []
    for _ in range(tries):
        x = rnd.uniform(x0 + r, x1 - r)
        y = rnd.uniform(y0 + r, y1 - r)
        if all((x - a) ** 2 + (y - b) ** 2 >= (2 * r) ** 2 for a, b in pts):
            pts.append((x, y))
    return pts


# --- геометрия ролика ------------------------------------------------------

R_MAIN = 78
# y0=985: слой 4/3/2 — клин, и при y0=900 он вставал в верхнюю половину зоны,
# а низ кадра пустовал. 985 ставит габарит 903…1337 по центру зоны 700…1560.
ROWS = hex_rows(R_MAIN, 300, 985, (4, 3, 2))
PACK = [p for row in ROWS for p in row]

SCATTER = [(285, 850), (540, 800), (800, 862), (268, 1120), (545, 1075),
           (818, 1132), (335, 1392), (600, 1420), (812, 1388)]

# пустоты: тройки взаимно касающихся шаров (вершиной вниз и вершиной вверх)
GAPS = ([(ROWS[0][i], ROWS[0][i + 1], ROWS[1][i]) for i in range(3)]
        + [(ROWS[1][i], ROWS[1][i + 1], ROWS[2][i]) for i in range(2)]
        + [(ROWS[1][i], ROWS[1][i + 1], ROWS[0][i + 1]) for i in range(2)]
        + [(ROWS[2][0], ROWS[2][1], ROWS[1][1])])

# пирамида растёт вверх от базы: каждый слой смещён на r и поднят на r·√3
PYR = []
for i, n in enumerate((4, 3, 2, 1)):
    PYR.append([(324 + i * 72 + k * 144, 1440 - i * 72 * math.sqrt(3)) for k in range(n)])

R_LAY = 88
LAY_BASE = [(276 + k * 2 * R_LAY, 1380.0) for k in range(4)]
LAY_TOP_Y = 1380.0 - R_LAY * math.sqrt(3)          # шар лежит в ложбине: касается обоих

DENSEST = hex_rows(56, 270, 880, (6, 5, 6, 5, 6))
PCT_PACK = hex_rows(34, 336, 830, (7, 6, 7))
SQUARE = [(300 + i * 124, 900 + j * 124) for j in range(4) for i in range(5)]
TRICKY = rand_pack((230, 830, 850, 1330), 52, seed=14)
# боксы опущены на 125px: при y 790…1118 вся конструкция стояла в верхней половине
# зоны 700…1560, а низ кадра пустовал
CMP_L = hex_rows(26, 244, 955, (6, 5, 6, 5, 6, 5))
CMP_R = rand_pack((566, 915, 894, 1235), 26, seed=41)
QUICK_PYR = []
for i, n in enumerate((3, 2, 1)):
    QUICK_PYR.append([(270 + i * 52 + k * 104, 1330 - i * 52 * math.sqrt(3)) for k in range(n)])


# --- графика планов -------------------------------------------------------

def g_y400(lay, lt):
    """«математикам понадобилось почти 400 лет»: число живёт только в графике."""
    d = ImageDraw.Draw(lay)
    p, o = blue_pop(lt - 0.12, 0.38)          # старт на резе, а не через 0.35с
    u = clamp01((lt - 0.55) / 0.55)
    if u > 0:
        bar(d, 300, 300 + 480 * ease_out(u), 1330, alpha=int(225 * u))
    if p > 0:
        lay.alpha_composite(text_layer((W, H), [
            gtext("400", (540, 1080), int(200 * (0.55 + 0.45 * p) * o), blue=True),
        ]))


def g_balls(lay, lt):
    """«как уложить одинаковые шары»: девять одинаковых шаров вразброс."""
    d = ImageDraw.Draw(lay)
    for i, (x, y) in enumerate(SCATTER):
        ball(d, x, y, R_MAIN, (lt - 0.10 - i * 0.075) / 0.30)


def g_dense(lay, lt):
    """«чтобы они занимали максимум места»: те же шары съезжаются в плотный слой."""
    d = ImageDraw.Draw(lay)
    u = ease_out(clamp01((lt - 0.10) / 0.85))
    for i, (x, y) in enumerate(SCATTER):
        tx, ty = PACK[i]
        ball(d, x + (tx - x) * u, y + (ty - y) * u, R_MAIN)


def g_gaps(lay, lt):
    """«между ними оставалось как можно меньше пустоты»: подсвечиваются сами пустоты."""
    d = ImageDraw.Draw(lay)
    for x, y in PACK:
        ball(d, x, y, R_MAIN, 1.0, alpha=205)
    for i, tri in enumerate(GAPS):
        u = clamp01((lt - 0.45 - i * 0.11) / 0.26)
        if u > 0:
            gap_patch(lay, tri, R_MAIN, 160 * ease_out(u))


def g_pyramid(lay, lt):
    """«их надо складывать пирамидкой»: слои ложатся снизу вверх."""
    d = ImageDraw.Draw(lay)
    for i, row in enumerate(PYR):
        for k, (x, y) in enumerate(row):
            ball(d, x, y, 72, (lt - 0.12 - i * 0.20 - k * 0.045) / 0.26)


def g_layers(lay, lt):
    """«каждый следующий слой в углубление между шарами предыдущего».

    Верхний шар опускается ровно в ложбину: расстояние до обоих нижних = 2R,
    значит он касается их обоих. Это и есть механика укладки.
    """
    d = ImageDraw.Draw(lay)
    for k, (x, y) in enumerate(LAY_BASE):
        ball(d, x, y, R_LAY, (lt - k * 0.05) / 0.20)
    for j, (tx, delay) in enumerate(((540.0, 0.62), (364.0, 1.55))):
        u = ease_out(clamp01((lt - delay) / 0.60))
        if u <= 0:
            continue
        y = 880 + (LAY_TOP_Y - 880) * u
        ball(d, tx, y, R_LAY)
        if u < 0.94:
            arrow(d, (tx, y + R_LAY + 22), (tx, y + R_LAY + 74),
                  alpha=int(150 * (1 - u)), width=5, head=16)


def g_y1611(lay, lt):
    """Год гипотезы Кеплера. Слово «гипотеза» в речи не звучит — это добавление."""
    p, o = blue_pop(lt - 0.12, 0.38)          # старт на резе, а не через 0.30с
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("1611", (540, 1090), int(190 * (0.55 + 0.45 * p) * o), blue=True),
        gtext("гипотеза", (540, 1300), 56, serif=True,
              op=clamp01((lt - 0.62) / 0.32)),
    ]))


def g_densest(lay, lt):
    """«что именно такая укладка»: плотный гексагональный слой целиком."""
    d = ImageDraw.Draw(lay)
    for i, row in enumerate(DENSEST):
        for k, (x, y) in enumerate(row):
            ball(d, x, y, 56, (lt - 0.10 - i * 0.13 - k * 0.028) / 0.24)


def g_pct(lay, lt):
    """«самая плотная из всех возможных»: доля занятого объёма.

    74,05 % = π/(3·√2) — плотность ГЦК-укладки. В речи число не звучит,
    поэтому оно и несёт кадр (brand-kit §5).
    """
    d = ImageDraw.Draw(lay)
    for i, row in enumerate(PCT_PACK):
        for k, (x, y) in enumerate(row):
            ball(d, x, y, 34, (lt - i * 0.09 - k * 0.02) / 0.20, width=5, alpha=210)
    p, o = blue_pop(lt - 0.55, 0.38)
    if p > 0:
        lay.alpha_composite(text_layer((W, H), [
            # 138, а не 150: на 150 свечение числа выходило за GFX_X (замерено, 751px > 730)
            gtext("74,05 %", (540, 1210), int(138 * (0.55 + 0.45 * p) * o), blue=True),
        ]))


def g_quick(lay, lt):
    """«интуитивно это кажется очевидным»: пирамидка складывается мгновенно, ответ — галка."""
    d = ImageDraw.Draw(lay)
    for i, row in enumerate(QUICK_PYR):       # складывается почти мгновенно — это и есть «интуитивно»
        for k, (x, y) in enumerate(row):
            ball(d, x, y, 52, (lt - i * 0.04 - k * 0.015) / 0.12)
    check(d, (620, 1200), (686, 1272), (866, 1080),
          clamp01((lt - 0.35) / 0.45), width=12)


def g_other(lay, lt):
    """«что никакая другая»: квадратная укладка — те же шары, а пустоты крупнее."""
    d = ImageDraw.Draw(lay)
    for i, (x, y) in enumerate(SQUARE):
        ball(d, x, y, 62, (lt - 0.05 - i * 0.016) / 0.20)


def g_tricky(lay, lt):
    """«даже самая хитрая укладка»: беспорядочная упаковка, шары не пересекаются."""
    d = ImageDraw.Draw(lay)
    for i, (x, y) in enumerate(TRICKY):
        ball(d, x, y, 52, (lt - 0.05 - i * 0.030) / 0.20)


def g_denser(lay, lt):
    """«не сможет оказаться плотнее»: пирамидка против хитрой укладки, полосы плотности.

    Правая полоса короче левой ровно во столько, во сколько случайная плотная
    упаковка (~64 %) уступает гексагональной (74,05 %).
    """
    d = ImageDraw.Draw(lay)
    for bi, (x0, pts, rad) in enumerate(((208, [p for r in CMP_L for p in r], 26),
                                         (566, CMP_R, 26))):
        u = clamp01((lt - bi * 0.14) / 0.28)
        if u <= 0:
            continue
        d.rounded_rectangle([x0, 915, x0 + 328, 1243], radius=18,
                            outline=WHITE + (int(120 * u),), width=4)
        for i, (x, y) in enumerate(pts):
            ball(d, x, y, rad, (lt - 0.12 - bi * 0.14 - i * 0.010) / 0.18,
                 width=4, alpha=225)
    # полосы заполняются быстро: на 0.80+0.50 конечное состояние держалось 0.26с,
    # и разницу длин (100 % против 86,4 %) просто не успевали прочитать
    v = ease_out(clamp01((lt - 0.50) / 0.38))
    if v > 0:
        for bi, (x0, frac) in enumerate(((208, 1.0), (566, 0.864))):
            d.rounded_rectangle([x0, 1311, x0 + 328, 1343], radius=16,
                                outline=WHITE + (110,), width=3)
            fw = 328 * frac * v
            if fw > 8:
                d.rounded_rectangle([x0, 1311, x0 + fw, 1343], radius=16,
                                    fill=WHITE + (215,))


def g_proof(lay, lt):
    """«строгое доказательство появилось»: страницы копятся, сверху знак конца доказательства."""
    d = ImageDraw.Draw(lay)
    for i in range(6):
        u = clamp01((lt - i * 0.11) / 0.22)
        if u <= 0:
            continue
        y = 1270 - i * 52
        a = int(215 * u)
        d.rounded_rectangle([375, y, 705, y + 44], radius=10,
                            outline=WHITE + (a,), width=4)
        for k in range(2):
            d.line([(400, y + 15 + k * 15), (400 + (150 + k * 90) * u, y + 15 + k * 15)],
                   fill=WHITE + (int(90 * u),), width=3)
    # знак конца доказательства стоит там, где ему и место, — в конце последней строки
    # верхней страницы. Отдельно висящим квадратом над стопкой он не читался.
    u = clamp01((lt - 0.85) / 0.28)
    if u > 0:
        s = int(18 * ease_out(u))
        d.rectangle([664, 1040 - s // 2, 664 + s, 1040 + s - s // 2], fill=WHITE + (235,))


def g_timeline(lay, lt):
    """«появилось только в начале двухтысячных»: путь от гипотезы до доказательства.

    Без чисел: годы уже прозвучали и уже показаны на отдельных планах.
    """
    d = ImageDraw.Draw(lay)
    u = ease_out(clamp01(lt / 0.35))
    if u > 0:
        d.line([(220, 1330), (220 + 640 * u, 1330)], fill=WHITE + (200,), width=6)
        for x in (250, 830):
            if 220 + 640 * u > x:
                d.line([(x, 1306), (x, 1354)], fill=WHITE + (200,), width=6)
    ball(d, 250, 1250, 30, clamp01((lt - 0.35) / 0.24), width=5)
    bez_arrow(d, (250, 1206), (540, 900), (830, 1206),
              clamp01((lt - 0.62) / 0.85), alpha=215, width=6)
    check(d, (792, 1250), (814, 1276), (866, 1198),
          clamp01((lt - 1.55) / 0.35), width=9)


def g_check4(lay, lt):
    """«потребовалось больше четырёх лет, чтобы убедиться»: четыре отрезка ручной проверки.

    Число «4» цифрой не пишется: четыре сегмента и четыре галки уже несут его,
    а дублировать смысл в одном кадре нельзя (brand-kit §5).
    """
    d = ImageDraw.Draw(lay)
    for i in range(3):
        u = clamp01((lt - i * 0.10) / 0.22)
        if u <= 0:
            continue
        y = 890 - i * 45
        a = int(200 * u)
        d.rounded_rectangle([400, y, 680, y + 34], radius=8, outline=WHITE + (a,), width=4)
        d.line([(422, y + 17), (422 + 150 * u, y + 17)], fill=WHITE + (80,), width=3)
    v = ease_out(clamp01((lt - 0.40) / 1.55))
    if v > 0.01:
        d.rounded_rectangle([250, 1170, 250 + 580 * v, 1202], radius=16, fill=WHITE + (210,))
    # рамка и перегородки рисуются ПОВЕРХ заливки — иначе четыре отрезка исчезают,
    # как только шкала доходит до конца, и «четыре года» перестают читаться
    d.rounded_rectangle([250, 1170, 830, 1202], radius=16, outline=WHITE + (150,), width=3)
    for i in range(1, 4):
        # перегородки чёрные: белая линия по белой заливке не видна вообще,
        # и шкала читалась одним сплошным бруском (проверено на контакт-листе)
        x = 250 + i * 145
        d.line([(x, 1168), (x, 1204)], fill=BLACK + (255,), width=6)
    for i in range(4):
        # -0.85, а не -(i+1): при ровно (i+1) четвёртая галка не появлялась никогда,
        # v упирается в 1.0 и v*4-4 = 0. Проверено на контакт-листе.
        done = clamp01((v * 4 - (i + 0.85)) / 0.20)
        if done <= 0:
            continue
        cx = 250 + 145 * i + 72
        check(d, (cx - 26, 1288), (cx - 8, 1310), (cx + 26, 1262), done, width=8)


GFX = {
    "y400": g_y400, "balls": g_balls, "dense": g_dense, "gaps": g_gaps,
    "pyramid": g_pyramid, "layers": g_layers, "y1611": g_y1611,
    "densest": g_densest, "pct": g_pct, "quick": g_quick, "other": g_other,
    "tricky": g_tricky, "denser": g_denser, "proof": g_proof,
    "timeline": g_timeline, "check4": g_check4,
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
            # wobble приглушён: по узлам сетки строится предметная графика
            GRID_CACHE[key] = grid_canvas(w, h, phase=(key / 10) * 0.7, wobble=0.55)
        canvas.paste(GRID_CACHE[key], (x, y), rounded_mask(w, h, R_A))
    return canvas


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    tmp = f"{BUILD}/assets/_video_kepler_raw.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    nf = int(round(DUR * FPS))
    stock_cap, stock_key = None, None

    for fno in range(nf):
        ok, fr = cap.read()
        if not ok:
            break
        t = fno / FPS
        t0, t1, kind, prm = shot_at(t)
        lt = t - t0

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

        canvas = background(kind, prm, fr, t, read_stock)
        gl = graphics_layer(kind, lt)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        wr.write(pil_to_cv(canvas.convert("RGB")))
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}")

    wr.release()
    cap.release()
    if stock_cap is not None:
        stock_cap.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp,
        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-pix_fmt", "yuv420p", VID,
    ], check=True)
    print("готово:", VID)


if __name__ == "__main__":
    main()

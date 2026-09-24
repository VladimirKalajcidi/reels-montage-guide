"""Сборка ролика 13 («парадокс Банаха — Тарского»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx13.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).
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
from storyboard13 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_banach.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/13/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GRID_CACHE = {}

# крупность A-roll: лицо детектором стоит стабильно (x~300 y~650 d~448 на 1080x1920),
# после hflip центр лица x=556, линия глаз y=838. Кроп ставит глаза на 42-43% карточки.
FRAMINGS = {
    "A1": dict(w=1000, h=1586, x=56, y=172),
    "A2": dict(w=880, h=1396, x=116, y=238),
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

    hflip безусловный: надпись на толстовке в исходнике читается зеркально —
    проверено по кадру, дубль снят фронталкой.
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

def circ(d, cx, cy, r, p=1.0, width=8, alpha=235, color=WHITE, start=-90):
    """Окружность, отрисовываемая по дуге (p: 0..1)."""
    p = clamp01(p)
    if p <= 0.001:
        return
    if p >= 0.999:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color + (alpha,), width=width)
    else:
        d.arc([cx - r, cy - r, cx + r, cy + r], start, start + 360 * p,
              fill=color + (alpha,), width=width)


def globe(d, cx, cy, r, p=1.0, width=8, alpha=235, mer=0.0):
    """Шар: окружность + меридиан и экватор — чтобы круг читался объёмным."""
    circ(d, cx, cy, r, p, width, alpha)
    if mer > 0:
        a = int(alpha * 0.55 * clamp01(mer))
        d.ellipse([cx - r * 0.40, cy - r, cx + r * 0.40, cy + r],
                  outline=WHITE + (a,), width=max(3, width - 4))
        d.ellipse([cx - r, cy - r * 0.28, cx + r, cy + r * 0.28],
                  outline=WHITE + (a,), width=max(3, width - 4))


def dot(d, x, y, r, alpha=210, color=WHITE):
    if r <= 0 or alpha <= 0:
        return
    d.ellipse([x - r, y - r, x + r, y + r], fill=color + (int(alpha),))


def arrow(d, a, b, alpha=215, width=7, head=24, color=WHITE):
    d.line([a, b], fill=color + (int(alpha),), width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (1, -1):
        d.line([b, (b[0] - head * math.cos(ang - s * 0.45),
                    b[1] - head * math.sin(ang - s * 0.45))],
               fill=color + (int(alpha),), width=width)


def bez_arrow(d, a, c, b, p=1.0, alpha=215, width=6, color=WHITE, head=20):
    """Квадратичная кривая a->b с контрольной точкой c, рисуется на долю p."""
    p = clamp01(p)
    if p <= 0.01:
        return
    n = 24
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


def bar(d, x0, x1, y, alpha=225, width=6, tick=18):
    """Мерная линейка с засечками на концах."""
    d.line([(x0, y), (x1, y)], fill=WHITE + (alpha,), width=width)
    d.line([(x0, y - tick), (x0, y + tick)], fill=WHITE + (alpha,), width=width)
    d.line([(x1, y - tick), (x1, y + tick)], fill=WHITE + (alpha,), width=width)


def cloud(cx, cy, r, n, seed):
    """Детерминированное облако точек внутри круга: (x, y, группа)."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        a = rnd.random() * math.tau
        rr = r * math.sqrt(rnd.random())
        out.append((cx + rr * math.cos(a), cy + rr * math.sin(a), i % 5))
    return out


CLOUD = cloud(540, 1150, 168, 150, 13)
CLOUD_BIG = cloud(540, 1140, 178, 240, 31)


# --- графика планов -------------------------------------------------------

def g_ball(lay, lt):
    """Шар. Слова несёт субтитр, картинку — графика, подписей нет."""
    d = ImageDraw.Draw(lay)
    globe(d, 540, 1140, 195, lt / 0.45, width=9,
          mer=clamp01((lt - 0.45) / 0.32))


def g_split(lay, lt):
    """«разобрать на куски»: шар распадается на 5 секторов."""
    d = ImageDraw.Draw(lay)
    cx, cy, r = 540, 1140, 178
    u = ease_out(clamp01((lt - 0.20) / 0.60))
    off = 42 * u
    for i in range(5):
        ap = clamp01((lt - i * 0.05) / 0.28)
        if ap <= 0:
            continue
        a0 = -90 + i * 72
        bis = math.radians(a0 + 36)
        ox, oy = cx + off * math.cos(bis), cy + off * math.sin(bis)
        d.pieslice([ox - r, oy - r, ox + r, oy + r], a0, a0 + 72,
                   outline=WHITE + (int(235 * ap),), width=6)


def g_two(lay, lt):
    """«собрать из них два шара»: исходный сверху, два новых снизу."""
    d = ImageDraw.Draw(lay)
    globe(d, 540, 872, 115, lt / 0.32, width=8, mer=clamp01((lt - 0.30) / 0.25))
    u = clamp01((lt - 0.45) / 0.30)
    if u > 0:
        e = ease_out(u)
        for sx in (-1, 1):
            arrow(d, (540, 1002),
                  (540 + sx * 125 * e, 1002 + 163 * e), alpha=int(200 * u), width=6)
    globe(d, 380, 1330, 115, (lt - 0.80) / 0.34, width=8,
          mer=clamp01((lt - 1.10) / 0.25))
    globe(d, 700, 1330, 115, (lt - 0.98) / 0.34, width=8,
          mer=clamp01((lt - 1.28) / 0.25))


def g_equal(lay, lt):
    """«оба полного размера, как и исходный»: два шара + равные мерные линейки.

    Выноски от краёв шара к линейке — иначе линейка читается как отдельная палка,
    а не как измерение диаметра именно этого шара.
    """
    d = ImageDraw.Draw(lay)
    r = 150
    for i, cx in enumerate((355, 725)):
        globe(d, cx, 1070, r, (lt - i * 0.22) / 0.38, width=8,
              mer=clamp01((lt - 0.40 - i * 0.22) / 0.28))
        u = clamp01((lt - 0.85 - i * 0.20) / 0.35)
        if u <= 0:
            continue
        e = ease_out(u)
        for s in (-1, 1):
            d.line([(cx + s * r, 1070), (cx + s * r, 1288)],
                   fill=WHITE + (int(85 * u),), width=3)
        bar(d, cx - r, cx - r + 2 * r * e, 1305, alpha=int(228 * u))


def g_pieces(lay, lt):
    """«на несколько особым образом подобранных частей»: разбиение на 5 частей.

    Число 5 в речи не звучит — это минимальное число кусков в теореме,
    добавление к сказанному, а не дубль субтитра (brand-kit §5).
    """
    d = ImageDraw.Draw(lay)
    cx, cy, r = 540, 1245, 168
    circ(d, cx, cy, r, lt / 0.30, width=8)
    for i in range(5):
        u = clamp01((lt - 0.30 - i * 0.085) / 0.24)
        if u <= 0:
            continue
        a = math.radians(-90 + i * 72)
        e = ease_out(u)
        d.line([(cx, cy), (cx + r * math.cos(a) * e, cy + r * math.sin(a) * e)],
               fill=WHITE + (int(205 * u),), width=5)
    p, o = blue_pop(lt - 0.90, 0.36)
    if p > 0:
        lay.alpha_composite(text_layer((W, H), [
            gtext("5", (540, 890), int(118 * (0.55 + 0.45 * p) * o), blue=True),
        ]))


def g_points(lay, lt):
    """«раскидав его точки по запутанным группам»: шар осыпается в облако точек."""
    d = ImageDraw.Draw(lay)
    fade = 1.0 - 0.72 * clamp01(lt / 0.55)
    circ(d, 540, 1150, 168, 1.0, width=6, alpha=int(200 * fade))
    for k, (x, y, g) in enumerate(CLOUD):
        u = clamp01((lt - 0.15 - (k % 30) * 0.020) / 0.26)
        if u <= 0:
            continue
        dot(d, x, y, (3.5 + g * 0.9) * ease_out(u), 90 + 24 * g)


def g_rule(lay, lt):
    """«по правилу, куда каждая точка должна попасть»: одна точка и её образы.

    Лаймовая точка — единственный элемент второго акцента на весь ролик
    (brand-kit §4).
    """
    d = ImageDraw.Draw(lay)
    circ(d, 540, 1150, 168, 1.0, width=6, alpha=55)
    for k, (x, y, g) in enumerate(CLOUD):
        dot(d, x, y, 3.5 + g * 0.9, 70 + 14 * g)
    src = (470, 1085)
    dot(d, src[0], src[1], 13, 250, LIME)
    bez_arrow(d, src, (620, 990), (672, 1210), clamp01((lt - 0.45) / 0.55),
              alpha=225, width=6)
    bez_arrow(d, src, (380, 1120), (432, 1298), clamp01((lt - 1.15) / 0.55),
              alpha=190, width=5)


def g_rot(lay, lt):
    """«вращать шар в разные стороны»: два кольца вращения вокруг разных осей."""
    d = ImageDraw.Draw(lay)
    cx, cy, r = 540, 1150, 162
    globe(d, cx, cy, r, lt / 0.28, width=8, mer=clamp01((lt - 0.22) / 0.22))
    u = clamp01((lt - 0.35) / 0.40)
    if u > 0:
        box = [cx - r - 38, cy - r * 0.34 - 38, cx + r + 38, cy + r * 0.34 + 38]
        d.arc(box, 200, 200 + 250 * ease_out(u), fill=WHITE + (int(210 * u),), width=6)
        if u > 0.9:
            arrow(d, (cx + r + 6, cy - 44), (cx + r + 30, cy + 6), alpha=210, width=6, head=18)
    v = clamp01((lt - 0.75) / 0.40)
    if v > 0:
        box = [cx - r * 0.40 - 38, cy - r - 38, cx + r * 0.40 + 38, cy + r + 38]
        d.arc(box, 120, 120 + 250 * ease_out(v), fill=WHITE + (int(190 * v),), width=6)


def g_move(lay, lt):
    """«вращая и перемещая эти группы точек»: кластеры едут на новые места."""
    d = ImageDraw.Draw(lay)
    groups = [((330, 980), (430, 1360)), ((730, 990), (320, 1120)),
              ((390, 1270), (760, 1330)), ((700, 1320), (700, 980))]
    for gi, (a, b) in enumerate(groups):
        u = ease_out(clamp01((lt - 0.15 - gi * 0.10) / 0.75))
        cxg, cyg = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
        if 0.02 < u < 0.99:
            d.line([a, (cxg, cyg)], fill=WHITE + (95,), width=3)
        rnd = random.Random(100 + gi)
        for k in range(18):
            ang = rnd.random() * math.tau + u * 2.4
            rr = 52 * math.sqrt(rnd.random())
            dot(d, cxg + rr * math.cos(ang), cyg + rr * math.sin(ang), 6.5, 215)


def g_nostretch(lay, lt):
    """«без растяжения»: рядом с шаром его растянутая копия, перечёркнута только она.

    Перечёркивать сам шар нельзя: кадр тогда читается как «шара нет», а речь
    запрещает не шар, а растяжение.
    """
    d = ImageDraw.Draw(lay)
    globe(d, 360, 1130, 118, lt / 0.30, width=8, mer=clamp01((lt - 0.25) / 0.25))
    u = clamp01((lt - 0.30) / 0.34)
    if u > 0:
        e = ease_out(u)
        rx = 118 + 42 * e
        d.ellipse([700 - rx, 1130 - 118, 700 + rx, 1130 + 118],
                  outline=WHITE + (int(215 * u),), width=7)
        for s in (-1, 1):
            arrow(d, (700 + s * 124, 1130), (700 + s * (rx + 20), 1130),
                  alpha=int(180 * u), width=6, head=17)
    v = clamp01((lt - 0.72) / 0.34)
    if v > 0:
        e = ease_out(v)
        d.line([(548, 1296), (548 + 306 * e, 1296 - 306 * e)],
               fill=WHITE + (242,), width=12)


def g_nonew(lay, lt):
    """«без добавления нового материала»: лишний кусок не подкладывается."""
    d = ImageDraw.Draw(lay)
    globe(d, 450, 1150, 128, lt / 0.30, width=8, mer=clamp01((lt - 0.25) / 0.25))
    u = clamp01((lt - 0.35) / 0.26)
    if u > 0:
        a = int(210 * u)
        d.line([(624, 1150), (706, 1150)], fill=WHITE + (a,), width=8)
        d.line([(665, 1109), (665, 1191)], fill=WHITE + (a,), width=8)
    fade = 1.0 - clamp01((lt - 1.35) / 0.40)
    globe(d, 822, 1150, 58, clamp01((lt - 0.55) / 0.30), width=6,
          alpha=int(225 * fade))
    v = clamp01((lt - 0.95) / 0.32)
    if v > 0:
        e = ease_out(v)
        d.line([(598, 1268), (598 + 292 * e, 1268 - 236 * e)],
               fill=WHITE + (240,), width=12)


def g_assemble(lay, lt):
    """«можно собрать из них два шара»: куски слетаются в две сферы."""
    d = ImageDraw.Draw(lay)
    r = 132
    for bi, cx in enumerate((395, 685)):
        u = ease_out(clamp01((lt - bi * 0.10) / 0.70))
        off = 55 * (1 - u)
        for i in range(3):
            a0 = -90 + i * 120
            bis = math.radians(a0 + 60)
            ox, oy = cx + off * math.cos(bis), 1200 + off * math.sin(bis)
            d.pieslice([ox - r, oy - r, ox + r, oy + r], a0, a0 + 120,
                       outline=WHITE + (int(150 + 60 * u),), width=5)
        circ(d, cx, 1200, r, clamp01((lt - 0.75 - bi * 0.10) / 0.32), width=8)


def g_measure(lay, lt):
    """«каждый будет точно такого же размера, как и исходный»: три равные линейки."""
    d = ImageDraw.Draw(lay)
    r = 105
    globe(d, 540, 880, r, lt / 0.30, width=7, alpha=190,
          mer=clamp01((lt - 0.25) / 0.25))
    u = clamp01((lt - 0.35) / 0.30)
    if u > 0:
        bar(d, 540 - r, 540 - r + 2 * r * ease_out(u), 1010, alpha=int(190 * u))
    for i, cx in enumerate((350, 730)):
        globe(d, cx, 1290, r, (lt - 0.75 - i * 0.18) / 0.32, width=8,
              mer=clamp01((lt - 1.05 - i * 0.18) / 0.25))
        v = clamp01((lt - 1.30 - i * 0.18) / 0.35)
        if v > 0:
            bar(d, cx - r, cx - r + 2 * r * ease_out(v), 1420, alpha=int(230 * v))


def g_physics(lay, lt):
    """«звучит как нарушение закона физики»: один шар физически не равен двум."""
    d = ImageDraw.Draw(lay)
    globe(d, 290, 1140, 78, lt / 0.22, width=7, mer=clamp01((lt - 0.18) / 0.18))
    u = clamp01((lt - 0.30) / 0.22)
    if u > 0:
        a = int(230 * u)
        d.line([(438, 1118), (522, 1118)], fill=WHITE + (a,), width=7)
        d.line([(438, 1162), (522, 1162)], fill=WHITE + (a,), width=7)
    v = clamp01((lt - 0.48) / 0.20)
    if v > 0:
        d.line([(452, 1188), (452 + 60 * ease_out(v), 1188 - 96 * ease_out(v))],
               fill=WHITE + (240,), width=7)
    globe(d, 645, 1140, 70, (lt - 0.26) / 0.22, width=7, mer=clamp01((lt - 0.44) / 0.18))
    globe(d, 812, 1140, 70, (lt - 0.38) / 0.22, width=7, mer=clamp01((lt - 0.56) / 0.18))


def g_theorem(lay, lt):
    """«строго доказанная математическая теорема»: галка и знак конца доказательства."""
    d = ImageDraw.Draw(lay)
    p = ease_out(clamp01((lt - 0.05) / 0.45))
    a, b, c = (400, 1150), (492, 1252), (704, 1006)
    if p > 0:
        f1 = min(1.0, p / 0.40)
        d.line([a, (a[0] + (b[0] - a[0]) * f1, a[1] + (b[1] - a[1]) * f1)],
               fill=WHITE + (238,), width=14)
        if p > 0.40:
            f2 = (p - 0.40) / 0.60
            d.line([b, (b[0] + (c[0] - b[0]) * f2, b[1] + (c[1] - b[1]) * f2)],
                   fill=WHITE + (238,), width=14)
    u = clamp01((lt - 0.66) / 0.30)
    if u > 0:                                  # знак конца доказательства
        s = int(64 * ease_out(u))
        d.rectangle([712, 1348 - s, 712 + s, 1348], fill=WHITE + (230,))


def g_y1924(lay, lt):
    """Год публикации Банаха и Тарского. В речи год не звучит — это добавление."""
    p, o = blue_pop(lt, 0.38)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("1924", (540, 1120), int(200 * (0.55 + 0.45 * p) * o), blue=True),
        gtext("год теоремы", (540, 1330), 56, serif=True,
              op=clamp01((lt - 0.45) / 0.32)),
    ]))


def g_infset(lay, lt):
    """«бесконечное множество точек»: точки всё не кончаются."""
    d = ImageDraw.Draw(lay)
    circ(d, 540, 1140, 178, 1.0, width=6, alpha=105)
    for k, (x, y, g) in enumerate(CLOUD_BIG):
        u = clamp01((lt - 0.05 - k * 0.0055) / 0.22)
        if u <= 0:
            continue
        dot(d, x, y, 4.2 * ease_out(u), 120 + 22 * g)


def g_finite(lay, lt):
    """«конечный набор кусков»: пять кусков, их можно пересчитать."""
    d = ImageDraw.Draw(lay)
    w, gap = 112, 24
    x0 = 540 - (5 * w + 4 * gap) // 2
    items = []
    for i in range(5):
        u = clamp01((lt - 0.10 - i * 0.10) / 0.26)
        if u <= 0:
            continue
        e = ease_out(u)
        x = x0 + i * (w + gap)
        h = int(150 * e)
        d.rounded_rectangle([x, 1155 - h // 2, x + w, 1155 + h // 2], radius=18,
                            outline=WHITE + (int(225 * u),), width=5)
        if u > 0.85:                       # штриховка: кусок вещества, а не пустая рамка
            for k in range(3):
                d.line([(x + 22 + k * 30, 1215), (x + 62 + k * 30, 1105)],
                       fill=WHITE + (70,), width=3)
        v = clamp01((lt - 0.28 - i * 0.10) / 0.20)
        if v > 0:
            items.append(gtext(str(i + 1), (x + w // 2, 1330), 46, op=v))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_infeq(lay, lt):
    """«забрать часть и получить снова такое же множество».

    Верхний ряд — исходное множество, у него вычёркиваются три элемента.
    Нижний ряд — то, что осталось, после перенумерации: выноски 4->1, 5->2, 6->3
    показывают, что это та же самая нумерация с начала. Ряды совпадают, отсюда «=».
    """
    d = ImageDraw.Draw(lay)
    xs = [285 + i * 58 for i in range(9)]
    for i, x in enumerate(xs):                      # исходное множество
        u = clamp01((lt - i * 0.045) / 0.22)
        if u > 0:
            dot(d, x, 920, 11 * ease_out(u), 225)
    for i in range(3):                              # забираем часть
        u = clamp01((lt - 0.62 - i * 0.08) / 0.22)
        if u <= 0:
            continue
        e, a = ease_out(u), int(240 * u)
        for s in (1, -1):
            d.line([(xs[i] - 17 * s, 920 - 17), (xs[i] - 17 * s + 34 * s * e, 920 + 17)],
                   fill=WHITE + (a,), width=5)
    for i in range(3):                              # перенумерация 4->1, 5->2, 6->3
        u = ease_out(clamp01((lt - 1.05 - i * 0.10) / 0.42))
        if u <= 0:
            continue
        a, b = (xs[i + 3], 948), (xs[i], 1152)
        d.line([a, (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)],
               fill=WHITE + (120,), width=3)
    for i, x in enumerate(xs):                      # то же множество заново
        u = clamp01((lt - 1.20 - i * 0.055) / 0.24)
        if u > 0:
            dot(d, x, 1180, 11 * ease_out(u), 225)
    items = []
    if lt > 0.50:
        items.append(gtext("…", (800, 906), 42, op=clamp01((lt - 0.50) / 0.25)))
    if lt > 1.75:
        items.append(gtext("…", (800, 1166), 42, op=clamp01((lt - 1.75) / 0.25)))
        items.append(gtext("=", (250, 1050), 58, op=clamp01((lt - 1.80) / 0.28)))
    p, o = blue_pop(lt - 2.05, 0.38)
    if p > 0:
        items.append(gtext("∞", (540, 1390), int(132 * (0.55 + 0.45 * p) * o), blue=True))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_hotel(lay, lt):
    """«всегда находится место для новых гостей»: гости сдвигаются, номер 1 освобождается.

    Новый гость нарисован кольцом, а не заливкой: иначе финальный кадр совпадает
    с начальным («все номера заняты») и механизм на паузе не читается.
    """
    d = ImageDraw.Draw(lay)
    w, gap = 92, 14
    x0 = 540 - (6 * w + 5 * gap) // 2
    step = w + gap
    shift = ease_out(clamp01((lt - 0.80) / 0.50))
    free = clamp01((lt - 1.28) / 0.22) * (1.0 - clamp01((lt - 2.05) / 0.20))
    items = []
    for i in range(6):
        u = clamp01((lt - 0.10 - i * 0.06) / 0.22)
        if u <= 0:
            continue
        x = x0 + i * step
        a = int(200 * u + 55 * (free if i == 0 else 0))
        d.rounded_rectangle([x, 1080, x + w, 1220], radius=14,
                            outline=WHITE + (min(255, a),), width=4 + (2 if i == 0 and free > 0.5 else 0))
        items.append(gtext(str(i + 1), (x + w // 2, 1012), 40, op=u))
    if 0.02 < shift < 0.99:                          # направление сдвига
        for i in range(5):
            arrow(d, (x0 + w // 2 + i * step + 22, 1268),
                  (x0 + w // 2 + i * step + step - 22, 1268),
                  alpha=110, width=4, head=12)
    for i in range(6):
        if clamp01((lt - 0.10 - i * 0.06) / 0.22) <= 0:
            continue
        x = x0 + w // 2 + (i + shift) * step
        a = 225 * (1.0 - clamp01((x - 826) / 44)) if i == 5 else 225
        dot(d, x, 1150, 16, max(0, a))
    v = clamp01((lt - 1.58) / 0.44)                  # новый гость входит в первый номер
    if v > 0:
        e = ease_out(v)
        gx, gy = x0 + w // 2, 1370 - 220 * e
        d.ellipse([gx - 16, gy - 16, gx + 16, gy + 16], outline=WHITE + (240,), width=5)
        if v < 0.92:
            arrow(d, (gx, 1382), (gx, gy + 32), alpha=int(160 * (1 - v)), width=5, head=16)
    if items:
        lay.alpha_composite(text_layer((W, H), items))


GFX = {
    "ball": g_ball, "split": g_split, "two": g_two, "equal": g_equal,
    "pieces": g_pieces, "points": g_points, "rule": g_rule, "rot": g_rot,
    "move": g_move, "nostretch": g_nostretch, "nonew": g_nonew,
    "assemble": g_assemble, "measure": g_measure, "physics": g_physics,
    "theorem": g_theorem, "y1924": g_y1924, "infset": g_infset,
    "finite": g_finite, "infeq": g_infeq, "hotel": g_hotel,
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


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    tmp = f"{BUILD}/assets/_video_banach_raw.mp4"
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

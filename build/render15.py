"""Сборка ролика 15 («доказательство теоремы Пифагора, придуманное Гарфилдом»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx15.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Вся предметная графика построена на одном честном чертеже (shot-recipes R10:
«геометрия должна быть проверяемой»):

* классическая картинка Пифагора — прямоугольный треугольник с катетами 120/160
  и квадратами, построенными наружу на каждой стороне; квадрат на гипотенузе
  строится по нормали к ней, а не «на глаз»;
* чертёж Гарфилда — два одинаковых прямоугольных треугольника (катеты a=230,
  b=330) и трапеция O-B-T-Q с параллельными сторонами b и a и высотой a+b.
  Третий треугольник P-B-T выходит равнобедренным прямоугольным: |PB| = |PT| =
  sqrt(a²+b²) = 402, |BT| = 402·√2 = 569 — проверено по координатам.
  Отсюда ab/2 + ab/2 + c²/2 = (a+b)²/2 и, после упрощения, c² = a² + b².
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard15 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_garfield.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/15/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GRID_CACHE = {}

# крупность A-roll: детектор даёт лицо x~304 y~659 d~433 на 1080x1920,
# после hflip центр лица x=560, линия глаз y=838. Кропы поставлены так,
# чтобы лицо село по центру карточки, а глаза — на 42-43% её высоты.
FRAMINGS = {
    "A1": dict(w=1000, h=1586, x=60, y=164),
    "A2": dict(w=880, h=1396, x=120, y=242),
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
    проверено по кадру на 12с, дубль снят фронталкой.
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

def seg(d, a, b, p=1.0, width=7, alpha=230, color=WHITE):
    """Отрезок a->b, прорисованный на долю p."""
    p = clamp01(p)
    if p <= 0.002:
        return
    d.line([a, (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)],
           fill=color + (int(alpha),), width=width)


def poly(d, pts, p=1.0, width=7, alpha=230, close=True, color=WHITE):
    """Замкнутый контур, прорисованный по периметру на долю p."""
    p = clamp01(p)
    if p <= 0.002:
        return
    seq = list(pts) + ([pts[0]] if close else [])
    lens = [math.dist(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    total = sum(lens)
    done = total * p
    for i, ln in enumerate(lens):
        if done <= 0:
            break
        seg(d, seq[i], seq[i + 1], min(1.0, done / ln), width, alpha, color)
        done -= ln


def fill_poly(d, pts, alpha, color=WHITE):
    if alpha <= 1:
        return
    d.polygon([(float(x), float(y)) for x, y in pts], fill=color + (int(alpha),))


def right_angle(d, corner, p1, p2, size=26, alpha=210, width=5):
    """Маркер прямого угла: квадратик в вершине corner между направлениями на p1 и p2."""
    def unit(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n
    u1, u2 = unit(corner, p1), unit(corner, p2)
    a = (corner[0] + u1[0] * size, corner[1] + u1[1] * size)
    b = (corner[0] + (u1[0] + u2[0]) * size, corner[1] + (u1[1] + u2[1]) * size)
    c = (corner[0] + u2[0] * size, corner[1] + u2[1] * size)
    d.line([a, b, c], fill=WHITE + (int(alpha),), width=width, joint="curve")


def arrow(d, a, b, alpha=215, width=7, head=24, color=WHITE):
    d.line([a, b], fill=color + (int(alpha),), width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (1, -1):
        d.line([b, (b[0] - head * math.cos(ang - s * 0.45),
                    b[1] - head * math.sin(ang - s * 0.45))],
               fill=color + (int(alpha),), width=width)


def rot(pts, ang, c):
    ca, sa = math.cos(ang), math.sin(ang)
    return [(c[0] + (x - c[0]) * ca - (y - c[1]) * sa,
             c[1] + (x - c[0]) * sa + (y - c[1]) * ca) for x, y in pts]


def shift(pts, dx, dy):
    return [(x + dx, y + dy) for x, y in pts]


def lerp_pts(a, b, u):
    return [(p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u) for p, q in zip(a, b)]


def centroid(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


# --- геометрия ролика ------------------------------------------------------

# 1. Классическая картинка Пифагора: треугольник 120/160/200 и три квадрата,
#    построенные наружу. Квадрат на гипотенузе строится по нормали (b, a)/c.
PA, PB_ = 120.0, 160.0            # катеты в «чертёжных» единицах
P_S, P_CX, P_CY = 1.55, 540, 1130


def _pm(x, y):
    return (P_CX + P_S * (x - 60), P_CY - P_S * (y - 80))


TRI_P = [_pm(0, 0), _pm(PA, 0), _pm(0, PB_)]                       # C, B, A
SQ_A = [_pm(0, 0), _pm(PA, 0), _pm(PA, -PA), _pm(0, -PA)]          # на нижнем катете
SQ_B = [_pm(0, 0), _pm(0, PB_), _pm(-PB_, PB_), _pm(-PB_, 0)]      # на левом катете
SQ_C = [_pm(0, PB_), _pm(PA, 0), _pm(PA + PB_, PA), _pm(PB_, PA + PB_)]

# 2. Подобие: та же форма крупнее + высота из прямого угла на гипотенузу.
#    Основание высоты считается точно: H = (a·b²/(a²+b²), a²·b/(a²+b²)).
S_S, S_CX, S_CY = 4.2, 540, 1130


def _sm(x, y):
    return (S_CX + S_S * (x - 60), S_CY - S_S * (y - 80))


SIM_C, SIM_B, SIM_A = _sm(0, 0), _sm(PA, 0), _sm(0, PB_)
_den = PA * PA + PB_ * PB_
SIM_H = _sm(PA * PB_ * PB_ / _den, PA * PA * PB_ / _den)

# 3. Чертёж Гарфилда. Катеты a=230 (вертикаль) и b=330 (горизонталь).
GO = (375.0, 1440.0)      # прямой угол первого треугольника
GB = (705.0, 1440.0)      # GO->GB = b
GP = (375.0, 1210.0)      # GO->GP = a, гипотенуза GP->GB = c = 402
GQ = (375.0, 880.0)       # GP->GQ = b
GT = (605.0, 880.0)       # GQ->GT = a, гипотенуза GP->GT = c
TRAP = [GO, GB, GT, GQ]
T1 = [GO, GB, GP]
T2 = [GQ, GT, GP]
T3 = [GP, GB, GT]         # равнобедренный прямоугольный: |PB| = |PT| = c
DY_TRI = -195.0           # одиночный треугольник стоит по центру зоны
T1_SOLO = shift(T1, 0, DY_TRI)


def pyth_figure(d, p_tri, p_sq, alpha=230, width=7):
    """Треугольник + три квадрата: общая картинка для планов pyth / sq / nosq."""
    poly(d, TRI_P, p_tri, width=width, alpha=alpha)
    for i, sq in enumerate((SQ_A, SQ_B, SQ_C)):
        poly(d, sq, clamp01((p_sq - i * 0.16) / 0.34), width=width - 1, alpha=alpha)


# --- графика планов -------------------------------------------------------

def g_pyth(lay, lt):
    """«теорему пифагора»: классический чертёж — треугольник и три квадрата."""
    d = ImageDraw.Draw(lay)
    pyth_figure(d, clamp01(lt / 0.38), clamp01((lt - 0.34) / 1.0))


def g_n370(lay, lt):
    """«больше 370 разными способами»: число живёт только в графике,
    под ним — счётные штрихи, которые всё не кончаются."""
    d = ImageDraw.Draw(lay)
    # штрихи ставятся редко и строго вертикально: с шагом 26 и наклоном они
    # сливались в штриховку и читались как «зачёркнуто», а не «много способов»
    for k in range(30):
        u = clamp01((lt - 0.55 - k * 0.030) / 0.18)
        if u <= 0:
            continue
        x = 333 + (k % 10) * 46
        y = 1300 + (k // 10) * 74
        d.line([(x, y), (x, y + 34 * ease_out(u))],
               fill=WHITE + (int(200 * u),), width=6)
    p, o = blue_pop(lt - 0.12, 0.38)          # старт на резе, а не через 0.35с
    if p > 0:
        lay.alpha_composite(text_layer((W, H), [
            gtext("370", (540, 1060), int(196 * (0.55 + 0.45 * p) * o), blue=True),
        ]))


def g_sq(lay, lt):
    """«через площади квадратов»: у той же картинки заливаются площади —
    два квадрата на катетах против одного на гипотенузе."""
    d = ImageDraw.Draw(lay)
    for i, sq in enumerate((SQ_A, SQ_B, SQ_C)):
        u = clamp01((lt - 0.08 - i * 0.16) / 0.26)
        fill_poly(d, sq, 86 * ease_out(u) if i < 2 else 104 * ease_out(u))
    pyth_figure(d, 1.0, 1.0, alpha=225)


def g_simil(lay, lt):
    """«через подобия треугольников»: высота из прямого угла делит треугольник
    на два, подобных исходному. Основание высоты посчитано, а не поставлено на глаз."""
    d = ImageDraw.Draw(lay)
    poly(d, [SIM_C, SIM_B, SIM_A], clamp01(lt / 0.34), width=7)
    # высота — граница между двумя подобными треугольниками, поэтому она ярче
    # и толще заливок: на равных тонах деление на два треугольника не читалось
    seg(d, SIM_C, SIM_H, clamp01((lt - 0.30) / 0.26), width=7, alpha=242)
    for i, tri in enumerate(([SIM_A, SIM_C, SIM_H], [SIM_C, SIM_B, SIM_H])):
        u = clamp01((lt - 0.52 - i * 0.16) / 0.24)
        fill_poly(d, tri, (104 if i == 0 else 42) * ease_out(u))
    if lt > 0.50:
        right_angle(d, SIM_H, SIM_C, SIM_B, size=22,
                    alpha=int(190 * clamp01((lt - 0.50) / 0.22)))
    poly(d, [SIM_C, SIM_B, SIM_A], 1.0, width=7, alpha=230)


def g_alg(lay, lt):
    """«через алгебру»: тот же путь записью, без итога — итог придёт в конце ролика."""
    items = []
    u = clamp01(lt / 0.26)
    if u > 0:
        items.append(gtext("(a + b)²", (540, 1010), 74, op=u))
    v = clamp01((lt - 0.30) / 0.26)
    if v > 0:
        items.append(gtext("= a² + 2ab + b²", (540, 1230), 64, op=v))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_music(lay, lt):
    """«музыку пропорций»: две струны — две доли и три доли, стоячая волна живая."""
    d = ImageDraw.Draw(lay)
    items = []
    for i, (y0, loops, amp, lbl) in enumerate(((1000, 2, 46, "1:2"),
                                               (1290, 3, 40, "2:3"))):
        u = clamp01((lt - 0.10 - i * 0.22) / 0.34)
        if u <= 0:
            continue
        x0, x1 = 250, 830
        # огибающая не проходит через ноль: на чистом cos струна в отдельных кадрах
        # выпрямлялась в прямую линию и переставала читаться как струна
        env = 0.58 + 0.42 * math.cos((lt - 0.10 - i * 0.22) * 7.2)
        pts = []
        n = 90
        for k in range(n + 1):
            x = x0 + (x1 - x0) * (k / n) * u
            f = (x - x0) / (x1 - x0)
            pts.append((x, y0 - amp * math.sin(loops * math.pi * f) * env))
        if len(pts) > 1:
            d.line(pts, fill=WHITE + (230,), width=6, joint="curve")
        for k in range(loops + 1):                       # узлы стоячей волны
            x = x0 + (x1 - x0) * k / loops
            if x <= x0 + (x1 - x0) * u + 2:
                d.ellipse([x - 8, y0 - 8, x + 8, y0 + 8], fill=WHITE + (225,))
        w = clamp01((lt - 0.45 - i * 0.22) / 0.28)
        if w > 0:
            items.append(gtext(lbl, (540, y0 + 118), 44, op=w))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_y1876(lay, lt):
    """Год доказательства. Слово «год» в речи звучит, число — тоже, но на экране
    число живёт только здесь, а субтитра на этом плане нет (brand-kit §5)."""
    p, o = blue_pop(lt - 0.12, 0.38)          # старт на резе
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("1876", (540, 1090), int(190 * (0.55 + 0.45 * p) * o), blue=True),
        gtext("год доказательства", (540, 1300), 50, serif=True,
              op=clamp01((lt - 0.62) / 0.32)),
    ]))


def g_tri(lay, lt):
    """«он взял прямоугольный треугольник»: катеты a и b, гипотенуза c."""
    d = ImageDraw.Draw(lay)
    o, b, p = T1_SOLO
    poly(d, T1_SOLO, clamp01(lt / 0.40), width=8)
    if lt > 0.42:
        right_angle(d, o, b, p, alpha=int(215 * clamp01((lt - 0.42) / 0.22)))
    items = []
    for i, (txt, xy) in enumerate((("b", (540, o[1] + 46)),
                                   ("a", (o[0] - 44, (o[1] + p[1]) / 2)),
                                   ("c", (584, (p[1] + b[1]) / 2 - 30)))):
        u = clamp01((lt - 0.50 - i * 0.09) / 0.24)
        if u > 0:
            items.append(gtext(txt, xy, 48, serif=True, op=u))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_copy(lay, lt):
    """«приложил к нему такой же, повёрнутый определённым образом».

    Первый треугольник сползает на своё место, второй приезжает, доворачиваясь
    на 90°: его катет a встаёт продолжением катета b первого. Именно из-за
    этого разворота угол при вершине P оказывается прямым.
    """
    d = ImageDraw.Draw(lay)
    u1 = ease_out(clamp01(lt / 0.55))
    cur1 = lerp_pts(T1_SOLO, T1, u1)
    poly(d, cur1, 1.0, width=8)
    right_angle(d, cur1[0], cur1[1], cur1[2], alpha=215)

    u2 = clamp01((lt - 0.65) / 0.95)
    if u2 > 0:
        e = ease_out(u2)
        c2 = centroid(T2)
        start = shift(rot(T2, -math.pi / 2, c2), 215, 150)
        cur2 = lerp_pts(start, T2, e)
        poly(d, cur2, 1.0, width=8, alpha=235)
        if e > 0.92:
            right_angle(d, cur2[0], cur2[1], cur2[2],
                        alpha=int(215 * clamp01((e - 0.92) / 0.08)))
        elif e < 0.85:
            arrow(d, (centroid(cur2)[0] + 96, centroid(cur2)[1] + 70),
                  (centroid(cur2)[0] + 46, centroid(cur2)[1] + 34),
                  alpha=int(150 * (1 - e)), width=5, head=16)


def g_trap(lay, lt):
    """«и получившуюся фигуру, трапецию»: внешний контур замыкается наклонной
    стороной B->T. Стороны b и a параллельны, высота ровно a + b."""
    d = ImageDraw.Draw(lay)
    for tri in (T1, T2):
        poly(d, tri, 1.0, width=5, alpha=120)
    poly(d, TRAP, clamp01((lt - 0.06) / 0.80), width=9, alpha=240)
    u = clamp01((lt - 0.95) / 0.30)
    if u > 0:
        right_angle(d, GO, GB, GQ, alpha=int(200 * u))
        right_angle(d, GQ, GT, GO, alpha=int(200 * u))


def g_three(lay, lt):
    """«как сумма площадей трёх треугольников»: три площади каскадом.

    Третий треугольник P-B-T прямоугольный и равнобедренный — |PB| = |PT| = c,
    поэтому его площадь ровно c²/2. Прямой угол при P помечен.
    """
    d = ImageDraw.Draw(lay)
    poly(d, TRAP, 1.0, width=6, alpha=170)
    items = []
    for i, (tri, lbl) in enumerate(((T1, "ab/2"), (T2, "ab/2"), (T3, "c²/2"))):
        u = clamp01((lt - 0.10 - i * 0.24) / 0.26)
        if u <= 0:
            continue
        # два одинаковых треугольника держат один тон, треугольник c²/2 — заметно
        # светлее: на близких заливках трапеция читалась одним залитым пятном
        fill_poly(d, tri, (64 if i < 2 else 118) * ease_out(u))
        poly(d, tri, 1.0, width=5, alpha=int(215 * u))
        v = clamp01((lt - 0.24 - i * 0.24) / 0.22)
        if v > 0:
            cx, cy = centroid(tri)
            items.append(gtext(lbl, (cx, cy), 46, op=v))
    if lt > 0.72:
        right_angle(d, GP, GB, GT, size=24,
                    alpha=int(210 * clamp01((lt - 0.72) / 0.24)))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_formula(lay, lt):
    """«и по обычной формуле площади трапеции»: параллельные стороны и высота
    подписаны, формула стоит над чертежом, в свободной части зоны."""
    d = ImageDraw.Draw(lay)
    poly(d, TRAP, 1.0, width=7, alpha=225)
    items = []
    u = clamp01((lt - 0.20) / 0.30)
    if u > 0:                                   # высота трапеции = a + b
        a = int(150 * u)
        d.line([(300, GQ[1]), (300, GO[1])], fill=WHITE + (a,), width=4)
        for y in (GQ[1], GO[1]):
            d.line([(282, y), (318, y)], fill=WHITE + (a,), width=4)
    for i, (txt, xy, sz, delay) in enumerate(
            # подпись верхней стороны стоит ПОД ней, внутри пустой трапеции: над
            # стороной она садилась вплотную к формуле и читалась как её кусок
            (("a", ((GQ[0] + GT[0]) / 2, GQ[1] + 48), 48, 0.10),
             ("b", (540, GO[1] + 46), 48, 0.24),
             ("a + b", (246, (GQ[1] + GO[1]) / 2), 44, 0.38))):
        v = clamp01((lt - delay) / 0.24)
        if v > 0:
            items.append(gtext(txt, xy, sz, serif=(txt != "a + b"), op=v))
    w = clamp01((lt - 0.62) / 0.30)
    if w > 0:
        items.append(gtext("(a + b)/2 · (a + b)", (540, 780), 50, op=w))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_equate(lay, lt):
    """«приравняв оба способа подсчёта друг к другу»: две записи одной площади."""
    items = []
    u = clamp01(lt / 0.28)
    if u > 0:
        items.append(gtext("ab/2 + ab/2 + c²/2", (540, 950), 58, op=u))
    v = clamp01((lt - 0.55) / 0.24)
    if v > 0:
        items.append(gtext("=", (540, 1120), 72, op=v))
    w = clamp01((lt - 0.80) / 0.28)
    if w > 0:
        items.append(gtext("(a + b)²/2", (540, 1290), 58, op=w))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


SIMP = "2ab + c² = a² + 2ab + b²"
SIMP_SIZE = 50      # 54 давало габарит 167…913 при зоне 175…905 (замерено qa15)
SIMP_Y = 1090


def g_simplify(lay, lt):
    """«после упрощения»: обе части умножены на 2, одинаковые 2ab вычёркиваются.

    Позиции вычеркиваний берутся из метрик шрифта, а не подбираются: это ровно
    подстроки «2ab» на индексах 0 и 16.
    """
    lay.alpha_composite(text_layer((W, H), [
        gtext(SIMP, (540, SIMP_Y), SIMP_SIZE, op=clamp01(lt / 0.22)),
    ]))
    f = font("sans", SIMP_SIZE)
    left = 540 - f.getlength(SIMP) / 2
    d = ImageDraw.Draw(lay)
    for i, idx in enumerate((0, 16)):
        u = clamp01((lt - 0.42 - i * 0.16) / 0.22)
        if u <= 0:
            continue
        x0 = left + f.getlength(SIMP[:idx])
        x1 = left + f.getlength(SIMP[:idx + 3])
        d.line([(x0 - 6, SIMP_Y + 16), (x0 - 6 + (x1 - x0 + 12) * ease_out(u), SIMP_Y - 18)],
               fill=WHITE + (240,), width=7)


def g_pythblue(lay, lt):
    """«ровно теорему Пифагора»: итог. Формулу несёт графика, субтитра нет."""
    p, o = blue_pop(lt - 0.12, 0.38)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("c² = a² + b²", (540, 1130), int(96 * (0.55 + 0.45 * p) * o), blue=True),
    ]))


def g_nosq(lay, lt):
    """«без единого чертежа с квадратами»: перечёркнута именно картинка
    с квадратами — то, что в речи и объявлено ненужным."""
    d = ImageDraw.Draw(lay)
    pyth_figure(d, clamp01(lt / 0.30), clamp01((lt - 0.16) / 0.50), alpha=150, width=6)
    u = clamp01((lt - 0.85) / 0.40)
    if u > 0:
        e = ease_out(u)
        d.line([(212, 1418), (212 + 656 * e, 1418 - 588 * e)],
               fill=WHITE + (242,), width=13)


GFX = {
    "pyth": g_pyth, "n370": g_n370, "sq": g_sq, "simil": g_simil,
    "alg": g_alg, "music": g_music, "y1876": g_y1876, "tri": g_tri,
    "copy": g_copy, "trap": g_trap, "three": g_three, "formula": g_formula,
    "equate": g_equate, "simplify": g_simplify, "pythblue": g_pythblue,
    "nosq": g_nosq,
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
    tmp = f"{BUILD}/assets/_video_garfield_raw.mp4"
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

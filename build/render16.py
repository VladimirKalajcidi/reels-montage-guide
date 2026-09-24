"""Сборка ролика 16 («почему А4 — это 1 : √2»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx16.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Геометрия листа общая для всех «бумажных» планов: 340 x 481 = 1 : 1.4147.
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
from storyboard16 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_a4.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/16/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GRID_CACHE = {}

# крупность A-roll: лицо детектором стоит стабильно (x~258 y~622 d~476 на 1080x1920),
# после hflip центр лица x=584, линия глаз y=815. Кроп ставит глаза на 42-43% карточки.
FRAMINGS = {
    "A1": dict(w=990, h=1570, x=89, y=156),
    "A2": dict(w=880, h=1396, x=144, y=215),
}

# --- лист А4 (общая геометрия «бумажных» планов) ---
SW, SH = 340, 481          # 481 / 340 = 1.4147 ≈ √2
CX, CY = 540, 1130
L, T, R, B = CX - SW // 2, CY - SH // 2, CX + SW // 2, CY + SH // 2   # 370 890 710 1370
MY = (T + B) / 2           # линия сгиба

CXL = 430                  # лист сдвинут влево — справа место под подпись √2
LL, RL = CXL - SW // 2, CXL + SW // 2                                  # 260 600


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
    """A-roll: hflip + кроп под карточку A. Пиксели больше ничем не трогаются.

    hflip безусловный: надпись на футболке в исходнике читается зеркально —
    проверено по кадру, дубль снят фронталкой.

    Грейда здесь НЕТ сознательно, по прямому требованию: картинку не менять.
    Это расхождение с brand-kit §1/§6 (там грейд «делается всегда» и задан
    ориентир ~130/112/104 по средним). Гамма-кривая, которая выводила кадр
    на этот ориентир, поднимала шум с 1.45 до 2.11 и заметно вымывала лицо,
    поэтому убрана вместе с балансом каналов и подъёмом насыщенности.
    """
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    f = FRAMINGS[kind]
    w, h = f["w"], f["h"]
    x = max(0, min(f["x"], iw - w))
    y = max(0, min(f["y"], ih - h))
    fr = fr[y:y + h, x:x + w]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_AREA)
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

def rrect(d, x0, y0, x1, y1, alpha=235, width=7, color=WHITE):
    if alpha <= 0:
        return
    d.rectangle([x0, y0, x1, y1], outline=color + (int(min(255, alpha)),), width=width)


def path_draw(d, pts, p, width=7, alpha=235, color=WHITE):
    """Прогрессивная отрисовка ломаной: p — доля общей длины (0..1)."""
    p = clamp01(p)
    if p <= 0.001 or alpha <= 0:
        return
    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    lens = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segs]
    want = sum(lens) * p
    col = color + (int(min(255, alpha)),)
    for (a, b), ln in zip(segs, lens):
        if want <= 0:
            break
        if want >= ln:
            d.line([a, b], fill=col, width=width)
            want -= ln
        else:
            u = want / max(ln, 1e-6)
            d.line([a, (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)],
                   fill=col, width=width)
            want = 0


def rect_path(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def dashes(d, x0, x1, y, alpha=180, width=4, dash=22, gap=16):
    x = x0
    col = WHITE + (int(min(255, alpha)),)
    while x < x1:
        d.line([(x, y), (min(x + dash, x1), y)], fill=col, width=width)
        x += dash + gap


def hbar(d, x0, x1, y, alpha=225, width=6, tick=18):
    col = WHITE + (int(min(255, alpha)),)
    d.line([(x0, y), (x1, y)], fill=col, width=width)
    d.line([(x0, y - tick), (x0, y + tick)], fill=col, width=width)
    d.line([(x1, y - tick), (x1, y + tick)], fill=col, width=width)


def vbar(d, y0, y1, x, alpha=225, width=6, tick=18):
    col = WHITE + (int(min(255, alpha)),)
    d.line([(x, y0), (x, y1)], fill=col, width=width)
    d.line([(x - tick, y0), (x + tick, y0)], fill=col, width=width)
    d.line([(x - tick, y1), (x + tick, y1)], fill=col, width=width)


def arrow(d, a, b, alpha=215, width=7, head=24, color=WHITE):
    col = color + (int(min(255, alpha)),)
    d.line([a, b], fill=col, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (1, -1):
        d.line([b, (b[0] - head * math.cos(ang - s * 0.45),
                    b[1] - head * math.sin(ang - s * 0.45))], fill=col, width=width)


def bez_arrow(d, a, c, b, p=1.0, alpha=215, width=6, color=WHITE, head=20):
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
    col = color + (int(min(255, alpha)),)
    d.line(pts, fill=col, width=width, joint="curve")
    if p > 0.92:
        ang = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
        for s in (1, -1):
            d.line([pts[-1], (pts[-1][0] - head * math.cos(ang - s * 0.45),
                              pts[-1][1] - head * math.sin(ang - s * 0.45))],
                   fill=col, width=width)


def rot_poly(cx, cy, hw, hh, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


def poly(d, pts, alpha=235, width=7, color=WHITE):
    if alpha <= 0:
        return
    d.line(list(pts) + [pts[0]], fill=color + (int(min(255, alpha)),),
           width=width, joint="curve")


def ruled(d, x0, x1, y, n, alpha, step=80):
    """Полоски «текста» внутри листа — чтобы прямоугольник читался как бумага."""
    for i in range(n):
        d.line([(x0, y + i * step), (x1, y + i * step)],
               fill=WHITE + (int(min(255, alpha)),), width=4)


# --- графика планов -------------------------------------------------------

def g_sheet(lay, lt):
    """«не случайно, он математически очень удобен»: вычерчивается лист А4."""
    d = ImageDraw.Draw(lay)
    path_draw(d, rect_path(L, T, R, B), ease_out(clamp01(lt / 0.55)), width=7, alpha=235)
    for i in range(4):
        u = clamp01((lt - 0.70 - i * 0.09) / 0.26)
        if u > 0:
            d.line([(L + 46, T + 120 + i * 95), (R - 46, T + 120 + i * 95)],
                   fill=WHITE + (int(95 * u),), width=4)


def g_fold(lay, lt):
    """«сложите его пополам до длинной стороны»: верхняя половина ложится на нижнюю."""
    d = ImageDraw.Draw(lay)
    v = ease_out(clamp01((lt - 0.60) / 0.80))
    path_draw(d, rect_path(L, T, R, B), clamp01(lt / 0.28), width=7,
              alpha=int(120 + 115 * (1 - v)))
    u = clamp01((lt - 0.30) / 0.20)
    if u > 0:
        dashes(d, L, L + (R - L) * u, MY, alpha=190)
    if v > 0.02:
        rrect(d, L, MY, R, B, alpha=235, width=7)
        y_edge = MY - (SH / 2) * math.cos(math.pi * v)
        rrect(d, L, min(MY, y_edge), R, max(MY, y_edge), alpha=235, width=7)
    bez_arrow(d, (R + 34, T + 70), (866, CY), (R + 34, B - 70),
              clamp01((lt - 0.45) / 0.55), alpha=190, width=6)


def g_half(lay, lt):
    """«получившийся прямоугольник — это формат а5»: половинка отделяется.

    Подписи «А5» на графике нет: это слово звучит в речи и его несёт субтитр
    (brand-kit §5, «одно и то же не пишется дважды»).
    """
    d = ImageDraw.Draw(lay)
    u = ease_out(clamp01((lt - 0.35) / 0.65))
    rrect(d, L, T, R, B, alpha=int(110 * (1 - u)), width=6)
    dy = -120 * u
    rt, rb = MY + dy, B + dy
    rrect(d, L, rt, R, rb, alpha=235, width=7)
    for i in range(2):
        w = clamp01((lt - 1.10 - i * 0.10) / 0.26)
        if w > 0:
            d.line([(L + 46, rt + 80 + i * 80), (R - 46, rt + 80 + i * 80)],
                   fill=WHITE + (int(95 * w),), width=4)


def g_same(lay, lt):
    """«окажется точно такой же, как у исходного листа».

    Половинка поворачивается на 90° и растёт в √2 раз — и точно ложится
    на контур исходного листа. Совпадение и есть доказательство кадра.
    """
    d = ImageDraw.Draw(lay)
    rrect(d, L, T, R, B, alpha=95, width=6)
    ang = 90 * ease_out(clamp01((lt - 0.25) / 0.55))
    s = 1 + (math.sqrt(2) - 1) * ease_out(clamp01((lt - 0.95) / 0.55))
    done = clamp01((lt - 1.55) / 0.30)
    poly(d, rot_poly(CX, CY, (SW / 2) * s, (SH / 4) * s, ang),
         alpha=int(225 + 25 * done), width=int(7 + 2 * done))
    if done > 0:
        # уголки ставятся СНАРУЖИ контура: изнутри они сливаются с самой рамкой
        # и совпадение половинки с исходным листом на стоп-кадре не читается
        a = int(235 * done)
        for px, py in ((L, T), (R, T), (R, B), (L, B)):
            sx = -1 if px == L else 1
            sy = -1 if py == T else 1
            ox, oy = px + sx * 20, py + sy * 20
            d.line([(ox, oy), (ox - sx * 40, oy)], fill=WHITE + (a,), width=6)
            d.line([(ox, oy), (ox, oy - sy * 40)], fill=WHITE + (a,), width=6)


def g_sides(lay, lt):
    """«благодаря конкретному соотношению сторон»: обе стороны берутся в размер."""
    d = ImageDraw.Draw(lay)
    path_draw(d, rect_path(LL, T, RL, B), clamp01(lt / 0.25), width=7, alpha=235)
    u = clamp01((lt - 0.30) / 0.30)
    if u > 0:
        for x in (LL, RL):
            d.line([(x, B), (x, 1430)], fill=WHITE + (int(85 * u),), width=3)
        hbar(d, LL, LL + SW * ease_out(u), 1430, alpha=int(228 * u))
    v = clamp01((lt - 0.60) / 0.30)
    if v > 0:
        for y in (T, B):
            d.line([(RL, y), (660, y)], fill=WHITE + (int(85 * v),), width=3)
        vbar(d, T, T + SH * ease_out(v), 660, alpha=int(228 * v))


def g_ratio(lay, lt):
    """R5b: «один к квадратному корню из двух» — числа у сторон листа.

    Субтитра на этом плане нет: смысл целиком несёт графика.
    """
    d = ImageDraw.Draw(lay)
    rrect(d, LL, T, RL, B, alpha=205, width=7)
    for x in (LL, RL):
        d.line([(x, B), (x, 1430)], fill=WHITE + (80,), width=3)
    for y in (T, B):
        d.line([(RL, y), (660, y)], fill=WHITE + (80,), width=3)
    hbar(d, LL, LL + SW, 1430, alpha=190)
    vbar(d, T, T + SH, 660, alpha=190)
    items = []
    p1 = pop(lt - 0.35, 0.12)
    if p1 > 0:
        items.append(gtext("1", (CXL, 1478), int(62 * (0.6 + 0.4 * p1)), op=p1))
    p2, o2 = blue_pop(lt - 0.85, 0.38)
    if p2 > 0:
        items.append(gtext("√2", (740, CY), int(124 * (0.55 + 0.45 * p2) * o2), blue=True))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_halfproof(lay, lt):
    """«только это соотношение сохраняет в себе определение пополам».

    Каскад вложенных половинок: каждая следующая — ровно половина предыдущей
    и при этом ровно та же форма. Это и есть «сохраняет определение пополам».
    """
    d = ImageDraw.Draw(lay)
    levels = [(L, T, R, B)]
    x0, y0, x1, y1 = L, T, R, B
    for i in range(4):
        if i % 2 == 0:                      # режем по горизонтали — берём низ
            y0 = (y0 + y1) / 2
        else:                               # режем по вертикали — берём право
            x0 = (x0 + x1) / 2
        levels.append((x0, y0, x1, y1))
    for i, (a, b, c, e) in enumerate(levels):
        u = clamp01((lt - 0.15 - i * 0.28) / 0.24)
        if u <= 0:
            continue
        rrect(d, a, b, c, e, alpha=int(235 * ease_out(u)), width=max(4, 7 - i))


def g_algebra(lay, lt):
    """«это можно доказать очень простой алгеброй» — сам вывод.

    Вслух формула не звучит: субтитр несёт слова, графика — выкладку.
    """
    d = ImageDraw.Draw(lay)
    items = []
    p1 = pop(lt - 0.05, 0.30)
    if p1 > 0:
        items.append(gtext("x : 1  =  1 : x/2", (CX, 880), 60, op=p1))
    if clamp01((lt - 0.55) / 0.22) > 0:
        u = clamp01((lt - 0.55) / 0.22)
        arrow(d, (CX, 946), (CX, 946 + 54 * ease_out(u)), alpha=int(150 * u),
              width=5, head=16)
    p2 = pop(lt - 0.70, 0.30)
    if p2 > 0:
        items.append(gtext("x² = 2", (CX, 1062), 66, op=p2))
    if clamp01((lt - 1.25) / 0.22) > 0:
        u = clamp01((lt - 1.25) / 0.22)
        arrow(d, (CX, 1130), (CX, 1130 + 54 * ease_out(u)), alpha=int(150 * u),
              width=5, head=16)
    p3, o3 = blue_pop(lt - 1.45, 0.38)
    if p3 > 0:
        items.append(gtext("x =", (466, 1272), 66, op=pop(lt - 1.42, 0.18), anchor="rm"))
        items.append(gtext("√2", (498, 1272), int(118 * (0.55 + 0.45 * p3) * o3),
                           blue=True, anchor="lm"))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


# Подписываются только крайние боксы. Подписать все семь нельзя физически:
# «А5» и «А6» шириной ~48px стоят на боксах шириной 30 и 21px, расстояние между
# центрами падает до 49px — подписи слипаются в «А4А5А6». Ряд из семи убывающих
# прямоугольников с «А0» под первым и «А6» под последним читается однозначно.
LADDER_W0, LADDER_GAP, LADDER_BASE = 168.0, 24.0, 1270.0
LADDER_CX = 526.0          # ряд смещён влево: подпись А6 иначе выходит за GFX_X


def ladder_boxes():
    ws = [LADDER_W0 / (math.sqrt(2) ** i) for i in range(7)]
    total = sum(ws) + LADDER_GAP * 6
    x = LADDER_CX - total / 2
    out = []
    for w in ws:
        out.append((x, w, w * math.sqrt(2)))
        x += w + LADDER_GAP
    return out


def g_ladder(lay, lt):
    """«от а0 до а6»: линейка форматов, каждый следующий вдвое меньше по площади.

    Подписи А0…А6 несёт графика, поэтому субтитра на этой фразе нет.
    """
    d = ImageDraw.Draw(lay)
    items = []
    for i, (x, w, h) in enumerate(ladder_boxes()):
        u = clamp01((lt - 0.12 - i * 0.11) / 0.22)
        if u <= 0:
            continue
        e = ease_out(u)
        rrect(d, x, LADDER_BASE - h * e, x + w, LADDER_BASE,
              alpha=int(235 * u), width=5 if i < 4 else 4)
        v = clamp01((lt - 0.26 - i * 0.11) / 0.20)
        if v > 0 and i in (0, 6):
            items.append(gtext(f"А{i}", (x + w / 2, LADDER_BASE + 54), 30, op=v))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_scale(lay, lt):
    """«идеально масштабируется друг в друга»: два А5 в точности заполняют А4.

    Родительский контур вынесен наружу на 11px: если рисовать его ровно по краю,
    половинки ложатся на него линия в линию и на стоп-кадре виден один
    прямоугольник с чертой посередине, а не «две части внутри целого».
    """
    d = ImageDraw.Draw(lay)
    o = 11
    path_draw(d, rect_path(L - o, T - o, R + o, B + o), clamp01(lt / 0.22),
              width=6, alpha=165)
    starts = [(820, 992), (820, 1288)]
    targets = [(CX, (T + MY) / 2), (CX, (MY + B) / 2)]
    for i in range(2):
        u = ease_out(clamp01((lt - 0.30 - i * 0.18) / 0.55))
        if u <= 0:
            continue
        s = 0.42 + 0.58 * u
        cx = starts[i][0] + (targets[i][0] - starts[i][0]) * u
        cy = starts[i][1] + (targets[i][1] - starts[i][1]) * u
        hw, hh = (SW / 2 - 4) * s, (SH / 4 - 4) * s
        rrect(d, cx - hw, cy - hh, cx + hw, cy + hh, alpha=int(150 + 88 * u), width=6)


def g_cut(lay, lt):
    """«разрежьте любой лист пополам»: рез идёт поперёк, половинки расходятся."""
    d = ImageDraw.Draw(lay)
    v = ease_out(clamp01((lt - 0.95) / 0.55))
    off = 26 * v
    if v <= 0.02:
        path_draw(d, rect_path(L, T, R, B), clamp01(lt / 0.20), width=7, alpha=235)
    else:
        rrect(d, L, T - off, R, MY - off, alpha=235, width=7)
        rrect(d, L, MY + off, R, B + off, alpha=235, width=7)
    u = ease_out(clamp01((lt - 0.30) / 0.55))
    if 0 < u and v <= 0.02:
        xe = L + SW * u
        d.line([(L, MY), (xe, MY)], fill=WHITE + (250,), width=5)
        if u < 0.99:                            # «лезвие» на кромке реза
            d.line([(xe, MY - 26), (xe + 30, MY)], fill=WHITE + (235,), width=5)
            d.line([(xe, MY + 26), (xe + 30, MY)], fill=WHITE + (235,), width=5)


def g_next(lay, lt):
    """«и получите следующий формат»: нижняя половина встаёт вертикально."""
    d = ImageDraw.Draw(lay)
    fade = 1.0 - clamp01((lt - 0.20) / 0.50)
    if fade > 0:
        rrect(d, L, T - 26, R, MY - 26, alpha=int(200 * fade), width=6)
    u = ease_out(clamp01((lt - 0.25) / 0.75))
    cx = CX
    cy = (MY + B) / 2 + 26 + (CY - ((MY + B) / 2 + 26)) * u
    poly(d, rot_poly(cx, cy, SW / 2, SH / 4, 90 * u), alpha=238, width=7)


def g_sameprop(lay, lt):
    """«с той же самой пропорцией сторон»: у листа и у половинки одна и та же метка."""
    d = ImageDraw.Draw(lay)
    boxes = [(370, 1090, 105.4, 149.1), (700, 1090, 74.5, 105.4)]
    for i, (cx, cy, hw, hh) in enumerate(boxes):
        u = ease_out(clamp01((lt - i * 0.12) / 0.26))
        if u <= 0:
            continue
        s = 0.7 + 0.3 * u
        rrect(d, cx - hw * s, cy - hh * s, cx + hw * s, cy + hh * s,
              alpha=int(238 * u), width=6)
    items = []
    for i, cx in enumerate((370, 700)):
        u = clamp01((lt - 0.55 - i * 0.15) / 0.26)
        if u > 0:
            items.append(gtext("1 : √2", (cx, 1310), 46, op=u))
    u = clamp01((lt - 0.95) / 0.26)
    if u > 0:
        items.append(gtext("=", (CX, 1090), 56, op=u))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_unique(lay, lt):
    """«ни один другой выбор не дал бы такого свойства».

    Лист 1 : 1.15 складывается пополам, половинка ставится вертикально — и она
    заметно уже той же формы той же высоты. Зазор справа и есть смысл кадра,
    поэтому он показан мерной скобкой, а перечёркнут только результат.
    Пропорция 1 : 1.15 выбрана не «на глаз»: при 1 : 1.2 несовпадение давало
    52px и на кадре читалось как погрешность рисунка, при 1 : 1.15 — 88px.
    """
    d = ImageDraw.Draw(lay)
    ux0, uy0, ux1, uy1 = 250, 927.5, 550, 1272.5      # 300 x 345 = 1 : 1.15
    umy = (uy0 + uy1) / 2
    v = ease_out(clamp01((lt - 0.35) / 0.45))
    m = ease_out(clamp01((lt - 0.85) / 0.40))
    if m <= 0.02:
        path_draw(d, rect_path(ux0, uy0, ux1, uy1), clamp01(lt / 0.25),
                  width=7, alpha=int(120 + 115 * (1 - v)))
        if clamp01((lt - 0.28) / 0.14) > 0:
            dashes(d, ux0, ux1, umy, alpha=180)
        if v > 0.02:
            rrect(d, ux0, umy, ux1, uy1, alpha=235, width=7)
            y_edge = umy - 172.5 * math.cos(math.pi * v)
            rrect(d, ux0, min(umy, y_edge), ux1, max(umy, y_edge), alpha=235, width=7)
    else:
        g = clamp01((lt - 1.10) / 0.26)
        if g > 0:                          # та же форма той же высоты — куда надо попасть
            rrect(d, 409.5, 970, 670.4, 1270, alpha=int(230 * g), width=6)
        cx = 400 + (495.75 - 400) * m
        cy = 1186.25 + (1120 - 1186.25) * m
        poly(d, rot_poly(cx, cy, 150, 86.25, 90 * m), alpha=245, width=8)
        k = clamp01((lt - 1.30) / 0.24)
        if k > 0:                          # до края нужной формы не хватает 88px
            hbar(d, 582, 582 + 88.4 * ease_out(k), 1308, alpha=int(238 * k),
                 width=7, tick=20)
        c = ease_out(clamp01((lt - 1.45) / 0.35))
        if c > 0:
            # перечёркивается только результат: контур нужной формы должен
            # остаться целым, иначе кадр читается как «обе формы плохие»
            d.line([(395, 1290), (395 + 201 * c, 1290 - 335 * c)],
                   fill=WHITE + (248,), width=12)


def g_trio(lay, lt):
    """Графика хука 1: «у всех форматов а3 а4 а5 они одинаковые».

    Три листа со сторонами в отношении √2 друг к другу, потом А4 и А5 наезжают
    на А3 и точно ложатся в его контур — совпадение и есть смысл кадра.
    Подписи А3/А4/А5 гаснут перед наложением: дальше они уже ничего не кодируют,
    а под общим контуром читались бы как три подписи к одному прямоугольнику.
    В теле ролика эта графика не используется — только в хуке.
    """
    d = ImageDraw.Draw(lay)
    base, gap = 1330.0, 36.0
    ws = [200.0, 200.0 / math.sqrt(2), 100.0]
    x = CX - (sum(ws) + 2 * gap) / 2
    boxes = []
    for w in ws:
        h = w * math.sqrt(2)
        boxes.append([x, base - h, x + w, base])
        x += w + gap
    tgt = list(boxes[0])
    m = ease_out(clamp01((lt - 2.15) / 0.70))
    items = []
    for i, bx in enumerate(boxes):
        u = clamp01((lt - i * 0.30) / 0.35)
        if u <= 0:
            continue
        a, b, c, e = bx
        if i and m > 0:
            a += (tgt[0] - a) * m; b += (tgt[1] - b) * m
            c += (tgt[2] - c) * m; e += (tgt[3] - e) * m
        path_draw(d, rect_path(a, b, c, e), ease_out(u),
                  width=7 if i == 0 else 6, alpha=235 if i == 0 else int(210 + 25 * m))
        lv = clamp01((lt - 0.84 - i * 0.45) / 0.16) * (1.0 - clamp01((lt - 2.15) / 0.30))
        if lv > 0:
            items.append(gtext(f"А{i + 3}", ((a + c) / 2, 1390), 34, op=lv))
    done = clamp01((lt - 2.90) / 0.28)
    if done > 0:
        a, b, c, e = tgt
        al = int(235 * done)
        for px, py in ((a, b), (c, b), (c, e), (a, e)):
            sx = -1 if px == a else 1
            sy = -1 if py == b else 1
            ox, oy = px + sx * 20, py + sy * 20
            d.line([(ox, oy), (ox - sx * 40, oy)], fill=WHITE + (al,), width=6)
            d.line([(ox, oy), (ox, oy - sy * 40)], fill=WHITE + (al,), width=6)
    if items:
        lay.alpha_composite(text_layer((W, H), items))


GFX = {
    "trio": g_trio,
    "sheet": g_sheet, "fold": g_fold, "half": g_half, "same": g_same,
    "sides": g_sides, "ratio": g_ratio, "halfproof": g_halfproof,
    "algebra": g_algebra, "ladder": g_ladder, "scale": g_scale,
    "cut": g_cut, "next": g_next, "sameprop": g_sameprop, "unique": g_unique,
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
            # wobble и alpha уменьшены: весь ролик держится на тонком контуре листа,
            # при штатной сетке (wobble=1.0, alpha=200) её линии сравнимы по яркости
            # с контуром и прямоугольник перестаёт читаться как объект
            GRID_CACHE[key] = grid_canvas(w, h, phase=(key / 10) * 0.7,
                                          alpha=165, wobble=0.45)
        canvas.paste(GRID_CACHE[key], (x, y), rounded_mask(w, h, R_A))
    return canvas


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    # Кадры уходят в ffmpeg сырыми, БЕЗ промежуточного файла.
    # Раньше здесь стоял cv2.VideoWriter с кодеком mp4v: он писал MPEG-4 Part 2
    # на ~3.8 Мбит/с (при исходнике 12.75 Мбит/с), и это потом ещё раз жалось
    # в H.264. Кадр кодировался дважды, первый раз — с потерями и блочностью.
    # Теперь кодек ровно один.
    wr = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-an",
        "-c:v", "libx264", "-crf", "12", "-preset", "slow",
        "-pix_fmt", "yuv420p", VID,
    ], stdin=subprocess.PIPE)
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
        wr.stdin.write(pil_to_cv(canvas.convert("RGB")).tobytes())
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}")

    wr.stdin.close()
    rc = wr.wait()
    cap.release()
    if stock_cap is not None:
        stock_cap.release()
    if rc != 0:
        raise SystemExit(f"ffmpeg вернул {rc}")
    print("готово:", VID)


if __name__ == "__main__":
    main()

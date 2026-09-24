"""Сборка ролика 11 («великая теорема Ферма»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx11.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры — выше y600 (brand-kit §5).
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard11 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_fermat.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/11/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GCY = (GY0 + GY1) // 2
GRID_CACHE = {}


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
    """A-roll: hflip + кроп под карточку A + грейд (brand-kit §1)."""
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    if kind == "A2":
        w, h = 900, 1428
        x, y = (iw - w) // 2 + 8, 230
    else:
        w, h = 1000, 1585
        x, y = (iw - w) // 2 + 10, 125
    x = max(0, min(x, iw - w))
    y = max(0, min(y, ih - h))
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
    """Вставка в карточке B: притемнение по экспозиции + лёгкое обесцвечивание.

    Притемняем УМНОЖЕНИЕМ, а не вычитанием, и клипаем сами.
    cv2.convertScaleAbs считает |x*alpha + beta|, поэтому при отрицательной beta
    тон-кривая V-образная: всё темнее ~46 отражается вверх (0 -> 48), причём каналы
    отражаются независимо — тени светлеют и уезжают в произвольный оттенок.
    Именно это давало «странный цвет» в тёмных местах стока.
    """
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


# --- формулы со степенями -------------------------------------------------

def formula_items(tokens, cx, cy, base, max_w=680, scale=1.0, op=1.0):
    """tokens: [(text, is_sup, is_blue, alpha[, advance])] -> items для text_layer.

    Верхний индекс — тот же гротеск кеглем 0.52 от базового, поднятый на 0.34 базового.
    advance=False означает «этот глиф делит слот со следующим»: так показатель 2 и
    показатель n стоят в одном месте строки, а не друг за другом.
    Место под невидимые (alpha=0) токены резервируется всегда — иначе формула
    прыгает по горизонтали, когда дописывается её правая часть.
    """
    def layout(bsize):
        fb, fs = font("sans", bsize), font("sans", max(12, int(bsize * 0.52)))
        widths = [(fs if t[1] else fb).getlength(t[0]) for t in tokens]
        slots, pend, total = [], [], 0.0
        for i, t in enumerate(tokens):
            if len(t) > 4 and not t[4]:
                pend.append(i)
                continue
            sw = max([widths[i]] + [widths[j] for j in pend])
            for j in pend + [i]:
                slots.append((j, total + (sw - widths[j]) / 2))
            total += sw
            pend = []
        for j in pend:                       # хвост без ведущего токена
            slots.append((j, total))
            total += widths[j]
        return fb, fs, slots, total

    bsize = max(16, int(base * scale))
    fb, fs, slots, total = layout(bsize)
    if total > max_w:
        bsize = max(16, int(bsize * max_w / total))
        fb, fs, slots, total = layout(bsize)

    x0 = cx - total / 2
    out = []
    for i, dx in sorted(slots):
        txt, is_sup, is_blue, a = tokens[i][:4]
        if a <= 0.004:
            continue
        y = cy - bsize * 0.34 if is_sup else cy
        out.append(dict(text=txt, font=(fs if is_sup else fb), xy=(x0 + dx, y),
                        anchor="lm", fill=BLUE if is_blue else WHITE,
                        glow=BLUE_GLOW if is_blue else WHITE,
                        glow_r=30 if is_blue else 15,
                        glow_a=0.84 if is_blue else 0.48, opacity=op * a))
    return out


def poly(d, pts, p, color=WHITE, width=6, alpha=225, closed=True):
    """Рисует контур с прогрессивной отрисовкой по периметру (p: 0..1)."""
    p = clamp01(p)
    if p <= 0:
        return
    pl = list(pts) + ([pts[0]] if closed else [])
    segs = [(pl[i], pl[i + 1]) for i in range(len(pl) - 1)]
    lens = [math.dist(a, b) for a, b in segs]
    tot = sum(lens) or 1.0
    left = tot * p
    for (a, b), L in zip(segs, lens):
        if left <= 0:
            break
        f = min(1.0, left / L) if L else 1.0
        d.line([a, (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)],
               fill=color + (alpha,), width=width)
        left -= L


# --- графика планов -------------------------------------------------------

def g_pyth(lay, lt):
    """Теорема Пифагора: замкнутый треугольник + квадраты на сторонах. Без подписей.

    Треугольник ведёт кадр (толстый штрих), квадраты — фоновая обвязка: иначе на
    1.26с фигура читается как куча прямоугольников, а не как знакомая картинка.
    """
    d = ImageDraw.Draw(lay)
    squares = [
        [(400, 1240), (640, 1240), (640, 1480), (400, 1480)],
        [(400, 1060), (400, 1240), (220, 1240), (220, 1060)],
        [(640, 1240), (400, 1060), (640, 880), (880, 1060)],
    ]
    for i, sq in enumerate(squares):
        poly(d, sq, (lt - 0.30 - i * 0.09) / 0.30, width=4, alpha=120)
    tri = [(400, 1240), (640, 1240), (400, 1060)]
    poly(d, tri, lt / 0.40, width=9, alpha=245)
    # прямой угол
    if lt > 0.40:
        a = int(200 * clamp01((lt - 0.40) / 0.25))
        d.line([(400, 1200), (440, 1200), (440, 1240)], fill=WHITE + (a,), width=4)


def g_swap(lay, lt):
    """«одна маленькая замена»: показатели 2 -> n."""
    sw = clamp01((lt - 0.55) / 0.35)
    p2, o2 = blue_pop(lt - 0.55, 0.35)
    a_old, a_new = 1.0 - sw, sw
    # «2» и «n» делят один слот строки (advance=False у первого)
    toks = [("x", 0, 0, 1.0), ("2", 1, 0, a_old, False), ("n", 1, 1, a_new),
            (" + ", 0, 0, 1.0),
            ("y", 0, 0, 1.0), ("2", 1, 0, a_old, False), ("n", 1, 1, a_new),
            (" = ", 0, 0, 1.0),
            ("z", 0, 0, 1.0), ("2", 1, 0, a_old, False), ("n", 1, 1, a_new)]
    lay.alpha_composite(text_layer((W, H), formula_items(
        toks, GCX, 1120, 128, max_w=660, scale=0.72 + 0.28 * pop(lt, 0.22))))
    if sw > 0:
        d = ImageDraw.Draw(lay)
        y = 1290
        wdt = int(330 * ease_out(sw))
        d.line([(GCX - wdt, y), (GCX + wdt, y)], fill=BLUE + (int(210 * p2 * o2 / max(o2, 1e-6)),), width=5)


def g_zero(lay, lt):
    """R5b: решений нет. Слова несёт субтитр, цифру — графика."""
    p, over = blue_pop(lt, 0.38)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("0", (GCX, 1120), int(265 * (0.55 + 0.45 * p) * over), blue=True),
    ]))
    d = ImageDraw.Draw(lay)
    u = clamp01((lt - 0.42) / 0.40)
    if u > 0:
        wdt = int(210 * ease_out(u))
        d.line([(GCX - wdt, 1330), (GCX + wdt, 1330)], fill=BLUE + (215,), width=6)


def _eqn_tokens(a_x, a_y, a_z, exp="n", exp2=None, a_exp2=0.0):
    """Полный набор слотов уравнения. Невидимые части резервируют место."""
    def sup(a):
        if exp2 is None:
            return [(exp, 1, False, a)]
        # два показателя в одном слоте: старый гаснет ровно на входе нового
        return [(exp, 1, False, a * (1.0 - a_exp2), False),
                (exp2, 1, False, a * a_exp2)]
    return ([("x", 0, 0, a_x)] + sup(a_x)
            + [(" + ", 0, 0, a_y), ("y", 0, 0, a_y)] + sup(a_y)
            + [(" = ", 0, 0, a_z), ("z", 0, 0, a_z)] + sup(a_z))


def g_eqn(lay, lt):
    """Формула читается вслух — её несёт только графика, субтитра на плане нет.

    Термы входят по словам: «х в степени n» 8.18, «плюс y в степени n» 9.02.
    """
    a_x = ease_out(clamp01((lt - 0.35) / 0.42))
    a_y = ease_out(clamp01((lt - 1.30) / 0.42))
    lay.alpha_composite(text_layer((W, H), formula_items(
        _eqn_tokens(a_x, a_y, 0.0), GCX, 1120, 130, max_w=670)))


def g_eqn2(lay, lt):
    """«равно z в степени n» — 10.14, то есть на 0.16с после начала плана."""
    a_z = ease_out(clamp01((lt - 0.16) / 0.42))
    lay.alpha_composite(text_layer((W, H), formula_items(
        _eqn_tokens(1.0, 1.0, a_z), GCX, 1120, 130, max_w=670)))


def g_n2(lay, lt):
    """n = 2 -> обычный прямоугольный треугольник. Показатель меняется на «равном 2» (12.24)."""
    sw = ease_out(clamp01((lt - 0.72) / 0.30))
    lay.alpha_composite(text_layer((W, H), formula_items(
        _eqn_tokens(1.0, 1.0, 1.0, exp="n", exp2="2", a_exp2=sw),
        GCX, 900, 104, max_w=600)))
    d = ImageDraw.Draw(lay)
    tri = [(420, 1470), (700, 1470), (420, 1260)]
    poly(d, tri, (lt - 0.55) / 0.55, width=7)
    if lt > 1.05:
        b = int(180 * clamp01((lt - 1.05) / 0.25))
        d.line([(420, 1432), (458, 1432), (458, 1470)], fill=WHITE + (b,), width=4)


def g_inf(lay, lt):
    """Бесконечно много решений в целых числах: ∞ + решётка целых точек."""
    p, o = blue_pop(lt, 0.36)
    lay.alpha_composite(text_layer((W, H), [
        gtext("∞", (GCX, 1000), int(190 * (0.55 + 0.45 * p) * o)),
    ]))
    d = ImageDraw.Draw(lay)
    for j in range(4):
        for i in range(7):
            k = i + j * 7
            u = clamp01((lt - 0.40 - k * 0.018) / 0.26)
            if u <= 0:
                continue
            x, y = 285 + i * 85, 1250 + j * 85
            r = 9 * ease_out(u)
            d.ellipse([x - r, y - r, x + r, y + r], fill=WHITE + (int(190 * u),))


def g_triple345(lay, lt):
    """Тройка 3-4-5 на замкнутом треугольнике. Числа звучат в речи — субтитра нет."""
    d = ImageDraw.Draw(lay)
    tri = [(360, 1420), (740, 1420), (360, 1135)]
    poly(d, tri, lt / 0.32, width=7)
    if lt > 0.30:
        a = int(180 * clamp01((lt - 0.30) / 0.20))
        d.line([(360, 1382), (398, 1382), (398, 1420)], fill=WHITE + (a,), width=4)
    labels = [("4", (550, 1495), 0.18), ("3", (292, 1277), 0.48), ("5", (648, 1226), 0.78)]
    items = []
    for txt, xy, at in labels:
        u = pop(lt - at, 0.09)          # R5a: резкий поп 65-100мс
        if u > 0:
            items.append(gtext(txt, xy, int(74 * (0.6 + 0.4 * u))))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_ngt2(lay, lt):
    """Любой показатель больше двух."""
    lay.alpha_composite(text_layer((W, H), formula_items(
        [("n", 0, 0, 1.0), (" > ", 0, 0, 1.0), ("2", 0, 0, 1.0)],
        GCX, 920, 118, max_w=420, scale=0.7 + 0.3 * pop(lt, 0.20))))
    items = []
    for i, txt in enumerate(["3", "4", "5", "6", "7"]):
        u = pop(lt - 0.30 - i * 0.075, 0.10)
        if u > 0:
            items.append(gtext(txt, (300 + i * 120, 1290), int(84 * (0.62 + 0.38 * u))))
    if items:
        lay.alpha_composite(text_layer((W, H), items))
    d = ImageDraw.Draw(lay)
    u = clamp01((lt - 0.85) / 0.35)
    if u > 0:
        items2 = []
        for i in range(3):
            v = clamp01((lt - 0.95 - i * 0.10) / 0.25)
            if v > 0:
                items2.append(gtext("·", (770 + i * 34, 1290), 60, op=v * 0.8))
        if items2:
            lay.alpha_composite(text_layer((W, H), items2))
    _ = d


def g_none(lay, lt):
    """Ни одного решения: пустое множество + вычеркнутые кандидаты.

    Знак ∅ рисуется вручную: в SF Pro глиф U+2205 отсутствует и подменяется
    блоком «нет глифа» — проверено на кадре.
    """
    p, o = blue_pop(lt, 0.34)
    d = ImageDraw.Draw(lay)
    if p > 0:
        r = 100 * (0.55 + 0.45 * p) * o
        cx, cy = GCX, 1075
        glow = _layer()
        gd = ImageDraw.Draw(glow)
        for lay_i, (rr, wd, al) in enumerate([(r, 12, 235)]):
            gd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                       outline=WHITE + (al,), width=wd)
            k = rr * 0.78
            gd.line([(cx - k, cy + k), (cx + k, cy - k)], fill=WHITE + (al,), width=wd)
        blur = glow.filter(ImageFilter.GaussianBlur(15))
        blur.putalpha(blur.split()[3].point(lambda v: int(v * 0.48)))
        lay.alpha_composite(blur)
        lay.alpha_composite(glow)
        d = ImageDraw.Draw(lay)
    for i in range(4):
        x = 318 + i * 148
        u = clamp01((lt - 0.55 - i * 0.10) / 0.24)
        if u <= 0:
            continue
        a = int(160 * u)
        d.rounded_rectangle([x - 52, 1348, x + 52, 1452], radius=18,
                            outline=WHITE + (a,), width=4)
        v = clamp01((lt - 0.75 - i * 0.10) / 0.22)
        if v > 0:
            e = ease_out(v)
            d.line([(x - 44, 1356), (x - 44 + 88 * e, 1356 + 88 * e)],
                   fill=WHITE + (215,), width=5)
            d.line([(x + 44, 1356), (x + 44 - 88 * e, 1356 + 88 * e)],
                   fill=WHITE + (215,), width=5)


def g_y1637(lay, lt):
    p, o = blue_pop(lt, 0.38)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("1637", (GCX, 1120), int(206 * (0.55 + 0.45 * p) * o), blue=True),
        gtext("ферма", (GCX, 1330), 58, serif=True, op=clamp01((lt - 0.45) / 0.32)),
    ]))


def g_margin(lay, lt):
    """Поля книги: доказательство длиннее, чем свободное поле страницы."""
    d = ImageDraw.Draw(lay)
    a = ease_out(clamp01(lt / 0.35))
    d.rounded_rectangle([250, 790, 730, 1470], radius=16,
                        outline=WHITE + (int(150 * a),), width=4)
    for k in range(11):                       # строки текста на странице
        u = clamp01((lt - 0.12 - k * 0.035) / 0.22)
        if u <= 0:
            continue
        y = 850 + k * 46
        wdt = int((330 if k % 3 else 262) * ease_out(u))
        d.line([(288, y), (288 + wdt, y)], fill=WHITE + (120,), width=5)
    # граница поля
    u = clamp01((lt - 0.45) / 0.30)
    if u > 0:
        for seg in range(14):
            y0 = 800 + seg * 48
            d.line([(668, y0), (668, y0 + 26)], fill=WHITE + (int(140 * u), ), width=3)
    # доказательство, которое не помещается
    v = clamp01((lt - 0.80) / 0.85)
    if v > 0:
        y = 1370
        x1 = 288 + (880 - 288) * ease_out(v)
        d.line([(288, y), (x1, y)], fill=BLUE + (235,), width=8)
        if x1 > 700:
            d.line([(x1, y), (x1 - 26, y - 18)], fill=BLUE + (235,), width=8)
            d.line([(x1, y), (x1 - 26, y + 18)], fill=BLUE + (235,), width=8)


def g_y358(lay, lt):
    p, o = blue_pop(lt, 0.38)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("358", (GCX, 1085), int(212 * (0.55 + 0.45 * p) * o), blue=True),
        gtext("лет", (GCX, 1268), 60, op=clamp01((lt - 0.38) / 0.28)),
        gtext("молчания", (GCX, 1398), 62, serif=True, op=clamp01((lt - 0.72) / 0.32)),
    ]))


def g_y1994(lay, lt):
    p, o = blue_pop(lt, 0.38)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("1994", (GCX, 1120), int(206 * (0.55 + 0.45 * p) * o), blue=True),
        gtext("уайлс", (GCX, 1330), 58, serif=True, op=clamp01((lt - 0.45) / 0.32)),
    ]))


def g_proof(lay, lt):
    """Доказано: галка + знак конца доказательства. Без кириллицы — слова у субтитра."""
    d = ImageDraw.Draw(lay)
    p = ease_out(clamp01((lt - 0.05) / 0.42))
    a = (410, 1150)
    b = (500, 1250)
    c = (700, 1010)
    if p > 0:
        f1 = min(1.0, p / 0.40)
        d.line([a, (a[0] + (b[0] - a[0]) * f1, a[1] + (b[1] - a[1]) * f1)],
               fill=WHITE + (235,), width=14)
        if p > 0.40:
            f2 = (p - 0.40) / 0.60
            d.line([b, (b[0] + (c[0] - b[0]) * f2, b[1] + (c[1] - b[1]) * f2)],
                   fill=WHITE + (235,), width=14)
    u = clamp01((lt - 0.62) / 0.30)
    if u > 0:
        s = int(42 * ease_out(u))
        d.rectangle([742, 1330 - s, 742 + s, 1330], fill=WHITE + (225,))


def g_timeline(lay, lt):
    """1637 -> 1994: разрыв, в котором нужной математики ещё не было."""
    d = ImageDraw.Draw(lay)
    y = 1200
    x0, x1 = 300, 800
    p = ease_out(clamp01(lt / 0.55))
    d.line([(x0, y), (x0 + (x1 - x0) * p, y)], fill=WHITE + (165,), width=4)
    for x in (x0, x1):
        if p > 0.9:
            d.line([(x, y - 26), (x, y + 26)], fill=WHITE + (200,), width=5)
    u = clamp01((lt - 0.70) / 1.05)
    if u > 0:
        d.line([(x0, y), (x0 + (x1 - x0) * ease_out(u), y)], fill=BLUE + (230,), width=9)
    # подпись длины разрыва здесь не ставится: 1994-1637 даёт 357, а в речи звучит 358
    lay.alpha_composite(text_layer((W, H), [
        gtext("1637", (x0 + 18, 1080), 66, op=clamp01((lt - 0.28) / 0.30)),
        gtext("1994", (x1 - 18, 1080), 66, op=clamp01((lt - 0.95) / 0.30)),
    ]))


GFX = {
    "pyth": g_pyth, "swap": g_swap, "zero": g_zero, "eqn": g_eqn, "eqn2": g_eqn2,
    "n2": g_n2, "inf": g_inf, "triple345": g_triple345, "ngt2": g_ngt2,
    "none": g_none, "y1637": g_y1637, "margin": g_margin, "y358": g_y358,
    "y1994": g_y1994, "proof": g_proof, "timeline": g_timeline,
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
    tmp = f"{BUILD}/assets/_video_fermat_raw.mp4"
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

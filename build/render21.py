"""Сборка ролика 21 («порог скидки: почему скидка начинается ровно с 3000»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx21.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр
  (оба клипа идут 25 fps — последовательное чтение дало бы замедление);
* кегль крупного числа считается от ширины зоны графики, а не берётся XL из гайда
  (`fit_size`): «+500 ₽» на кегле 150 не влезает в 730px зоны вместе со свечением.

Предметная графика ролика — одна **шкала чека** от 0 до 3600 ₽ с отметкой порога
на 3000. Все планы построены на ней, поэтому длины на экране сравнимы между собой:
разрыв 800 ₽ и скидка 300 ₽ измеряются одной и той же линейкой.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard21 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, THRESHOLD, PLANNED, GAP, DISC_PCT,
                          DISCOUNT, MARGIN, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/21/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор (81 замер по ролику)
# даёт лицо стабильно: cx 390.5, линия глаз y 669.7, ширина 267.
# Кроп ставит глаза на 43% высоты карточки у A1 и на 41.5% у A2 — как в ролике 20.
# У A1 центр кропа уехал на 4.5px левее лица: кадр 720px шириной, шире не подвинуть.
FRAMINGS = {
    "A1": dict(w=668, h=1060, x=52, y=214),
    "A2": dict(w=601, h=954, x=90, y=274),
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
          color=None, glow_r=None, glow_a=None):
    kind = "serif_it" if serif else "sans"
    fill = color if color is not None else (BLUE if blue else WHITE)
    glow = BLUE_GLOW if blue else (color if color is not None else WHITE)
    gr = glow_r if glow_r is not None else (34 if blue else 15)
    ga = glow_a if glow_a is not None else (0.86 if blue else 0.48)
    return dict(text=text, font=font(kind, size), xy=xy, anchor=anchor,
                fill=fill, glow=glow, glow_r=gr, glow_a=ga, opacity=op)


def fit_size(text, size, max_w):
    """Кегль подбирается под ширину зоны, а не берётся из шкалы XL (см. шапку)."""
    while size > 40 and font("sans", size).getlength(text) > max_w:
        size -= 2
    return size


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


def dashed_line(d, a, b, p=1.0, dash=18, gap=14, width=3, alpha=150, color=WHITE):
    if p <= 0:
        return
    ln = math.hypot(b[0] - a[0], b[1] - a[1])
    if ln < 1:
        return
    n = max(1, int(ln // (dash + gap)) + 1)
    lim = clamp01(p)
    for i in range(n):
        u0 = i * (dash + gap) / ln
        u1 = min(1.0, (i * (dash + gap) + dash) / ln)
        if u0 > lim:
            break
        u1 = min(u1, lim)
        d.line([(a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0),
                (a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)],
               fill=color + (int(alpha),), width=width)


def rrect(d, box, r, fill=None, outline=None, width=4):
    x0, y0, x1, y1 = box
    if x1 - x0 < 2:
        return
    rr = int(max(0, min(r, (x1 - x0) / 2 - 1, (y1 - y0) / 2 - 1)))
    d.rounded_rectangle([x0, y0, x1, y1], radius=rr, fill=fill,
                        outline=outline, width=width)


# --- шкала чека -------------------------------------------------------------
# Одна линейка на весь ролик: 0 ... 3600 ₽ по горизонтали.
# шкала сдвинута влево от центра зоны: справа от отметки порога должно хватить
# места плашке «−10%» (при 205..875 она вылезала за GFX_X, qa21.gfx_in_zone это ловил)
BAR_X0, BAR_X1 = 195, 845
VMAX = 3600
BAR_TOP, BAR_H = 1148, 112
BAR_BOT = BAR_TOP + BAR_H            # 1260
MARK_TOP, MARK_BOT = 1104, 1304      # вертикальная отметка порога, пересекает шкалу
TAG_Y = 1350                         # ярлыки со значениями под шкалой
TAG_SZ = 50
BIG_Y = 930                          # центр крупного числа над шкалой
TICKS_Y = 1088                       # низ «чеков»-штрихов над шкалой
ARROW_Y = 952


def bx(v):
    return BAR_X0 + (BAR_X1 - BAR_X0) * max(0.0, min(float(v), VMAX)) / VMAX


X_THR = bx(THRESHOLD)     # 760.0
X_PLAN = bx(PLANNED)      # 613.3


def bar_track(d, alpha=120, p=1.0):
    """Шкала чека: пустой жёлоб со скруглением, слева направо от 0 до 3600 ₽."""
    p = clamp01(p)
    if p <= 0.01:
        return
    rrect(d, (BAR_X0, BAR_TOP, BAR_X0 + (BAR_X1 - BAR_X0) * p, BAR_BOT), 54,
          outline=WHITE + (int(alpha),), width=5)


def bar_fill(d, v, p=1.0, alpha=205, x_from=None):
    x0 = BAR_X0 if x_from is None else x_from
    x1 = x0 + (bx(v) - x0) * clamp01(p)
    if x1 - x0 < 6:
        return
    rrect(d, (x0 + 6, BAR_TOP + 6, x1 - 2, BAR_BOT - 6), 48, fill=WHITE + (int(alpha),))


def threshold_mark(d, p=1.0, alpha=235, flag=True):
    p = clamp01(p)
    if p <= 0.02:
        return
    a = int(alpha * p)
    y1 = MARK_TOP + (MARK_BOT - MARK_TOP) * p
    d.line([(X_THR, MARK_TOP), (X_THR, y1)], fill=WHITE + (a,), width=11)


def tags(op_plan=0.0, op_thr=0.0):
    """Ярлыки под шкалой: «2200 ₽» слева под залитой частью, «3000 ₽» на отметке."""
    out = []
    if op_plan > 0.02:
        out.append(gtext(f"{PLANNED} ₽", (BAR_X0 + 12, TAG_Y), TAG_SZ,
                         op=op_plan, anchor="lm"))
    if op_thr > 0.02:
        out.append(gtext(f"{THRESHOLD} ₽", (X_THR, TAG_Y), TAG_SZ, op=op_thr))
    return out


# «чеки» отдельных покупателей — детерминированный разброс вокруг PLANNED
def _checks():
    rng = np.random.default_rng(21)
    xs = np.clip(rng.normal(PLANNED, 520, 13), 800, 2950)
    xs = xs - (xs.mean() - PLANNED)     # среднее облака = ровно PLANNED
    hs = rng.integers(44, 96, 13)
    return sorted(zip([float(bx(v)) for v in xs], [int(h) for h in hs]))


CHECKS = _checks()


def draw_checks(d, p=1.0, alpha=170, stagger=0.055):
    n = len(CHECKS)
    for i, (x, h) in enumerate(CHECKS):
        kp = clamp01((p - i * stagger) / 0.16)
        if kp <= 0.02:
            continue
        hh = h * ease_out(kp)
        d.line([(x, TICKS_Y), (x, TICKS_Y - hh)],
               fill=WHITE + (int(alpha * kp),), width=8)


# --- графика планов ---------------------------------------------------------

def g_thresh(lay, lt):
    """«от 3 тысяч рублей» — появляется шкала, на ней отметка порога, над ней число."""
    d = ImageDraw.Draw(lay)
    bar_track(d, p=ease_out(clamp01(lt / 0.32)))
    threshold_mark(d, ease_out(clamp01((lt - 0.32) / 0.28)))
    items = []
    p = pop(lt - 0.52, 0.09)          # R5a: белое число, резкий поп 0.6 -> 1.0
    if p > 0.02:
        sz = fit_size(f"{THRESHOLD} ₽", 140, 620)
        items.append(gtext(f"{THRESHOLD} ₽", (540, BIG_Y),
                           int(sz * (0.60 + 0.40 * p)), op=p))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


CAND = [1200, 1800, 2400, THRESHOLD]


def g_calc(lay, lt):
    """«это конкретный расчёт» — порог выбирают из вариантов, а не берут наугад:
    несколько пробных отметок гаснут, остаётся одна."""
    d = ImageDraw.Draw(lay)
    bar_track(d)
    lock = ease_out(clamp01((lt - 1.05) / 0.35))
    for i, v in enumerate(CAND[:-1]):
        kp = ease_out(clamp01((lt - i * 0.20) / 0.26))
        if kp <= 0.02:
            continue
        a = int(150 * kp * (1.0 - 0.88 * lock))
        dashed_line(d, (bx(v), MARK_TOP + 30), (bx(v), MARK_BOT),
                    p=1.0, dash=18, gap=14, width=7, alpha=a)
    kp = ease_out(clamp01((lt - 0.60) / 0.26))
    if kp <= 0.02:
        return
    dashed_line(d, (X_THR, MARK_TOP + 30), (X_THR, MARK_BOT),
                p=1.0, dash=18, gap=14, width=7, alpha=int(150 * kp * (1 - lock)))
    threshold_mark(d, lock)
    if lock > 0.5:
        lay.alpha_composite(text_layer((W, H), tags(op_thr=(lock - 0.5) * 2)))


def g_check(lay, lt):
    """«насколько в среднем вырастет чек» — облако чеков разных покупателей
    и стрелка роста в сторону порога. Значений тут ещё нет — только направление."""
    d = ImageDraw.Draw(lay)
    bar_track(d)
    threshold_mark(d, 1.0)
    draw_checks(d, clamp01(lt / 0.95))
    ap = ease_out(clamp01((lt - 0.90) / 0.45))
    if ap > 0.02:
        arrow(d, (450, ARROW_Y), (450 + (X_THR - 20 - 450) * ap, ARROW_Y),
              alpha=int(225 * ap), width=7, head=22)
    lay.alpha_composite(text_layer((W, H), tags(op_thr=1.0)))


def g_n2200(lay, lt):
    """«ну скажем 2200 рублей» — шкала заливается до планируемого чека."""
    d = ImageDraw.Draw(lay)
    bar_track(d)
    bar_fill(d, PLANNED, p=ease_out(clamp01(lt / 0.52)))
    threshold_mark(d, 1.0)
    items = tags(op_thr=1.0)
    p = pop(lt - 0.60, 0.09)
    if p > 0.02:
        sz = fit_size(f"{PLANNED} ₽", 140, 620)
        items.append(gtext(f"{PLANNED} ₽", (540, BIG_Y),
                           int(sz * (0.60 + 0.40 * p)), op=p))
    lay.alpha_composite(text_layer((W, H), items))


def g_gap800(lay, lt):
    """«остаётся совсем немного» — разрыв между чеком и порогом = 800 ₽ (R5b)."""
    d = ImageDraw.Draw(lay)
    bar_track(d)
    bar_fill(d, PLANNED)
    threshold_mark(d, 1.0)
    gp = ease_out(clamp01(lt / 0.40))
    if gp > 0.02:                       # сам разрыв — пунктирная заливка на шкале
        x1 = X_PLAN + (X_THR - X_PLAN) * gp
        rrect(d, (X_PLAN - 40, BAR_TOP + 6, x1, BAR_BOT - 6), 48,
              fill=WHITE + (int(78 * gp),))
    items = tags(op_plan=1.0, op_thr=1.0)
    p, sc = blue_pop(lt - 0.90)
    if p > 0.02:
        sz = fit_size(f"{GAP} ₽", 158, 600)
        items.append(gtext(f"{GAP} ₽", (540, BIG_Y),
                           int(sz * (0.55 + 0.45 * p) * sc), blue=True, op=p))
    lay.alpha_composite(text_layer((W, H), items))


def g_reach(lay, lt):
    """«лишь бы дотянуть до порога» — разрыв закрывается тремя добавленными товарами."""
    d = ImageDraw.Draw(lay)
    bar_track(d)
    bar_fill(d, PLANNED)
    threshold_mark(d, 1.0)
    seg = (X_THR - X_PLAN) / 3
    for i in range(3):
        kp = ease_out(clamp01((lt - 0.16 - i * 0.28) / 0.30))
        if kp <= 0.02:
            continue
        x0 = X_PLAN + i * seg
        rrect(d, (x0 + 5, BAR_TOP + 6, x0 + seg * kp, BAR_BOT - 6), 34,
              fill=WHITE + (int(210 * kp),))
    lay.alpha_composite(text_layer((W, H), tags(op_plan=0.5, op_thr=1.0)))


def g_unlock(lay, lt):
    """«и получить желанную скидку» — порог взят, скидка включилась.

    «−10%» на экране — размер скидки, в речи он не звучит (brand-kit §5:
    подпись на графике допустима, только если добавляет то, чего нет в речи).
    """
    d = ImageDraw.Draw(lay)
    bar_track(d)
    bar_fill(d, THRESHOLD)
    puls = 1.0 if lt < 0.30 else 0.80 + 0.20 * abs(math.sin(lt * 4.2))
    threshold_mark(d, 1.0, alpha=int(235 * puls))
    items = tags(op_thr=1.0)
    p = pop(lt - 0.52, 0.12)
    if p > 0.02:
        txt = f"−{DISC_PCT}%"
        sz = int(72 * (0.62 + 0.38 * p))
        f = font("sans", sz)
        wpx, hpx = f.getlength(txt), sz * 0.78
        top = 986
        box = (X_THR - wpx / 2 - 26, top, X_THR + wpx / 2 + 26, top + hpx + 30)
        rrect(d, box, 16, outline=WHITE + (int(190 * p),), width=4)
        items.append(gtext(txt, (X_THR, (box[1] + box[3]) / 2), sz, op=p))
    lay.alpha_composite(text_layer((W, H), items))


# --- сравнение «добавил» против «скидка» -----------------------------------
# Отдельная схема на общем масштабе 0.55 px/₽: длины двух полос сравнимы напрямую.
CMP_X0 = 262
CMP_K = 0.72
ROW1_Y, ROW2_Y = 1000, 1200
ROW_H = 96
LBL_SZ = 58


def cmp_bar(d, y, val, p=1.0, alpha=205, fill=True, x_from=None):
    x0 = CMP_X0 if x_from is None else x_from
    x1 = x0 + val * CMP_K * clamp01(p)
    if x1 - x0 < 3:
        return
    rrect(d, (x0, y, x1, y + ROW_H), 46,
          fill=WHITE + (int(alpha),) if fill else None,
          outline=None if fill else WHITE + (int(alpha),), width=4)


def g_added(lay, lt):
    """«если добавленная сумма превышает размер самой скидки» —
    две полосы на одной линейке: 800 ₽ добавлено против 300 ₽ скидки."""
    d = ImageDraw.Draw(lay)
    p1 = ease_out(clamp01(lt / 0.42))
    p2 = ease_out(clamp01((lt - 0.52) / 0.42))
    cmp_bar(d, ROW1_Y, GAP, p=p1)
    cmp_bar(d, ROW2_Y, DISCOUNT, p=p2, alpha=150)
    items = []
    if p1 > 0.15:
        items.append(gtext(f"+{GAP} ₽", (CMP_X0, ROW1_Y - 52), LBL_SZ,
                           op=clamp01((p1 - 0.15) / 0.4), anchor="lm"))
    if p2 > 0.15:
        items.append(gtext(f"−{DISCOUNT} ₽", (CMP_X0, ROW2_Y - 52), LBL_SZ,
                           op=clamp01((p2 - 0.15) / 0.4), anchor="lm"))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_plus(lay, lt):
    """«магазин всё равно остаётся в плюсе» — из полосы «добавлено» вычитается
    скидка, светится ровно остаток: 800 − 300 = 500 (R5b)."""
    d = ImageDraw.Draw(lay)
    sp = ease_out(clamp01(lt / 0.42))
    x_cut = CMP_X0 + DISCOUNT * CMP_K
    cmp_bar(d, ROW1_Y, GAP, alpha=int(90 + 60 * (1 - sp)))
    # отданная часть гаснет, остаток разгорается
    rrect(d, (CMP_X0, ROW1_Y, x_cut, ROW1_Y + ROW_H), 46,
          fill=WHITE + (int(150 * (1 - 0.72 * sp)),))
    x1 = x_cut + (CMP_X0 + GAP * CMP_K - x_cut) * sp
    if x1 - x_cut > 3:
        rrect(d, (x_cut, ROW1_Y, x1, ROW1_Y + ROW_H), 46, fill=WHITE + (240,))
    dashed_line(d, (x_cut, ROW1_Y - 34), (x_cut, ROW1_Y + ROW_H + 46),
                p=sp, dash=14, gap=12, width=4, alpha=180)
    if sp > 0.5:                        # отданная скидкой часть — скобка под ней
        a = int(150 * (sp - 0.5) * 2)
        y = ROW1_Y + ROW_H + 46
        d.line([(CMP_X0, y), (x_cut, y)], fill=WHITE + (a,), width=4)
        d.line([(CMP_X0, y - 22), (CMP_X0, y)], fill=WHITE + (a,), width=4)
    items = []
    p, sc = blue_pop(lt - 0.84)
    if p > 0.02:
        sz = fit_size(f"+{MARGIN} ₽", 152, 600)
        items.append(gtext(f"+{MARGIN} ₽", (540, 862),
                           int(sz * (0.55 + 0.45 * p) * sc), blue=True, op=p))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


def g_thrcalc(lay, lt):
    """«порог скидки рассчитан математически» — облако чеков, их среднее,
    и порог, поставленный чуть правее среднего. Это и есть весь расчёт."""
    d = ImageDraw.Draw(lay)
    bar_track(d)
    draw_checks(d, clamp01(lt / 0.55), alpha=120, stagger=0.030)
    mp = ease_out(clamp01((lt - 0.62) / 0.30))
    if mp > 0.02:                       # среднее по чекам
        a = int(235 * mp)
        d.line([(X_PLAN, TICKS_Y + 10), (X_PLAN, TICKS_Y - 118 * mp)],
               fill=WHITE + (a,), width=9)
        d.polygon([(X_PLAN - 19, TICKS_Y + 10), (X_PLAN + 19, TICKS_Y + 10),
                   (X_PLAN, TICKS_Y + 40)], fill=WHITE + (a,))
    threshold_mark(d, ease_out(clamp01((lt - 1.00) / 0.30)))
    gp = ease_out(clamp01((lt - 1.34) / 0.40))
    if gp > 0.02:                       # зазор «среднее -> порог»
        a = int(200 * gp)
        d.line([(X_PLAN, ARROW_Y), (X_PLAN, TICKS_Y - 118)], fill=WHITE + (a,), width=4)
        d.line([(X_THR, ARROW_Y), (X_THR, MARK_TOP)], fill=WHITE + (a,), width=4)
        dashed_line(d, (X_PLAN, ARROW_Y), (X_THR, ARROW_Y), p=gp,
                    dash=16, gap=12, width=5, alpha=a)
        if gp > 0.9:
            arrow(d, (X_THR - 40, ARROW_Y), (X_THR, ARROW_Y),
                  alpha=a, width=5, head=18)
    lay.alpha_composite(text_layer((W, H),
                                   tags(op_plan=mp, op_thr=clamp01((lt - 1.0) / 0.3))))


def g_rise(lay, lt):
    """«средний чек вырос сильнее» — среднее переезжает с 2200 на 3000."""
    d = ImageDraw.Draw(lay)
    bar_track(d)
    threshold_mark(d, 1.0, alpha=120)
    d.line([(X_PLAN, TICKS_Y + 10), (X_PLAN, TICKS_Y - 118)],
           fill=WHITE + (70,), width=9)
    mp = ease_out(clamp01((lt - 0.30) / 0.80))
    x = X_PLAN + (X_THR - X_PLAN) * mp
    bar_fill(d, PLANNED, alpha=120)
    if x - X_PLAN > 6:                  # пройденный путь подсвечен на шкале
        rrect(d, (X_PLAN, BAR_TOP + 6, x, BAR_BOT - 6), 48, fill=WHITE + (228,))
    d.line([(x, TICKS_Y + 10), (x, TICKS_Y - 118)], fill=WHITE + (235,), width=9)
    d.polygon([(x - 19, TICKS_Y + 10), (x + 19, TICKS_Y + 10), (x, TICKS_Y + 40)],
              fill=WHITE + (235,))
    if mp > 0.06:
        arrow(d, (X_PLAN, ARROW_Y), (x, ARROW_Y), alpha=215, width=6, head=20)
    lay.alpha_composite(text_layer((W, H), tags(op_plan=0.45, op_thr=1.0)))


# Вердикт: обе величины столбиками от общей базовой линии. На основной шкале
# 800 ₽ и 300 ₽ дают полосы 145 и 55px — на глаз это не сравнение, а два пятна;
# столбики того же отношения (2.67:1) читаются с паузы за доли секунды.
VER_BASE = 1400
VER_K = 0.50
VER_W = 120
VER_CX = {"gain": 440, "loss": 680}
VER_LBL = 44


def _column(d, cx, val, p, alpha):
    h = val * VER_K * clamp01(p)
    if h < 4:
        return
    rrect(d, (cx - VER_W / 2, VER_BASE - h, cx + VER_W / 2, VER_BASE), 20,
          fill=WHITE + (int(alpha),))


def g_verdict(lay, lt):
    """«вырос сильнее, чем магазин потеряет на скидке» — прирост чека против
    размера скидки: два столбика от одной базовой линии, отношение 800:300."""
    d = ImageDraw.Draw(lay)
    d.line([(VER_CX["gain"] - VER_W, VER_BASE), (VER_CX["loss"] + VER_W, VER_BASE)],
           fill=WHITE + (120,), width=4)
    items = []
    p1 = ease_out(clamp01((lt - 0.10) / 0.45))
    _column(d, VER_CX["gain"], GAP, p1, 238)
    if p1 > 0.35:
        items.append(gtext(f"+{GAP} ₽", (VER_CX["gain"], VER_BASE + 52), VER_LBL,
                           op=clamp01((p1 - 0.35) / 0.4)))
    p2 = ease_out(clamp01((lt - 0.78) / 0.45))
    _column(d, VER_CX["loss"], DISCOUNT, p2, 150)
    if p2 > 0.35:
        items.append(gtext(f"−{DISCOUNT} ₽", (VER_CX["loss"], VER_BASE + 52), VER_LBL,
                           op=clamp01((p2 - 0.35) / 0.4) * 0.85))
    if items:
        lay.alpha_composite(text_layer((W, H), items))


GFX = {
    "thresh": g_thresh, "calc": g_calc, "check": g_check, "n2200": g_n2200,
    "gap800": g_gap800, "reach": g_reach, "unlock": g_unlock, "added": g_added,
    "plus": g_plus, "thrcalc": g_thrcalc, "rise": g_rise, "verdict": g_verdict,
}


def graphics_layer(kind, lt):
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    return lay


# --- крупное слово в карточке (R6) -----------------------------------------
BIG_WORD_Y = 440
BIG_WORD_W = 800


def big_word_layer(word, lt):
    """R6: слово капсом во всю ширину карточки, на стене над головой.
    Без маски по силуэту (shot-recipes R6: на нашем материале маска ломает слово)."""
    p = ease_out(clamp01(lt / 0.30))
    if p <= 0.02:
        return None
    sz = 40
    while sz < 260 and font("sans", sz + 4).getlength(word) <= BIG_WORD_W:
        sz += 4
    it = gtext(word, (540, BIG_WORD_Y), sz, op=0.62 * p, glow_r=24, glow_a=0.35)
    return text_layer((W, H), [it])


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
        bx_, by = slot["x"], slot["y"] + pos * slot["step"]
        if slot["scatter"]:
            calm = bid % 3 == 0                      # каждый третий блок — спокойный
            bx_ += (0 if calm else [-48, 40, -22][min(pos, 2)])
        lt = t - t0
        nchars = sum(len(r[0]) for r in runs)
        type_dur = min(0.55, max(0.30, nchars * 0.030))
        reveal = nchars if lt >= type_dur else nchars * lt / type_dur
        op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
        max_w = slot["x_max"] - bx_ if slot["anchor"][0] == "l" else \
            2 * min(bx_ - slot["x_min"], slot["x_max"] - bx_)
        items += line_items(runs, (bx_, by), slot["anchor"], opacity=op,
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
    if prm.get("big"):
        bw = big_word_layer(prm["big"], t - t0)
        if bw is not None:
            canvas.alpha_composite(bw)
    gl = graphics_layer(kind, t - t0)
    if gl is not None:
        canvas.alpha_composite(gl)
    cl = caption_layer(t)
    if cl is not None:
        canvas.alpha_composite(cl)
    return canvas


class StockReader:
    """Сток конформится к 30 fps проекта по времени: клип идёт со своей скоростью,
    а не «кадр исходника на кадр проекта» (оба клипа 25 fps — играли бы замедленно)."""

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

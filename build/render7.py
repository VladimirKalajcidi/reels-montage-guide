"""Рендер ролика 7: правило 37% (задача оптимальной остановки).

Кадр собирается слоями, чтобы их можно было проверить по отдельности:
  фон (лицо / сток / сетка) → слой графики → слой субтитров
Слой графики и слой субтитров не должны иметь общих пикселей (qa7.py).
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard7 import (SRC, SHOTS, CAPS, SLOTS, DUR, FACE_KINDS, STOCK_DIR,
                         GFX_ZONE, GFX_X, BARS, N_BARS, MARK, BEST_SKIP, PICK,
                         EARLY_MARK, LATE_MARK, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_secretary.mp4"

GY0, GY1 = GFX_ZONE          # 700 … 1560
GX0, GX1 = GFX_X             # 175 … 905

# --- геометрия «лестницы вариантов» ---
PITCH = (GX1 - GX0) / N_BARS          # 52.14
BAR_W = 36
BASE_Y = 1500                          # общая база столбиков
TOP_Y = BASE_Y - max(BARS)             # 880 — верх самого высокого


def bar_x(i):
    return GX0 + i * PITCH + (PITCH - BAR_W) / 2


def mark_x(k):
    """x отметки после k-го столбика."""
    return GX0 + k * PITCH


# ----------------------------------------------------------------- источник --

def cv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# Кроп под карточку A. Соотношение 870/1380 = 0.6304.
# Линия глаз в исходнике ≈ y 825; ставим её на ~42% высоты карточки.
CROPS = {
    "A1": (90, 240, 900, 1428),
    "A2": (130, 279, 820, 1301),
}


def source_card(frame, kind):
    fr = cv2.flip(frame, 1)                      # фронталка: снимаем зеркало
    x, y, w, h = CROPS[kind]
    fr = fr[y:y + h, x:x + w]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_AREA)
    # грейд: гамма вверх, чуть теплее, чуть контраста (brand-kit §1)
    fr = cv2.convertScaleAbs(fr, alpha=1.06, beta=8)
    b, g, r = cv2.split(fr.astype(np.float32))
    r = np.clip(r * 1.04 + 3, 0, 255)
    b = np.clip(b * 0.97, 0, 255)
    fr = cv2.merge([b, g, r]).astype(np.uint8)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
    fr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    fr = cv2.filter2D(fr, -1, np.array([[0, -0.09, 0],
                                        [-0.09, 1.36, -0.09],
                                        [0, -0.09, 0]], np.float32))
    return cv_to_pil(fr)


def stock_frame_from_bgr(fr):
    """Кино-кадр 16:9 обрезается по бокам до 1.385:1 и притемняется на ~27%."""
    ih, iw = fr.shape[:2]
    target = CARD_B[2] / CARD_B[3]
    if iw / ih > target:
        nw = int(ih * target)
        fr = fr[:, (iw - nw) // 2:(iw - nw) // 2 + nw]
    else:
        nh = int(iw / target)
        fr = fr[(ih - nh) // 2:(ih - nh) // 2 + nh, :]
    fr = cv2.resize(fr, (CARD_B[2], CARD_B[3]), interpolation=cv2.INTER_AREA)
    fr = cv2.convertScaleAbs(fr, alpha=1.03, beta=-42)
    return cv_to_pil(fr)


# ------------------------------------------------------------------ графика --

def _layer():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def ease(t):
    return ease_out(max(0.0, min(1.0, t)))


def pop(lt, dur=0.09):
    """R5a — белое число: 65–100мс, 0.6 → 1.0, без оверщута."""
    return ease(lt / dur) if lt >= 0 else 0.0


def blue_pop(lt, dur=0.38):
    """R5b — синее число: 0.35–0.40с, 0.55 → 1.0, оверщут ~3%."""
    if lt < 0:
        return 0.0, 1.0
    p = ease(lt / dur)
    return p, 1.0 + 0.03 * math.sin(min(1.0, lt / dur) * math.pi)


def gtext(text, xy, size, blue=False, serif=False, op=1.0, anchor="mm"):
    f = font("serif_it" if serif else "sans", size)
    return dict(text=text, font=f, xy=xy, anchor=anchor,
                fill=BLUE if blue else WHITE,
                glow=BLUE_GLOW if blue else WHITE,
                glow_r=34 if blue else 16,
                glow_a=0.62 if blue else 0.45, opacity=op)


def glow_box(lay, box, radius, color, ga=0.55, gr=20):
    """Мягкое свечение под фигурой — считается в локальном боксе, не по всему кадру."""
    pad = gr * 3
    x0, y0, x1, y1 = [int(v) for v in box]
    bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
    bx1, by1 = min(W, x1 + pad), min(H, y1 + pad)
    if bx1 <= bx0 or by1 <= by0:
        return
    m = Image.new("L", (bx1 - bx0, by1 - by0), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [x0 - bx0, y0 - by0, x1 - bx0, y1 - by0], radius=radius, fill=255)
    m = m.filter(ImageFilter.GaussianBlur(gr))
    gl = Image.new("RGBA", m.size, color + (0,))
    gl.putalpha(m.point(lambda v: int(v * ga)))
    lay.alpha_composite(gl, (bx0, by0))


def draw_bar(lay, d, i, alpha=200, color=WHITE, scale=1.0, glow=0.0):
    h = BARS[i] * scale
    if h < 2:
        return
    x = bar_x(i)
    box = (x, BASE_Y - h, x + BAR_W, BASE_Y)
    if glow > 0:
        glow_box(lay, box, 10, color, ga=glow, gr=22)
    d.rounded_rectangle(box, radius=9, fill=color + (int(alpha),))


def dashed_h(d, y, x0, x1, color=WHITE, alpha=150, dash=18, gap=14, width=3):
    x = x0
    while x < x1:
        d.line([(x, y), (min(x + dash, x1), y)], fill=color + (alpha,), width=width)
        x += dash + gap


def dashed_v(d, x, y0, y1, color=WHITE, alpha=150, dash=16, gap=12, width=3):
    y = y0
    while y < y1:
        d.line([(x, y), (x, min(y + dash, y1))], fill=color + (alpha,), width=width)
        y += dash + gap


def cross(d, cx, cy, r=15, alpha=210, width=5):
    d.line([(cx - r, cy - r), (cx + r, cy + r)], fill=WHITE + (alpha,), width=width)
    d.line([(cx - r, cy + r), (cx + r, cy - r)], fill=WHITE + (alpha,), width=width)


# --- планы -------------------------------------------------------------------

def g_row_scan(lay, lt):
    """Варианты идут один за другим; пройденный сразу «запирается»."""
    d = ImageDraw.Draw(lay)
    step = 0.088                       # каскад тянется на всю длину плана (1.5с)
    for i in range(N_BARS):
        p = ease((lt - i * step) / 0.15)
        if p <= 0:
            continue
        lock = ease((lt - i * step - 0.22) / 0.30)
        draw_bar(lay, d, i, alpha=int(245 - 130 * lock), scale=p)
    cx = GX0 + min(1.0, lt / (N_BARS * step)) * (GX1 - GX0)
    d.line([(cx, TOP_Y - 20), (cx, BASE_Y + 22)], fill=WHITE + (180,), width=3)


def g_noreturn(lay, lt):
    """Всё пройденное закрыто крестом, движение только вперёд."""
    d = ImageDraw.Draw(lay)
    for i in range(N_BARS):
        draw_bar(lay, d, i, alpha=95)
    k = lt / 0.15
    for i in range(N_BARS):
        if i < k:
            a = int(210 * ease((k - i) / 0.7))
            cross(d, bar_x(i) + BAR_W / 2, BASE_Y - BARS[i] - 36, r=13, alpha=a, width=4)
    prog = ease(lt / 2.1)
    ax = GX0 + prog * (GX1 - GX0)
    d.line([(GX0, 1540), (ax, 1540)], fill=WHITE + (190,), width=4)
    if prog > 0.04:
        d.polygon([(ax + 16, 1540), (ax - 8, 1528), (ax - 8, 1552)], fill=WHITE + (220,))


def g_split37(lay, lt):
    """Отметка: первые 37% уходят в наблюдение. Число несёт графика."""
    d = ImageDraw.Draw(lay)
    fade = ease((lt - 0.55) / 0.45)
    for i in range(N_BARS):
        draw_bar(lay, d, i, alpha=int(200 - 140 * fade) if i < MARK else 200)
    drop = ease((lt - 0.15) / 0.5)
    if drop > 0:
        dashed_v(d, mark_x(MARK), 920, 920 + (BASE_Y + 26 - 920) * drop,
                 alpha=205, width=4)
    p, over = blue_pop(lt - 0.36)
    if p > 0:
        lay.alpha_composite(text_layer(
            (W, H), [gtext("37%", (345, 790), max(1, int(145 * (0.55 + 0.45 * p) * over)),
                           blue=True)]))


def g_bestfirst(lay, lt):
    """Из пропущенных запоминаем лучшего — он становится планкой."""
    d = ImageDraw.Draw(lay)
    for i in range(N_BARS):
        if i == BEST_SKIP:
            continue
        draw_bar(lay, d, i, alpha=60 if i < MARK else 190)
    dashed_v(d, mark_x(MARK), 920, BASE_Y + 26, alpha=150, width=3)
    hi = ease((lt - 0.2) / 0.45)
    draw_bar(lay, d, BEST_SKIP, alpha=int(120 + 135 * hi), glow=0.30 * hi)
    line = ease((lt - 0.55) / 0.95)
    if line > 0:
        y = BASE_Y - BARS[BEST_SKIP]
        x0 = bar_x(BEST_SKIP) + BAR_W
        dashed_h(d, y, x0, x0 + (GX1 - x0) * line, alpha=190, width=4)


def g_pick(lay, lt):
    """После отметки берём первого, кто выше планки."""
    d = ImageDraw.Draw(lay)
    y_line = BASE_Y - BARS[BEST_SKIP]
    for i in range(N_BARS):
        if i < MARK:
            draw_bar(lay, d, i, alpha=95 if i == BEST_SKIP else 55)
    dashed_v(d, mark_x(MARK), 920, BASE_Y + 26, alpha=130, width=3)
    dashed_h(d, y_line, bar_x(BEST_SKIP) + BAR_W, GX1, alpha=180, width=4)
    for i in range(MARK, N_BARS):
        at = (i - MARK) * 0.17
        if lt < at:
            draw_bar(lay, d, i, alpha=55)
            continue
        if i == PICK:
            p, over = blue_pop(lt - at, 0.34)
            draw_bar(lay, d, i, alpha=255, color=BLUE, scale=0.6 + 0.4 * p,
                     glow=0.55 * p)
        elif i < PICK:
            flash = ease((lt - at) / 0.12) * (1 - ease((lt - at - 0.18) / 0.3))
            draw_bar(lay, d, i, alpha=int(60 + 150 * flash))
        else:
            draw_bar(lay, d, i, alpha=55)


def _stop_at(lay, d, mark, pick_idx, lt, show_best=False):
    """Общая картинка «отметка на mark, взяли pick_idx»."""
    y_line = BASE_Y - max(BARS[:mark])
    for i in range(N_BARS):
        if i == pick_idx:
            continue
        dim = 55 if i < mark else 120
        draw_bar(lay, d, i, alpha=dim)
    dashed_v(d, mark_x(mark), 920, BASE_Y + 26, alpha=150, width=3)
    dashed_h(d, y_line, GX0, GX1, alpha=170, width=3)
    p, over = blue_pop(lt - 0.25, 0.34)
    if p > 0:
        draw_bar(lay, d, pick_idx, alpha=255, color=BLUE,
                 scale=0.6 + 0.4 * p, glow=0.5 * p)
    else:
        draw_bar(lay, d, pick_idx, alpha=120)
    if show_best:
        hi = ease((lt - 0.8) / 0.5)
        if hi > 0:
            draw_bar(lay, d, PICK, alpha=int(120 + 135 * hi), glow=0.35 * hi)
            dashed_h(d, BASE_Y - BARS[PICK], GX0,
                     GX0 + (GX1 - GX0) * hi, alpha=200, width=4)


def g_tooearly(lay, lt):
    """Остановились слишком рано — взяли первое, что выше почти пустой планки."""
    _stop_at(lay, ImageDraw.Draw(lay), EARLY_MARK, EARLY_MARK, lt)


def g_missed(lay, lt):
    """…а настоящий лучший был дальше по ряду."""
    d = ImageDraw.Draw(lay)
    _stop_at(lay, d, EARLY_MARK, EARLY_MARK, lt + 0.6, show_best=True)
    ar = ease((lt - 1.25) / 0.4)
    if ar > 0:
        x = bar_x(PICK) + BAR_W / 2
        y = BASE_Y - BARS[PICK] - 60
        d.polygon([(x, y + 26 * ar), (x - 17, y - 6), (x + 17, y - 6)],
                  fill=WHITE + (int(220 * ar),))


def g_toolate(lay, lt):
    """Смотрели слишком долго — лучший остался в зоне наблюдения."""
    d = ImageDraw.Draw(lay)
    for i in range(N_BARS):
        if i < LATE_MARK:
            fade = ease((lt - i * 0.09) / 0.35)
            draw_bar(lay, d, i, alpha=int(190 - 135 * fade))
        else:
            draw_bar(lay, d, i, alpha=150)
    mv = ease(lt / 1.15)
    dashed_v(d, GX0 + (mark_x(LATE_MARK) - GX0) * mv, 920, BASE_Y + 26,
             alpha=205, width=4)
    if lt > 1.3:
        hi = ease((lt - 1.3) / 0.4)
        dashed_h(d, BASE_Y - BARS[PICK], GX0, GX1, alpha=int(200 * hi), width=4)
        for i in range(LATE_MARK, N_BARS):
            if lt > 1.7 + (i - LATE_MARK) * 0.16:
                cross(d, bar_x(i) + BAR_W / 2, BASE_Y - BARS[i] - 36, r=13,
                      alpha=210, width=4)


def g_observed(lay, lt):
    """Зона наблюдения: смотрели и пропускали, лучший был здесь."""
    d = ImageDraw.Draw(lay)
    for i in range(N_BARS):
        if i < LATE_MARK:
            puls = max(0.0, 1 - abs(lt * 6.2 - i) / 1.6)
            draw_bar(lay, d, i, alpha=int(60 + 120 * puls))
        else:
            draw_bar(lay, d, i, alpha=45)
    d.line([(GX0, 1540), (mark_x(LATE_MARK), 1540)], fill=WHITE + (185,), width=4)
    d.line([(GX0, 1518), (GX0, 1540)], fill=WHITE + (185,), width=4)
    d.line([(mark_x(LATE_MARK), 1518), (mark_x(LATE_MARK), 1540)],
           fill=WHITE + (185,), width=4)
    if lt > 0.75:
        hi = ease((lt - 0.75) / 0.5)
        draw_bar(lay, d, PICK, alpha=int(70 + 175 * hi), glow=0.4 * hi)
        cross(d, bar_x(PICK) + BAR_W / 2, BASE_Y - BARS[PICK] - 40, r=15,
              alpha=int(220 * ease((lt - 1.35) / 0.35)), width=5)


def g_stats(lay, lt):
    """Пока собирали статистику — по каждому пройденному ставится отсечка."""
    d = ImageDraw.Draw(lay)
    for i in range(N_BARS):
        draw_bar(lay, d, i, alpha=55 if i < LATE_MARK else 45)
    d.line([(GX0, 1540), (mark_x(LATE_MARK), 1540)], fill=WHITE + (185,), width=4)
    for i in range(LATE_MARK):
        p = ease((lt - i * 0.075) / 0.18)
        if p <= 0:
            continue
        cx = bar_x(i) + BAR_W / 2
        y = BASE_Y - BARS[i] - 34
        r = 9 * p
        d.ellipse([cx - r, y - r, cx + r, y + r], fill=WHITE + (int(215 * p),))


def g_n37_100(lay, lt):
    """Итог: 37% синим, 100% белым и перечёркнуто."""
    items = []
    p, over = blue_pop(lt - 0.04)
    if p > 0:
        items.append(gtext("37%", (540, 1010),
                           max(1, int(185 * (0.55 + 0.45 * p) * over)), blue=True))
    q = pop(lt - 1.74, 0.10)
    if q > 0:
        items.append(gtext("100%", (540, 1310),
                           max(1, int(105 * (0.6 + 0.4 * q))), op=0.72))
    if items:
        lay.alpha_composite(text_layer((W, H), items))
    if lt > 1.95:
        s = ease((lt - 1.95) / 0.22)
        f = font("sans", 105)
        half = f.getlength("100%") / 2 + 22
        d = ImageDraw.Draw(lay)
        d.line([(540 - half, 1310), (540 - half + 2 * half * s, 1310)],
               fill=WHITE + (225,), width=7)


def g_curve(lay, lt):
    """P(доля наблюдения) = -x·ln x. Максимум ровно на 1/e ≈ 37%."""
    d = ImageDraw.Draw(lay)
    ax0, ax1, ay0, ay1 = 230, 870, 1480, 960
    d.line([(ax0, ay1 - 24), (ax0, ay0)], fill=WHITE + (140,), width=3)
    d.line([(ax0, ay0), (ax1 + 24, ay0)], fill=WHITE + (140,), width=3)

    def px(u):
        return ax0 + u * (ax1 - ax0)

    def py(p):
        return ay0 - (p / 0.42) * (ay0 - ay1)

    prog = ease((lt - 0.2) / 2.0)
    pts = []
    steps = 90
    for i in range(1, int(steps * prog) + 1):
        u = i / steps
        pts.append((px(u), py(-u * math.log(u))))
    if len(pts) > 1:
        d.line(pts, fill=WHITE + (230,), width=6, joint="curve")
    if lt > 2.25:
        hi = ease((lt - 2.25) / 0.4)
        ux = 1 / math.e
        cx, cy = px(ux), py(ux)
        dashed_v(d, cx, cy + 10, cy + 10 + (ay0 - cy - 10) * hi,
                 color=BLUE_GLOW, alpha=200, width=4)
        r = 13 * hi
        glow_box(lay, (cx - r, cy - r, cx + r, cy + r), int(r), BLUE_GLOW,
                 ga=0.5 * hi, gr=20)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BLUE + (255,))
        if lt > 2.6:
            op = ease((lt - 2.6) / 0.35)
            lay.alpha_composite(text_layer(
                (W, H), [gtext("37%", (cx, 905), 56, op=op * 0.95)]))


GFX = {
    "row_scan": g_row_scan, "noreturn": g_noreturn, "split37": g_split37,
    "bestfirst": g_bestfirst, "pick": g_pick, "tooearly": g_tooearly,
    "missed": g_missed, "toolate": g_toolate, "observed": g_observed,
    "stats": g_stats, "n37_100": g_n37_100, "curve": g_curve,
}


def topword_layer(lay, word, lt):
    """R6 — крупное слово КАПСОМ внутри карточки A, на стене над головой."""
    size = 40
    for s in range(40, 320, 2):
        if font("sans", s).getlength(word) > 800:
            break
        size = s
    op = ease(lt / 0.3) * 0.62
    lay.alpha_composite(text_layer((W, H), [dict(
        text=word, font=font("sans", size), xy=(540, 440), anchor="mm",
        fill=WHITE, glow=WHITE, glow_r=24, glow_a=0.22, opacity=op)]))


def graphics_layer(kind, prm, lt):
    """Только элементы переднего плана, без сетки-подложки."""
    if kind not in GFX and not prm.get("topword"):
        return None
    lay = _layer()
    if kind in GFX:
        GFX[kind](lay, lt)
    if prm.get("topword"):
        topword_layer(lay, prm["topword"], lt)
    return lay


# ---------------------------------------------------------------- субтитры --

KMAP = {"r": "sans", "i": "sans_it", "s": "serif_it"}


def line_items(runs, xy, anchor, opacity=1.0, reveal_chars=None, max_w=None):
    sizes = [sz for _, _, sz in runs]
    fonts = [font(KMAP[k], sz) for (_, k, sz) in runs]
    widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
    total = sum(widths) + 8 * (len(runs) - 1)
    if max_w and total > max_w:                     # ужать кегль, детерминированно
        sc = max_w / total
        sizes = [max(20, int(sz * sc)) for sz in sizes]
        fonts = [font(KMAP[runs[i][1]], sizes[i]) for i in range(len(runs))]
        widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
        total = sum(widths) + 8 * (len(runs) - 1)
    x, y = xy
    cx = x - total / 2 if anchor[0] == "m" else x
    out, cum = [], 0
    for i, (txt, k, _) in enumerate(runs):
        f = fonts[i]
        it = dict(text=txt, font=f, xy=(cx, y), anchor="l" + anchor[1],
                  fill=WHITE, glow=WHITE, glow_r=int(12 + sizes[i] * .08),
                  glow_a=0.58 if k == "s" else 0.48, opacity=opacity)
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


def cap_size(idx):
    return max(sz for _, _, sz in CAPS[idx][2])


def build_blocks():
    """Блок: до 3 фраз; рвётся на паузе >0.30с и на смене плана."""
    blocks, cur = [], []
    for i, c in enumerate(CAPS):
        if cur:
            prev = CAPS[cur[-1]]
            same_shot = shot_at(c[0]) is shot_at(prev[0])
            if c[0] - prev[1] > 0.30 or len(cur) >= 3 or not same_shot:
                blocks.append(cur)
                cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    info = {}
    for bid, b in enumerate(blocks):
        end = max(CAPS[i][1] for i in b)
        # смещение строки по вертикали: шаг растёт вместе с кеглем (brand-kit §5)
        offs, y = [], 0.0
        for pos, i in enumerate(b):
            if pos:
                y += max(60.0, (cap_size(b[pos - 1]) + cap_size(i)) * 0.70)
            offs.append(y)
        for pos, i in enumerate(b):
            info[i] = (bid, pos, len(b), end, offs[pos], offs[-1])
    return info


BLOCKS = build_blocks()
DIM = [1.0, 0.74, 0.58]


def caption_layer(t):
    kind = shot_at(t)[2]
    slot = SLOTS[slot_for(kind)]
    active = [i for i, c in enumerate(CAPS) if c[0] <= t < BLOCKS[i][3]]
    items = []
    for idx in active[-3:]:
        t0, _, runs = CAPS[idx]
        bid, pos, _, _, off, tot = BLOCKS[idx]
        newer = sum(1 for j in active if j > idx and BLOCKS[j][0] == bid)
        dim = DIM[min(newer, 2)]
        # блок двигается целиком, если не влезает в свою зону
        shift = 0.0
        if slot_for(kind) == "G":
            over = (slot["y"] + tot) - 560
            if over > 0:
                shift = -over
        bx = slot["x"]
        by = slot["y"] + off + shift
        if bid % 3 != 0:                     # каждый третий блок — спокойный
            bx += slot["scatter"][min(pos, 2)]
        lt = t - t0
        nchars = sum(len(r[0]) for r in runs)
        type_dur = min(0.55, max(0.30, nchars * 0.030))
        reveal = nchars if lt >= type_dur else nchars * lt / type_dur
        op = (0.35 + 0.65 * ease(lt / 0.40)) * dim
        max_w = slot["x_max"] - bx if slot["anchor"][0] == "l" else \
            2 * min(bx - slot["x_min"], slot["x_max"] - bx)
        items += line_items(runs, (bx, by), slot["anchor"], opacity=op,
                            reveal_chars=reveal, max_w=max_w)
    if not items:
        return None
    return text_layer((W, H), items)


# ------------------------------------------------------------------- сборка --

def background(kind, fr, t, read_stock):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        canvas.paste(source_card(fr, kind), (CARD_A[0], CARD_A[1]),
                     rounded_mask(CARD_A[2], CARD_A[3], R_A))
    elif kind == "stock":
        canvas.paste(stock_frame_from_bgr(read_stock()), (CARD_B[0], CARD_B[1]),
                     rounded_mask(CARD_B[2], CARD_B[3], R_B))
    else:
        x, y, w, h = CARD_A
        canvas.paste(grid_canvas(w, h, phase=t * 0.7), (x, y),
                     rounded_mask(w, h, R_A))
    return canvas


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    tmp = f"{BUILD}/assets/_video_secretary_raw.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    nf = int(round(DUR * FPS))
    stock_cap, stock_key = None, None
    last_face = None

    for fno in range(nf):
        ok, fr = cap.read()
        if ok:
            last_face = fr
        else:
            fr = last_face
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

        canvas = background(kind, fr, t, read_stock)
        gl = graphics_layer(kind, prm, lt)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        wr.write(pil_to_cv(canvas.convert("RGB")))
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}", flush=True)

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

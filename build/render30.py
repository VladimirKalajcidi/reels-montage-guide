"""Сборка ролика 30 («парадокс дружбы»).

Слойность: фон (лицо / сетка) -> графика -> субтитры. Аудио добавляет
sfx30.py, исходная речь не режется. Графика — только своя (сеть из точек
и рёбер), сток не используется: тема абстрактная, сеть показывает механику
буквально и без потери смысла (assets-manifest §2).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* межстрочный шаг считается от кеглей соседних строк, а не берётся фиксированным;
* кегль числа подбирается под ширину зоны графики, а не по шкале XL.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard30 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, TWO, HUNDRED, MULT, NUM_XY, NUM_SIZE,
                          MULT_XY, MULT_SIZE, NET_EDGES, NET_DEGREE, HUB,
                          ORDINARY, ru, shot_at, slot_for)
from style import *
import graph as G

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (29 замеров по всему дублю: cx 402±10, линия глаз y 688±4, ширина лица 288±6).
# Кроп ставит глаза на 42% высоты карточки — тот же приём, что в роликах 22-28.
FRAMINGS = {
    "A1": dict(w=630, h=1000, x=87, y=268),
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


# --- маска зоны графики -----------------------------------------------------
_ZMASK = None


def zone_mask():
    global _ZMASK
    if _ZMASK is None:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([GX0 + 16, GY0 + 16, GX1 - 16, GY1 - 16], fill=255)
        _ZMASK = m.filter(ImageFilter.GaussianBlur(5))
    return _ZMASK


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


# --- сеть: общий макет для netYou / random-person / random-friend / bias ----
# Хаб (0) в центре зоны, 8 узлов вокруг него — тот же граф, что в storyboard30
# (NET_EDGES): степень хаба 5, у остальных 1-3. Один и тот же макет во всех
# планах, чтобы зритель узнавал граф, а не видел три разных картинки.
# graph.py центрирует verts() на собственных (G.CX, G.CY) = (540, 905), но эта
# точка и радиусы RAD6/RAD5 не рассчитаны на зону графики этого ролика
# (y 700...1560, x 175...905, brand-kit §5) — сверху зоны остаётся всего 205px
# от центра до границы. Поэтому центр графа сдвинут ниже (NET_CY=1020) и радиус
# уменьшен так, чтобы кольцо с запасом на узел+свечение не касалось краёв зоны;
# offset() переносит любой выход verts()/G.verts в нужный центр.
NET_CY = 1020
NET_RAD = 220


def offset(pts, cx, cy):
    return [(px - G.CX + cx, py - G.CY + cy) for px, py in pts]


NET_VS = [(540, NET_CY)] + offset(G.verts(8, rad=NET_RAD, rot=-90), 540, NET_CY)


def net_layer(lt, highlight=None, dim_rest=False, appear=True):
    """highlight: None | набор индексов узлов, подсвеченных синим.
    dim_rest: остальные узлы притушены (используется на bias/compare)."""
    prog = stagger(lt, 9, t0=0.02, step=0.05, dur=0.30) if appear else [1.0] * 9
    eprog = stagger(lt, len(NET_EDGES), t0=0.10, step=0.04, dur=0.26) if appear else [1.0] * len(NET_EDGES)
    edges = []
    hl = highlight or set()
    for k, (i, j) in enumerate(NET_EDGES):
        kind = "white" if (i in hl and j in hl) else ("grey" if dim_rest else "white")
        edges.append((i, j, kind, eprog[k]))
    node_state = {}
    for i in range(9):
        sc = prog[i]
        if i in hl:
            node_state[i] = (sc, "b")
        elif dim_rest:
            node_state[i] = (sc * 0.9, "dim")
        else:
            node_state[i] = (sc, "w")
    return G.draw_graph(NET_VS, edges=edges, node_state=node_state)


def g_netYou(lay, lt):
    """«друзей чем у вас и это математический факт»: небольшая сеть — вы
    в центре и несколько друзей вокруг. Обычная эго-сеть одного человека."""
    l2 = net_layer(lt)
    lay.alpha_composite(l2)


TWO_Y0 = NET_CY - 90
TWO_DX = 165
TWO_Y1 = NET_CY + 190     # 930 / 375-705 / 1210 — все внутри зоны 700-1560 / 175-905


def g_twoFriends(lay, lt):
    """«у которого всего два друга»: узел с ровно двумя рёбрами. Цифра 2
    выходит резким попом (R5a) после того, как субтитр «всего» уже погас."""
    d = ImageDraw.Draw(lay)
    p0 = ease_out(clamp01(lt / 0.22))
    center = (540, TWO_Y0)
    left = (540 - TWO_DX, TWO_Y1)
    right = (540 + TWO_DX, TWO_Y1)
    for a, b in ((center, left), (center, right)):
        d.line([a, (a[0] + (b[0] - a[0]) * p0, a[1] + (b[1] - a[1]) * p0)],
               fill=_col(WHITE, 0.85), width=6)
    for i, pt in enumerate((center, left, right)):
        r = 24 * ease_out(clamp01((lt - i * 0.06) / 0.22))
        if r > 0:
            d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=_col(WHITE, 1.0))
    q = clamp01((lt - 1.65) / 0.10)              # 3 кадра, поп после «всего»
    if q > 0:
        draw_number(lay, str(TWO), NUM_XY, NUM_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


HUB_RING = 15
HUB_RING_RAD = 200


def g_hundredFriends(lay, lt):
    """«у которого 100 друзей»: плотное кольцо узлов вокруг хаба — условно
    «много», точное число несёт текст. Хаб крупнее — это тот же приём
    масштаба узла по степени, что и на netRandomPerson/netBias."""
    d = ImageDraw.Draw(lay)
    cx, cy = 540, NET_CY
    sats = offset(G.verts(HUB_RING, rad=HUB_RING_RAD, rot=-90), cx, cy)
    prog = stagger(lt, HUB_RING, t0=0.02, step=0.028, dur=0.24)
    for i, (sx, sy) in enumerate(sats):
        p = prog[i]
        if p <= 0.004:
            continue
        d.line([(cx, cy), (cx + (sx - cx) * p, cy + (sy - cy) * p)],
               fill=_col(WHITE, 0.55 * p), width=4)
    for i, (sx, sy) in enumerate(sats):
        p = prog[i]
        if p <= 0.004:
            continue
        r = 10 * ease_out(p)
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=_col(WHITE, 0.9 * p))
    rc = 30 * ease_out(clamp01(lt / 0.20))
    d.ellipse([cx - rc, cy - rc, cx + rc, cy + rc], fill=_col(WHITE, 1.0))
    q = clamp01((lt - 1.35) / 0.10)
    if q > 0:
        draw_number(lay, str(HUNDRED), NUM_XY, NUM_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


MUL_TOP_Y, MUL_BOT_Y = NET_CY - 200, NET_CY + 240   # 820 / 1260 — внутри зоны


def g_multiplier(lay, lt):
    """«50 раз больше возможностей»: маленькие 2 и 100 (напоминание) сверху
    и снизу, между ними — синее × 50 (R5b). × 50 в речи не звучит как число
    графики отдельно — это то, что добавляет график поверх сказанного:
    множитель, а не сама сотня/двойка (brand-kit §5)."""
    d = ImageDraw.Draw(lay)
    a = ease_out(clamp01(lt / 0.20))
    draw_number(lay, str(TWO), (540, MUL_TOP_Y), 84, color=WHITE, glow=WHITE,
                glow_a=0.5, alpha=a)
    draw_number(lay, str(HUNDRED), (540, MUL_BOT_Y), 84, color=WHITE, glow=WHITE,
                glow_a=0.5, alpha=a)
    q = ease_out(clamp01((lt - 0.08) / 0.28))
    for y0, y1 in ((MUL_TOP_Y + 62, MULT_XY[1] - 84), (MULT_XY[1] + 84, MUL_BOT_Y - 62)):
        d.line([(540, y0), (540 + (0) * q, y0 + (y1 - y0) * q)], fill=_col(WHITE, 0.8), width=6)
        if q > 0.9:
            for sx in (-1, 1):
                d.line([(540, y1), (540 + sx * 17, y1 - 20)], fill=_col(WHITE, 0.8), width=6)
    r = clamp01((lt - 0.18) / 0.38)
    if r > 0:
        sc = 0.55 + 0.45 * ease_out(r) + 0.03 * math.sin(math.pi * min(1.0, r))
        draw_number(lay, f"× {ru(MULT)}", MULT_XY, MULT_SIZE, color=BLUE,
                    glow=BLUE_GLOW, glow_a=0.85, scale=sc, reveal=clamp01((lt - 0.18) / 0.30))


def g_netRandomPerson(lay, lt):
    """«не случайного человека»: равновероятный выбор узла — стрелка падает
    на обычный узел (степень 1), он остаётся белым: это не «результат»,
    просто иллюстрация исходного, наивного способа выбирать."""
    lay.alpha_composite(net_layer(lt, dim_rest=False))
    d = ImageDraw.Draw(lay)
    tx, ty = NET_VS[ORDINARY]
    q = ease_out(clamp01((lt - 0.20) / 0.30))
    if q > 0.004:
        y0 = ty - 140 - (1 - q) * 60
        d.line([(tx, y0), (tx, ty - 30)], fill=_col(WHITE, 0.9 * q), width=6)
        for sx in (-1, 1):
            d.line([(tx, ty - 30), (tx + sx * 14, ty - 52)], fill=_col(WHITE, 0.9 * q), width=6)
        r = 30 * ease_out(clamp01((lt - 0.42) / 0.16))
        if r > 0:
            d.ellipse([tx - r, ty - r, tx + r, ty + r], outline=_col(WHITE, 0.9), width=5)


def g_netRandomFriend(lay, lt):
    """«а случайного друга человека»: выбор через случайное ребро — тот же
    граф, но стрелка падает на хаб. Хаб загорается синим: это и есть вывод,
    ради которого нарисован граф (brand-kit: синий = результат)."""
    d = ImageDraw.Draw(lay)
    hub_on = lt > 0.36
    lay.alpha_composite(net_layer(lt, highlight={HUB} if hub_on else None, appear=False))
    hx, hy = NET_VS[HUB]
    q = ease_out(clamp01((lt - 0.06) / 0.30))
    if q > 0.004:
        y0 = hy - 170 - (1 - q) * 60
        d.line([(hx, y0), (hx, hy - 38)], fill=_col(BLUE_GLOW, 0.95 * q), width=7)
        for sx in (-1, 1):
            d.line([(hx, hy - 38), (hx + sx * 16, hy - 62)], fill=_col(BLUE_GLOW, 0.95 * q), width=7)
    r = 34 * ease_out(clamp01((lt - 0.36) / 0.18))
    if r > 0:
        d.ellipse([hx - r, hy - r, hx + r, hy + r], outline=_col(BLUE_GLOW, 0.95), width=6)


BIAS_PICKS = [HUB, ORDINARY, HUB, HUB]
BIAS_STEP = 0.56


def g_netBias(lay, lt):
    """«популярные люди встречаются намного чаще»: несколько выборов подряд —
    хаб загорается почти каждый раз и остаётся подсвеченным."""
    k = min(len(BIAS_PICKS) - 1, int(lt / BIAS_STEP))
    e = lt - k * BIAS_STEP
    lit = {p for p in BIAS_PICKS[:k]}
    cur = BIAS_PICKS[k]
    q = ease_out(clamp01(e / 0.22))
    hl = lit | ({cur} if q > 0.3 else set())
    lay.alpha_composite(net_layer(lt, highlight=hl if hl else None,
                                  dim_rest=bool(hl), appear=False))
    d = ImageDraw.Draw(lay)
    x, y = NET_VS[cur]
    r = 40 * (1 - ease_out(clamp01(e / 0.30))) * (0.4 + 0.6 * q)
    if q > 0.02 and r > 1:
        d.ellipse([x - r, y - r, x + r, y + r], outline=_col(BLUE_GLOW, 0.7 * (1 - q + 0.2)), width=5)


SK_L, SK_R = 400, 680
SK_BASE_Y = NET_CY + 260
SK_H_LOW, SK_H_HIGH = 90, 340


def g_netSkew(lay, lt):
    """«и это создает систематическое смещение»: без чисел и подписей —
    просто два столбика, левый низкий (случайный человек), правый высокий
    (случайный друг). Смещение и есть разница высот."""
    d = ImageDraw.Draw(lay)
    q1 = ease_out(clamp01(lt / 0.30))
    q2 = ease_out(clamp01((lt - 0.16) / 0.42))
    w = 130
    h1 = SK_H_LOW * q1
    d.rounded_rectangle([SK_L - w / 2, SK_BASE_Y - h1, SK_L + w / 2, SK_BASE_Y],
                        radius=14, fill=_col(WHITE, 0.85))
    h2 = SK_H_HIGH * q2
    d.rounded_rectangle([SK_R - w / 2, SK_BASE_Y - h2, SK_R + w / 2, SK_BASE_Y],
                        radius=14, fill=_col(BLUE, 0.90 * (0.3 + 0.7 * q2)))
    d.line([(SK_L - w, SK_BASE_Y), (SK_R + w, SK_BASE_Y)], fill=_col(WHITE, 0.35), width=4)


# зона графики шире по вертикали (860px), чем по горизонтали (730px) —
# при двух эго-сетях рядом лимитирующая ось x, поэтому радиусы разные:
# правая (7 узлов) 130px, левая (2 узла) 90px, с зазором между кластерами.
EGO_L_CX, EGO_R_CX = 400, 680
EGO_CY = NET_CY + 30


def _ego(d, cx, cy, n, rad, prog, node_r=16):
    pts = offset(G.verts(n, rad=rad, rot=-90), cx, cy)
    for i, (sx, sy) in enumerate(pts):
        p = prog[i]
        if p <= 0.004:
            continue
        d.line([(cx, cy), (cx + (sx - cx) * p, cy + (sy - cy) * p)],
               fill=_col(WHITE, 0.55 * p), width=4)
    for i, (sx, sy) in enumerate(pts):
        p = prog[i]
        if p <= 0.004:
            continue
        r = node_r * ease_out(p)
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=_col(WHITE, 0.9 * p))
    rc = 22 * ease_out(clamp01(min(1.0, prog[0] * 3)))
    d.ellipse([cx - rc, cy - rc, cx + rc, cy + rc], fill=_col(BLUE, 1.0))


def g_netCompareA(lay, lt):
    """«в среднем количество друзей у вашего друга оказывается больше»:
    правая эго-сеть (друг) растёт первой и заметно гуще — 7 связей."""
    d = ImageDraw.Draw(lay)
    prog = stagger(lt, 7, t0=0.04, step=0.07, dur=0.30)
    _ego(d, EGO_R_CX, EGO_CY, 7, 130, prog)


def g_netCompareB(lay, lt):
    """«чем количество друзей у случайного человека»: слева достраивается
    редкая эго-сеть (случайный человек) — 2 связи, для контраста с правой."""
    d = ImageDraw.Draw(lay)
    prog_r = [1.0] * 7
    _ego(d, EGO_R_CX, EGO_CY, 7, 130, prog_r)
    prog_l = stagger(lt, 2, t0=0.04, step=0.10, dur=0.30)
    _ego(d, EGO_L_CX, EGO_CY, 2, 90, prog_l)


def g_netRecall(lay, lt):
    """«отчасти может быть просто математикой выборки»: короткое эхо того же
    механизма — маленький хаб с тремя узлами, выбор снова падает на хаб."""
    d = ImageDraw.Draw(lay)
    cx, cy = 540, NET_CY
    pts = offset(G.verts(3, rad=170, rot=-90), cx, cy)
    prog = stagger(lt, 3, t0=0.02, step=0.06, dur=0.24)
    for i, (sx, sy) in enumerate(pts):
        p = prog[i]
        if p <= 0.004:
            continue
        d.line([(cx, cy), (cx + (sx - cx) * p, cy + (sy - cy) * p)],
               fill=_col(WHITE, 0.6 * p), width=5)
        r = 16 * ease_out(p)
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=_col(WHITE, 0.9 * p))
    q = ease_out(clamp01((lt - 0.34) / 0.20))
    r = 26 * (0.6 + 0.4 * q)
    if q > 0.004:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_col(BLUE, 1.0))
        d.ellipse([cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8],
                  outline=_col(BLUE_GLOW, 0.7 * q), width=5)
    else:
        r0 = 20
        d.ellipse([cx - r0, cy - r0, cx + r0, cy + r0], fill=_col(WHITE, 1.0))


def g_word_paradox(lt):
    """R6: крупное слово ПАРАДОКС поверх карточки A, брендовая позиция y≈440."""
    text = "ПАРАДОКС"
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
    "netYou": g_netYou, "twoFriends": g_twoFriends,
    "hundredFriends": g_hundredFriends, "multiplier": g_multiplier,
    "netRandomPerson": g_netRandomPerson, "netRandomFriend": g_netRandomFriend,
    "netBias": g_netBias, "netSkew": g_netSkew,
    "netCompareA": g_netCompareA, "netCompareB": g_netCompareB,
    "netRecall": g_netRecall,
}


def graphics_layer(kind, lt):
    if kind == "A1word":
        return g_word_paradox(lt)
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    lay.putalpha(ImageChops.multiply(lay.split()[3], zone_mask()))
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


STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/30/stock"
STOCK_GAIN = 0.72      # притемнение вставки на 28% по экспозиции (brand-kit §6: 25-30%)


def stock_card(fr, cx=0.5, cy=0.5):
    fr = fit_bgr(fr, CARD_B[2], CARD_B[3], cx, cy)
    fr = np.clip(fr.astype(np.float32) * STOCK_GAIN, 0, 255)
    hsv = cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255)
    fr = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


def fit_bgr(fr, w, h, cx=0.5, cy=0.5):
    """Кроп до пропорции карточки. cx/cy — куда смещать окно кропа."""
    ih, iw = fr.shape[:2]
    target = w / h
    if iw / ih > target:
        nw = int(ih * target)
        x = int(round((iw - nw) * cx))
        x = max(0, min(iw - nw, x))
        fr = fr[:, x:x + nw]
    else:
        nh = int(iw / target)
        y = int(round((ih - nh) * cy))
        y = max(0, min(ih - nh, y))
        fr = fr[y:y + nh, :]
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)


class StockReader:
    """Сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр."""

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
        if want < self.pos - 0.05:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, want * 1000)
            self.pos = want
            ok, fr = self.cap.read()
            if ok:
                self.last = fr
            return self.last
        guard = 0
        while self.pos < want and guard < 400:
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


def background(kind, prm, fr, t, stock_reader=None):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        card = source_card(fr, kind)
        canvas.paste(card, (CARD_A[0], CARD_A[1]), rounded_mask(CARD_A[2], CARD_A[3], R_A))
    elif kind == "stock":
        card = stock_card(stock_reader(), prm.get("cx", 0.5), prm.get("cy", 0.5))
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
    enc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-an", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
        "-pix_fmt", "yuv420p", VID,
    ], stdin=subprocess.PIPE)
    stock = StockReader()

    last = None
    for fno in range(nf):
        ok, fr = cap.read()
        if ok:
            last = fr
        else:
            fr = last
        t = fno / FPS
        t0, t1, kind, prm = shot_at(t)
        canvas = compose(kind, prm, fr, t, t0,
                         lambda: stock.frame(prm["clip"], prm.get("ss", 0), t - t0))
        enc.stdin.write(pil_to_cv(canvas.convert("RGB")).tobytes())
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}", flush=True)

    enc.stdin.close()
    enc.wait()
    cap.release()
    stock.release()
    print("готово:", VID)


if __name__ == "__main__":
    main()

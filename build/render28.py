"""Сборка ролика 28 («парадокс двух конвертов»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx28.py, исходная речь не режется.
Вся графика живёт в GFX_ZONE/GFX_X, субтитры на сетке — выше y600 (brand-kit §5).

Унаследовано из разбора прошлых роликов:
* A-roll не грейдится — только hflip и кроп;
* кадры уходят в ffmpeg сырыми, без промежуточного cv2.VideoWriter(mp4v);
* сток конформится к 30 fps проекта по времени, а не читается кадр-в-кадр;
* межстрочный шаг считается от кеглей соседних строк, а не берётся фиксированным;
* кегль числа подбирается под ширину зоны графики, а не по шкале XL.

Визуальный язык (без единой надписи на графике):
* конверт = конверт из задачи, купюры внутри = сколько там денег;
* дуговая стрелка = обмен конвертами;
* ветвление = «во втором либо ..., либо ...»;
* ось с двумя метками = две суммы, метка посередине = их среднее;
* полоса под ветвью = вес ветви;
* синее = результат, лайм = единственное место, где рассуждение ломается.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard28 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, TAG, ENV_W, ENV_R, PAIR_DX, PAIR_CX,
                          BILL_W, BILL_H, BILL_STEP, BILL_UNIT,
                          FOUND_XY, FOUND_SIZE, EV_XY, EV_SIZE,
                          MULT_XY, MULT_SIZE,
                          FOUND, LOW, HIGH, EV, ANY_SUMS, ru,
                          shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_{TAG}.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/28/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GRID_CACHE = {}

# крупность A-roll. Исходник 720x1280; после hflip детектор даёт лицо стабильно
# (35 замеров по всему дублю: cx 387±7, линия глаз y 690±3, ширина лица 291±7).
# Кроп ставит глаза на 42% высоты карточки в обоих вариантах.
FRAMINGS = {
    "A1": dict(w=636, h=1009, x=69, y=266),
    "A2": dict(w=580, h=920, x=97, y=304),
}


def cv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def fit_bgr(fr, w, h, cx=0.5, cy=0.5):
    """Кроп до пропорции карточки. cx/cy — куда смещать окно кропа:
    композицию клипа кадрируем сами, важный объект держим в центре
    (brand-kit §2), а не полагаемся на центр исходного кадра."""
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


def stock_card(fr, cx=0.5, cy=0.5):
    fr = fit_bgr(fr, CARD_B[2], CARD_B[3], cx, cy)
    fr = np.clip(fr.astype(np.float32) * STOCK_GAIN, 0, 255)
    hsv = cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255)
    fr = cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
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


# --- маска зоны графики ----------------------------------------------------
_ZMASK = None


def zone_mask():
    """Страховка от вылета графики за GFX_ZONE/GFX_X.

    Врезка 16px при blur 5 гарантирует, что за границами зоны не остаётся
    ни одного пикселя с ненулевой альфой. Рабочее кадрирование делает сама
    геометрия: композиции считаются от центра зоны.
    """
    global _ZMASK
    if _ZMASK is None:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([GX0 + 16, GY0 + 16, GX1 - 16, GY1 - 16], fill=255)
        _ZMASK = m.filter(ImageFilter.GaussianBlur(5))
    return _ZMASK


# --- примитивы -------------------------------------------------------------

def _col(color, alpha):
    return tuple(color) + (max(0, min(255, int(round(255 * alpha)))),)


def seg(d, a, b, p=1.0, alpha=1.0, width=5, color=WHITE):
    """Отрезок, прорисованный на долю p от a к b."""
    p = clamp01(p)
    if p <= 0.002 or alpha <= 0.002:
        return
    d.line([a, (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)],
           fill=_col(color, alpha), width=width)


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
    """Крупное число со свечением (R5a/R5b).

    scale — масштаб появления, reveal — доля проявленных слева направо
    символов (набор по разрядам у синего числа).
    """
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


def num_width(text, size):
    return font("sans", size).getlength(text)


# --- конверт ---------------------------------------------------------------

def env_flap(d, cx, cy, w, alpha=1.0, color=WHITE, width=6, p=1.0):
    """Открытый клапан — треугольник вверх. Рисуется ДО купюр, чтобы стопка
    выходила из конверта, а не лежала поверх него."""
    p = clamp01(p)
    if p <= 0.004 or alpha <= 0.002:
        return
    s = 0.62 + 0.38 * ease_out(p)
    w = w * s
    h = w * ENV_R
    x0, x1, y0 = cx - w / 2, cx + w / 2, cy - h / 2
    d.line([(x0 + width * 0.4, y0 + width * 0.4), (cx, y0 - h * 0.44),
            (x1 - width * 0.4, y0 + width * 0.4)],
           fill=_col(color, alpha * p), width=width, joint="curve")


def envelope(d, cx, cy, w, alpha=1.0, flap=0.0, color=WHITE, width=6, p=1.0):
    """Корпус конверта: скруглённый прямоугольник; при flap=0 сверху «галка»
    закрытого клапана. Открытый клапан рисуется отдельно (env_flap)."""
    p = clamp01(p)
    if p <= 0.004 or alpha <= 0.002:
        return
    a = alpha * p
    s = 0.62 + 0.38 * ease_out(p)
    w = w * s
    h = w * ENV_R
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    d.rounded_rectangle([x0, y0, x1, y1], radius=w * 0.055,
                        outline=_col(color, a), width=width)
    if flap < 0.5:
        d.line([(x0 + width * 0.4, y0 + width * 0.4), (cx, y0 + h * 0.60),
                (x1 - width * 0.4, y0 + width * 0.4)],
               fill=_col(color, a), width=width, joint="curve")


def bills(d, cx, cy, w, n, alpha=1.0, color=WHITE, width=4, prog=None):
    """Стопка купюр, торчащая из открытого конверта: n купюр вверх от верхнего
    края корпуса. Количество купюр и есть «сколько там денег»: вдвое больше
    купюр — вдвое больше денег. Никаких подписей."""
    h = w * ENV_R
    bw, bh = w * BILL_W, h * BILL_H
    step = bh * BILL_STEP
    base = cy - h / 2 - bh * 0.12
    for i in range(n - 1, -1, -1):
        p = prog[i] if prog else 1.0
        if p <= 0.004:
            continue
        by = base - i * step - (1 - ease_out(p)) * 24
        jx = ((i % 2) * 2 - 1) * w * 0.020
        box = [cx - bw / 2 + jx, by - bh / 2, cx + bw / 2 + jx, by + bh / 2]
        d.rounded_rectangle(box, radius=bh * 0.20, fill=_col(BLACK, alpha * p),
                            outline=_col(color, alpha * p), width=width)
        d.line([(box[0] + bw * 0.30, by), (box[2] - bw * 0.30, by)],
               fill=_col(color, alpha * p * 0.55), width=max(2, width - 1))


def open_envelope(d, cx, cy, w, n=0, alpha=1.0, color=WHITE, width=6, p=1.0,
                  bill_prog=None):
    """Открытый конверт со стопкой: клапан -> купюры -> корпус спереди."""
    env_flap(d, cx, cy, w, alpha=alpha, color=color, width=width, p=p)
    if n:
        bills(d, cx, cy, w, n, alpha=alpha * clamp01(p * 1.4), color=color,
              width=max(3, width - 2), prog=bill_prog)
    envelope(d, cx, cy, w, alpha=alpha, flap=1.0, color=color, width=width, p=p)


def arc_arrow(d, a, b, rise, p=1.0, alpha=1.0, color=WHITE, width=6, head=26):
    """Дуговая стрелка a -> b: квадратичная кривая с подъёмом rise и наконечником."""
    p = clamp01(p)
    if p <= 0.01 or alpha <= 0.002:
        return
    cxp = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - rise)

    def bez(t):
        return ((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * cxp[0] + t * t * b[0],
                (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * cxp[1] + t * t * b[1])

    n = 48
    pts = [bez(i / n * p) for i in range(n + 1)]
    d.line(pts, fill=_col(color, alpha), width=width, joint="curve")
    if p > 0.5:
        tip = pts[-1]
        prev = pts[-4]
        ang = math.atan2(tip[1] - prev[1], tip[0] - prev[0])
        for s in (+1, -1):
            aa = ang + s * 2.55
            d.line([tip, (tip[0] + head * math.cos(aa), tip[1] + head * math.sin(aa))],
                   fill=_col(color, alpha), width=width)


def slot_box(d, cx, cy, w, h, alpha=1.0, color=WHITE, width=5, p=1.0):
    p = clamp01(p)
    if p <= 0.004:
        return
    s = 0.7 + 0.3 * ease_out(p)
    d.rounded_rectangle([cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2],
                        radius=h * 0.18, outline=_col(color, alpha * p), width=width)


def stagger(lt, n, t0=0.10, step=0.09, dur=0.42):
    return [ease_out(clamp01((lt - t0 - i * step) / dur)) for i in range(n)]


# --- планы -----------------------------------------------------------------

LOOP_Y = 1290          # линия конвертов на планах с обменом
LOOP_W = 290
ARR_Y = LOOP_Y - LOOP_W * ENV_R / 2 - 24
ARR_RISE = 108


def _swap_pair(d, alpha=1.0, appear=None):
    left = (PAIR_CX - PAIR_DX, LOOP_Y)
    right = (PAIR_CX + PAIR_DX, LOOP_Y)
    p = appear or (1.0, 1.0)
    envelope(d, left[0], left[1], LOOP_W, alpha=alpha, p=p[0])
    envelope(d, right[0], right[1], LOOP_W, alpha=alpha, p=p[1])
    return left, right


def g_swapLoop(lay, lt):
    """Хук «бесконечно менять своё решение»: два конверта и стрелка обмена,
    которая перекидывается туда-сюда."""
    d = ImageDraw.Draw(lay)
    left, right = _swap_pair(d, appear=stagger(lt, 2, t0=0.02, step=0.08, dur=0.22))
    if lt < 0.30:
        return
    e = lt - 0.30
    period = 0.40
    k = int(e / period)
    q = clamp01((e % period) / (period * 0.72))
    a = (left[0], ARR_Y) if k % 2 == 0 else (right[0], ARR_Y)
    b = (right[0], ARR_Y) if k % 2 == 0 else (left[0], ARR_Y)
    if k > 0:
        arc_arrow(d, b, a, ARR_RISE, p=1.0, alpha=0.16, width=5)
    arc_arrow(d, a, b, ARR_RISE, p=ease_out(q), alpha=0.95, width=7)


def g_loopAlways(lay, lt):
    """«менять конверт якобы выгодно всегда, даже если только что поменяли»:
    та же пара конвертов, но период броска сокращается — обмен не кончается."""
    d = ImageDraw.Draw(lay)
    left, right = _swap_pair(d)
    e, per, k = 0.0, 0.46, 0
    while True:                       # период падает с 0.46 до 0.17с
        per = max(0.17, 0.46 - 0.055 * k)
        if e + per > lt:
            break
        e += per
        k += 1
    q = clamp01((lt - e) / (per * 0.72))
    a = (left[0], ARR_Y) if k % 2 == 0 else (right[0], ARR_Y)
    b = (right[0], ARR_Y) if k % 2 == 0 else (left[0], ARR_Y)
    if k > 0:
        arc_arrow(d, b, a, ARR_RISE, p=1.0, alpha=0.18, width=5)
    arc_arrow(d, a, b, ARR_RISE, p=ease_out(q), alpha=0.95, width=7)


def g_twoEnv(lay, lt):
    """«итак перед вами два закрытых конверта»: ровно два конверта, закрытых."""
    d = ImageDraw.Draw(lay)
    pr = stagger(lt, 2, t0=0.02, step=0.10, dur=0.26)
    envelope(d, PAIR_CX - PAIR_DX, 1130, 300, p=pr[0])
    envelope(d, PAIR_CX + PAIR_DX, 1130, 300, p=pr[1])


DBL_Y = 1400
DBL_W = 290


def g_double(lay, lt):
    """«в одном лежит в два раза денег больше, чем в другом»: два открытых
    конверта, в левом две купюры, в правом четыре. Ровно вдвое, без подписей."""
    d = ImageDraw.Draw(lay)
    lx, rx = PAIR_CX - PAIR_DX, PAIR_CX + PAIR_DX
    open_envelope(d, lx, DBL_Y, DBL_W, n=FOUND // BILL_UNIT,
                  bill_prog=stagger(lt, FOUND // BILL_UNIT, t0=0.12, step=0.11,
                                    dur=0.30))
    open_envelope(d, rx, DBL_Y, DBL_W, n=HIGH // BILL_UNIT,
                  bill_prog=stagger(lt, HIGH // BILL_UNIT, t0=0.12, step=0.11,
                                    dur=0.30))


OPEN_Y = 1385
OPEN_W = 360


def g_open1000(lay, lt):
    """«там например тысяча рублей»: выбранный конверт открыт, из него
    поднимаются купюры, над ним — 1 000 ₽ (R5a, резкий поп)."""
    d = ImageDraw.Draw(lay)
    open_envelope(d, PAIR_CX, OPEN_Y, OPEN_W, n=FOUND // BILL_UNIT,
                  p=ease_out(clamp01(lt / 0.26)),
                  bill_prog=stagger(lt, FOUND // BILL_UNIT, t0=0.14, step=0.10,
                                    dur=0.26))
    # 0.76с — не «покрасивее», а обязательное условие: субтитр «там например»
    # гаснет на 12.757, число выходит уже в тишине текста (brand-kit §5)
    q = clamp01((lt - 0.76) / 0.10)             # 3 кадра, 0.6 -> 1.0, без оверщута
    if q > 0:
        draw_number(lay, f"{ru(FOUND)} \u20bd", FOUND_XY, FOUND_SIZE, color=WHITE,
                    glow=WHITE, glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


BR_TOP = (540, 870)
BR_TOP_W = 285
BR_L, BR_R = 350, 730
BR_Y = 1350
BR_W = 250


def g_branch(lay, lt):
    """«кажется, что во втором...»: второй конверт наверху и две ветви вниз
    к двум таким же конвертам — двум вариантам того, что в нём лежит.
    Оба варианта пока закрыты: открывает их следующий план."""
    d = ImageDraw.Draw(lay)
    envelope(d, BR_TOP[0], BR_TOP[1], BR_TOP_W, p=ease_out(clamp01(lt / 0.26)))
    a = (BR_TOP[0], BR_TOP[1] + BR_TOP_W * ENV_R / 2 + 12)
    q = ease_out(clamp01((lt - 0.30) / 0.42))
    seg(d, a, (BR_L, BR_Y - BR_W * ENV_R / 2 - 14), p=q, width=6)
    seg(d, a, (BR_R, BR_Y - BR_W * ENV_R / 2 - 14), p=q, width=6)
    pr = stagger(lt, 2, t0=0.72, step=0.10, dur=0.30)
    envelope(d, BR_L, BR_Y, BR_W, alpha=0.80, p=pr[0])
    envelope(d, BR_R, BR_Y, BR_W, alpha=0.80, p=pr[1])


BN_Y = 1370
BN_W = 285
BN_NUM_Y = 950
BN_NUM_SIZE = 80


def g_branchNums(lay, lt):
    """«либо 500, либо 2000»: две возможности материализуются — конверт
    с одной купюрой и конверт с двумя. Субтитра на плане нет: числа звучат."""
    d = ImageDraw.Draw(lay)
    lx, rx = PAIR_CX - PAIR_DX, PAIR_CX + PAIR_DX
    pr = stagger(lt, 2, t0=0.02, step=0.09, dur=0.24)
    open_envelope(d, lx, BN_Y, BN_W, n=LOW // BILL_UNIT, p=pr[0],
                  bill_prog=stagger(lt, LOW // BILL_UNIT, t0=0.16, step=0.09, dur=0.26))
    open_envelope(d, rx, BN_Y, BN_W, n=HIGH // BILL_UNIT, p=pr[1],
                  bill_prog=stagger(lt, HIGH // BILL_UNIT, t0=0.16, step=0.09, dur=0.26))
    for i, (x, val) in enumerate(((lx, LOW), (rx, HIGH))):
        q = clamp01((lt - 0.50 - i * 0.22) / 0.10)
        if q > 0:
            draw_number(lay, ru(val), (x, BN_NUM_Y), BN_NUM_SIZE, color=WHITE,
                        glow=WHITE, glow_a=0.55, scale=0.6 + 0.4 * ease_out(q))


AX_Y = 1110
AX_X0, AX_X1 = 330, 750
AX_LBL_SIZE = 68


def g_ev1250(lay, lt):
    """«средняя ожидаемая сумма — 1250»: 500 и 2 000 стоят на оси в своих
    точках, метка едет ровно на середину между ними — и там загорается
    синее 1 250 ₽ (R5b). Середина отрезка [500, 2 000] и есть 1 250."""
    d = ImageDraw.Draw(lay)
    q = ease_out(clamp01(lt / 0.20))
    d.line([(AX_X0, AX_Y), (AX_X0 + (AX_X1 - AX_X0) * q, AX_Y)],
           fill=_col(WHITE, 0.55), width=5)
    for i, (x, val) in enumerate(((AX_X0, LOW), (AX_X1, HIGH))):
        p = ease_out(clamp01((lt - 0.06 - i * 0.09) / 0.22))
        if p <= 0.004:
            continue
        d.line([(x, AX_Y - 30), (x, AX_Y + 30)], fill=_col(WHITE, 0.85 * p), width=6)
        draw_number(lay, ru(val), (x, AX_Y - 110), AX_LBL_SIZE, color=WHITE,
                    glow=WHITE, glow_a=0.5, alpha=p)
    m = ease_out(clamp01((lt - 0.52) / 0.42))    # метка едет к середине
    if m > 0.004:
        mx = AX_X0 + (AX_X1 - AX_X0) * 0.5 * m
        d.line([(mx, AX_Y - 40), (mx, AX_Y + 40)], fill=_col(BLUE_GLOW, 0.95), width=8)
    r = clamp01((lt - 0.94) / 0.38)
    if r > 0:
        sc = 0.55 + 0.45 * ease_out(r) + 0.03 * math.sin(math.pi * min(1.0, r))
        draw_number(lay, f"{ru(EV)} \u20bd", EV_XY, EV_SIZE, color=BLUE, glow=BLUE_GLOW,
                    glow_a=0.85, scale=sc, reveal=clamp01((lt - 0.94) / 0.30))


SG_Y = 1390
SG_W = 285
SG_NUM_Y = 1010
SG_NUM_SIZE = 66


def g_swapGood(lay, lt):
    """«получается, менять конверт выгодно»: в своём конверте 1 000,
    во втором — ожидаемые 1 250, и синяя стрелка ведёт туда."""
    d = ImageDraw.Draw(lay)
    lx, rx = PAIR_CX - PAIR_DX, PAIR_CX + PAIR_DX
    open_envelope(d, lx, SG_Y, SG_W, n=FOUND // BILL_UNIT,
                  bill_prog=stagger(lt, FOUND // BILL_UNIT, t0=0.04, step=0.08,
                                    dur=0.22))
    envelope(d, rx, SG_Y, SG_W)
    draw_number(lay, f"{ru(FOUND)} \u20bd", (lx, SG_NUM_Y), SG_NUM_SIZE, color=WHITE,
                glow=WHITE, glow_a=0.5, alpha=ease_out(clamp01((lt - 0.10) / 0.26)))
    q = clamp01((lt - 0.44) / 0.30)
    if q > 0:
        draw_number(lay, f"{ru(EV)} \u20bd", (rx, SG_NUM_Y), SG_NUM_SIZE, color=BLUE,
                    glow=BLUE_GLOW, glow_a=0.85, scale=0.6 + 0.4 * ease_out(q))
    y = SG_Y - SG_W * ENV_R / 2 - 26
    arc_arrow(d, (lx, y), (rx, y), 96, p=ease_out(clamp01((lt - 0.80) / 0.42)),
              alpha=0.95, color=BLUE_GLOW, width=8)


AS_TOP_Y, AS_BOT_Y = 880, 1390
AS_SIZE = 96


def g_anySum(lay, lt):
    """«точно такое же рассуждение при любой увиденной сумме»: сумма наверху
    меняется, результат внизу меняется вместе с ней, а множитель между ними
    не меняется никогда. × 1,25 в речи не звучит — это то, что добавляет
    графика (brand-kit §5)."""
    d = ImageDraw.Draw(lay)
    hold = 0.66
    i = min(len(ANY_SUMS) - 1, int(lt / hold))
    s = ANY_SUMS[i]
    a = ease_out(clamp01((lt - i * hold) / 0.20))
    draw_number(lay, f"{ru(s)} \u20bd", (540, AS_TOP_Y), AS_SIZE, color=WHITE,
                glow=WHITE, glow_a=0.55, alpha=a)
    draw_number(lay, f"{ru(int(s * 1.25))} \u20bd", (540, AS_BOT_Y), AS_SIZE,
                color=WHITE, glow=WHITE, glow_a=0.55, alpha=a)
    q = ease_out(clamp01((lt - 0.10) / 0.30))
    for y0, y1 in ((AS_TOP_Y + 76, MULT_XY[1] - 86), (MULT_XY[1] + 86, AS_BOT_Y - 76)):
        seg(d, (540, y0), (540, y1), p=q, alpha=0.8, width=6)
        if q > 0.9:                                   # наконечник вниз
            for sx in (-1, 1):
                d.line([(540, y1), (540 + sx * 17, y1 - 20)],
                       fill=_col(WHITE, 0.8), width=6)
    r = clamp01((lt - 0.24) / 0.38)
    if r > 0:
        sc = 0.55 + 0.45 * ease_out(r) + 0.03 * math.sin(math.pi * min(1.0, r))
        draw_number(lay, "\u00d7 1,25", MULT_XY, MULT_SIZE, color=BLUE, glow=BLUE_GLOW,
                    glow_a=0.85, scale=sc, reveal=clamp01((lt - 0.24) / 0.30))


FL_TOP = (540, 810)
FL_L, FL_R = 330, 750
FL_NUM_Y = 1180
FL_NUM_SIZE = 70
FL_BAR_Y = 1390
FL_BAR_W = 250


def g_flaw(lay, lt):
    """«мы без основания считаем две суммы равновероятными»: от 1 000 идут
    две ветви к 500 и 2 000, под каждой — полоса веса. Полосы приходят
    равными, а потом начинают разъезжаться и не останавливаются: ничто
    в задаче не закрепляет их пополам. Разъехавшиеся полосы — единственный
    лаймовый элемент ролика."""
    d = ImageDraw.Draw(lay)
    draw_number(lay, f"{ru(FOUND)} \u20bd", FL_TOP, 84, color=WHITE, glow=WHITE,
                glow_a=0.5, alpha=ease_out(clamp01(lt / 0.24)))
    a = (FL_TOP[0], FL_TOP[1] + 76)
    q = ease_out(clamp01((lt - 0.12) / 0.30))
    seg(d, a, (FL_L, FL_NUM_Y - 62), p=q, width=6)
    seg(d, a, (FL_R, FL_NUM_Y - 62), p=q, width=6)
    for i, (x, val) in enumerate(((FL_L, LOW), (FL_R, HIGH))):
        p = ease_out(clamp01((lt - 0.42 - i * 0.09) / 0.24))
        draw_number(lay, ru(val), (x, FL_NUM_Y), FL_NUM_SIZE, color=WHITE, glow=WHITE,
                    glow_a=0.5, alpha=p)
    w = ease_out(clamp01((lt - 0.78) / 0.30))
    if w <= 0.004:
        return
    drift = clamp01((lt - 1.40) / 0.55)
    k = 0.5 + drift * 0.34 * math.sin((lt - 1.40) * 2.0)
    col = LIME if drift > 0.02 else WHITE
    for x, frac in ((FL_L, 1 - k), (FL_R, k)):
        d.line([(x - FL_BAR_W / 2, FL_BAR_Y), (x + FL_BAR_W / 2, FL_BAR_Y)],
               fill=_col(WHITE, 0.22 * w), width=14)
        ln = FL_BAR_W * frac * w
        d.line([(x - FL_BAR_W / 2, FL_BAR_Y), (x - FL_BAR_W / 2 + ln, FL_BAR_Y)],
               fill=_col(col, 0.95 * w), width=14)


# Две пары, при которых в руках могла оказаться 1 000: (500, 1 000) и
# (1 000, 2 000). Пары показаны конвертами в том же масштабе, что и весь
# ролик (одна купюра = 500 ₽), поэтому новых чисел на экране не появляется:
# конверт с двумя купюрами — тот самый, что уже в руках, он обведён рамкой
# в обеих парах. Какая пара разыгрывалась — из условия не следует, поэтому
# подсветка ходит между ними и ни на одной не останавливается.
PR_PAIRS = ((LOW, FOUND), (FOUND, HIGH))
PR_Y = (975, 1330)
PR_X = (380, 700)
PR_ENV_W = 205


def g_prior(lay, lt):
    """«для конкретного расчёта нужно знать, как изначально выбрали суммы»:
    увиденный конверт (две купюры) одинаково хорошо ложится и в пару
    «одна купюра — две», и в пару «две купюры — четыре»."""
    d = ImageDraw.Draw(lay)
    hi = 0 if math.sin(lt * 2.3) >= 0 else 1
    h = PR_ENV_W * ENV_R
    for i, pair in enumerate(PR_PAIRS):
        y = PR_Y[i]
        p = ease_out(clamp01((lt - 0.02 - i * 0.12) / 0.26))
        if p <= 0.004:
            continue
        a = (1.0 if i == hi else 0.52) * p
        for j, val in enumerate(pair):
            n = val // BILL_UNIT
            open_envelope(d, PR_X[j], y, PR_ENV_W, n=n, alpha=a, p=p, width=5)
            if val == FOUND:                     # конверт, который уже в руках
                top = y - h / 2 - n * h * BILL_H * BILL_STEP - 26
                d.rounded_rectangle([PR_X[j] - PR_ENV_W / 2 - 20, top,
                                     PR_X[j] + PR_ENV_W / 2 + 20, y + h / 2 + 18],
                                    radius=22, outline=_col(WHITE, a * 0.80), width=5)


GFX = {
    "swapLoop": g_swapLoop, "twoEnv": g_twoEnv, "double": g_double,
    "open1000": g_open1000, "branch": g_branch, "branchNums": g_branchNums,
    "ev1250": g_ev1250, "swapGood": g_swapGood, "anySum": g_anySum,
    "loopAlways": g_loopAlways, "flaw": g_flaw, "prior": g_prior,
}


def graphics_layer(kind, lt):
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    lay.putalpha(ImageChops.multiply(lay.split()[3], zone_mask()))
    return lay


# --- субтитры --------------------------------------------------------------

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
    """Блок = до 2 фраз в пределах одного плана, разрыв по паузе >0.30с.

    Вертикальный шаг внутри блока считается от кеглей соседних строк
    (brand-kit §5), поэтому строка 54pt не садится на соседнюю 42pt.
    """
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
            calm = bid % 3 == 0                      # каждый третий блок — спокойный
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

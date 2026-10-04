"""Инфографика ролика 57 (слот 157): два конверта + вопрос «50%?» + RED «?».

Сцена R10 (каскад) на сетке y 720–1360:
  - левый конверт «500 ₽»  появляется на t_env1 (0.25с ease-out, подъём 18px)
  - правый конверт «2 000 ₽» на t_env2
  - метка «50%?  50%?»       на t_label
  - большой RED «?»          на t_qm (impact в sfx157)

Числа 1000/500/2000/1250 — числа R5a/R5b: pop_item из envelopes157 (не dirichlet50).
"""
import math
from PIL import Image, ImageDraw, ImageFont
from style import W, H, WHITE, ease_out, SANS

NUM_BLUE = (91, 124, 255)
RED_COL = (255, 59, 48)
LABEL_COL = (200, 200, 200)

_wf = {}


def wide(size):
    if size not in _wf:
        f = ImageFont.truetype(SANS, size)
        f.set_variation_by_axes([150, 28, 400, 1000])
        _wf[size] = f
    return _wf[size]


def _ramp(t, t0, dur):
    return min(1.0, max(0.0, (t - t0) / dur))


def pop_item(t, t0, text, xy, size, col=None, anchor="mm"):
    """R5b (синее): 0.38с, 0.55→1.0 с оверщутом ~3%.
    R5a (белое): 80мс, 0.6→1.0, без оверщута."""
    if col is None:
        col = WHITE
    lt = t - t0
    if lt < 0:
        return []
    blue = (col == NUM_BLUE)
    if blue:
        p = min(1.0, lt / 0.38)
        sc = 0.55 + 0.45 * ease_out(p)
        if 0.55 < p < 1.0:
            sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    else:
        p = min(1.0, lt / 0.08)
        sc = 0.6 + 0.4 * ease_out(p)
    f = wide(max(8, int(size * sc)))
    it = dict(text=text, font=f, xy=xy, anchor=anchor, fill=col,
              glow_r=0, opacity=min(1.0, lt / (0.12 if blue else 0.06)))
    if blue and p < 1.0:
        wdt = f.getlength(text)
        it["reveal"] = ("wipe", xy[0] - wdt / 2 + wdt * min(1.0, p / 0.8) + 4)
    return [it]


def fit_size(text, size, max_w):
    return min(size, int(size * max_w / wide(size).getlength(text)))


# ---------------------------------------------------------------- числа R5a/R5b
NUM_Y = 1010


def num_1000_items(t, prm):
    label = "НАПРИМЕР"
    digits = "1 000 РУБЛЕЙ"
    sz = fit_size(digits, 160, 700)
    out = []
    out += pop_item(t, prm["t_label"], label, (540, NUM_Y - sz // 2 - 60), 56)
    out += pop_item(t, prm["t_num"], digits, (540, NUM_Y), sz)
    return out


def num_500_2000_items(t, prm):
    d500 = "500"
    d2000 = "2 000 РУБЛЕЙ"
    sz500 = fit_size(d500, 180, 340)
    sz2000 = fit_size(d2000, 130, 700)
    out = []
    out += pop_item(t, prm["t_500"],  d500,  (540, 870),  sz500)
    out += pop_item(t, prm["t_2000"], d2000, (540, 1090), sz2000)
    return out


def num_1250_items(t, prm):
    label = "ОЖИДАЕМАЯ СТОИМОСТЬ"
    digits = "1 250 РУБЛЕЙ"
    sz = fit_size(digits, 160, 700)
    out = []
    out += pop_item(t, prm["t_label"], label, (540, NUM_Y - sz // 2 - 60), 44)
    out += pop_item(t, prm["t_num"], digits, (540, NUM_Y), sz, NUM_BLUE)
    return out


# ---------------------------------------------------------------- два конверта R10
ZX0, ZY0, ZX1, ZY1 = 130, 720, 950, 1360
LW, LH = ZX1 - ZX0, ZY1 - ZY0
SS = 2

ENV_W, ENV_H = 310, 370
ENV_L_CX, ENV_R_CX = 270, 810
ENV_CY = 990
LABEL_Y = 1240
QM_Y = 820


def _envelope(d, cx, cy, a, dy, text_items):
    """Рисует контур конверта (rounded rect) + V-образный лоскут + текст суммы."""
    x0 = (cx - ENV_W / 2 - ZX0) * SS
    y0 = (cy + dy - ENV_H / 2 - ZY0) * SS
    x1 = (cx + ENV_W / 2 - ZX0) * SS
    y1 = (cy + dy + ENV_H / 2 - ZY0) * SS
    col = WHITE + (int(230 * a),)
    d.rounded_rectangle([x0, y0, x1, y1], radius=14 * SS, outline=col, width=5 * SS)
    # V-образный клапан вверху
    mx = (x0 + x1) / 2
    top = y0 + 28 * SS
    pts = [(x0 + 2 * SS, y0 + 2 * SS), (mx, top), (x1 - 2 * SS, y0 + 2 * SS)]
    d.line(pts, fill=col, width=4 * SS)
    return text_items


def envelopes_gfx(t, shot):
    prm = shot[3]
    lay = Image.new("RGBA", (LW * SS, LH * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    items = []

    for key, cx, amount, t_key in [
        ("t_env1", ENV_L_CX, "500 ₽",   prm["t_env1"]),
        ("t_env2", ENV_R_CX, "2 000 ₽", prm["t_env2"]),
    ]:
        p = _ramp(t, t_key, 0.25)
        if p <= 0:
            continue
        a = ease_out(p)
        dy = 18 * (1 - a)
        _envelope(d, cx, ENV_CY, a, dy, [])
        sz = fit_size(amount, 80, ENV_W - 20)
        # cold: amount as text_layer items (canvas coords)
        items += pop_item(t, t_key, amount,
                          (cx, ENV_CY + dy - 10), sz, WHITE)

    # «50%?  50%?» метка
    lbl_p = _ramp(t, prm["t_label"], 0.20)
    if lbl_p > 0:
        lbl_a = ease_out(lbl_p)
        lbl_dy = int(14 * (1 - lbl_a))
        lbl_t = wide(48)
        lbl_text = "50%?       50%?"
        lbl_col = LABEL_COL + (int(230 * lbl_a),)
        d.text(((LW * SS) / 2, (LABEL_Y - ZY0 + lbl_dy) * SS), lbl_text,
               font=lbl_t, anchor="mm", fill=lbl_col)

    # RED «?» большой
    qm_p = _ramp(t, prm["t_qm"], 0.10)
    if qm_p > 0:
        qm_a = ease_out(qm_p)
        qm_dy = int(22 * (1 - qm_a))
        qm_t = wide(int(200 * (0.6 + 0.4 * ease_out(min(1.0, (t - prm["t_qm"]) / 0.08)))))
        qm_col = RED_COL + (int(255 * min(1.0, (t - prm["t_qm"]) / 0.06)),)
        d.text(((LW * SS) / 2, (QM_Y - ZY0 + qm_dy) * SS), "?",
               font=qm_t, anchor="mm", fill=qm_col)

    loc = lay.resize((LW, LH), Image.LANCZOS)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.alpha_composite(loc, (ZX0, ZY0))
    return out, items

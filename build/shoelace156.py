"""Графика ролика 56: формула Шнурки.

gfx(t, shot) → (RGBA-слой или None, items)
Вызывается из render156.py для каждого кадра с kind ∈ GRID_KINDS.
"""
import math
from PIL import Image, ImageDraw

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from root125 import wide

W, H = 1080, 1920

# ─── координатная плоскость ────────────────────────────────────────────────
ORIG = (235, 1390)   # начало координат на холсте
SC   = 75            # пикселей на единицу

PTS_MATH  = [(1, 1), (5, 1), (6, 4), (3, 6), (1, 4)]
VERT_T    = [9.40, 10.02, 10.70, 11.42, 12.10]  # абс. время появления каждой вершины
VERT_LBL  = ["(1,1)", "(5,1)", "(6,4)", "(3,6)", "(1,4)"]
VERT_OFF  = [(14, -18), (14, -18), (14, -18), (-52, -28), (-14, -18)]  # смещение метки

def _mp(x, y):
    return (ORIG[0] + int(x * SC), ORIG[1] - int(y * SC))

# ─── таблица координат ──────────────────────────────────────────────────────
MCX     = 440   # центр x-столбца
MCY_COL = 640   # центр y-столбца
M_HDR_Y = 800   # y базовой линии заголовка
M_DY    = 80    # шаг строки

ROWS   = [(1,1),(5,1),(6,4),(3,6),(1,4),(1,1)]
R_PROD = [1, 20, 36, 12, 1]
L_PROD = [5,  6, 12,  6, 4]

# ─── палитра ────────────────────────────────────────────────────────────────
W255  = (255, 255, 255, 255)
GRAY  = (180, 180, 180, 180)
TEAL4 = ( 45, 225, 194, 255)
RED4  = (255,  59,  48, 255)
BLUE4 = ( 59,  75, 232, 255)
DIM4  = (160, 160, 160, 180)


def _lay():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def _dt(img, text, f, xy, anchor="lt", fill=W255):
    ImageDraw.Draw(img).text(xy, text, font=f, anchor=anchor, fill=fill)


# ─── вспомогательные рисовалки ─────────────────────────────────────────────

def _axes(img):
    d = ImageDraw.Draw(img)
    col = (200, 200, 200, 200)
    # оси
    d.line([_mp(0, 0), _mp(7.6, 0)], fill=col, width=3)
    d.line([_mp(0, 0), _mp(0, 7.6)], fill=col, width=3)
    # стрелки
    xe = _mp(7.6, 0); d.polygon([(xe[0]+8,xe[1]),(xe[0]-4,xe[1]-5),(xe[0]-4,xe[1]+5)], fill=col)
    ye = _mp(0, 7.6); d.polygon([(ye[0],ye[1]-8),(ye[0]-5,ye[1]+4),(ye[0]+5,ye[1]+4)], fill=col)
    # деления и числа
    f_ax = wide(28)
    for i in range(1, 7):
        px, py = _mp(i, 0)
        d.line([(px, py-5), (px, py+5)], fill=col, width=2)
        _dt(img, str(i), f_ax, (px, py+14), anchor="mt", fill=col)
    for j in range(1, 7):
        px, py = _mp(0, j)
        d.line([(px-5, py), (px+5, py)], fill=col, width=2)
        _dt(img, str(j), f_ax, (px-14, py), anchor="rm", fill=col)


def _pentagon(img, alpha=255):
    pts = [_mp(x, y) for x, y in PTS_MATH]
    col_fill = (45, 225, 194, max(0, alpha - 210))
    col_line = (45, 225, 194, alpha)
    d = ImageDraw.Draw(img)
    if col_fill[3] > 0:
        d.polygon(pts, fill=col_fill, outline=col_line, width=3)
    else:
        for k in range(len(pts)):
            d.line([pts[k], pts[(k+1) % len(pts)]], fill=col_line, width=3)


def _matrix(img, n_rows, diag_r=0.0, diag_l=0.0):
    """Таблица до n_rows строк (из 6); правые / левые диагонали по прозрачности 0–1."""
    f_h = wide(44)
    f_d = wide(46)
    f_p = wide(32)
    d = ImageDraw.Draw(img)

    _dt(img, "x", f_h, (MCX, M_HDR_Y), anchor="ms", fill=W255)
    _dt(img, "y", f_h, (MCY_COL, M_HDR_Y), anchor="ms", fill=W255)
    sep_y = M_HDR_Y + 16
    mid_x = (MCX + MCY_COL) // 2
    d.line([(MCX-70, sep_y), (MCY_COL+70, sep_y)], fill=GRAY, width=2)
    d.line([(mid_x, M_HDR_Y-50), (mid_x, M_HDR_Y + max(n_rows,1)*M_DY + 20)], fill=GRAY, width=2)

    ry = []
    for k in range(n_rows):
        row_y = M_HDR_Y + (k+1) * M_DY
        ry.append(row_y)
        xi, yi = ROWS[k]
        col = TEAL4 if (xi, yi) == (1, 1) and k > 0 else W255
        _dt(img, str(xi), f_d, (MCX, row_y), anchor="ms", fill=col)
        _dt(img, str(yi), f_d, (MCY_COL, row_y), anchor="ms", fill=col)

    # правые диагонали: x_i → y_{i+1}
    if diag_r > 0.01 and len(ry) >= 2:
        a = int(255 * diag_r)
        rc = (TEAL4[0], TEAL4[1], TEAL4[2], a)
        for k in range(min(len(ry)-1, 5)):
            d.line([(MCX, ry[k]), (MCY_COL, ry[k+1])], fill=rc, width=3)
            mx = (MCX + MCY_COL)//2 + 30
            my = (ry[k] + ry[k+1])//2
            _dt(img, str(R_PROD[k]), f_p, (mx, my), anchor="lm",
                fill=(TEAL4[0], TEAL4[1], TEAL4[2], a))

    # левые диагонали: y_i → x_{i+1}
    if diag_l > 0.01 and len(ry) >= 2:
        a = int(255 * diag_l)
        lc = (RED4[0], RED4[1], RED4[2], a)
        for k in range(min(len(ry)-1, 5)):
            d.line([(MCY_COL, ry[k]), (MCX, ry[k+1])], fill=lc, width=3)
            mx = (MCX + MCY_COL)//2 - 30
            my = (ry[k] + ry[k+1])//2
            _dt(img, str(L_PROD[k]), f_p, (mx, my), anchor="rm",
                fill=(RED4[0], RED4[1], RED4[2], a))


# ─── фазы ───────────────────────────────────────────────────────────────────

def _polygon(t, shot):
    img = _lay(); _axes(img); _pentagon(img); return img, []


def _labels(t, shot):
    img = _lay(); _axes(img); _pentagon(img)
    f_lb = wide(28)
    for i, (mx, my) in enumerate(PTS_MATH):
        px, py = _mp(mx, my)
        d = ImageDraw.Draw(img)
        d.ellipse([(px-8, py-8),(px+8, py+8)], fill=TEAL4, outline=(255,255,255,200), width=2)
        ox, oy = VERT_OFF[i]
        _dt(img, VERT_LBL[i], f_lb, (px+ox, py+oy), anchor="lt", fill=TEAL4)
    return img, []


def _points(t, shot):
    img = _lay(); _axes(img)
    n = sum(1 for vt in VERT_T if t >= vt)
    for i in range(n):
        mx, my = PTS_MATH[i]
        px, py = _mp(mx, my)
        age = t - VERT_T[i]
        alp = min(1.0, age / 0.30)
        a = int(255 * alp)
        dy_off = int(10 * (1 - alp))
        d = ImageDraw.Draw(img)
        d.ellipse([(px-10, py-10-dy_off),(px+10, py+10-dy_off)],
                  fill=(TEAL4[0],TEAL4[1],TEAL4[2],a),
                  outline=(255,255,255,a), width=2)
        ox, oy = VERT_OFF[i]
        _dt(img, VERT_LBL[i], wide(28), (px+ox, py+oy-dy_off),
            anchor="lt", fill=(TEAL4[0],TEAL4[1],TEAL4[2],a))
    return img, []


def _matrix_phase(t, shot):
    t0, dur = shot[0], shot[1]-shot[0]
    rel = t - t0
    n = max(1, min(5, int(5 * rel / max(dur-0.2, 0.5) + 0.5)))
    img = _lay(); _matrix(img, n); return img, []


def _right_diag(t, shot):
    t0 = shot[0]; rel = t - t0
    diag_r = min(1.0, max(0.0, (rel - 0.6) / 1.5))
    img = _lay(); _matrix(img, 6, diag_r=diag_r); return img, []


def _num70(t, shot):
    t0 = shot[0]; p = min(1.0, (t - t0) / 0.30)
    img = _lay()
    # матрица с диагоналями на фоне
    _matrix(img, 6, diag_r=1.0)
    # = 70 справа от матрицы
    a70 = int(255 * p)
    _dt(img, "= 70", wide(40), (MCY_COL+70, M_HDR_Y+4*M_DY),
        anchor="lm", fill=(TEAL4[0],TEAL4[1],TEAL4[2],a70))
    # R5a: крупная белая «70»
    if p < 0.05:
        return img, []
    os_ = 1.0 + 0.03 * math.sin(math.pi * min(1.0, p / 0.5)) if p < 1.0 else 1.0
    sz = int(220 * os_)
    items = [dict(text="70", font=wide(sz), xy=(540, 930), anchor="mm",
                  fill=(255, 255, 255), glow_r=0, opacity=min(1.0, p*2))]
    return img, items


def _left_diag(t, shot):
    t0 = shot[0]; rel = t - t0
    diag_l = min(1.0, max(0.0, (rel - 0.5) / 1.0))
    img = _lay()
    _matrix(img, 6, diag_r=1.0, diag_l=diag_l)
    # «= 33» когда сказано «Получаем»
    if t >= 27.80:
        a = int(255 * min(1.0, (t - 27.80) / 0.30))
        _dt(img, "= 33", wide(40), (MCY_COL+70, M_HDR_Y+4*M_DY+22),
            anchor="lm", fill=(RED4[0],RED4[1],RED4[2],a))
    return img, []


def _calc(t, shot):
    img = _lay()
    cy = 900

    def _fad(start):
        return min(1.0, max(0.0, (t - start) / 0.35))

    f62 = wide(62); f72 = wide(72)
    # «70 − 33»
    if t >= 29.32:
        p = int(255 * _fad(29.32))
        _dt(img, "70", f72, (440, cy), anchor="rm", fill=(255,255,255,p))
        _dt(img, "−",  f62, (480, cy), anchor="lm", fill=(180,180,180,p))
        _dt(img, "33", f72, (540, cy), anchor="lm", fill=(255,255,255,p))
    # «= 37»
    if t >= 30.24:
        p = int(255 * _fad(30.24))
        _dt(img, "= 37", f62, (630, cy), anchor="lm", fill=(255,255,255,p))
    # «|37| ÷ 2»
    cy2 = cy + 130
    if t >= 31.14:
        p = int(255 * _fad(31.14))
        _dt(img, "|37|", f72, (440, cy2), anchor="rm", fill=(255,255,255,p))
        _dt(img, "÷ 2", f62, (480, cy2), anchor="lm", fill=(180,180,180,p))
    # «= 18.5»
    if t >= 32.26:
        p = int(255 * _fad(32.26))
        _dt(img, "= 18.5", f62, (590, cy2), anchor="lm",
            fill=(BLUE4[0],BLUE4[1],BLUE4[2],p))
    return img, []


def _result(t, shot):
    t0 = shot[0]; p = min(1.0, (t - t0) / 0.35)
    if p < 0.05:
        return None, []
    os_ = 1.0 + 0.03 * math.sin(math.pi * min(1.0, p / 0.6)) if p < 1.0 else 1.0
    sz = int(210 * os_)
    items = [dict(text="18.5", font=wide(sz), xy=(540, 930), anchor="mm",
                  fill=(59, 75, 232), glow_r=0, opacity=min(1.0, p*2))]
    return None, items


def _shoelace(t, shot):
    t0 = shot[0]; rel = t - t0
    diag = min(1.0, max(0.0, (rel - 0.4) / 1.5))
    img = _lay()
    _matrix(img, 6, diag_r=diag, diag_l=diag)
    if t >= 43.26:
        a = int(255 * min(1.0, (t - 43.26) / 0.35))
        _dt(img, "ШНУРОВКА", wide(48), (540, 1440),
            anchor="mm", fill=(TEAL4[0],TEAL4[1],TEAL4[2],a))
    return img, []


def _exam(t, shot):
    img = _lay(); _axes(img); _pentagon(img)
    f_lb = wide(28)
    for i, (mx, my) in enumerate(PTS_MATH):
        px, py = _mp(mx, my)
        ImageDraw.Draw(img).ellipse([(px-8,py-8),(px+8,py+8)],
                                    fill=TEAL4, outline=(255,255,255,200), width=2)
        ox, oy = VERT_OFF[i]
        _dt(img, VERT_LBL[i], f_lb, (px+ox, py+oy), anchor="lt", fill=TEAL4)
    if t >= 47.62:
        a = int(255 * min(1.0, (t - 47.62) / 0.35))
        _dt(img, "ЗАДАЧА", wide(52), (540, 745), anchor="mm",
            fill=(RED4[0],RED4[1],RED4[2],a))
        _dt(img, "ОГЭ", wide(68), (540, 815), anchor="mm",
            fill=(RED4[0],RED4[1],RED4[2],a))
    return img, []


# ─── публичный API ───────────────────────────────────────────────────────────

_DISPATCH = {
    "polygon":    _polygon,
    "labels":     _labels,
    "points":     _points,
    "matrix":     _matrix_phase,
    "right_diag": _right_diag,
    "num70":      _num70,
    "left_diag":  _left_diag,
    "calc":       _calc,
    "result":     _result,
    "shoelace":   _shoelace,
    "exam":       _exam,
}


def gfx(t, shot):
    fn = _DISPATCH.get(shot[2])
    if fn is None:
        return None, []
    return fn(t, shot)

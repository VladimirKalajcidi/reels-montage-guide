"""Графика ролика 56 (v2, слот 156): формула шнуровки (Гаусса) на клетке.

gfx(t, shot) -> (RGBA-слой на весь холст или None, []). Всё рисуется в габарите карточки A
с суперсэмплингом ×2 (сглаженные линии), зона графики x 175…905, y 700…1360 (ниже — субтитры v3,
база 1500). Шрифт — SF Pro Expanded Black, без свечения; синий числа #5B7CFF.

Координатная плоскость (сцены PLOT_KINDS): сетка R4 с клеткой 84 (render156.grid_bg), её линии
x = 65.75 + 84·i, y = 198.25 + 84·j (style.grid_canvas, seed 7) — начало координат в узле
(233.75, 1290.25), единица = клетка, вершины пятиугольника стоят в узлах.
Таблица x | y — столбцы 430 / 650 (центр 540), произведения ↘ — справа (830), ↙ — слева (250),
строки 864…1314 через 90. Задача ОГЭ — на узлах штатной сетки 168 (как pick128).

Цвета серий: ↘ (x·y следующей строки, сумма 70) — бирюзовый, ↙ (y·x следующей строки, 33) — красный;
те же цвета у 70 и 33 в вычислении. Математика закреплена assert'ами ниже.
"""
import math
from PIL import Image, ImageDraw, ImageFont
from style import W, H, CARD_A, SANS, ease_out

WHITE = (255, 255, 255)
BLUE = (91, 124, 255)
TEAL = (45, 225, 194)
RED = (255, 59, 48)
CX0, CY0, CW, CH = CARD_A
SS = 2
GX0, GX1, GY0, GY1 = 175, 905, 700, 1360
_wf = {}


def wide(size):
    size = max(6, int(round(size)))
    if size not in _wf:
        f = ImageFont.truetype(SANS, size)
        f.set_variation_by_axes([150, 28, 400, 1000])
        _wf[size] = f
    return _wf[size]


def ramp(t, t0, dur):
    return min(1.0, max(0.0, (t - t0) / dur)) if dur > 0 else float(t >= t0)


def eo(t, t0, dur):
    return ease_out(ramp(t, t0, dur))


def fit(texts, size, avail):
    while max(wide(size).getlength(s) for s in texts) > avail and size > 24:
        size -= 2
    return size


# ---------------------------------------------------------------- холст ×2
class G:
    def __init__(self):
        self.im = Image.new("RGBA", (CW * SS, CH * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def p(self, q):
        return ((q[0] - CX0) * SS, (q[1] - CY0) * SS)

    def line(self, pts, col=WHITE, a=255, w=5):
        if a <= 2 or len(pts) < 2:
            return
        self.d.line([self.p(q) for q in pts], fill=col + (int(a),), width=int(w * SS), joint="curve")
        for q in (pts[0], pts[-1]):
            self.dot(q, w / 2, col, a)

    def dot(self, q, r, col=WHITE, a=255):
        if r <= 0.3 or a <= 2:
            return
        x, y = self.p(q)
        r *= SS
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=col + (int(a),))

    def ring(self, q, r, w, col=WHITE, a=255):
        if r <= 0.3 or a <= 2:
            return
        x, y = self.p(q)
        r *= SS
        self.d.ellipse([x - r, y - r, x + r, y + r], outline=col + (int(a),), width=int(w * SS))

    def fill(self, pts, col=WHITE, a=50):
        if a > 1:
            self.d.polygon([self.p(q) for q in pts], fill=col + (int(a),))

    def text(self, q, s, size, col=WHITE, a=255, anchor="ms"):
        if a > 2:
            self.d.text(self.p(q), s, font=wide(size * SS), fill=col + (int(a),), anchor=anchor)

    def path(self, pts, prog, col=WHITE, a=255, w=6, closed=False):
        """Ломаная, нарисованная на долю prog своей длины (0..1)."""
        prog = min(1.0, max(0.0, prog))
        if prog <= 0:
            return None
        pl = list(pts) + ([pts[0]] if closed else [])
        segs = list(zip(pl, pl[1:]))
        lens = [math.dist(u, v) for u, v in segs]
        left = prog * sum(lens)
        out = [pl[0]]
        for (u, v), L in zip(segs, lens):
            if left <= 0:
                break
            k = min(1.0, left / L) if L else 1.0
            out.append((u[0] + (v[0] - u[0]) * k, u[1] + (v[1] - u[1]) * k))
            left -= L
        out = [q for i, q in enumerate(out) if i == 0 or math.dist(q, out[i - 1]) > 0.01]
        if len(out) >= 2:
            self.line(out, col, a, w)
        return out[-1]

    def arrow(self, u, v, prog, col, a=255, w=6, head=20):
        """Стрелка u→v, растёт на долю prog; наконечник — на текущем конце."""
        prog = min(1.0, max(0.0, prog))
        if prog <= 0:
            return
        e = (u[0] + (v[0] - u[0]) * prog, u[1] + (v[1] - u[1]) * prog)
        L = math.dist(u, e)
        if L < 2:
            return
        dx, dy = (e[0] - u[0]) / L, (e[1] - u[1]) / L
        base = (e[0] - dx * head * 0.9, e[1] - dy * head * 0.9)
        self.line([u, base], col, a, w)
        nx, ny = -dy, dx
        tri = [e, (base[0] + nx * head * 0.55, base[1] + ny * head * 0.55),
               (base[0] - nx * head * 0.55, base[1] - ny * head * 0.55)]
        self.d.polygon([self.p(q) for q in tri], fill=col + (int(a),))

    def dashed(self, u, v, prog, col=WHITE, a=200, w=3, dash=12, gap=10):
        prog = min(1.0, max(0.0, prog))
        L = math.dist(u, v) * prog
        if L <= 0:
            return
        dx, dy = (v[0] - u[0]) / math.dist(u, v), (v[1] - u[1]) / math.dist(u, v)
        s = 0.0
        while s < L:
            e = min(L, s + dash)
            self.line([(u[0] + dx * s, u[1] + dy * s), (u[0] + dx * e, u[1] + dy * e)], col, a, w)
            s += dash + gap

    def layer(self):
        out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        out.paste(self.im.resize((CW, CH), Image.LANCZOS), (CX0, CY0))
        return out


# ---------------------------------------------------------------- математика
PENT = [(1, 1), (5, 1), (6, 4), (3, 6), (1, 4)]
ROWS = PENT + [PENT[0]]                                          # «повторяем первую точку»
R_PROD = [ROWS[i][0] * ROWS[i + 1][1] for i in range(5)]         # ↘ x_i·y_{i+1}
L_PROD = [ROWS[i][1] * ROWS[i + 1][0] for i in range(5)]         # ↙ y_i·x_{i+1}
assert R_PROD == [1, 20, 36, 12, 1] and sum(R_PROD) == 70
assert L_PROD == [5, 6, 12, 6, 4] and sum(L_PROD) == 33
assert abs(sum(R_PROD) - sum(L_PROD)) / 2 == 18.5


def area(p):
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
                   for i in range(len(p)))) / 2


def centroid(p):
    a = c_x = c_y = 0.0
    for i in range(len(p)):
        x0, y0 = p[i]
        x1, y1 = p[(i + 1) % len(p)]
        k = x0 * y1 - x1 * y0
        a += k
        c_x += (x0 + x1) * k
        c_y += (y0 + y1) * k
    return (c_x / (3 * a), c_y / (3 * a))


def _simple(p):
    """Многоугольник без самопересечений (непересекающиеся несоседние рёбра)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if j in (i, (i + 1) % n) or i == (j + 1) % n:
                continue
            a, b, c, d = p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]
            if cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0:
                return False
    return True


assert area(PENT) == 18.5 and _simple(PENT)

# ---------------------------------------------------------------- координатная плоскость
OX, OY, U = 233.75, 1290.25, 84.0


def P(x, y):
    return (OX + U * x, OY - U * y)


AX_LEN = 6.7
TICK_SZ = 34
LBL_SZ = fit(["(6; 4)"], 42, 150)
# подписи вершин: (якорь, смещение) — снаружи контура, внутри зоны (проверено превью preview156)
LBL = {(1, 1): ("lt", (14, 14)), (5, 1): ("lt", (14, 14)), (6, 4): ("lm", (18, 0)),
       (3, 6): ("ls", (16, -14)), (1, 4): ("lm", (18, 22))}


def lbl_text(q):
    return f"({q[0]}; {q[1]})"


for _q, (_anc, _off) in LBL.items():
    _x, _y = P(*_q)
    _x0, _y0, _x1, _y1 = wide(LBL_SZ).getbbox(lbl_text(_q), anchor=_anc)
    assert GX0 <= _x + _off[0] + _x0 and _x + _off[0] + _x1 <= GX1, _q
    assert GY0 <= _y + _off[1] + _y0 and _y + _off[1] + _y1 <= GY1, _q


def axes(g, lt, a=235, num_a=160, draw=None, hi=()):
    """Оси с делениями 1…6. draw=None — уже нарисованы; иначе lt-момент начала прорисовки.
    hi — координаты, чьи деления подсвечены: [('x', 5), ('y', 1)]."""
    px = 1.0 if draw is None else eo(lt, draw, 0.35)
    py = 1.0 if draw is None else eo(lt, draw + 0.10, 0.35)
    if px > 0:
        g.arrow(P(0, 0), P(AX_LEN, 0), px, WHITE, a, 5, 22)
    if py > 0:
        g.arrow(P(0, 0), P(0, AX_LEN), py, WHITE, a, 5, 22)
    for k in range(1, 7):
        u = 1.0 if draw is None else eo(lt, draw + 0.25 + 0.04 * k, 0.2)
        if u <= 0:
            continue
        for ax in ("x", "y"):
            on = (ax, k) in hi
            na = 255 if on else num_a
            sz = TICK_SZ + (6 if on else 0)
            if ax == "x":
                x, y = P(k, 0)
                g.line([(x, y - 7), (x, y + 7)], WHITE, a * u, 3)
                g.text((x, y + 16), str(k), sz, WHITE, na * u, "mt")
            else:
                x, y = P(0, k)
                g.line([(x - 7, y), (x + 7, y)], WHITE, a * u, 3)
                g.text((x - 16, y), str(k), sz, WHITE, na * u, "rm")
    la = 255 * (1.0 if draw is None else eo(lt, draw + 0.4, 0.2))
    xe, ye = P(AX_LEN, 0)
    g.text((xe - 6, ye + 14), "x", 36, WHITE, la, "lt")
    xe, ye = P(0, AX_LEN)
    g.text((xe + 22, ye + 10), "y", 36, WHITE, la, "lm")


def pent_px():
    return [P(*q) for q in PENT]


# ---------------------------------------------------------------- сцены плоскости
HOOK_Q = (1.18, 0.28)        # «S = ?» на «клетке» (2.76 → lt 1.14)


def hook(g, lt):
    pts = pent_px()
    for k, q in enumerate(pts):
        g.dot(q, 11 * eo(lt, 0.02 + 0.05 * k, 0.18), WHITE, 240)
    g.fill(pts, WHITE, 46 * eo(lt, 0.70, 0.40))
    g.path(pts, (lt - 0.10) / 0.60, WHITE, 245, 7, closed=True)
    u = eo(lt, HOOK_Q[0], HOOK_Q[1])
    if u > 0:
        c = P(*centroid(PENT))
        g.text((c[0], c[1] + 30 + 14 * (1 - u)), "S = ?", 84, WHITE, 255 * u, "ms")


def axes_scene(g, lt):      # 4.92 «координатную»; вершины — на «вершин» 7.26 → lt 2.34
    pts = pent_px()
    g.fill(pts, WHITE, 30)
    g.path(pts, 1, WHITE, 150, 5, closed=True)
    axes(g, lt, draw=0.02)
    for k, q in enumerate(pts):
        u = eo(lt, 2.34 + 0.07 * k, 0.2)
        g.dot(q, 7 + 5 * u, WHITE, 150 + 100 * u)


PT_T = [9.30, 10.02, 10.70, 11.42, 12.10]      # «1 1», «5 1», «6 4», «3 6», «1 4» — первое число пары


def points(g, t, lt):
    pts = pent_px()
    hi = []
    out = eo(lt, 3.50, 0.40)                    # 12.46: контур собирается из вершин
    g.path(pts, 1, WHITE, 60 + 175 * out, 5 + 2 * out, closed=True)
    for k, q in enumerate(PENT):
        if t < PT_T[k] - 0.02:
            continue
        x, y = P(*q)
        nxt = PT_T[k + 1] if k + 1 < 5 else 99
        live = 1 - ramp(t, nxt - 0.02, 0.25)    # проекции — только у текущей точки
        pr = eo(t, PT_T[k], 0.22)
        if live > 0:
            g.dashed((x, y), (x, OY), pr, TEAL, 210 * live, 3)
            g.dashed((x, y), (OX, y), pr, TEAL, 210 * live, 3)
            if pr > 0.6:
                hi += [("x", q[0]), ("y", q[1])]
        u = eo(t, PT_T[k], 0.18)
        g.dot((x, y), 13 * u, TEAL if live > 0.5 else WHITE, 255)
        anc, off = LBL[q]
        g.text((x + off[0], y + off[1] + 10 * (1 - u)), lbl_text(q), LBL_SZ, WHITE, 255 * u, anc)
    axes(g, lt, hi=hi)


SHAPES = [[(0, 1), (2, 0), (3, 2), (5, 0), (6, 3), (4, 4), (5, 6), (2, 5), (1, 3)],
          [(1, 0), (6, 1), (4, 2), (6, 5), (3, 4), (1, 6), (0, 3), (2, 2)]]
for _s in SHAPES:
    assert _simple(_s) and all(0 <= x <= 6 and 0 <= y <= 6 for x, y in _s)
ANY_T = (0.00, 0.56)            # вторая фигура — на «любой» 37.80 → lt 0.56


def anyshape(g, lt):
    for k, sh in enumerate(SHAPES):
        t0 = ANY_T[k]
        gone = ramp(lt, ANY_T[1] - 0.02, 0.10) if k == 0 else 0
        a = 1 - gone
        if lt < t0 or a <= 0:
            continue
        pts = [P(*q) for q in sh]
        for j, q in enumerate(pts):
            g.dot(q, 10 * eo(lt, t0, 0.10), WHITE, 240 * a)
        g.fill(pts, WHITE, 46 * eo(lt, t0 + 0.24, 0.16) * a)
        g.path(pts, (lt - t0) / 0.26, WHITE, 245 * a, 7, closed=True)


ORD_T = (40.84, 42.10)          # «по порядку» → «границы»: точка обходит контур


def order(g, t, lt):
    axes(g, lt, a=110, num_a=90)
    pts = pent_px()
    g.fill(pts, WHITE, 26)
    g.path(pts, 1, WHITE, 90, 5, closed=True)
    prog = ramp(t, *ORD_T[:1], ORD_T[1] - ORD_T[0])
    head = g.path(pts, prog, WHITE, 255, 8, closed=True) if prog > 0 else pts[0]
    per = [math.dist(pts[i], pts[(i + 1) % 5]) for i in range(5)]
    reach = [sum(per[:i]) / sum(per) for i in range(5)]
    c = P(*centroid(PENT))
    for k, q in enumerate(pts):
        g.dot(q, 9, WHITE, 230)
        tk = ORD_T[0] + reach[k] * (ORD_T[1] - ORD_T[0])
        u = eo(t, tk - 0.04, 0.18)
        if u > 0:
            dx, dy = q[0] - c[0], q[1] - c[1]
            L = math.hypot(dx, dy)
            b = (q[0] + dx / L * 44, q[1] + dy / L * 44)
            g.dot(b, 24 * (0.6 + 0.4 * u), TEAL, 255 * u)
            g.text((b[0], b[1] + 1), str(k + 1), 30, (0, 0, 0), 255 * u, "mm")
    if 0 < prog < 1:
        g.dot(head, 15, TEAL, 255)


# ---------------------------------------------------------------- таблица
XL, XC, YC, XR = 250, 430, 650, 830
HDR_Y = 774
ROW_Y = [864 + 90 * k for k in range(6)]
NUM_SZ, HDR_SZ, PROD_SZ = 72, 60, 60
MID_Y = [(ROW_Y[i] + ROW_Y[i + 1]) / 2 for i in range(5)]


def _cap(size):
    return -wide(size).getbbox("0", anchor="ls")[1]


def num(g, x, yc, s, size, col=WHITE, a=255, rise=0.0):
    g.text((x, yc + _cap(size) / 2 + rise), s, size, col, a, "ms")


def table(g, rows_a, hdr_a=1.0, num_a=255, row_col=None):
    """rows_a — непрозрачность 0..1 по строкам (с подъёмом при появлении)."""
    if hdr_a > 0:
        num(g, XC, HDR_Y - 24, "x", HDR_SZ, WHITE, 200 * hdr_a)
        num(g, YC, HDR_Y - 24, "y", HDR_SZ, WHITE, 200 * hdr_a)
        g.line([(XC - 80, HDR_Y + 14), (YC + 80, HDR_Y + 14)], WHITE, 130 * hdr_a, 3)
        g.line([(540, HDR_Y - 50), (540, ROW_Y[-1] + 40)], WHITE, 90 * hdr_a, 3)
    for k, u in enumerate(rows_a):
        if u <= 0:
            continue
        col = (row_col or {}).get(k, WHITE)
        x, y = ROWS[k]
        num(g, XC, ROW_Y[k], str(x), NUM_SZ, col, num_a * u, 14 * (1 - u))
        num(g, YC, ROW_Y[k], str(y), NUM_SZ, col, num_a * u, 14 * (1 - u))


ROW_T = [14.02, 14.36, 14.70, 15.04, 15.38]     # «координаты всех вершин друг под другом»
NW = wide(NUM_SZ).getlength("0")


def r_arrow(i):     # ↘ от x_i к y_{i+1}
    return (XC + NW / 2 + 16, ROW_Y[i] + 8), (YC - NW / 2 - 16, ROW_Y[i + 1] - 8)


def l_arrow(i):     # ↙ от y_i к x_{i+1}
    return (YC - NW / 2 - 16, ROW_Y[i] + 8), (XC + NW / 2 + 16, ROW_Y[i + 1] - 8)


def table_scene(g, t, lt):
    table(g, [eo(t, ROW_T[k], 0.22) for k in range(5)], hdr_a=eo(lt, 0.0, 0.25))


CLOSE_T = 17.90


def close_scene(g, t, lt):
    u = eo(t, CLOSE_T, 0.24)
    table(g, [1, 1, 1, 1, 1, u], row_col={0: TEAL if u > 0.5 else WHITE, 5: TEAL})
    xb = XC - 70
    br = [(XC - 44, ROW_Y[0]), (xb - 20, ROW_Y[0]), (xb - 20, ROW_Y[5]), (XC - 44, ROW_Y[5])]
    g.path(br[:3], ramp(t, CLOSE_T + 0.04, 0.30), TEAL, 235, 5)
    if t >= CLOSE_T + 0.30:
        g.arrow(br[2], br[3], eo(t, CLOSE_T + 0.30, 0.14), TEAL, 235, 5, 18)


D1_T = [20.80 + 0.36 * i for i in range(5)]     # «диагонали» 20.72 → «складываем» 22.18


def products(g, side, a=255, upto=5, t=None, times=None):
    xs, vals, col = (XR, R_PROD, TEAL) if side == "r" else (XL, L_PROD, RED)
    for i in range(upto):
        u = 1.0 if times is None else eo(t, times[i] + 0.20, 0.18)
        if u > 0:
            num(g, xs, MID_Y[i], str(vals[i]), PROD_SZ, col, a * u, 10 * (1 - u))


def diag1(g, t, lt):
    table(g, [1] * 6)
    for i in range(5):
        u, v = r_arrow(i)
        g.arrow(u, v, eo(t, D1_T[i], 0.24), TEAL, 245, 6, 20)
    products(g, "r", t=t, times=D1_T)


SUM_T = 24.60                    # «70»
SUM_TERMS = ["1", "+", "20", "+", "36", "+", "12", "+", "1"]
SUM_SZ = fit(["1 + 20 + 36 + 12 + 1"], 64, GX1 - GX0 - 40)
BIG_SZ = fit(["= 70"], 230, GX1 - GX0 - 60)


def _row_layout(tokens, size, gap):
    f = wide(size)
    ws = [f.getlength(s) for s in tokens]
    x = 540 - (sum(ws) + gap * (len(tokens) - 1)) / 2
    out = []
    for s, w in zip(tokens, ws):
        out.append((x + w / 2, s))
        x += w + gap
    return out


def sum70(g, t, lt):
    for k, (x, s) in enumerate(_row_layout(SUM_TERMS, SUM_SZ, SUM_SZ * 0.22)):
        u = eo(lt, 0.02 + 0.05 * k, 0.2)
        num(g, x, 880, s, SUM_SZ, WHITE if s == "+" else TEAL, (170 if s == "+" else 255) * u, 10 * (1 - u))
    p = ramp(t, SUM_T, 0.08)                   # R5a: 0.6 → 1.0 за 0.08с, без оверщута
    if p > 0:
        sz = BIG_SZ * (0.6 + 0.4 * ease_out(p))
        num(g, 540, 1170, "= 70", sz, WHITE, 255 * min(1, p * 1.6))


D2_T = [26.70 + 0.22 * i for i in range(5)]     # «в другую сторону» 26.62 → «Получаем» 27.80
T33 = 27.96


def diag2(g, t, lt):
    table(g, [1] * 6)
    products(g, "r", a=110)
    num(g, XR, HDR_Y - 24, "70", HDR_SZ, TEAL, 255)
    for i in range(5):
        u, v = l_arrow(i)
        g.arrow(u, v, eo(t, D2_T[i], 0.20), RED, 245, 6, 20)
    products(g, "l", t=t, times=D2_T)
    u = eo(t, T33, 0.18)
    if u > 0:
        num(g, XL, HDR_Y - 24, "33", HDR_SZ, RED, 255 * u, 10 * (1 - u))


# вычисление: | 70 − 33 | над чертой, 2 под ней; место под все токены — сразу (ничего не сдвигается)
CALC_TOK = [("|", WHITE), ("70", TEAL), ("−", WHITE), ("33", RED), ("|", WHITE)]
CALC_SZ = fit(["|70 − 33|"], 130, GX1 - GX0 - 80)
CALC_T = dict(n70=29.32, minus=29.82, n33=30.24, bars=31.20, bar=31.88, two=32.26)
NUM_Y, BAR_Y, DEN_Y = 980, 1060, 1180


def _calc_layout(size):
    f = wide(size)
    gaps = [0.10, 0.22, 0.22, 0.10]
    ws = [f.getlength(s) for s, _ in CALC_TOK]
    tot = sum(ws) + sum(gaps) * size
    x = 540 - tot / 2
    out = []
    for k, (w, (s, col)) in enumerate(zip(ws, CALC_TOK)):
        out.append((x + w / 2, s, col))
        x += w + (gaps[k] * size if k < len(gaps) else 0)
    return out, tot


CALC_L, CALC_W = _calc_layout(CALC_SZ)


def calc(g, t, lt):
    when = [CALC_T["bars"], CALC_T["n70"], CALC_T["minus"], CALC_T["n33"], CALC_T["bars"]]
    for (x, s, col), t0 in zip(CALC_L, when):
        u = eo(t, t0, 0.22)
        if u > 0:
            num(g, x, NUM_Y, s, CALC_SZ, col, 255 * u, 14 * (1 - u))
    b = eo(t, CALC_T["bar"], 0.28)
    if b > 0:
        g.line([(540 - CALC_W / 2 * b, BAR_Y), (540 + CALC_W / 2 * b, BAR_Y)], WHITE, 245, 8)
    u = eo(t, CALC_T["two"], 0.22)
    if u > 0:
        num(g, 540, DEN_Y, "2", CALC_SZ, WHITE, 255 * u, 14 * (1 - u))


RES_T = 33.82                    # «18,5» — R5b
RES_TOP = ["S", "=", "|70 − 33|", ":", "2"]
RES_TOP_C = [WHITE, WHITE, WHITE, WHITE, WHITE]
RES_TOP_SZ = fit(["S = |70 − 33| : 2"], 72, GX1 - GX0 - 40)
RES_SZ = fit(["18,5"], 230, GX1 - GX0 - 60)


def result(g, t, lt):
    for k, (x, s) in enumerate(_row_layout(RES_TOP, RES_TOP_SZ, RES_TOP_SZ * 0.24)):
        u = eo(lt, 0.03 * k, 0.2)
        num(g, x, 860, s, RES_TOP_SZ, WHITE, 255 * u, 10 * (1 - u))
    num(g, 540, 1000, "=", 90, WHITE, 200 * eo(lt, 0.2, 0.2))
    lb = t - RES_T
    if lb >= 0:                  # 0.38с, 0.55 → 1.0, оверщут ~3%
        p = min(1.0, lb / 0.38)
        sc = 0.55 + 0.45 * ease_out(p)
        if 0.55 < p < 1.0:
            sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
        num(g, 540, 1190, "18,5", RES_SZ * sc, BLUE, 255 * min(1.0, lb / 0.12))


LACE_T = [44.30 + 0.22 * i for i in range(5)]   # «диагональные» 44.30 → «координатами» 45.48


def laces(g, t, lt):
    for k in range(6):
        for x in (XC, YC):
            g.ring((x, ROW_Y[k]), 36, 3, WHITE, 70)
    table(g, [1] * 6, num_a=150)
    for i in range(5):
        u, v = r_arrow(i)
        g.path([u, v], eo(t, LACE_T[i], 0.20), TEAL, 250, 9)
        u, v = l_arrow(i)
        g.path([u, v], eo(t, LACE_T[i] + 0.09, 0.20), RED, 250, 9)


# ---------------------------------------------------------------- задача ОГЭ (узлы сетки 168)
STEP = 168
NX0 = CARD_A[0] + STEP - STEP * 0.5 + (7 * 13 % 40) / 4.0       # 191.75
NY0 = CARD_A[1] + 7 * STEP - STEP * 0.5 + (7 * 7 % 40) / 4.0    # 1332.25
EXAM = [(0, 0), (4, 1), (3, 3), (1, 2)]
assert area(EXAM) == 6 and _simple(EXAM)


def npos(q):
    return (NX0 + STEP * q[0], NY0 - STEP * q[1])


EX_Q_T = 48.16                  # «площадь»
SOLVE_T, CHECK_T = 52.10, 53.02  # синее «6»; ✓ на «ошибок»
EX_SZ = 84


def _exam_fig(g, lt, draw=True):
    pts = [npos(q) for q in EXAM]
    for k, q in enumerate(pts):
        g.dot(q, 11 * (eo(lt, 0.02 + 0.05 * k, 0.16) if draw else 1), WHITE, 240)
    g.fill(pts, WHITE, 46 * (eo(lt, 0.62, 0.35) if draw else 1))
    g.path(pts, (lt - 0.08) / 0.55 if draw else 1, WHITE, 245, 7, closed=True)
    return P_c(pts)


def P_c(pts):
    c = centroid(EXAM)
    return npos(c)


def exam(g, t, lt):
    c = _exam_fig(g, lt)
    u = eo(t, EX_Q_T, 0.25)
    if u > 0:
        num(g, c[0], c[1] + 10 * (1 - u), "S = ?", EX_SZ, WHITE, 255 * u)


def solved(g, t, lt):
    c = _exam_fig(g, lt, draw=False)
    f = wide(EX_SZ)
    left = "S = "
    lb = t - SOLVE_T
    p = min(1.0, max(0.0, lb / 0.38))
    sc = 0.55 + 0.45 * ease_out(p)
    if 0.55 < p < 1.0:
        sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    w6 = f.getlength("6")
    tot = f.getlength(left) + w6
    x0 = c[0] - tot / 2 - 30
    g.text((x0, c[1] + _cap(EX_SZ) / 2), left, EX_SZ, WHITE, 255, "ls")
    if lb >= 0:
        num(g, x0 + f.getlength(left) + w6 / 2, c[1], "6", EX_SZ * sc, BLUE, 255 * min(1.0, lb / 0.12))
    ck = ramp(t, CHECK_T, 0.26)
    if ck > 0:
        bx = x0 + tot + 34
        tick = [(bx, c[1]), (bx + 22, c[1] + 24), (bx + 62, c[1] - 30)]
        g.path(tick, ease_out(ck), TEAL, 255, 10)


# ---------------------------------------------------------------- диспетчер
def _plot(fn):
    return lambda g, t, lt: fn(g, lt)


SCENES = dict(hook=_plot(hook), axes=_plot(axes_scene), points=points, table=table_scene,
              close=close_scene, diag1=diag1, sum70=sum70, diag2=diag2, calc=calc, result=result,
              anyshape=_plot(anyshape), order=order, laces=laces, exam=exam, solved=solved)

# sfx156: impact — синие «18,5» и «6»; tick — каскады/стрелки/новые члены
IMPACT_T = [RES_T, SOLVE_T]
TICK_T = [1.64, 4.94, 7.26] + PT_T + [14.02, CLOSE_T, D1_T[0], SUM_T, D2_T[0], T33,
                                    CALC_T["n70"], CALC_T["bars"], CALC_T["two"], 37.24, 37.80,
                                    ORD_T[0], LACE_T[0], 47.66, EX_Q_T, CHECK_T]


def gfx(t, shot):
    fn = SCENES.get(shot[2])
    if fn is None:
        return None, []
    g = G()
    fn(g, t, t - shot[0])
    return g.layer(), []

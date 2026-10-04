"""Графика ролика 57 v2 (слот 157): парадокс двух конвертов.
Предметная графика на вертикальной сетке (R4 + R10) + числа (R5a белые, R5b синие #5B7CFF). Синих чисел 2:
«1 250 ₽» и «+25%» (на обеих дугах петли). Красный — только «?» у «50%» (слово-акцент разгадки). Лайма нет.

Конверт — светлая карточка со скруглением (R10: «белые/светлые карточки»), клапан — тёмная галочка; открытый —
клапан вверх, из кармана торчит купюра. Объекты появляются каскадом (0.22с ease-out, 0.85→1.0, стаггер 80мс).

Сцены (storyboard157, времена — слова words.json после fix157):
  env   — два закрытых конверта на резе (6.60), «2X» под левым на «два» (7.15), «X» под правым на «другом» (8.32).
  g1000 — открытый конверт с купюрой на резе (11.44), «1 000 ₽» (R5a) на «тысячу» (12.08). Субтитр — «например».
  g500  — конверт «?» на резе (15.60), «500 ₽» на «500» (15.66), «2 000 ₽» на «две тысячи» (16.70). Субтитра нет.
  g1250 — «½ · 500 + ½ · 2 000» на резе (20.40, две половины каскадом), синее «1 250 ₽» (R5b) на «1250» (20.50),
          «> 1 000 ₽» на «получается» (22.70) — менять «выгодно».
  loop  — два конверта на резе (29.96), верхняя дуга →  на «конверт» (30.24), «+25%» на «выгодно» (30.90),
          нижняя дуга ← на «всегда» (31.25), «+25%» (31.45); на «даже» (32.12) точка пошла по петле (период 2.4с,
          спокойно — урок 17), конверт, в который она пришла, подсвечивается.
  g5050 — дерево: конверт «?» на резе (36.47), ветки на «две» (36.78), «500 ₽» / «2 000 ₽» на «суммы» (37.48),
          «50%» на «равновероятными» (38.82), красный «?» у обоих (39.45).
Арифметика — assert'ами: ½·500 + ½·2000 = 1250 > 1000; 1.25·X > X для любого X (петля).
Зона графики x 175…905, y 700…1340 (субтитры v3 — база 1500): assert по всем кадрам каждого плана (урок 38).
gfx(t, shot) -> (слой RGBA на весь холст, []).
"""
import math
import numpy as np
from style import WHITE, ease_out
from root125 import wide
from poly127 import Shapes, _ramp

BLUE = (91, 124, 255)            # цифры графики — #5B7CFF (brand-kit, «Субтитры v3»)
RED = (255, 59, 48)
PAPER = (246, 244, 240)          # светлая карточка конверта
PAPER2 = (214, 210, 204)         # клапан открытого конверта
INK = (52, 50, 48)               # линии на бумаге
GX0, GX1, GY0, GY1 = 175, 905, 700, 1340
CXM = 540
STAG = 0.08

# ---------------------------------------------------------------- арифметика ролика
assert 0.5 * 500 + 0.5 * 2000 == 1250 and 1250 > 1000
assert all(0.5 * (x / 2) + 0.5 * (2 * x) == 1.25 * x > x for x in (100, 1000, 7777))   # «при любой сумме»

T = dict(
    e_l=6.60, e_r=6.68, e_2x=7.15, e_x=8.32,
    k_env=11.44, k_num=12.08,
    f_env=15.60, f_500=15.66, f_2000=16.70,
    q_f1=20.40, q_f2=20.48, q_num=20.50, q_gt=22.70,
    l_l=29.96, l_r=30.04, l_top=30.24, l_p1=30.90, l_bot=31.25, l_p2=31.45, l_dot=32.12,
    t_env=36.47, t_br=36.78, t_500=37.48, t_2000=37.56, t_p1=38.82, t_p2=38.90, t_q1=39.45, t_q2=39.53,
)
PERIOD = 2.4


# ---------------------------------------------------------------- примитивы
def _pop(t, t0, dur=0.22):
    """Каскад R10: 0.85→1.0 и прозрачность за dur, ease-out. None — ещё не появился."""
    lt = t - t0
    if lt < 0:
        return None, 0
    e = ease_out(min(1.0, lt / dur))
    return 0.85 + 0.15 * e, int(255 * e)


def _text(sh, cx, y, s, sz, sc=1.0, A=255, col=WHITE, anchor="ms"):
    if A <= 0:
        return
    f = wide(max(8, int(sz * 2 * sc)))
    x, yy = sh._p((cx, y))
    sh.d.text((x, yy), s, font=f, fill=col + (A,), anchor=anchor)


def _r5a(sh, cx, y, s, sz, t, t0, col=WHITE, anchor="ms"):
    """R5a: 80мс, 0.6→1.0 от базовой линии, без оверщута."""
    lt = t - t0
    if lt < 0:
        return
    sc = 0.6 + 0.4 * ease_out(min(1.0, lt / 0.08))
    _text(sh, cx, y, s, sz, sc, int(255 * min(1.0, lt / 0.06)), col, anchor)


def _r5b(sh, cx, y, s, sz, t, t0, anchor="ms"):
    """R5b: 0.38с, 0.55→1.0, микро-оверщут ~3% и возврат."""
    lt = t - t0
    if lt < 0:
        return
    p = min(1.0, lt / 0.38)
    sc = 0.55 + 0.45 * ease_out(p)
    if 0.55 < p < 1.0:
        sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    _text(sh, cx, y, s, sz, sc, int(255 * min(1.0, lt / 0.12)), BLUE, anchor)


def _poly(sh, pts, fill=None, outline=None, width=0, A=255):
    sh.d.polygon([sh._p(q) for q in pts], fill=(fill + (A,)) if fill else None)
    if outline and width:
        q = [sh._p(p) for p in pts] + [sh._p(pts[0])]
        sh.d.line(q, fill=outline + (A,), width=int(width * 2), joint="curve")


def _rrect(sh, x0, y0, x1, y1, r, fill, A=255, outline=None, width=0):
    a, b = sh._p((x0, y0)), sh._p((x1, y1))
    sh.d.rounded_rectangle([a[0], a[1], b[0], b[1]], radius=int(r * 2), fill=fill + (A,),
                           outline=(outline + (A,)) if outline else None, width=int(width * 2))


def envelope(sh, c, w, h, sc=1.0, A=255, mark=None, glow=0.0):
    """Закрытый конверт: светлая карточка, клапан — тёмная галочка до 58% высоты; mark — тёмный знак на кармане.
    glow (0..1) — подсветка: белое кольцо вокруг (точка петли пришла в этот конверт)."""
    if A <= 0:
        return
    w, h = w * sc, h * sc
    x0, y0, x1, y1 = c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2
    if glow > 0:
        g = 10 + 6 * glow
        _rrect(sh, x0 - g, y0 - g, x1 + g, y1 + g, 22 + g, WHITE, int(A * 0.30 * glow))
    _rrect(sh, x0, y0, x1, y1, 16 * sc, PAPER, A)
    m = 10 * sc
    v = [(x0 + m, y0 + m), (c[0], y0 + 0.58 * h), (x1 - m, y0 + m)]
    sh.d.line([sh._p(q) for q in v], fill=INK + (A,), width=int(7 * sc * 2), joint="curve")
    if mark:
        _text(sh, c[0], y1 - 0.10 * h, mark, 0.30 * h / sc, sc, A, INK)


def envelope_open(sh, c, w, h, sc=1.0, A=255):
    """Открытый конверт: клапан вверх (за купюрой), купюра торчит из кармана, карман спереди."""
    if A <= 0:
        return
    w, h = w * sc, h * sc
    x0, y0, x1, y1 = c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2
    _poly(sh, [(x0 + 4, y0 + 2), (c[0], y0 - 0.48 * h), (x1 - 4, y0 + 2)], PAPER2, INK, 4, A)
    bx0, bx1, by0 = c[0] - 0.36 * w, c[0] + 0.36 * w, y0 - 0.34 * h
    _rrect(sh, bx0, by0, bx1, y0 + 0.5 * h, 8 * sc, (236, 236, 230), A, (120, 118, 112), 3)
    _text(sh, c[0], by0 + 0.25 * h, "₽", 0.22 * h / sc, sc, A, (120, 118, 112), "mm")
    _rrect(sh, x0, y0, x1, y1, 16 * sc, PAPER, A)
    sh.d.line([sh._p(q) for q in [(x0 + 10, y1 - 8), (c[0], y0 + 0.40 * h), (x1 - 10, y1 - 8)]],
              fill=INK + (A,), width=int(6 * sc * 2), joint="curve")


def _bez(a, ctrl, b, n=48):
    return [((1 - u) ** 2 * a[0] + 2 * (1 - u) * u * ctrl[0] + u * u * b[0],
             (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * ctrl[1] + u * u * b[1]) for u in (k / n for k in range(n + 1))]


def _arrow(sh, pts, p, head=26, width=7):
    """Дуга рисуется на долю p; наконечник — когда дошла."""
    if p <= 0:
        return
    sh.path(pts, p, closed=False, width=width)
    if p >= 1.0:
        b, a = pts[-1], pts[-4]
        L = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        for s in (1, -1):
            q = (b[0] - head * ux + s * head * 0.6 * -uy, b[1] - head * uy + s * head * 0.6 * ux)
            sh.path([q, b], 1.0, closed=False, width=width)


def _at(pts, u):
    """Точка на ломаной на доле длины u."""
    lens = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    left = u * sum(lens)
    for (a, b), L in zip(zip(pts, pts[1:]), lens):
        if left <= L:
            k = left / L
            return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
        left -= L
    return pts[-1]


# ---------------------------------------------------------------- env: два закрытых конверта, 2X / X
E_W, E_H = 290, 190
E_L, E_R = (330, 945), (750, 945)
E_LAB_Y, E_LAB_SZ = 1215, 120


def scene_env(sh, t, t0):
    for c, tt in ((E_L, T["e_l"]), (E_R, T["e_r"])):
        sc, A = _pop(t, tt)
        if sc is not None:
            envelope(sh, c, E_W, E_H, sc, A)
    _r5a(sh, E_L[0], E_LAB_Y, "2X", E_LAB_SZ, t, T["e_2x"])
    _r5a(sh, E_R[0], E_LAB_Y, "X", E_LAB_SZ, t, T["e_x"])


# ---------------------------------------------------------------- g1000: открытый конверт, «1 000 ₽»
K_C, K_W, K_H = (540, 905), 300, 190
K_SZ = next(sz for sz in range(170, 60, -2) if wide(sz).getlength("1 000 ₽") <= GX1 - GX0 - 20)
K_Y = 1225


def scene_g1000(sh, t, t0):
    sc, A = _pop(t, T["k_env"])
    if sc is not None:
        envelope_open(sh, K_C, K_W, K_H, sc, A)
    _r5a(sh, CXM, K_Y, "1 000 ₽", K_SZ, t, T["k_num"])


# ---------------------------------------------------------------- g500: конверт «?», 500 ₽ / 2 000 ₽
F_C, F_W, F_H = (540, 815), 250, 165
F_SZ = next(sz for sz in range(150, 60, -2) if wide(sz).getlength("2 000 ₽") <= GX1 - GX0 - 40)
F_Y1, F_Y2 = 1075, 1270


def scene_g500(sh, t, t0):
    sc, A = _pop(t, T["f_env"])
    if sc is not None:
        envelope(sh, F_C, F_W, F_H, sc, A, mark="?")
    _r5a(sh, CXM, F_Y1, "500 ₽", F_SZ, t, T["f_500"])
    _r5a(sh, CXM, F_Y2, "2 000 ₽", F_SZ, t, T["f_2000"])


# ---------------------------------------------------------------- g1250: ½·500 + ½·2000 = 1 250 ₽ > 1 000 ₽
Q_F1, Q_F2 = "½ · 500 ", "+ ½ · 2 000"
Q_FSZ = next(sz for sz in range(80, 30, -2) if wide(sz).getlength(Q_F1 + Q_F2) <= GX1 - GX0 - 20)
# строки — между линиями сетки (y ≈ 824 / 992 / 1160 / 1328): на 845 линия шла через середину формулы,
# на кадре читалось как зачёркнутое (контакт-лист v2)
Q_FY = 905
Q_NSZ = next(sz for sz in range(170, 60, -2) if wide(sz).getlength("1 250 ₽") <= GX1 - GX0 - 30)
Q_NY = 1112
Q_GSZ, Q_GY = 84, 1290


def scene_g1250(sh, t, t0):
    f = wide(Q_FSZ)
    x = CXM - f.getlength(Q_F1 + Q_F2) / 2
    for s, tt in ((Q_F1, T["q_f1"]), (Q_F2, T["q_f2"])):
        sc, A = _pop(t, tt, 0.15)
        if sc is not None:
            _text(sh, x + f.getlength(s) / 2, Q_FY, s, Q_FSZ, sc, A)
        x += f.getlength(s)
    _r5b(sh, CXM, Q_NY, "1 250 ₽", Q_NSZ, t, T["q_num"])
    _r5a(sh, CXM, Q_GY, "> 1 000 ₽", Q_GSZ, t, T["q_gt"])


# ---------------------------------------------------------------- loop: петля обмена
L_W, L_H = 230, 152
L_L, L_R = (320, 1020), (760, 1020)
L_TOP = _bez((L_L[0] + 40, L_L[1] - L_H / 2 - 16), (CXM, 760), (L_R[0] - 40, L_R[1] - L_H / 2 - 16))
L_BOT = _bez((L_R[0] - 40, L_R[1] + L_H / 2 + 16), (CXM, 1290), (L_L[0] + 40, L_L[1] + L_H / 2 + 16))
L_PSZ = 64
L_PY1 = min(y for _, y in L_TOP) - 26
L_PY2 = max(y for _, y in L_BOT) + 26 + 48


def _dot_phase(t):
    """Доля круга: 0–0.4 верхняя дуга (Л→П), 0.4–0.5 в правом конверте, 0.5–0.9 нижняя (П→Л), 0.9–1 в левом."""
    return ((t - T["l_dot"]) / PERIOD) % 1.0


def scene_loop(sh, t, t0):
    gl = gr = 0.0
    if t >= T["l_dot"]:
        ph = _dot_phase(t)
        gr = max(0.0, 1 - abs(ph - 0.45) / 0.10)
        gl = max(0.0, 1 - min(abs(ph - 0.95), abs(ph + 0.05)) / 0.10)
    _arrow(sh, L_TOP, ease_out(_ramp(t, T["l_top"], 0.35)))
    _arrow(sh, L_BOT, ease_out(_ramp(t, T["l_bot"], 0.35)))
    for c, tt, g in ((L_L, T["l_l"], gl), (L_R, T["l_r"], gr)):
        sc, A = _pop(t, tt)
        if sc is not None:
            envelope(sh, c, L_W, L_H, sc, A, glow=g)
    _r5b(sh, CXM, L_PY1, "+25%", L_PSZ, t, T["l_p1"])
    _r5b(sh, CXM, L_PY2, "+25%", L_PSZ, t, T["l_p2"])
    if t >= T["l_dot"]:
        ph = _dot_phase(t)
        A = int(255 * min(1.0, (t - T["l_dot"]) / 0.1))
        if ph < 0.4:
            sh.dot(_at(L_TOP, ph / 0.4), 16, alpha=A)
        elif 0.5 <= ph < 0.9:
            sh.dot(_at(L_BOT, (ph - 0.5) / 0.4), 16, alpha=A)


# ---------------------------------------------------------------- g5050: дерево 50% / 50% с красным «?»
D_C, D_W, D_H = (540, 800), 220, 145
D_TOP = (CXM, D_C[1] + D_H / 2 + 14)
D_LX, D_RX = 320, 735
D_END_Y = 1050
D_NSZ = 62
D_NY = 1132
D_PSZ = 64
D_PY = 1232


def scene_g5050(sh, t, t0):
    sc, A = _pop(t, T["t_env"])
    if sc is not None:
        envelope(sh, D_C, D_W, D_H, sc, A, mark="?")
    for x, tt in ((D_LX, T["t_br"]), (D_RX, T["t_br"] + STAG)):
        sh.path([D_TOP, (x, D_END_Y)], ease_out(_ramp(t, tt, 0.25)), closed=False, width=7)
    _r5a(sh, D_LX, D_NY, "500 ₽", D_NSZ, t, T["t_500"])
    _r5a(sh, D_RX, D_NY, "2 000 ₽", D_NSZ, t, T["t_2000"])
    f = wide(D_PSZ)
    for x, tp, tq in ((D_LX, T["t_p1"], T["t_q1"]), (D_RX, T["t_p2"], T["t_q2"])):
        x0 = x - f.getlength("50%?") / 2           # раскладка с «?» заранее — «50%» не сдвигается
        _r5a(sh, x0 + f.getlength("50%") / 2, D_PY, "50%", D_PSZ, t, tp)
        _r5a(sh, x0 + f.getlength("50%") + f.getlength("?") / 2, D_PY, "?", D_PSZ, t, tq, RED)


SCENES = dict(env=scene_env, g1000=scene_g1000, g500=scene_g500, g1250=scene_g1250, loop=scene_loop,
              g5050=scene_g5050)


def gfx(t, shot):
    t0, t1, kind, prm = shot
    sh = Shapes()
    SCENES[kind](sh, t, t0)
    return sh.layer(), []


# ---------------------------------------------------------------- проверки
def _static_checks():
    av = GX1 - GX0
    for s, sz in (("1 000 ₽", K_SZ), ("2 000 ₽", F_SZ), (Q_F1 + Q_F2, Q_FSZ), ("1 250 ₽", Q_NSZ), ("> 1 000 ₽", Q_GSZ)):
        assert wide(sz).getlength(s) <= av, (s, sz)
    assert K_SZ >= 120 and F_SZ >= 110 and Q_NSZ >= 120, (K_SZ, F_SZ, Q_NSZ)   # R5 — крупно, не 60px как в v1
    # дерево: числа и проценты не выходят из зоны и не слипаются
    f, fp = wide(D_NSZ), wide(D_PSZ)
    assert D_LX + f.getlength("500 ₽") / 2 < D_RX - f.getlength("2 000 ₽") / 2 - 60
    assert D_RX + f.getlength("2 000 ₽") / 2 <= GX1 and D_RX + fp.getlength("50%?") / 2 <= GX1
    # петля: «+25%» над верхней дугой и под нижней — в зоне
    assert L_PY1 - 50 >= GY0 and L_PY2 <= GY1, (L_PY1, L_PY2)


_static_checks()


def check_zone(shots, fps=30):
    """Урок ролика 38: альфа слоя графики вне зоны x 175…905, y 700…1340 — по всем кадрам каждого плана."""
    bad = 0
    for t0, t1, kind, prm in shots:
        for fr in range(int(round(t0 * fps)), int(round(t1 * fps))):
            a = np.asarray(gfx(fr / fps, (t0, t1, kind, prm))[0].split()[3])
            ys, xs = np.nonzero(a > 40)
            if len(xs) and (xs.min() < GX0 - 6 or xs.max() > GX1 + 6 or ys.min() < GY0 - 6 or ys.max() > GY1 + 6):
                bad += 1
                print("вне зоны:", kind, round(fr / fps, 2), xs.min(), xs.max(), ys.min(), ys.max())
    return bad


# sfx157: impact — синие числа (R5b), soft tick — появление объектов и белых чисел, дуги, старт точки, красный «?»
IMPACT_T = [T["q_num"], T["l_p1"], T["l_p2"]]
TICK_T = [T["e_l"], T["e_2x"], T["e_x"], T["k_env"], T["k_num"], T["f_env"], T["f_500"], T["f_2000"],
          T["q_f1"], T["q_gt"], T["l_l"], T["l_top"], T["l_bot"], T["l_dot"], T["t_env"], T["t_br"],
          T["t_500"], T["t_p1"], T["t_q1"]]

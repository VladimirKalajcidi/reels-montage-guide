"""Графика ролика 66 (слот 166): теорема о бесконечных обезьянах.
Предметная графика на вертикальной сетке (R4 + R10) + числа (R5a белые, R5b синие #5B7CFF). Синих чисел 2:
«29 000 000 000 000 000 000 000 000» (одно число в 4 строки) и «100%». Красного/лайма в графике нет.

Сцены (storyboard166, времена — слова words.json):
  inf   — знак бесконечности (лемниската Бернулли) рисуется с реза (1.95) за 0.55с — дорисован к «бесконечность» (2.45);
          затем точка спокойно бежит по нему (период 3с, урок 17). Субтитр «КАК РАБОТАЕТ» — слово «бесконечность» несёт знак.
  tobe  — «TO BE OR / NOT TO BE» печатается по словам речи: курсор-блок на резе (12.30, сетка не пустая), «TO BE» на «быть»
          (13.11), «OR» на «или» (13.29), «NOT» на «не» (13.41), «TO BE» на «быть» (13.57); курсор идёт за словом.
          Субтитры «ДАЖЕ ФРАЗА» и «НА АНГЛИЙСКОМ» — английский текст фразы несёт графика.
  odds  — «29 000 000 / 000 000 / 000 000 / 000 000» синим (R5b): первая строка на «29» (17.78), остальные каскадом на
          «септиллионам» (18.68, +80мс). 29 септиллионов = 29·10²⁴ — 24 нуля, assert. Субтитр «ПРИМЕРНО 1 К» — до числа.
  k130  — «130 000» белым (R5a) на «130» (20.57), «СИМВОЛОВ» на «символов» (21.51). Субтитра нет (речь = графика).
  p100  — «ВЕРОЯТНОСТЬ» на резе (30.62, слова нет в речи — добавляет смысл), «100%» синим (R5b) на «обязательно» (31.16).
          Субтитр «СЛУЧАЕТСЯ».
Строки графики — в ячейках сетки, не на линиях (урок 57: линия через строку читается как зачёркивание).
Зона графики x 175…905, y 700…1340 (субтитры v3 — база 1500): assert по всем кадрам каждого плана (урок 38).
gfx(t, shot) -> (слой RGBA на весь холст, []).
"""
import math
import numpy as np
from style import WHITE, ease_out
from root125 import wide
from poly127 import Shapes, _ramp
from conv157 import _pop, _text, _r5a, _r5b, BLUE

GX0, GX1, GY0, GY1 = 175, 905, 700, 1340
CXM = 540
STAG = 0.08
AV = GX1 - GX0

T = dict(
    i_draw=1.95, i_dot=2.50,
    b_cur=12.30, b_w1=13.11, b_w2=13.29, b_w3=13.41, b_w4=13.57,
    o_r1=17.78, o_r2=18.68,
    k_num=20.57, k_lab=21.51,
    p_lab=30.62, p_num=31.16,
)
I_DUR, PERIOD = 0.55, 3.0

# ---------------------------------------------------------------- арифметика ролика
ODDS = ["29 000 000", "000 000", "000 000", "000 000"]
assert int("".join(ODDS).replace(" ", "")) == 29 * 10 ** 24          # 29 септиллионов (короткая шкала)
assert 2.9e25 < 26 ** 18 < 3.0e25                                   # «1 к 29 септиллионам» = 26^18 (18 знаков фразы)


# ---------------------------------------------------------------- inf: лемниската
I_C, I_A = (CXM, 1010), 320          # полуширина лемнискаты = a → ширина знака 640


def _lem(u):
    """Лемниската Бернулли x = a·cos/(1+sin²), y = a·sin·cos/(1+sin²); u ∈ [0, 1) — полный обход от правого конца."""
    th = 2 * math.pi * u
    s, c = math.sin(th), math.cos(th)
    d = 1 + s * s
    return (I_C[0] + I_A * c / d, I_C[1] - I_A * s * c / d * 1.25)


I_PTS = [_lem(k / 240) for k in range(241)]


def scene_inf(sh, t, t0):
    p = ease_out(_ramp(t, T["i_draw"], I_DUR))
    if p > 0:
        sh.path(I_PTS, p, closed=False, width=11)
    if t >= T["i_dot"]:
        u = ((t - T["i_dot"]) / PERIOD) % 1.0
        sh.dot(_lem(u), 19, alpha=int(255 * min(1.0, (t - T["i_dot"]) / 0.1)))


# ---------------------------------------------------------------- tobe: печать фразы
B_L1, B_L2 = ["TO BE", "OR"], ["NOT", "TO BE"]
B_SZ = next(sz for sz in range(130, 60, -2)   # строка + пробел + курсор-блок 0.42 кегля
            if wide(sz).getlength("NOT TO BE ") + sz * 0.42 <= AV - 20)
B_Y1, B_Y2 = 950, 1120              # ячейки 820…990 и 990…1160
_fb = wide(B_SZ)
_sp = _fb.getlength(" ")
CUR_W, CUR_H = int(B_SZ * 0.42), int(B_SZ * 0.72)
CUR_SLOT = _sp * 0.5 + CUR_W          # место под курсор справа — строка центрируется вместе с ним


def _line_x(words):
    w = _fb.getlength(" ".join(words))
    x, out = CXM - (w + CUR_SLOT) / 2, []
    for s in words:
        out.append(x + _fb.getlength(s) / 2)
        x += _fb.getlength(s) + _sp
    return out, CXM - (w + CUR_SLOT) / 2 + w


B_X1, B_E1 = _line_x(B_L1)
B_X2, B_E2 = _line_x(B_L2)
B_WORDS = [(B_L1[0], B_X1[0], B_Y1, T["b_w1"]), (B_L1[1], B_X1[1], B_Y1, T["b_w2"]),
           (B_L2[0], B_X2[0], B_Y2, T["b_w3"]), (B_L2[1], B_X2[1], B_Y2, T["b_w4"])]


def scene_tobe(sh, t, t0):
    for s, cx, y, tt in B_WORDS:
        sc, A = _pop(t, tt, 0.15)
        if sc is not None:
            _text(sh, cx, y, s, B_SZ, sc, A)
    # курсор: стоит после последнего напечатанного слова (на резе — в начале первой строки)
    done = [w for w in B_WORDS if t >= w[3]]
    if not done:
        x, y = B_X1[0] - _fb.getlength(B_L1[0]) / 2, B_Y1
    else:
        s, cx, y, _ = done[-1]
        x = cx + _fb.getlength(s) / 2 + _sp * 0.5
        if s == B_L1[1]:
            x, y = B_X2[0] - _fb.getlength(B_L2[0]) / 2, B_Y2
    if not (done and done[-1] is B_WORDS[-1]):   # после последнего слова курсора нет — у «BE» он читался как «BEI»
        x0, y0 = sh._p((x, y - CUR_H))
        x1, y1 = sh._p((x + CUR_W, y))
        sh.d.rectangle([x0, y0, x1, y1], fill=WHITE + (230,))


# ---------------------------------------------------------------- odds: 29 септиллионов
O_SZ = next(sz for sz in range(110, 50, -2) if wide(sz).getlength(ODDS[0]) <= AV - 30)
O_YS = [800, 960, 1130, 1300]       # ячейки 700…820, 820…990, 990…1160, 1160…1330


def scene_odds(sh, t, t0):
    for k, (s, y) in enumerate(zip(ODDS, O_YS)):
        _r5b(sh, CXM, y, s, O_SZ, t, T["o_r1"] if k == 0 else T["o_r2"] + (k - 1) * STAG)


# ---------------------------------------------------------------- k130: 130 000 символов
K_SZ = next(sz for sz in range(190, 60, -2) if wide(sz).getlength("130 000") <= AV - 30)
K_Y, K_LSZ, K_LY = 1120, 70, 1290


def scene_k130(sh, t, t0):
    _r5a(sh, CXM, K_Y, "130 000", K_SZ, t, T["k_num"])
    _r5a(sh, CXM, K_LY, "СИМВОЛОВ", K_LSZ, t, T["k_lab"])


# ---------------------------------------------------------------- p100: вероятность 100%
P_LSZ, P_LY = 64, 940
P_SZ = next(sz for sz in range(220, 60, -2) if wide(sz).getlength("100%") <= AV - 40)
P_Y = 1230


def scene_p100(sh, t, t0):
    _r5a(sh, CXM, P_LY, "ВЕРОЯТНОСТЬ", P_LSZ, t, T["p_lab"])
    _r5b(sh, CXM, P_Y, "100%", P_SZ, t, T["p_num"])


SCENES = dict(inf=scene_inf, tobe=scene_tobe, odds=scene_odds, k130=scene_k130, p100=scene_p100)


def gfx(t, shot):
    t0, t1, kind, prm = shot
    sh = Shapes()
    SCENES[kind](sh, t, t0)
    return sh.layer(), []


# ---------------------------------------------------------------- проверки
def _static_checks():
    for s, sz in (("NOT TO BE", B_SZ), (ODDS[0], O_SZ), ("130 000", K_SZ), ("СИМВОЛОВ", K_LSZ),
                  ("ВЕРОЯТНОСТЬ", P_LSZ), ("100%", P_SZ)):
        assert wide(sz).getlength(s) <= AV, (s, sz)
    assert B_SZ >= 80 and O_SZ >= 70 and K_SZ >= 110 and P_SZ >= 170, (B_SZ, O_SZ, K_SZ, P_SZ)
    assert max(B_E1, B_E2) + CUR_SLOT <= GX1 and min(B_X1[0], B_X2[0]) - _fb.getlength("NOT") / 2 >= GX0     # курсор в конце строки — в зоне
    xs = [p[0] for p in I_PTS]; ys = [p[1] for p in I_PTS]
    assert min(xs) - 20 >= GX0 and max(xs) + 20 <= GX1 and min(ys) - 20 >= GY0 and max(ys) + 20 <= GY1


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


# sfx166: impact — синие числа (R5b), soft tick — появление элементов графики
IMPACT_T = [T["o_r1"], T["p_num"]]
TICK_T = [T["i_draw"], T["i_dot"], T["b_w1"], T["b_w2"], T["b_w3"], T["b_w4"], T["o_r2"], T["k_num"], T["k_lab"],
          T["p_lab"]]

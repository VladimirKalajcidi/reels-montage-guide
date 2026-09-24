"""Предметная математика ролика 27 — считается, а не вписывается руками.

Многоугольник взят из речи: (1;1) (5;1) (6;4) (3;6) (1;4). Всё остальное —
производные величины, и они обязаны сойтись с тем, что произносится вслух:
70, 33 и площадь 18,5.

Проверка независимая: площадь по шнуровке сверяется с формулой Пика
(i + b/2 − 1), посчитанной по узлам решётки. Две разные формулы должны
дать одно число — иначе на экран выйдет неверный результат.
"""
from math import gcd

POLY = [(1, 1), (5, 1), (6, 4), (3, 6), (1, 4)]
ROWS = POLY + [POLY[0]]                      # первая точка повторяется в конце

D1 = [ROWS[i][0] * ROWS[i + 1][1] for i in range(len(POLY))]   # вниз-вправо
D2 = [ROWS[i][1] * ROWS[i + 1][0] for i in range(len(POLY))]   # вниз-влево
S1, S2 = sum(D1), sum(D2)
DIFF = S1 - S2
AREA = DIFF / 2

# Второй многоугольник — невыпуклый, с «дыркой» в контуре: им закрывается
# фраза про то, что формула работает для любого многоугольника.
POLY2 = [(1, 1), (6, 1), (6, 3), (3, 3), (3, 4), (6, 4), (6, 6), (1, 6)]


def shoelace(p):
    s = 0
    for i in range(len(p)):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % len(p)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def boundary(p):
    """Целых точек на границе: на ребре (dx, dy) их ровно gcd(|dx|, |dy|)."""
    return sum(gcd(abs(p[(i + 1) % len(p)][0] - p[i][0]),
                   abs(p[(i + 1) % len(p)][1] - p[i][1])) for i in range(len(p)))


def _bnd_points(p):
    out = []
    for i in range(len(p)):
        x1, y1 = p[i]
        x2, y2 = p[(i + 1) % len(p)]
        g = gcd(abs(x2 - x1), abs(y2 - y1))
        for k in range(g):
            out.append((x1 + (x2 - x1) * k // g, y1 + (y2 - y1) * k // g))
    return out


def _inside(x, y, p):
    r = False
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            r = not r
        j = i
    return r


def interior(p):
    b = set(_bnd_points(p))
    xs = [q[0] for q in p]
    ys = [q[1] for q in p]
    return [(x, y)
            for x in range(min(xs), max(xs) + 1)
            for y in range(min(ys), max(ys) + 1)
            if (x, y) not in b and _inside(x, y, p)]


def pick(p):
    return len(interior(p)) + boundary(p) / 2 - 1


def _sgn(a, b, c):
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (v > 0) - (v < 0)


def _cross(a, b, c, d):
    if a in (c, d) or b in (c, d):
        return False
    return _sgn(a, b, c) * _sgn(a, b, d) < 0 and _sgn(c, d, a) * _sgn(c, d, b) < 0


def self_intersections(p):
    e = [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]
    return [(i, j) for i in range(len(e)) for j in range(i + 1, len(e))
            if _cross(*e[i], *e[j])]


def convex(p):
    s = {_sgn(p[i], p[(i + 1) % len(p)], p[(i + 2) % len(p)]) for i in range(len(p))}
    return len(s - {0}) == 1


if __name__ == "__main__":
    for name, p in (("POLY", POLY), ("POLY2", POLY2)):
        print(f"{name}: {p}")
        print(f"   шнуровка {shoelace(p)}  Пик {pick(p)}  сходятся={shoelace(p) == pick(p)}"
              f"  самопересечений={len(self_intersections(p))}  выпуклый={convex(p)}")
    print("вниз-вправо:", D1, "= ", S1)
    print("вниз-влево :", D2, "= ", S2)
    print("разность", DIFF, " площадь", AREA)

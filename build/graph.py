"""Граф-инфографика для теоремы Рамсея — рецепт R10 на сетке."""
import math
from PIL import Image, ImageDraw, ImageFilter
from style import W, H, WHITE, BLUE, BLUE_GLOW, ease_out

CX, CY = 540, 905          # центр графа на холсте
RAD6, RAD5 = 292, 288      # радиус окружности вершин


def verts(n, rad=None, rot=-90.0):
    rad = rad if rad is not None else (RAD6 if n == 6 else RAD5)
    out = []
    for i in range(n):
        a = math.radians(rot + i * 360.0 / n)
        out.append((CX + rad * math.cos(a), CY + rad * math.sin(a)))
    return out


def _glow(layer, radius, strength=0.7, color=None):
    a = layer.split()[3]
    blur = a.filter(ImageFilter.GaussianBlur(radius))
    gl = Image.new("RGBA", layer.size, (color or (255, 255, 255)) + (0,))
    gl.putalpha(blur.point(lambda v: int(v * strength)))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(gl)
    out.alpha_composite(layer)
    return out


def draw_graph(vs, edges=(), tri=None, node_state=None, ss=2):
    """edges: список (i, j, kind, prog) kind: 'blue'|'white'|'dim'; prog 0..1 — отрисовка ребра
    tri: (i, j, k, kind, prog) — заливка треугольника
    node_state: dict i -> (scale 0..1, kind) kind: 'w'|'b'|'dim'"""
    lay = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    S = lambda p: (p[0] * ss, p[1] * ss)

    # треугольник-заливка
    if tri:
        i, j, k, kind, prog = tri
        if prog > 0:
            col = BLUE if kind == "blue" else WHITE
            pts = [S(vs[i]), S(vs[j]), S(vs[k])]
            fill = Image.new("RGBA", lay.size, (0, 0, 0, 0))
            ImageDraw.Draw(fill).polygon(pts, fill=col + (int(60 * prog),))
            lay.alpha_composite(fill)

    # рёбра
    for (i, j, kind, prog) in edges:
        if prog <= 0:
            continue
        p, q = vs[i], vs[j]
        qx = p[0] + (q[0] - p[0]) * prog
        qy = p[1] + (q[1] - p[1]) * prog
        if kind == "blue":
            col, wdt = BLUE + (255,), int(6 * ss)
        elif kind == "white":
            col, wdt = WHITE + (215,), int(5 * ss)
        elif kind == "grey":
            col, wdt = WHITE + (130,), int(4 * ss)
        else:
            col, wdt = WHITE + (55,), int(3 * ss)
        if kind == "white":
            _dashed(d, S(p), (qx * ss, qy * ss), col, wdt, dash=int(17 * ss), gap=int(13 * ss))
        else:
            d.line([S(p), (qx * ss, qy * ss)], fill=col, width=wdt)

    # вершины
    ns = node_state or {}
    for i, v in enumerate(vs):
        sc, kind = ns.get(i, (1.0, "w"))
        if sc <= 0:
            continue
        r = 21 * ss * sc
        if kind == "b":
            col, ring = BLUE + (255,), 8 * ss
        elif kind == "dim":
            col, ring = WHITE + (90,), 0
        else:
            col, ring = WHITE + (255,), 0
        if ring:
            d.ellipse([v[0] * ss - r - ring, v[1] * ss - r - ring,
                       v[0] * ss + r + ring, v[1] * ss + r + ring],
                      outline=BLUE + (140,), width=int(3 * ss))
        d.ellipse([v[0] * ss - r, v[1] * ss - r, v[0] * ss + r, v[1] * ss + r], fill=col)

    lay = lay.resize((W, H), Image.LANCZOS)
    return _glow(lay, 16, 0.45)


def _dashed(d, p, q, col, wdt, dash, gap):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    if L < 1:
        return
    ux, uy = dx / L, dy / L
    t = 0.0
    while t < L:
        e = min(t + dash, L)
        d.line([(p[0] + ux * t, p[1] + uy * t), (p[0] + ux * e, p[1] + uy * e)],
               fill=col, width=wdt)
        t = e + gap


def all_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def stagger(t, n, dur_each=0.30, step=0.07):
    """Каскад: список прогрессов 0..1 для n элементов (стаггер 70мс)."""
    out = []
    for i in range(n):
        p = (t - i * step) / dur_each
        out.append(ease_out(max(0.0, min(1.0, p))))
    return out

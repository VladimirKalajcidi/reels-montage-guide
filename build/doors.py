"""Инфографика ролика «две двери и два стражника» — рецепт R10 на сетке.

Одна сцена на весь ролик: две двери сверху, два стражника под ними.
Все планы — состояния этой сцены, поэтому зритель не переучивает картинку.
Геометрия под карточку A (105..975 / 238..1618).
"""
import math
from PIL import Image, ImageDraw, ImageFilter
from style import W, H, WHITE, BLUE, BLUE_GLOW, LIME, ease_out, font

# --- геометрия сцены ---
DOOR_W, DOOR_H, DOOR_TOP = 250, 420, 520
DOOR_BOT = DOOR_TOP + DOOR_H            # 940
LX, RX = 330, 750                       # центры дверей и стражников
GY = 1120                               # центр головы стражника
GBODY = (1162, 1272)                    # тело: верх, низ
GLBL_Y = 1312                           # подпись под стражником
DLBL_Y = 1000                           # подпись под дверью (когда стражников нет)
NUM_XY = (832, 440)                     # число-ревил в правом верхнем углу
FLIP_Y = DOOR_TOP + DOOR_H // 2         # 730 — высота центральной стрелки


def _glow(layer, radius=16, strength=0.45, color=None):
    a = layer.split()[3]
    blur = a.filter(ImageFilter.GaussianBlur(radius))
    gl = Image.new("RGBA", layer.size, (color or (255, 255, 255)) + (0,))
    gl.putalpha(blur.point(lambda v: int(v * strength)))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(gl)
    out.alpha_composite(layer)
    return out


# ---------------------------------------------------------------- примитивы
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


DOOR_STYLE = {
    "dim":   (WHITE + (62,), None, 4),
    "white": (WHITE + (215,), None, 6),
    "named": (WHITE + (255,), WHITE + (26,), 7),
    "blue":  (BLUE + (255,), BLUE + (60,), 7),
    "lime":  (LIME + (255,), LIME + (55,), 6),
}


def _door_path(cx):
    """Замкнутый контур двери-арки одной ломаной — без стыков между дугой и стойками."""
    r = DOOR_W / 2
    ay = DOOR_TOP + r
    pts = [(cx + r * math.cos(math.pi + math.pi * i / 40),
            ay + r * math.sin(math.pi + math.pi * i / 40)) for i in range(41)]
    pts += [(cx + r, DOOR_BOT), (cx - r, DOOR_BOT), (cx - r, ay)]
    return pts


def _dash_path(d, pts, col, wdt, dash, gap):
    """Пунктир вдоль ломаной."""
    on, rest = True, dash
    for i in range(len(pts) - 1):
        p, q = pts[i], pts[i + 1]
        seg = math.dist(p, q)
        pos = 0.0
        while pos < seg:
            step = min(rest, seg - pos)
            if on:
                u0, u1 = pos / seg, (pos + step) / seg
                d.line([(p[0] + (q[0] - p[0]) * u0, p[1] + (q[1] - p[1]) * u0),
                        (p[0] + (q[0] - p[0]) * u1, p[1] + (q[1] - p[1]) * u1)],
                       fill=col, width=wdt)
            pos += step
            rest -= step
            if rest <= 1e-6:
                on = not on
                rest = dash if on else gap


def door(d, cx, kind, prog, ss, dashed=False):
    """Дверь-арка. prog: 0..1 — прозрачность появления."""
    if prog <= 0:
        return
    outline, fill, wd = DOOR_STYLE[kind]
    a = max(0.0, min(1.0, prog))
    outline = outline[:3] + (int(outline[3] * a),)
    if fill:
        fill = fill[:3] + (int(fill[3] * a),)
    wd = max(1, int(wd * ss))
    pts = [(x * ss, y * ss) for x, y in _door_path(cx)]
    if fill:
        d.polygon(pts, fill=fill)
    if dashed:
        _dash_path(d, pts + [pts[0]], outline, wd, 19 * ss, 14 * ss)
    else:
        d.line(pts + [pts[0]], fill=outline, width=wd, joint="curve")
    # ручка — с внутренней стороны
    kx = (cx + DOOR_W / 2 - 40) if cx < 540 else (cx - DOOR_W / 2 + 40)
    kr = 10 * ss
    d.ellipse([kx * ss - kr, 800 * ss - kr, kx * ss + kr, 800 * ss + kr], fill=outline)


GUARD_STYLE = {"dim": WHITE + (80,), "w": WHITE + (255,), "b": BLUE + (255,)}


def guard(d, cx, kind, prog, ss, ring=0.0):
    """Фигурка стражника: голова + плечи. prog — проявление (масштаб + альфа)."""
    if prog <= 0:
        return
    a = max(0.0, min(1.0, prog))
    col = GUARD_STYLE[kind]
    col = col[:3] + (int(col[3] * a),)
    sc = 0.62 + 0.38 * a
    hr = 31 * sc * ss
    d.ellipse([cx * ss - hr, GY * ss - hr, cx * ss + hr, GY * ss + hr], fill=col)
    t, b = GBODY
    hw_t, hw_b = 22 * sc * ss, 64 * sc * ss
    d.polygon([(cx * ss - hw_t, t * ss), (cx * ss + hw_t, t * ss),
               (cx * ss + hw_b, b * ss), (cx * ss - hw_b, b * ss)], fill=col)
    if ring > 0:
        rr = (86 * sc + 16 * ring) * ss
        rc = (BLUE if kind == "b" else WHITE) + (int(150 * ring),)
        d.ellipse([cx * ss - rr, (GY + 78) * ss - rr, cx * ss + rr, (GY + 78) * ss + rr],
                  outline=rc, width=int(3 * ss))


def arrow(d, p, q, col, wdt, ss, prog=1.0, head=24, dashed=False, curve=None):
    """Стрелка p→q. curve — точка-контроль для квадратичной кривой."""
    if prog <= 0:
        return
    if curve is None:
        pts = [p, q]
    else:
        pts = []
        for i in range(41):
            u = i / 40
            x = (1 - u) ** 2 * p[0] + 2 * (1 - u) * u * curve[0] + u ** 2 * q[0]
            y = (1 - u) ** 2 * p[1] + 2 * (1 - u) * u * curve[1] + u ** 2 * q[1]
            pts.append((x, y))
    # обрезаем по прогрессу
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(seg)
    want = total * max(0.0, min(1.0, prog))
    acc, out = 0.0, [pts[0]]
    for i, s in enumerate(seg):
        if acc + s >= want:
            u = (want - acc) / max(s, 1e-6)
            out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * u,
                        pts[i][1] + (pts[i + 1][1] - pts[i][1]) * u))
            break
        acc += s
        out.append(pts[i + 1])
    S = [(x * ss, y * ss) for x, y in out]
    wd = max(1, int(wdt * ss))
    if dashed:
        for i in range(len(S) - 1):
            if i % 2 == 0:
                d.line([S[i], S[i + 1]], fill=col, width=wd)
        if len(S) > 1 and len(S) % 2 == 0:
            d.line([S[-2], S[-1]], fill=col, width=wd)
    else:
        d.line(S, fill=col, width=wd, joint="curve")
    if prog > 0.55 and len(S) > 1:
        ex, ey = S[-1]
        px, py = S[max(0, len(S) - 3)]
        ang = math.atan2(ey - py, ex - px)
        hl = head * ss * min(1.0, (prog - 0.55) / 0.25)
        for s in (2.6, -2.6):
            d.line([(ex, ey), (ex + hl * math.cos(ang + s), ey + hl * math.sin(ang + s))],
                   fill=col, width=wd)


def cross(d, cx, cy, size, col, ss, prog=1.0, wdt=11):
    """Крест «неверная дверь» — две диагонали, вторая с задержкой."""
    if prog <= 0:
        return
    h = size / 2
    wd = max(1, int(wdt * ss))
    p1 = min(1.0, prog / 0.55)
    p2 = max(0.0, min(1.0, (prog - 0.45) / 0.55))
    if p1 > 0:
        d.line([((cx - h) * ss, (cy - h) * ss),
                ((cx - h + size * p1) * ss, (cy - h + size * p1) * ss)], fill=col, width=wd)
    if p2 > 0:
        d.line([((cx + h) * ss, (cy - h) * ss),
                ((cx + h - size * p2) * ss, (cy - h + size * p2) * ss)], fill=col, width=wd)


def cell(d, x0, y0, x1, y1, outline, fill, ss, r=26, wdt=4):
    if fill:
        d.rounded_rectangle([x0 * ss, y0 * ss, x1 * ss, y1 * ss], radius=r * ss, fill=fill)
    d.rounded_rectangle([x0 * ss, y0 * ss, x1 * ss, y1 * ss], radius=r * ss,
                        outline=outline, width=max(1, int(wdt * ss)))


# ---------------------------------------------------------------- сцена
def stage(door_l=("dim", 1.0), door_r=("dim", 1.0),
          guard_l=None, guard_r=None, ring_l=0.0, ring_r=0.0,
          arc=None, up=None, ask=0.0, cross_r=0.0, cross_l=0.0, flip=0.0,
          dash_r=False, ss=2):
    """Собирает кадр сцены. Все параметры — состояния, а не анимации:
    прогрессы считаются в раскадровке, здесь только отрисовка."""
    lay = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)

    door(d, LX, door_l[0], door_l[1], ss)
    door(d, RX, door_r[0], door_r[1], ss, dashed=dash_r)

    if guard_l:
        guard(d, LX, guard_l[0], guard_l[1], ss, ring_l)
    if guard_r:
        guard(d, RX, guard_r[0], guard_r[1], ss, ring_r)

    # дуга «спрашиваю про напарника» между стражниками.
    # Направления лежат на РАЗНОЙ глубине — иначе на плане, где нарисованы обе,
    # вторая дуга просто закрывает первую и «двойное искажение» не читается.
    if arc:
        direction, prog, kind = arc
        col = (BLUE if kind == "blue" else WHITE) + (235,)
        lr = direction == "lr"
        p = (LX + 78, GY + 96) if lr else (RX - 78, GY + 96)
        q = (RX - 78, GY + 96) if lr else (LX + 78, GY + 96)
        depth = GY + (400 if lr else 214)
        arrow(d, p, q, col, 6, ss, prog, head=24, dashed=True, curve=(540, depth))

    # стрелка «стражник → дверь»
    if up:
        side, prog = up
        x = LX if side == "l" else RX
        arrow(d, (x, GY - 48), (x, DOOR_BOT + 26), WHITE + (230,), 6, ss, prog, head=24)

    # стрелка «мой вопрос» — коротким крюком снизу к левому стражнику
    if ask > 0:
        arrow(d, (556, 1462), (LX + 74, GBODY[1] - 14), BLUE + (240,), 7, ss, ask,
              head=26, curve=(470, 1440))

    if cross_r > 0:
        cross(d, RX, DOOR_TOP + 210, 190, WHITE + (255,), ss, cross_r)
    if cross_l > 0:
        cross(d, LX, DOOR_TOP + 210, 190, WHITE + (255,), ss, cross_l)

    # большая дуга над дверями «идём в противоположную»
    if flip > 0:
        arrow(d, (RX, DOOR_TOP - 34), (LX, DOOR_TOP - 34), BLUE + (255,), 9, ss, flip,
              head=32, curve=(540, DOOR_TOP - 200))

    lay = lay.resize((W, H), Image.LANCZOS)
    return _glow(lay, 16, 0.45)


def table(rows, ss=2):
    """Таблица «кого спросил → что назовёт». rows: [(prog, hi)] — прогресс и подсветка."""
    lay = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    ys = [(745, 895), (955, 1105)]
    for i, (prog, hi) in enumerate(rows):
        if prog <= 0:
            continue
        a = max(0.0, min(1.0, prog))
        y0, y1 = ys[i]
        cell(d, 145, y0, 565, y1, WHITE + (int(190 * a),), None, ss)
        lc = LIME + (int(235 * a),) if hi else WHITE + (int(190 * a),)
        lf = LIME + (int(52 * a),) if hi else None
        cell(d, 605, y0, 935, y1, lc, lf, ss)
        arrow(d, (571, (y0 + y1) / 2), (599, (y0 + y1) / 2),
              WHITE + (int(210 * a),), 5, ss, a, head=16)
    lay = lay.resize((W, H), Image.LANCZOS)
    return _glow(lay, 14, 0.40)


# ---------------------------------------------------------------- текст сцены
KMAP = {"r": "sans", "i": "sans_it", "s": "serif_it"}


def txt(s, xy, size, kind="r", col=None, glow=None, gr=None, ga=0.55, op=1.0, anchor="mm"):
    """Служебная подпись внутри инфографики (не субтитр)."""
    col = col or WHITE
    return dict(text=s, font=font(KMAP[kind], size), xy=xy, anchor=anchor,
                fill=col, glow=glow or col,
                glow_r=gr if gr is not None else int(10 + size * 0.10),
                glow_a=ga, opacity=op)


def stagger(t, n, dur_each=0.30, step=0.08):
    return [ease_out(max(0.0, min(1.0, (t - i * step) / dur_each))) for i in range(n)]


def num(t, t0, digits, xy, size, blue=True, prog_digits=True):
    """R5b — синее число: разряды слева направо + scale 0.55→1.0 с оверщутом ~3%."""
    lt = t - t0
    if lt < 0:
        return []
    dur = 0.38
    p = min(1.0, lt / dur)
    e = ease_out(p)
    sc = 0.55 + 0.45 * e
    if 0.55 < p < 1.0:
        sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    n = len(digits)
    shown = digits if not prog_digits else digits[:max(1, int(math.ceil(n * min(1.0, lt / (dur * 0.8)))))]
    f = font("sans", max(8, int(size * sc)))
    col = BLUE if blue else WHITE
    gl = BLUE_GLOW if blue else WHITE
    return [dict(text=shown, font=f, xy=xy, anchor="mm", fill=col, glow=gl,
                 glow_r=int(size * 0.20), glow_a=1.0 if blue else 0.7,
                 opacity=min(1.0, lt / 0.12))]


def pop(t, t0, text, xy, size, kind="s"):
    """R5a — белое слово/число: резкий поп за 2-3 кадра."""
    lt = t - t0
    if lt < 0:
        return []
    p = min(1.0, lt / (3 / 30.0))
    sc = 0.6 + 0.4 * ease_out(p)
    f = font(KMAP[kind], max(8, int(size * sc)))
    return [dict(text=text, font=f, xy=xy, anchor="mm", fill=WHITE, glow=WHITE,
                 glow_r=int(size * 0.16), glow_a=0.7, opacity=min(1.0, lt / 0.08))]


def cl(v):
    return max(0.0, min(1.0, v))

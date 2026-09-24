"""Инфографика ролика 5 — «задача трёх тел и рождение теории хаоса» (рецепты R4/R10).

Одна сцена на весь ролик: тела на орбитах вокруг общего центра. Планы — её состояния:
два тела → три тела → расходящиеся траектории → клубок. Зритель не переучивает картинку.

Геометрия под карточку A (105..975 / 238..1618). Верх карточки до y≈560 занят блоком
субтитров (слот «T»), поэтому вся сцена живёт в полосе y 600..1590.

Общие примитивы (`txt` / `num` / `pop` / `stagger` / `cl` / `cell` / `_glow` / `_dash_path`)
переиспользуются из `doors.py` — они не привязаны к сюжету того ролика.
"""
import math
from PIL import Image, ImageDraw
from style import W, H, WHITE, BLUE, LIME, ease_out, font           # noqa: F401
from doors import _glow, _dash_path, txt, num, pop, stagger, cl, cell   # noqa: F401

SC = (540, 1000)        # центр орбитальной сцены
NUM_XY = (540, 1462)    # число-ревил — под сценой, по центру карточки

# орбиты: (a, b, наклон, угловая скорость, фаза)
ORB = [(252, 150, -0.22, 0.90, 0.00),
       (392, 246, 0.44, -0.55, 1.10)]


# ---------------------------------------------------------------- служебное
def _lay(ss):
    lay = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    return lay, ImageDraw.Draw(lay)


def _fin(lay, ss, r=16, s=0.45):
    return _glow(lay.resize((W, H), Image.LANCZOS), r, s)


def _clip(pts, prog):
    """Обрезать ломаную по доле её длины — «линия рисуется»."""
    if prog >= 1.0:
        return pts
    if prog <= 0.0 or len(pts) < 2:
        return []
    seg = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    want = sum(seg) * prog
    acc, out = 0.0, [pts[0]]
    for i, s in enumerate(seg):
        if acc + s >= want:
            u = (want - acc) / max(s, 1e-6)
            out.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * u,
                        pts[i][1] + (pts[i + 1][1] - pts[i][1]) * u))
            break
        acc += s
        out.append(pts[i + 1])
    return out


def stroke(d, pts, col, wdt, ss, prog=1.0, dashed=False, dash=17, gap=12):
    pts = _clip(pts, prog)
    if len(pts) < 2:
        return
    S = [(x * ss, y * ss) for x, y in pts]
    wd = max(1, int(wdt * ss))
    if dashed:
        _dash_path(d, S, col, wd, dash * ss, gap * ss)
    else:
        d.line(S, fill=col, width=wd, joint="curve")


def dot(d, xy, r, col, ss, prog=1.0, ring=0.0):
    """Тело: круг с проявлением (масштаб + альфа)."""
    if prog <= 0:
        return
    a = max(0.0, min(1.0, prog))
    rr = r * (0.55 + 0.45 * a) * ss
    c = col + (int(255 * a),)
    x, y = xy[0] * ss, xy[1] * ss
    d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=c)
    if ring > 0:
        gr = rr + (20 + 12 * ring) * ss
        d.ellipse([x - gr, y - gr, x + gr, y + gr],
                  outline=col + (int(150 * ring),), width=int(3 * ss))


def check(d, cx, cy, size, col, ss, prog=1.0, wdt=10):
    """Галочка — двумя отрезками, второй с задержкой."""
    if prog <= 0:
        return
    wd = max(1, int(wdt * ss))
    p1 = min(1.0, prog / 0.45)
    p2 = max(0.0, min(1.0, (prog - 0.35) / 0.65))
    a = (cx - size * 0.46, cy + size * 0.02)
    b = (cx - size * 0.13, cy + size * 0.34)
    c = (cx + size * 0.48, cy - size * 0.38)
    if p1 > 0:
        d.line([(a[0] * ss, a[1] * ss),
                ((a[0] + (b[0] - a[0]) * p1) * ss, (a[1] + (b[1] - a[1]) * p1) * ss)],
               fill=col, width=wd)
    if p2 > 0:
        d.line([(b[0] * ss, b[1] * ss),
                ((b[0] + (c[0] - b[0]) * p2) * ss, (b[1] + (c[1] - b[1]) * p2) * ss)],
               fill=col, width=wd)


def xmark(d, cx, cy, size, col, ss, prog=1.0, wdt=10):
    """Крест «не работает» — две диагонали, вторая с задержкой."""
    if prog <= 0:
        return
    h, wd = size / 2, max(1, int(wdt * ss))
    p1 = min(1.0, prog / 0.55)
    p2 = max(0.0, min(1.0, (prog - 0.45) / 0.55))
    if p1 > 0:
        d.line([((cx - h) * ss, (cy - h) * ss),
                ((cx - h + size * p1) * ss, (cy - h + size * p1) * ss)], fill=col, width=wd)
    if p2 > 0:
        d.line([((cx + h) * ss, (cy - h) * ss),
                ((cx + h - size * p2) * ss, (cy - h + size * p2) * ss)], fill=col, width=wd)


# ---------------------------------------------------------------- орбитальная сцена
def _pt(i, ang, jit=0.0, t=0.0):
    a, b, rot, _, _ = ORB[i]
    rr = 1.0 + jit * 0.17 * math.sin(3 * ang + t * 1.6)
    x, y = a * rr * math.cos(ang), b * rr * math.sin(ang)
    return (SC[0] + x * math.cos(rot) - y * math.sin(rot),
            SC[1] + x * math.sin(rot) + y * math.cos(rot))


def _orb_pts(i, jit=0.0, t=0.0, n=150):
    return [_pt(i, 2 * math.pi * k / n, jit, t) for k in range(n + 1)]


def _body_pt(i, t, jit=0.0):
    _, _, _, w, ph = ORB[i]
    return _pt(i, w * t + ph, jit, t)


def orbits(t, sun=1.0, o1=0.0, b1=0.0, o2=0.0, b2=0.0,
           pred=0.0, jit=0.0, dim=1.0, ss=2):
    """Состояние сцены. Прогрессы считаются в раскадровке — здесь только отрисовка."""
    lay, d = _lay(ss)
    ow = int(150 * dim)
    if o1 > 0:
        stroke(d, _orb_pts(0, jit, t), WHITE + (ow,), 4, ss, o1)
    if o2 > 0:
        stroke(d, _orb_pts(1, jit, t), WHITE + (ow,), 4, ss, o2)

    # пунктирное продолжение орбиты «в будущее» + мишень на предсказанной точке
    if pred > 0:
        _, _, _, w, ph = ORB[0]
        a0 = w * t + ph
        pts = [_pt(0, a0 + 1.7 * math.pi * k / 70, jit, t) for k in range(71)]
        stroke(d, pts, BLUE + (235,), 5, ss, pred, dashed=True)
        if pred > 0.72:
            q = _pt(0, a0 + 1.7 * math.pi, jit, t)
            r = 27 * ss
            d.ellipse([q[0] * ss - r, q[1] * ss - r, q[0] * ss + r, q[1] * ss + r],
                      outline=BLUE + (255,), width=int(4 * ss))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                d.line([(q[0] * ss + dx * r * 0.4, q[1] * ss + dy * r * 0.4),
                        (q[0] * ss + dx * r * 1.7, q[1] * ss + dy * r * 1.7)],
                       fill=BLUE + (255,), width=int(4 * ss))

    if sun > 0:
        dot(d, SC, 34, WHITE, ss, sun)
    if b1 > 0:
        dot(d, _body_pt(0, t, jit), 17, BLUE, ss, b1)
    if b2 > 0:
        dot(d, _body_pt(1, t, jit), 14, WHITE, ss, b2)
    return _fin(lay, ss)


# ---------------------------------------------------------------- траектории
def _rose(seed, u, eps=0.0, lam=0.42):
    """Квази-орбита трёх тел. `eps` — микро-сдвиг старта, растущий экспоненциально:
    в начале траектории неразличимы, дальше расходятся."""
    if eps:
        u = u + eps * (math.exp(lam * u) - 1.0)
    ph = 0.6 + 0.9 * seed
    return (540 + 208 * math.cos(0.90 * u + ph) + 112 * math.cos(2.31 * u + 1.7 * ph),
            1000 + 208 * math.sin(0.90 * u + ph) + 112 * math.sin(1.73 * u + 0.5 * ph))


def rose_pts(seed, eps=0.0, n=320, U=13.0):
    return [_rose(seed, U * k / n, eps) for k in range(n + 1)]


def tangle(seeds, progs, blue=(), ss=2):
    """Клубок траекторий — каскадом. `blue` — индексы синих линий."""
    lay, d = _lay(ss)
    for i, s in enumerate(seeds):
        col = (BLUE if i in blue else WHITE) + (200,)
        stroke(d, rose_pts(s), col, 3, ss, progs[i])
    return _fin(lay, ss, 14, 0.38)


def diverge(seed, prog, eps=0.0042, dots=1.0, ss=2):
    """Две траектории из практически одной точки: белая и синяя."""
    lay, d = _lay(ss)
    A, B = rose_pts(seed, 0.0), rose_pts(seed, eps)
    stroke(d, A, WHITE + (225,), 4, ss, prog)
    stroke(d, B, BLUE + (235,), 4, ss, prog)
    if dots > 0:
        dot(d, A[0], 11, WHITE, ss, dots)
    return _fin(lay, ss)


def magnifier(prog, gap=0.0, ss=2):
    """Лупа на старте: две точки, разъезжающиеся на «сколь угодно малую» разницу."""
    lay, d = _lay(ss)
    cx, cy, R = 540, 1030, 178
    a = max(0.0, min(1.0, prog))
    rr = R * (0.72 + 0.28 * a) * ss
    d.ellipse([cx * ss - rr, cy * ss - rr, cx * ss + rr, cy * ss + rr],
              outline=WHITE + (int(190 * a),), width=int(4 * ss))
    g = 34 * gap
    dot(d, (cx - g, cy), 13, WHITE, ss, a)
    dot(d, (cx + g, cy), 13, BLUE, ss, a)
    if gap > 0.15:
        y = cy + 62
        col = WHITE + (int(210 * min(1.0, gap)),)
        d.line([((cx - g) * ss, y * ss), ((cx + g) * ss, y * ss)], fill=col, width=int(3 * ss))
        for x in (cx - g, cx + g):
            d.line([(x * ss, (y - 14) * ss), (x * ss, (y + 14) * ss)], fill=col, width=int(3 * ss))
    return _fin(lay, ss)


# ---------------------------------------------------------------- рамка «формулы нет»
def formula_box(prog, cross=0.0, ss=2):
    lay, d = _lay(ss)
    a = max(0.0, min(1.0, prog))
    if a > 0:
        cell(d, 205, 1268, 815, 1398, WHITE + (int(200 * a),), None, ss, r=30)
    if cross > 0:
        xmark(d, 886, 1333, 96, WHITE + (255,), ss, cross, wdt=9)
    return _fin(lay, ss, 14, 0.40)


# ---------------------------------------------------------------- таблица-вердикт
PANEL_ROWS = [(830, 990), (1050, 1210)]


def panel(rows, ss=2):
    """rows: [(prog, 'ok'|'no')] — две строки-ячейки со знаком справа."""
    lay, d = _lay(ss)
    for i, (prog, kind) in enumerate(rows):
        if prog <= 0:
            continue
        a = max(0.0, min(1.0, prog))
        y0, y1 = PANEL_ROWS[i]
        base = LIME if kind == "no" else WHITE
        cell(d, 165, y0, 915, y1, base + (int(200 * a),),
             (LIME + (int(46 * a),) if kind == "no" else None), ss, r=28)
        cy = (y0 + y1) / 2
        if kind == "ok":
            check(d, 828, cy, 78, WHITE + (255,), ss, min(1.0, a / 0.7))
        else:
            xmark(d, 828, cy, 70, LIME + (255,), ss, min(1.0, a / 0.7), wdt=9)
    return _fin(lay, ss, 14, 0.40)


# ---------------------------------------------------------------- приложения теории
def _spiral(cx, cy, r0, r1, turns, n=170, ph=0.0):
    return [(cx + (r0 + (r1 - r0) * k / n) * math.cos(2 * math.pi * turns * k / n + ph),
             cy + (r0 + (r1 - r0) * k / n) * math.sin(2 * math.pi * turns * k / n + ph))
            for k in range(n + 1)]


def _loop(cx, cy, a, b, rot, n=64):
    """Замкнутый эллипс-«крыло» одной ломаной."""
    return [(cx + a * math.cos(2 * math.pi * k / n) * math.cos(rot)
             - b * math.sin(2 * math.pi * k / n) * math.sin(rot),
             cy + a * math.cos(2 * math.pi * k / n) * math.sin(rot)
             + b * math.sin(2 * math.pi * k / n) * math.cos(rot)) for k in range(n + 1)]


def _butterfly(cx, cy, s):
    """Бабочка: четыре крыла-петли + тельце и усики. Читается за полсекунды."""
    wings = []
    for sx in (1, -1):
        wings.append(_loop(cx + sx * 0.80 * s, cy - 0.62 * s, 1.05 * s, 0.66 * s, sx * -0.52))
        wings.append(_loop(cx + sx * 0.64 * s, cy + 0.70 * s, 0.76 * s, 0.50 * s, sx * 0.48))
    body = [(cx, cy - 0.92 * s), (cx, cy + 0.88 * s)]
    ant = [[(cx, cy - 0.90 * s), (cx + sx * 0.42 * s, cy - 1.42 * s)] for sx in (1, -1)]
    return wings + [body] + ant


def icons(kind, prog, t, ss=2):
    lay, d = _lay(ss)
    if kind == "butterfly":
        p1 = ease_out(cl(prog / 0.40))
        p2 = ease_out(cl((prog - 0.30) / 0.70))
        for part in _butterfly(332, 1236, 46):
            stroke(d, part, WHITE + (235,), 4, ss, p1)
        sp = _spiral(672, 882, 26, 168, 2.3, ph=t * 0.5)
        # хвост от бабочки к спирали — гладкая дуга, без излома на стыке
        p0, q0 = (352, 1170), sp[0]
        tail = [((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * 380 + u ** 2 * q0[0],
                 (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * 1010 + u ** 2 * q0[1])
                for u in [k / 44 for k in range(45)]] + sp
        stroke(d, tail, BLUE + (230,), 4, ss, p2)
    elif kind == "turb":
        st = stagger(prog, 3, 0.45, 0.10)
        for i, (cx, cy, r1, tn) in enumerate(((330, 1210, 118, 2.4),
                                              (620, 900, 172, 2.8),
                                              (830, 1230, 96, 2.0))):
            stroke(d, _spiral(cx, cy, 16, r1, tn, ph=t * (0.6 + 0.2 * i)),
                   (BLUE if i == 1 else WHITE) + (225,), 4, ss, st[i])
    elif kind == "asteroid":
        p1 = ease_out(cl(prog / 0.50))
        p2 = ease_out(cl((prog - 0.35) / 0.65))
        orb = [(540 + 302 * math.cos(u * 2 * math.pi / 120) * math.cos(0.2)
                - 168 * math.sin(u * 2 * math.pi / 120) * math.sin(0.2),
                1030 + 302 * math.cos(u * 2 * math.pi / 120) * math.sin(0.2)
                + 168 * math.sin(u * 2 * math.pi / 120) * math.cos(0.2)) for u in range(121)]
        stroke(d, orb, WHITE + (170,), 4, ss, p1)
        dot(d, (540, 1030), 26, WHITE, ss, p1)
        ang = t * 0.8
        bx = (540 + 302 * math.cos(ang) * math.cos(0.2) - 168 * math.sin(ang) * math.sin(0.2))
        by = (1030 + 302 * math.cos(ang) * math.sin(0.2) + 168 * math.sin(ang) * math.cos(0.2))
        dot(d, (bx, by), 13, BLUE, ss, p1)
        # веер неопределённости: три пунктирные дуги вперёд по орбите
        for k in (-0.34, 0.0, 0.34):
            pts = [(540 + (302 + k * 300 * u / 40) * math.cos(ang + 2.2 * u / 40) * math.cos(0.2)
                    - (168 + k * 180 * u / 40) * math.sin(ang + 2.2 * u / 40) * math.sin(0.2),
                    1030 + (302 + k * 300 * u / 40) * math.cos(ang + 2.2 * u / 40) * math.sin(0.2)
                    + (168 + k * 180 * u / 40) * math.sin(ang + 2.2 * u / 40) * math.cos(0.2))
                   for u in range(41)]
            stroke(d, pts, BLUE + (200,), 3, ss, p2, dashed=True, dash=15, gap=12)
    return _fin(lay, ss, 14, 0.40)

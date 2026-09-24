"""Инфографика ролика 4 — «четыре краски и ошибка Кэмпе». Рецепт R10 на сетке.

Карта настоящая: диаграмма Вороного по заданным центрам, координаты предварительно
искривлены синусом — границы получаются «от руки», как и сетка бренда.

ЧЕТЫРЕ КРАСКИ — четыре настоящих цвета. Первая версия рисовала их как
синий / белый / белая штриховка / синяя штриховка, чтобы не нарушать «третий цвет нельзя»
из brand-kit §7. На кадре это провалилось: штриховка читается как текстура, а не как краска,
и перекраска выглядела дёрганьем. Палитра расширена по прямому решению автора и держится
вокруг бренда: ведущий — синий #3B4BE8, лайм #C3DB4E уже был в ките вторым акцентом,
костяной белый — цвет текста, янтарный добавлен один.

Читаемость механики держится не на цвете, а на трёх приёмах:
  · ПАЛИТРА — полоса из четырёх образцов под картой: видно, сколько красок вообще есть
    и какие из них уже заняты вокруг проблемного участка;
  · СЕРЫЙ ФОН — всё, о чём сейчас не говорят, уходит в нейтральный серый,
    так что «цепочка областей двух цветов» буквально остаётся двумя цветами на сером;
  · ЯВНЫЕ МЕТКИ — «?» в участке, которому не хватает краски, и знак ↔ на обмене.

Раскраски и цепи не «нарисованы на глаз», а проверяются функцией audit() ниже:
STATE_OK правильная, STATE_STUCK правильная и вокруг центра стоят все 4 краски,
цепь {c1,c3} после обмена освобождает c1 — это и есть аргумент Кэмпа.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from style import W, H, WHITE, BLUE, BLUE_GLOW, LIME, ease_out, font
# generic-хелперы текста и чисел живут в doors.py — они не завязаны на ту раскадровку
from doors import txt, num, pop, cl, stagger, cross, KMAP  # noqa: F401

# --- габарит карты внутри карточки A (105..975 / 238..1618) ---
MX, MY, MW, MH = 150, 585, 780, 760   # ниже блока субтитров: 3 строки в слоте T доходят до y≈570
MR = 44                       # скругление листа карты

# --- центры областей: колесо (центр + кольцо из 5) + внешнее кольцо из 5 ---
P, R0, R1, R2, R3, R4, O0, O1, O2, O3, O4 = range(11)
RING = [R0, R1, R2, R3, R4]
OUTER = [O0, O1, O2, O3, O4]


def _seeds(rot=0.0, rr=196, ro=338):
    cx, cy = MX + MW / 2, MY + MH / 2
    out = [(cx, cy)]
    for i in range(5):
        a = math.radians(rot - 90 + i * 72)
        out.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    for i in range(5):
        a = math.radians(rot - 54 + i * 72)
        out.append((cx + ro * math.cos(a), cy + ro * math.sin(a)))
    return out


SEEDS = {"a": _seeds(0.0), "b": _seeds(36.0, rr=186, ro=326)}
PHASE = {"a": (1.3, 0.0), "b": (0.4, 2.1)}

# --- краски ---
BONE = (231, 236, 247)         # костяной белый
AMBER = (226, 133, 74)         # единственный добавленный цвет
GREY = (60, 66, 78)            # нейтральный фон для областей вне фокуса
PAINT = {
    0: None,
    1: BLUE,                   # #3B4BE8 — ведущий цвет бренда
    2: BONE,
    3: LIME,                   # #C3DB4E — второй акцент из brand-kit
    4: AMBER,
}
PAINT_NAME = {1: "синий", 2: "белый", 3: "лайм", 4: "янтарь"}
# WHITE / LIME / AMBER реэкспортируются для раскадровки — цвета контуров цепей
FILL_A = 236
SHEET_A = 200                  # лист карты гасит сетку под собой, иначе заливки не читаются
EDGE_A = 235

# --- палитра под картой ---
PAL_Y, PAL_W, PAL_H, PAL_GAP = 1392, 150, 64, 30
PAL_X0 = 540 - (4 * PAL_W + 3 * PAL_GAP) / 2

# карта раскрашена честно — все соседи разного цвета
STATE_OK = {P: 1, R0: 2, R1: 3, R2: 2, R3: 3, R4: 4, O0: 1, O1: 4, O2: 1, O3: 2, O4: 3}
# «застрявшая» конфигурация: вокруг центра стоят все четыре краски, ему не хватает цвета
STATE_STUCK = {P: 0, R0: 1, R1: 2, R2: 3, R3: 4, R4: 2, O0: 3, O1: 4, O2: 1, O3: 3, O4: 4}

CHAIN_13 = [R0, O0]      # цепь красок 1↔3, содержит R0
CHAIN_14 = [R0, O4]      # цепь красок 1↔4, тоже содержит R0 — цепи пересекаются


def swapped(state, chain, a, b):
    s = dict(state)
    for i in chain:
        s[i] = b if s[i] == a else (a if s[i] == b else s[i])
    return s


STATE_SWAPPED = swapped(STATE_STUCK, CHAIN_13, 1, 3)
STATE_FIXED = dict(STATE_SWAPPED)
STATE_FIXED[P] = 1                      # освободившаяся краска встаёт в проблемный участок


# ---------------------------------------------------------------- геометрия
_CACHE = {}


def _lab(key):
    """Метки областей: ближайший центр в искривлённых координатах."""
    if key in _CACHE:
        return _CACHE[key]
    ph = PHASE[key]
    yy, xx = np.mgrid[0:MH, 0:MW]
    X = (xx + MX).astype(np.float32)
    Y = (yy + MY).astype(np.float32)
    Xw = X + 24 * np.sin(Y / 71.0 + ph[0])
    Yw = Y + 24 * np.sin(X / 86.0 + ph[1])
    best, lab = None, np.zeros((MH, MW), np.int8)
    for i, (sx, sy) in enumerate(SEEDS[key]):
        d = (Xw - sx) ** 2 + (Yw - sy) ** 2
        if best is None:
            best = d
        else:
            m = d < best
            lab[m] = i
            best = np.minimum(best, d)
    # границы между областями
    e = np.zeros((MH, MW), bool)
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    e[:-1, :] |= lab[:-1, :] != lab[1:, :]
    edge = np.zeros((MH, MW), np.uint8)
    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):
            edge |= np.roll(np.roll(e, dy, 0), dx, 1).astype(np.uint8)
    # лист карты со скруглением
    clip = Image.new("L", (MW, MH), 0)
    ImageDraw.Draw(clip).rounded_rectangle([0, 0, MW - 1, MH - 1], radius=MR, fill=255)
    _CACHE[key] = (lab, edge.astype(bool), np.array(clip))
    return _CACHE[key]


def adjacency(key):
    lab, _, _ = _lab(key)
    adj = {i: set() for i in range(11)}
    a, b = lab[:, :-1], lab[:, 1:]
    for u, v in set(zip(a[a != b].tolist(), b[a != b].tolist())):
        adj[u].add(v); adj[v].add(u)
    a, b = lab[:-1, :], lab[1:, :]
    for u, v in set(zip(a[a != b].tolist(), b[a != b].tolist())):
        adj[u].add(v); adj[v].add(u)
    return adj


# ---------------------------------------------------------------- отрисовка
def _fill_rgba(key, style):
    """style: {region: (краска, альфа-множитель)}. Краска 'g' — нейтральный серый."""
    lab, edge, clip = _lab(key)
    out = np.zeros((MH, MW, 4), np.float32)
    for i, (c, k) in style.items():
        if not c or k <= 0:
            continue
        col = GREY if c == "g" else PAINT[c]
        if col is None:
            continue
        m = lab == i
        if not m.any():
            continue
        out[m, 0], out[m, 1], out[m, 2] = col
        # серый должен отступать на второй план, иначе фон спорит с фокусом
        out[m, 3] = FILL_A * k * (0.72 if c == "g" else 1.0)
    return out


def _compose(key, layers, edge_a=EDGE_A, sheet=SHEET_A):
    lab, edge, clip = _lab(key)
    acc = np.zeros((MH, MW, 4), np.float32)
    acc[..., 3] = sheet
    for st in layers:
        src = _fill_rgba(key, st)
        sa = src[..., 3:4] / 255.0
        da = acc[..., 3:4] / 255.0
        oa = sa + da * (1 - sa)
        rgb = np.where(oa > 0,
                       (src[..., :3] * sa + acc[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6), 0)
        acc = np.concatenate([rgb, oa * 255], axis=2)
    ea = np.zeros((MH, MW), np.float32)
    ea[edge] = edge_a
    keep = ea > acc[..., 3]
    acc[..., 0][keep] = 255; acc[..., 1][keep] = 255; acc[..., 2][keep] = 255
    acc[..., 3] = np.maximum(acc[..., 3], ea)
    acc[..., 3] *= clip / 255.0
    return Image.fromarray(acc.clip(0, 255).astype(np.uint8), "RGBA")


def _glow(layer, radius=15, strength=0.42):
    a = layer.split()[3]
    blur = a.filter(ImageFilter.GaussianBlur(radius))
    gl = Image.new("RGBA", layer.size, WHITE + (0,))
    gl.putalpha(blur.point(lambda v: int(v * strength)))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(gl)
    out.alpha_composite(layer)
    return out


def _outline_of(key, regions, width=9):
    """Контур объединения областей — для подсветки цепи."""
    lab, _, clip = _lab(key)
    m = np.isin(lab, regions)
    e = np.zeros_like(m)
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:-1, :] |= m[:-1, :] != m[1:, :]
    o = np.zeros_like(e)
    r = width // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            o |= np.roll(np.roll(e, dy, 0), dx, 1)
    return o & (clip > 0)


def _paste_mask(canvas, mask, col):
    arr = np.zeros((MH, MW, 4), np.uint8)
    arr[mask] = tuple(col) + (255,)
    im = Image.fromarray(arr, "RGBA")
    canvas.paste(im, (MX, MY), im)


def centroid(key, i):
    lab, _, _ = _lab(key)
    ys, xs = np.where(lab == i)
    return (float(xs.mean()) + MX, float(ys.mean()) + MY)


def draw(key, state, alphas=None, layers2=None, focus=None,
         chain=None, chain_col=None, chain2=None, chain2_col=None,
         crosses=(), pair=None, empty_ring=None, qmark=None, swap=None, dim=1.0):
    """Кадр карты.
    state    — {область: краска}
    alphas   — {область: множитель альфы} (каскад появления)
    layers2  — (state2, {область: k}) вторым слоем: кроссфейд красок на обмене
    focus    — если задан, все области вне списка уходят в нейтральный серый
    chain/chain2 — списки областей, обвести контуром (chain_col — цвет контура)
    pair     — пара соседей, обвести общим контуром (демонстрация правила)
    qmark    — область, поставить «?» (не хватает краски)
    swap     — (a, b, prog) — знак ↔ между центрами двух областей
    """
    alphas = alphas or {}
    fset = set(focus) if focus is not None else None
    st = {}
    for i in range(11):
        c = state.get(i, 0)
        if fset is not None and i not in fset and c:
            c = "g"
        st[i] = (c, alphas.get(i, 1.0) * dim)
    ls = [st]
    if layers2:
        s2, k2 = layers2
        ls.append({i: (s2.get(i, 0), k2.get(i, 0.0)) for i in range(11)})

    sheet = _compose(key, ls, edge_a=int(EDGE_A * dim))
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(sheet, (MX, MY), sheet)

    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle(
        [MX, MY, MX + MW - 1, MY + MH - 1], radius=MR,
        outline=WHITE + (int(130 * dim),), width=4)
    canvas.alpha_composite(ov)

    for ch, cc, cw in ((chain2, chain2_col, 14), (chain, chain_col, 10),
                       (pair, WHITE, 10)):
        if ch:
            _paste_mask(canvas, _outline_of(key, ch, width=cw), cc or WHITE)
    if empty_ring is not None:
        _paste_mask(canvas, _outline_of(key, [empty_ring], width=11), WHITE)

    d = ImageDraw.Draw(canvas)
    for i in crosses:
        cx, cy = centroid(key, i)
        cross(d, cx, cy, 120, WHITE + (255,), 1, 1.0, wdt=11)

    if qmark is not None:
        cx, cy = centroid(key, qmark)
        f = font("sans", 130)
        d.text((cx, cy), "?", font=f, anchor="mm", fill=WHITE + (255,))

    if swap:
        a, b, prog = swap
        if prog > 0:
            pa, pb = centroid(key, a), centroid(key, b)
            mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
            ln = 92 * min(1.0, prog)
            ang = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
            ux, uy = math.cos(ang), math.sin(ang)
            for sgn in (1, -1):
                ex, ey = mx + ux * ln * sgn, my + uy * ln * sgn
                d.line([(mx - ux * ln * sgn * 0.15, my - uy * ln * sgn * 0.15), (ex, ey)],
                       fill=WHITE + (255,), width=9)
                for t2 in (2.5, -2.5):
                    d.line([(ex, ey), (ex + 30 * math.cos(ang + t2 if sgn > 0 else ang + math.pi + t2),
                                       ey + 30 * math.sin(ang + t2 if sgn > 0 else ang + math.pi + t2))],
                           fill=WHITE + (255,), width=9)

    return _glow(canvas)


# ---------------------------------------------------------------- палитра красок
def palette(show=(1, 2, 3, 4), marks=None, prog=1.0, dim=1.0):
    """Полоса из четырёх образцов под картой — «вот все краски, которые есть».
    marks: {краска: 'x' | 'free'} — занята вокруг проблемного участка / освободилась."""
    marks = marks or {}
    ss = 2
    lay = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    st = stagger(prog * 0.6, 4, 0.22, 0.06) if prog < 1 else [1.0] * 4
    for k in range(4):
        c = k + 1
        p = st[k]
        if p <= 0:
            continue
        on = c in show
        a = int(FILL_A * p * dim * (1.0 if on else 0.30))
        x0 = PAL_X0 + k * (PAL_W + PAL_GAP)
        box = [x0 * ss, PAL_Y * ss, (x0 + PAL_W) * ss, (PAL_Y + PAL_H) * ss]
        d.rounded_rectangle(box, radius=16 * ss, fill=PAINT[c] + (a,))
        m = marks.get(c)
        if m == "free":
            d.rounded_rectangle([box[0] - 9 * ss, box[1] - 9 * ss, box[2] + 9 * ss, box[3] + 9 * ss],
                                radius=22 * ss, outline=WHITE + (255,), width=int(5 * ss))
        elif m == "x":
            cx, cy = (x0 + PAL_W / 2) * ss, (PAL_Y + PAL_H / 2) * ss
            h = 22 * ss
            for sx in (1, -1):
                d.line([(cx - h * sx, cy - h), (cx + h * sx, cy + h)],
                       fill=WHITE + (235,), width=int(7 * ss))
    lay = lay.resize((W, H), Image.LANCZOS)
    return _glow(lay, 12, 0.35)


# ---------------------------------------------------------------- таймлайн
# Лист таймлайна повторяет лист карты — одна графическая система.
# Годы на оси идут одним кеглем; крупное число-ревил живёт в общем слоте под листом,
# иначе увеличенный год налезает на соседний (проверено на кадре).
TL_X, TL_Y0, TL_W, TL_H = 150, 700, 780, 530
TL_AXIS = 900
TL_YEAR_Y = 790
TICKS = [(300, "1879"), (540, "1890"), (780, "1976")]
SPAN_Y = [990, 1108]          # своя строка на каждый промежуток — скобки не сталкиваются
TL_GAP_Y = [1052, 1170]
GAP_X = [420, 660]


def timeline(show=3, spans=(), dim=1.0):
    """show — сколько засечек показать; spans — [(i, j, prog)] скобки между засечками."""
    ss = 2
    lay = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle([TL_X * ss, TL_Y0 * ss, (TL_X + TL_W) * ss, (TL_Y0 + TL_H) * ss],
                        radius=MR * ss, fill=(0, 0, 0, int(SHEET_A * dim)),
                        outline=WHITE + (int(120 * dim),), width=int(4 * ss))
    a = int(190 * dim)
    d.line([(210 * ss, TL_AXIS * ss), (870 * ss, TL_AXIS * ss)],
           fill=WHITE + (a,), width=int(4 * ss))
    for k, (x, _) in enumerate(TICKS):
        if k >= show:
            continue
        hi = (k == show - 1)
        d.line([(x * ss, (TL_AXIS - 30) * ss), (x * ss, (TL_AXIS + 30) * ss)],
               fill=WHITE + (255 if hi else a,), width=int((7 if hi else 4) * ss))
    for (i, j, prog) in spans:
        if prog <= 0:
            continue
        x0, x1 = TICKS[i][0], TICKS[j][0]
        xe = x0 + (x1 - x0) * min(1.0, prog)
        y, col = SPAN_Y[i], WHITE + (215,)
        d.line([(x0 * ss, y * ss), (xe * ss, y * ss)], fill=col, width=int(4 * ss))
        for xx in (x0, xe):
            d.line([(xx * ss, (y - 15) * ss), (xx * ss, (y + 15) * ss)], fill=col, width=int(4 * ss))
    lay = lay.resize((W, H), Image.LANCZOS)
    return _glow(lay, 14, 0.40)


# ---------------------------------------------------------------- самопроверка
def audit():
    """Проверяет, что раскраски правильные, а аргумент Кэмпа на этой карте действительно работает."""
    msg = []
    for key in ("a", "b"):
        adj = adjacency(key)
        for name, st in (("OK", STATE_OK), ("STUCK", STATE_STUCK), ("SWAPPED", STATE_SWAPPED),
                         ("FIXED", STATE_FIXED)):
            bad = [(u, v) for u in adj for v in adj[u]
                   if u < v and st[u] and st[u] == st[v]]
            if bad:
                msg.append(f"[{key}/{name}] соседи одного цвета: {bad}")
        around = {STATE_STUCK[v] for v in adj[P]}
        if around != {1, 2, 3, 4}:
            msg.append(f"[{key}] вокруг центра не все 4 краски: {sorted(around)}")
        # цепь {1,3} от R0 должна быть замкнутой и не задевать остальных
        seen, stack = set(), [R0]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for v in adj[u]:
                if STATE_STUCK[v] in (1, 3) and v not in seen:
                    stack.append(v)
        if seen != set(CHAIN_13):
            msg.append(f"[{key}] цепь 1-3 от R0 = {sorted(seen)}, ожидали {sorted(CHAIN_13)}")
        seen2, stack = set(), [R0]
        while stack:
            u = stack.pop()
            if u in seen2:
                continue
            seen2.add(u)
            for v in adj[u]:
                if STATE_STUCK[v] in (1, 4) and v not in seen2:
                    stack.append(v)
        if seen2 != set(CHAIN_14):
            msg.append(f"[{key}] цепь 1-4 от R0 = {sorted(seen2)}, ожидали {sorted(CHAIN_14)}")
        if 1 in {STATE_SWAPPED[v] for v in adj[P]}:
            msg.append(f"[{key}] после обмена краска 1 у центра не освободилась")
    return msg


if __name__ == "__main__":
    for key in ("a", "b"):
        print(key, "соседства:", {k: sorted(v) for k, v in adjacency(key).items()})
    problems = audit()
    print("аудит:", problems or "всё сходится")

"""Раскадровка ролика 6 — профессия геофизика-интерпретатора (сейсморазведка,
обратная задача, мостик к школьной геометрии и тригонометрии).

Тайминги — из videos/6/words.json (whisper large-v3 по речевым отрезкам).
Речь не режется: резы только по картинке, по границам слов.
Whisper ослышался в одном месте: «в УЗИ» -> по смыслу «в вузе» (высшее учебное
заведение), исправлено только в отображаемом субтитре, words.json не трогаем.
"""
import math
from PIL import Image, ImageDraw, ImageFilter

from style import W, H, WHITE, BLUE, BLUE_GLOW, LIME, font, ease_out

TAG = "v6"
SRC = "/Users/vladimirkalajcidi/reels_good2/videos/6/source.mov"
OUT = "/Users/vladimirkalajcidi/reels_good2/videos/6/geofizik_edit.mp4"
DUR = 52.657

# лицо после кропа (без hflip — дубль уже не зеркальный, проверено по нашивке
# на футболке): глаза около 40-41% высоты карточки на обеих рамках.
FRAMINGS = {"A1": (720, 1142, 0, 120), "A2": (560, 888, 80, 227)}

S = "r"

SHOTS = [
    (0.000,  1.519, "A1", {}),
    (1.519,  2.574, "stock", {"clip": "oilrig1", "ss": 1.0}),
    (2.574,  5.204, "stock", {"clip": "oilrig2", "ss": 1.0}),
    (5.204,  7.638, "A2", {}),
    (7.638,  9.261, "depth1", {}),
    (9.261, 12.054, "depth2", {}),
    (12.054, 13.596, "money_pop", {}),
    (13.596, 15.549, "stock", {"clip": "money", "ss": 2.0}),
    (15.549, 16.885, "loss_pop", {}),
    (16.885, 18.716, "stock", {"clip": "desert", "ss": 0.5}),
    (18.716, 20.322, "A1", {}),
    (20.322, 21.589, "wave_down", {}),
    (21.589, 23.346, "wave_up", {}),
    (23.346, 26.302, "layers2", {}),
    (26.302, 27.881, "A2", {}),
    (27.881, 30.318, "wireframe", {}),
    (30.318, 33.271, "A1", {"topword": "ОБРАТНАЯ"}),
    (33.271, 35.302, "A2", {}),
    (35.302, 36.701, "terms1", {}),
    (36.701, 39.558, "terms2", {}),
    (39.558, 40.949, "stock", {"clip": "geometry", "ss": 0.5}),
    (40.949, 43.158, "A1", {"topword": "ТРИГОНОМЕТРИЯ"}),
    (43.158, 45.175, "A2", {}),
    (45.175, 47.960, "stock", {"clip": "lecture", "ss": 3.0}),
    (47.960, 49.629, "A1", {}),
    (49.629, 52.657, "A2", {}),
]

# ---------------------------------------------------------------- субтитры
CAPS = [
    (0.000, 1.519, [("это самая дорогая профессия", S, 52)], "A"),

    (1.519, 2.574, [("в нефтяной отрасли", S, 50)], "T"),
    (2.574, 3.524, [("и это не буровик", S, 46)], "T"),
    (3.524, 5.204, [("и не директор", S, 46), (" месторождения", "s", 40)], "T"),

    (5.204, 6.604, [("это ", S, 46), ("геофизик-интерпретатор", "i", 38)], "A"),
    (6.604, 7.638, [("человек который видит", S, 44)], "A"),

    (7.638, 8.774, [("что находится под землёй", S, 42)], "T"),
    (8.774, 9.261, [("на глубине", S, 46)], "T"),

    (10.133, 12.054, [("не выкопав", S, 46), (" ни одной ямы", "s", 40)], "T"),

    (12.054, 12.712, [("его доходы", S, 50)], "T"),

    (13.596, 14.644, [("в месяц потому что", S, 42)], "T"),
    (14.644, 15.549, [("одна его ошибка", S, 46)], "T"),

    (15.549, 16.104, [("это скважина", S, 50)], "T"),

    (16.885, 18.386, [("пробуренная ", S, 44), ("в пустоту", "s", 52)], "T"),

    (18.716, 19.115, [("и вот чем", S, 46)], "A"),
    (19.115, 20.322, [("он занимается", S, 48)], "A"),

    (20.322, 21.241, [("по земле бьют сигналы", S, 42)], "T"),

    (21.589, 22.304, [("возвращается", S, 46)], "T"),
    (22.304, 22.937, [("и в этом эхе", S, 42)], "T"),
    (22.937, 23.346, [("он читает", S, 46)], "T"),

    (23.346, 25.001, [("картину где залегает ", S, 40), ("нефть", "s", 56)], "T"),
    (25.001, 26.302, [("а где просто ", S, 40), ("вода", "s", 56)], "T"),

    (26.302, 26.905, [("стоит ли вообще", S, 46)], "A"),
    (26.905, 27.881, [("здесь бурить", S, 50)], "A"),

    (27.881, 29.347, [("он восстанавливает", S, 40), (" трёхмерный объект", S, 40)], "T"),
    (29.347, 30.318, [("по отражённой волне", S, 44)], "T"),

    (30.318, 31.351, [("это задача которую", S, 44)], "A"),
    (31.351, 33.271, [("математики называют", S, 44)], "A"),

    (33.271, 33.968, [("и она на порядок", S, 44)], "A"),
    (33.968, 35.302, [("сложнее ", S, 46), ("прямой", "i", 60)], "A"),

    (39.558, 40.364, [("а вырастает всё это", S, 40)], "T"),
    (40.364, 40.949, [("из обычной", S, 46)], "T"),

    (40.949, 41.937, [("школьной геометрии", S, 46)], "A"),

    (43.158, 43.940, [("в вузе это догнать", S, 42)], "A"),
    (43.940, 45.175, [("уже очень тяжело", S, 44)], "A"),

    (45.175, 45.690, [("там на этой базе", S, 42)], "T"),
    (45.690, 46.390, [("строят а не", S, 46)], "T"),
    (46.390, 47.960, [("проходят её ", S, 44), ("с нуля", "s", 50)], "T"),

    (47.960, 48.866, [("пишите слово ", S, 44), ("пробное", "s", 56)], "A"),
    (48.866, 49.629, [("в комментариях", S, 48)], "A"),

    (49.629, 50.288, [("и я проведу", S, 46)], "A"),
    (50.288, 51.153, [("для вашего ребёнка", S, 44)], "A"),
    (51.153, 51.998, [("первое занятие", S, 48)], "A"),
    (51.998, 52.657, [("по математике", S, 50)], "A"),
]

SLOTS = {
    "A": (540, 1305, "mm"),
    "T": (540, 470, "mm"),
    "B": (540, 1470, "mm"),
}

# ---------------------------------------------------------------- графика
GX0, GX1, GY0, GY1 = 175, 905, 700, 1560  # зона графики карточки A (brand-kit §5)
GCX = (GX0 + GX1) // 2


def cl(v):
    return max(0.0, min(1.0, v))


def layer():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def txt(text, xy, size=54, kind="r", op=1.0, blue=False, anchor="mm", glow_r=None):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    col = BLUE if blue else WHITE
    gl = BLUE_GLOW if blue else WHITE
    return dict(text=text, font=font(kmap[kind], size), xy=xy, anchor=anchor,
                fill=col, glow=gl, glow_r=glow_r if glow_r else (34 if blue else 14),
                glow_a=0.9 if blue else 0.55, opacity=op)


def pop(t, t0, text, xy, size, kind="r"):
    """R5a — белое число резким попом за 2-3 кадра."""
    lt = t - t0
    if lt < 0:
        return []
    p = cl(lt / 0.10)
    sc = 0.6 + 0.4 * ease_out(p)
    return [txt(text, xy, max(8, int(size * sc)), kind, op=cl(lt / 0.08), glow_r=int(size * 0.16))]


def num_blue(t, t0, text, xy, size):
    """R5b — синее число: масштаб 0.55->1.0 с оверщутом, разряды слева направо."""
    lt = t - t0
    if lt < 0:
        return []
    dur = 0.38
    p = cl(lt / dur)
    e = ease_out(p)
    sc = 0.55 + 0.45 * e
    if 0.55 < p < 1.0:
        sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    n = len(text)
    shown = text[:max(1, math.ceil(n * cl(lt / (dur * 0.8))))]
    return [txt(shown, xy, max(8, int(size * sc)), op=cl(lt / 0.12), blue=True, glow_r=int(size * 0.20))]


GRAPH_MAXW = GX1 - GX0 - 60  # запас 30px внутрь с каждого края зоны графики


def fit_size(text, size, kind="r", maxw=GRAPH_MAXW):
    """Автоподбор кегля под ширину зоны графики — та же логика, что и у
    субтитров (сначала не превышаем ширину, кегль не задаётся на глаз)."""
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    w = font(kmap[kind], size).getlength(text)
    if w <= maxw:
        return size
    return max(24, int(size * maxw / w))


def grid_line(d, a, b, color=WHITE, width=6, alpha=235):
    d.line([a, b], fill=color + (alpha,), width=width)


# --- R4: разрез земли, глубина -------------------------------------------
def depth_ruler(p_build):
    """Вертикальная линейка глубины в зоне графики, засечки, подпись числа."""
    img = layer()
    d = ImageDraw.Draw(img)
    x = GCX - 120
    y0, y1 = GY0 + 40, GY0 + 40 + (GY1 - GY0 - 80) * ease_out(p_build)
    grid_line(d, (x, y0), (x, y1), WHITE, 6, 210)
    # засечки через 25% построенной длины
    ticks = 4
    for i in range(ticks + 1):
        ty = y0 + (y1 - y0) * i / ticks
        if ty > y1 + 1:
            continue
        grid_line(d, (x - 22, ty), (x + 22, ty), WHITE, 5, 190)
    # стрелка-наконечник внизу линейки
    if p_build > 0.05:
        ah = 22 * ease_out(p_build)
        d.polygon([(x - 16, y1 - ah), (x + 16, y1 - ah), (x, y1 + ah * 0.6)],
                  fill=WHITE + (220,))
    return img


def layers(kind, prm, lt, dur, t):
    L, T = [], []

    if kind == "depth1":
        L.append(depth_ruler(cl(lt / 0.9)))

    elif kind == "depth2":
        L.append(depth_ruler(1.0))
        T += pop(t, t - lt + 0.05, "3 км", (GCX + 90, GY0 + 260), 175)

    elif kind == "money_pop":
        sz = fit_size("900 000 ₽", 150)
        T += pop(t, t - lt + 0.05, "900 000 ₽", (GCX, (GY0 + GY1) // 2), sz)
        T.append(txt("в месяц", (GCX, (GY0 + GY1) // 2 + 150), 54, "s",
                     op=cl((lt - 0.35) / 0.3)))

    elif kind == "loss_pop":
        sz = fit_size("1 000 000 000 ₽", 92)
        T += num_blue(t, t - lt + 0.05, "1 000 000 000 ₽", (GCX, (GY0 + GY1) // 2), sz)
        T.append(txt("рублей в пустоту", (GCX, (GY0 + GY1) // 2 + 150), 50, "s",
                     op=cl((lt - 0.42) / 0.3)))

    elif kind == "wave_down":
        img = layer()
        d = ImageDraw.Draw(img)
        p = ease_out(cl(lt / 0.9))
        y0, y1 = GY0 + 30, GY0 + 30 + (GY1 - GY0 - 60) * p
        pts = []
        n = 40
        amp0 = 70
        for i in range(n + 1):
            yy = y0 + (y1 - y0) * i / n
            damp = amp0 * (1 - i / n * 0.3)
            xx = GCX + damp * math.sin(i * 0.9)
            pts.append((xx, yy))
        d.line(pts, fill=WHITE + (225,), width=7, joint="curve")
        if p > 0.02:
            d.ellipse([GCX - 14, y0 - 14, GCX + 14, y0 + 14], fill=WHITE + (240,))
        L.append(img)

    elif kind == "wave_up":
        img = layer()
        d = ImageDraw.Draw(img)
        # затухающий след вниз (память предыдущего кадра) + бегущий импульс вверх
        y_bot = GY0 + 30 + (GY1 - GY0 - 60)
        y_top = GY0 + 30
        n = 40
        pts_full = []
        for i in range(n + 1):
            yy = y_bot - (y_bot - y_top) * i / n
            damp = 70 * (1 - (n - i) / n * 0.3)
            xx = GCX + damp * math.sin(i * 0.9 + math.pi)
            pts_full.append((xx, yy))
        d.line(pts_full, fill=WHITE + (90,), width=6, joint="curve")
        p = ease_out(cl(lt / 0.9))
        k = max(1, int(len(pts_full) * p))
        d.line(pts_full[:k], fill=BLUE_GLOW + (235,), width=8, joint="curve")
        if k > 0:
            hx, hy = pts_full[k - 1]
            d.ellipse([hx - 15, hy - 15, hx + 15, hy + 15], fill=BLUE_GLOW + (245,))
        L.append(img)

    elif kind == "layers2":
        img = layer()
        d = ImageDraw.Draw(img)
        midy = (GY0 + GY1) // 2
        p = ease_out(cl(lt / 0.6))
        # верхний слой — вода (контур белым)
        d.rounded_rectangle([GX0, GY0 + 20, GX1, midy - 10], radius=14,
                            outline=WHITE + (int(210 * p),), width=6)
        # нижний слой — нефть (лайм-заливка, единственный второй акцент ролика)
        h2 = (GY1 - 20 - (midy + 10)) * p
        d.rounded_rectangle([GX0, GY1 - 20 - h2, GX1, GY1 - 20], radius=14,
                            fill=LIME + (int(150 * p),), outline=LIME + (230,), width=5)
        L.append(img)

    elif kind == "wireframe":
        img = layer()
        d = ImageDraw.Draw(img)
        cx, cy, s = GCX, (GY0 + GY1) // 2 + 20, 230
        # изометрический куб из 8 вершин, рёбра проявляются каскадом
        off = (0.5 * s, 0.28 * s)
        front = [(cx - s / 2, cy - s / 2), (cx + s / 2, cy - s / 2),
                 (cx + s / 2, cy + s / 2), (cx - s / 2, cy + s / 2)]
        back = [(x + off[0], y - off[1]) for x, y in front]
        edges = list(zip(front, [front[1], front[2], front[3], front[0]]))
        edges += list(zip(back, [back[1], back[2], back[3], back[0]]))
        edges += list(zip(front, back))
        n = len(edges)
        for i, (a, b) in enumerate(edges):
            ap = cl((lt - i * 0.045) / 0.28)
            if ap <= 0:
                continue
            e = ease_out(ap)
            mx, my = a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e
            d.line([a, (mx, my)], fill=WHITE + (225,), width=6)
        L.append(img)

    elif kind == "terms1":
        sz = fit_size("волновые уравнения", 78)
        T += pop(t, t - lt + 0.05, "волновые уравнения", (GCX, (GY0 + GY1) // 2), sz)

    elif kind == "terms2":
        sz1 = fit_size("преобразования фурье", 66)
        sz2 = fit_size("численные методы", 66)
        T += pop(t, t - lt + 0.05, "преобразования фурье", (GCX, (GY0 + GY1) // 2 - 90), sz1)
        T += pop(t, t - lt + 0.55, "численные методы", (GCX, (GY0 + GY1) // 2 + 90), sz2)

    return L, T

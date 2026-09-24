"""Визуальная система — реализация brand-kit.md. Все числа для холста 1080x1920."""
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1080, 1920
FPS = 30

# --- карточки (brand-kit.md §2) ---
CARD_A = (105, 238, 870, 1380)   # x, y, w, h   говорящая голова
CARD_B = (110, 614, 858, 620)    # вставка
R_A, R_B = 60, 50                # радиусы скругления
CENTER = (540, 925)

# --- палитра (brand-kit.md §4) ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (59, 75, 232)
BLUE_GLOW = (107, 135, 249)
LIME = (195, 219, 78)

# --- шрифты ---
SANS = "/System/Library/Fonts/SFNS.ttf"
SANS_IT = "/System/Library/Fonts/SFNSItalic.ttf"
SERIF_IT = "/System/Library/Fonts/Supplemental/Baskerville.ttc"
SERIF_IT_IDX = 2  # Baskerville Italic

_fc = {}


def font(kind, size):
    key = (kind, size)
    if key in _fc:
        return _fc[key]
    if kind == "sans":
        f = ImageFont.truetype(SANS, size)
        try:
            f.set_variation_by_axes([500])   # SF Pro variable: вес ~Medium
        except Exception:
            pass
    elif kind == "sans_it":
        f = ImageFont.truetype(SANS_IT, size)
        try:
            f.set_variation_by_axes([500])
        except Exception:
            pass
    elif kind == "serif_it":
        f = ImageFont.truetype(SERIF_IT, size, index=SERIF_IT_IDX)
    else:
        raise ValueError(kind)
    _fc[key] = f
    return f


# --- скруглённая маска ---
_mc = {}


def rounded_mask(w, h, r, ss=4):
    key = (w, h, r)
    if key in _mc:
        return _mc[key]
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=r * ss, fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    _mc[key] = m
    return m


def place_card(canvas, img, rect, radius):
    """Вписать img в карточку rect со скруглением (crop-to-fill)."""
    x, y, w, h = rect
    src = fit_cover(img, w, h)
    canvas.paste(src, (x, y), rounded_mask(w, h, radius))
    return canvas


def fit_cover(img, w, h):
    iw, ih = img.size
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    img = img.resize((nw, nh), Image.LANCZOS)
    return img.crop(((nw - w) // 2, (nh - h) // 2, (nw - w) // 2 + w, (nh - h) // 2 + h))


# --- сетка-холст (brand-kit / shot-recipes R4) ---
def grid_canvas(w, h, phase=0.0, cell=168, line=3, alpha=200, seed=7, wobble=1.0):
    """Кривая «от руки» сетка на чёрном, с радиальным вигнетом.

    wobble — множитель кривизны линий. По умолчанию 1.0 (как было).
    Уменьшать только там, где по узлам сетки строится предметная графика:
    при полной кривизне линия уходит от номинального пересечения до ~14px,
    и «вершина точно в узле» перестаёт быть точной (ролик 12, формула Пика).
    """
    ss = 2
    g = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(g)
    lw = max(1, int(round(line * ss)))
    amp = 9 * ss * wobble  # амплитуда кривизны
    step = 14 * ss

    def wob(i, t, k):
        return (math.sin(t * 0.011 + i * 1.7 + phase * 0.9 + k) * amp
                + math.sin(t * 0.0031 + i * 0.7 - phase * 0.6 + k * 2) * amp * 0.6)

    n = int(w / cell) + 3
    for i in range(-1, n):
        x0 = (i * cell - cell * 0.5) * ss + (seed * 13 % 40) * ss / 4
        pts = []
        for t in range(0, h * ss + step, step):
            pts.append((x0 + wob(i, t, 0.0), t))
        d.line(pts, fill=alpha, width=lw, joint="curve")
    m = int(h / cell) + 3
    for j in range(-1, m):
        y0 = (j * cell - cell * 0.5) * ss + (seed * 7 % 40) * ss / 4
        pts = []
        for t in range(0, w * ss + step, step):
            pts.append((t, y0 + wob(j + 50, t, 1.3)))
        d.line(pts, fill=alpha, width=lw, joint="curve")
    g = g.resize((w, h), Image.LANCZOS)

    # радиальный вигнет — эталон: видимая часть ~75% ширины / ~88% высоты карточки
    vg = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(vg)
    ex, ey = w * 0.22, h * 0.26
    vd.ellipse([ex, ey, w - ex, h - ey], fill=255)
    vg = vg.filter(ImageFilter.GaussianBlur(min(w, h) * 0.15))
    vg = vg.point(lambda v: min(255, int(v * 2.8)))
    g = Image.composite(g, Image.new("L", (w, h), 0), vg)

    out = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    out.putalpha(255)
    px = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    px.putalpha(g)
    out.alpha_composite(px)
    return out


# --- текст со свечением ---
def text_layer(size, items):
    """items: список dict(text, font, xy, anchor, fill, glow, glow_r, glow_a, opacity, reveal)
    reveal: None | ('wipe', x_px)  — обрезать всё правее x_px (печатная машинка)"""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    for it in items:
        sub = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(sub)
        d.text(it["xy"], it["text"], font=it["font"], fill=it["fill"] + (255,),
               anchor=it.get("anchor", "mm"))
        gr = it.get("glow_r", 14)
        if gr:
            a = sub.split()[3]
            blur = a.filter(ImageFilter.GaussianBlur(gr))
            gl = Image.new("RGBA", size, it.get("glow", it["fill"]) + (0,))
            gl.putalpha(blur.point(lambda v: int(v * it.get("glow_a", 0.55))))
            comp = Image.new("RGBA", size, (0, 0, 0, 0))
            comp.alpha_composite(gl)
            comp.alpha_composite(sub)
            sub = comp
        rev = it.get("reveal")
        if rev and rev[0] == "wipe":
            m = Image.new("L", size, 0)
            ImageDraw.Draw(m).rectangle([0, 0, int(rev[1]), size[1]], fill=255)
            m = m.filter(ImageFilter.GaussianBlur(1.2))
            sub.putalpha(Image.composite(sub.split()[3], Image.new("L", size, 0), m))
        op = it.get("opacity", 1.0)
        if op < 1.0:
            sub.putalpha(sub.split()[3].point(lambda v: int(v * op)))
        layer.alpha_composite(sub)
    return layer


def measure(text, f):
    b = f.getbbox(text)
    return b[2] - b[0], b[3] - b[1]


def char_offsets(text, f):
    """x-смещения границ символов относительно начала строки."""
    out = [0.0]
    for i in range(1, len(text) + 1):
        out.append(f.getlength(text[:i]))
    return out


def ease_out(t):
    return 1 - (1 - t) ** 3

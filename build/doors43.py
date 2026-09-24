"""Инфографика ролика 10 (слот 43): сцена «три двери» — рецепт R10 на сетке.

Одна сцена на все графические планы; state задаёт, что в ней показано
(см. storyboard43.DOOR_STATES). Всё рисуется в координатах холста 1080x1920,
в зоне графики brand-kit §5: y 700…1560, x 175…905.

gfx(t, shot) -> (слой-графика RGBA на весь холст, список текстовых items для чисел).
Текст чисел — тоже графика: в qa43.py он идёт в слой графики, не в слой субтитров.
"""
import math
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from style import W, H, WHITE, BLUE, BLUE_GLOW, LIME, ease_out, font, SANS

# v3 (2026-09-24): цифры графики — тем же шрифтом, что субтитры (SF Pro Expanded Black),
# без свечения; синий — светлый край рабочего диапазона brand-kit (#5B7CFF), чтобы без
# свечения читался на чёрном
NUM_BLUE = (91, 124, 255)
_wf = {}


def wide(size):
    if size not in _wf:
        f = ImageFont.truetype(SANS, size)
        f.set_variation_by_axes([150, 28, 400, 1000])   # Width, opsz, GRAD, Weight
        _wf[size] = f
    return _wf[size]

DOOR_W, DOOR_H = 200, 400
TOP = 710                             # v3: сцена поднята на 50px — внизу теперь субтитры
BOT = TOP + DOOR_H                    # 1110
CXS = [295, 540, 785]                 # центры дверей
MARK_Y = 1146                         # лаймовая метка выбора под дверью 1
BR_Y = 1162                           # скобка под дверями 2 и 3
NUM_Y = 1268                          # числа под дверями
NUM_SIZE = 84
BR_X0, BR_X1 = CXS[1] - DOOR_W // 2 + 10, CXS[2] + DOOR_W // 2 - 10
BR_CX = (BR_X0 + BR_X1) // 2
SS = 2                                # суперсэмплинг рисунка

_goat = None


def goat_img(width=176):
    """Белый силуэт козы: альфа эмодзи Apple Color Emoji, без его цветов."""
    global _goat
    if _goat is None:
        f = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", 160)
        im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((20, 20), "🐐", font=f, embedded_color=True)
        a = im.split()[3]
        a = a.crop(a.getbbox())
        h = int(a.height * width / a.width)
        a = a.resize((width, h), Image.LANCZOS)
        g = Image.new("RGBA", a.size, (255, 255, 255, 0))
        g.putalpha(a)
        _goat = g
    return _goat


def _glow(layer, radius=12, strength=0.40, color=(255, 255, 255)):
    a = layer.split()[3]
    blur = a.filter(ImageFilter.GaussianBlur(radius))
    gl = Image.new("RGBA", layer.size, color + (0,))
    gl.putalpha(blur.point(lambda v: int(v * strength)))
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(gl)
    out.alpha_composite(layer)
    return out


def _fade(layer, op):
    if op >= 1.0:
        return layer
    layer = layer.copy()
    layer.putalpha(layer.split()[3].point(lambda v: int(v * max(0.0, op))))
    return layer


def _door(i, open_p=0.0, goat_a=1.0, frame_a=1.0, hl=False, appear=1.0):
    """Одна дверь на своём прозрачном слое холста. open_p 0..1 — створка распахнута."""
    lay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    cx = CXS[i]
    sc = 0.90 + 0.10 * appear
    w, h = DOOR_W * sc, DOOR_H * sc
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = (TOP + BOT) / 2 - h / 2, (TOP + BOT) / 2 + h / 2
    S = lambda *p: [v * SS for v in p]
    fa = int(255 * frame_a)
    lw = (9 if hl else 6) * SS

    # проём: тёмное нутро, коза видна, когда створка отходит
    d.rounded_rectangle(S(x0, y0, x1, y1), radius=14 * SS, fill=(16, 16, 20, 255))
    if open_p > 0:
        g = goat_img(int(176 * sc))
        g = _fade(g, goat_a).resize((g.width * SS, g.height * SS), Image.LANCZOS)
        lay.alpha_composite(g, (int((cx - g.width / SS / 2) * SS),
                                int((y1 - 26 - g.height / SS) * SS)))
        d = ImageDraw.Draw(lay)

    # створка на петлях слева: сужается и уходит в перспективу к зрителю
    e = ease_out(open_p)
    lwid = w * (1 - 0.80 * e)
    sk = 22 * e
    leaf = [(x0, y0), (x0 + lwid, y0 - sk), (x0 + lwid, y1 + sk), (x0, y1)]
    leaf_fill = (26, 26, 28, 255) if hl else (0, 0, 0, 255)
    d.polygon([(px * SS, py * SS) for px, py in leaf], fill=leaf_fill)
    d.line([(px * SS, py * SS) for px, py in leaf + [leaf[0]]],
           fill=(255, 255, 255, fa), width=lw, joint="curve")
    if e < 0.35:
        k = 1 - e / 0.35
        ins = 26 * sc
        d.rounded_rectangle(S(x0 + ins, y0 + ins + 70 * sc, x1 - ins, y1 - ins),
                            radius=8 * SS, outline=(255, 255, 255, int(110 * k * frame_a)),
                            width=3 * SS)
        kr = 9 * sc
        d.ellipse(S(x1 - 34 * sc - kr, (y0 + y1) / 2 - kr, x1 - 34 * sc + kr, (y0 + y1) / 2 + kr),
                  fill=(255, 255, 255, int(230 * k * frame_a)))
        f = wide(int(54 * sc) * SS)
        d.text((cx * SS, (y0 + 58 * sc) * SS), str(i + 1), font=f, anchor="mm",
               fill=(255, 255, 255, int(255 * k * frame_a)))
    # контур проёма поверх (когда створка открыта, проём остаётся очерченным)
    if open_p > 0:
        d.rounded_rectangle(S(x0, y0, x1, y1), radius=14 * SS,
                            outline=(255, 255, 255, int(150 * frame_a)), width=4 * SS)
    lay = lay.resize((W, H), Image.LANCZOS)
    return _fade(lay, appear)


def _mark(p):
    """Лаймовая метка выбора под дверью 1 — единственный элемент #C3DB4E в ролике."""
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if p <= 0:
        return lay
    e = ease_out(min(1.0, p))
    half = 75 * e
    ImageDraw.Draw(lay).rounded_rectangle([CXS[0] - half, MARK_Y - 6, CXS[0] + half, MARK_Y + 6],
                                          radius=6, fill=LIME + (255,))
    return _glow(lay, 10, 0.55, LIME)


def _bracket(p, op=1.0):
    lay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    if p <= 0 or op <= 0:
        return lay.resize((W, H))
    e = ease_out(min(1.0, p))
    d = ImageDraw.Draw(lay)
    half = (BR_X1 - BR_X0) / 2 * e
    a = int(235 * op)
    pts = [(BR_CX - half, BR_Y - 22), (BR_CX - half, BR_Y), (BR_CX + half, BR_Y), (BR_CX + half, BR_Y - 22)]
    d.line([(x * SS, y * SS) for x, y in pts], fill=(255, 255, 255, a), width=5 * SS, joint="curve")
    d.line([(BR_CX * SS, BR_Y * SS), (BR_CX * SS, (BR_Y + 18) * SS)], fill=(255, 255, 255, a), width=5 * SS)
    return _glow(lay.resize((W, H), Image.LANCZOS), 10, 0.35)


def num_item(t, t0, text, xy, size, blue, max_w=None):
    """R5b (синее): 0.38с, 0.55→1.0 с оверщутом ~3%, разряды слева направо.
    R5a (белое): 80мс, 0.6→1.0, без оверщута. v3: широкий Black, без свечения."""
    lt = t - t0
    if lt < 0:
        return []
    if max_w:
        size = min(size, int(size * max_w / wide(size).getlength(text)))
    if blue:
        p = min(1.0, lt / 0.38)
        sc = 0.55 + 0.45 * ease_out(p)
        if 0.55 < p < 1.0:
            sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    else:
        p = min(1.0, lt / 0.08)
        sc = 0.6 + 0.4 * ease_out(p)
    f = wide(max(8, int(size * sc)))
    it = dict(text=text, font=f, xy=xy, anchor="mm", fill=NUM_BLUE if blue else WHITE,
              glow_r=0, opacity=min(1.0, lt / (0.12 if blue else 0.06)))
    if blue and p < 1.0:
        wdt = f.getlength(text)
        it["reveal"] = ("wipe", xy[0] - wdt / 2 + wdt * min(1.0, p / 0.8) + 4)
    return [it]


_cache = {}


def _static(key, fn):
    if key not in _cache:
        if len(_cache) > 30:
            _cache.clear()
        _cache[key] = fn()
    return _cache[key]


def doors_gfx(t, shot):
    t0, t1, kind, prm = shot
    st = prm["state"]
    lt = t - t0
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    nums = []

    if st == "intro":
        for i in range(3):
            ap = min(1.0, max(0.0, (lt - i * 0.08) / 0.30))
            ap = ease_out(ap)
            q = round(ap, 2)
            lay.alpha_composite(_static(("in", i, q), lambda: _glow(_door(i, appear=q))))
        return lay, nums

    open3, goat_a, frame3, hl2 = 0.0, 1.0, 1.0, False
    if st == "open3":
        open3 = round(min(1.0, max(0.0, (lt - 0.05) / 0.45)), 2)
    if st == "transfer":
        open3, goat_a, frame3 = 1.0, 0.45, 0.6
        hl2 = t >= prm["t_move"] + 0.40
    for i in range(3):
        if i == 2:
            key = ("d", 2, open3, goat_a, frame3)
            lay.alpha_composite(_static(key, lambda: _glow(_door(2, open3, goat_a, frame3))))
        elif i == 1:
            lay.alpha_composite(_static(("d", 1, hl2), lambda: _glow(_door(1, hl=hl2))))
        else:
            lay.alpha_composite(_static(("d", 0), lambda: _glow(_door(0))))

    if st == "pick":
        pm = round(min(1.0, max(0.0, (t - prm["t_pick"]) / 0.30)), 2)
        lay.alpha_composite(_static(("m", pm), lambda: _mark(pm)))
    else:
        lay.alpha_composite(_static(("m", 1.0), lambda: _mark(1.0)))

    if st == "third":
        nums += num_item(t, prm["t_num"], "1/3", (CXS[0], NUM_Y), NUM_SIZE, False)
    if st == "two_thirds":
        nums += num_item(t, t0 - 1.0, "1/3", (CXS[0], NUM_Y), NUM_SIZE, False)
        bp = round(min(1.0, lt / 0.30), 2)
        lay.alpha_composite(_static(("b", bp, 1.0), lambda: _bracket(bp)))
        nums += num_item(t, prm["t_num"], "2/3", (BR_CX, NUM_Y), NUM_SIZE, True)
    if st == "transfer":
        nums += num_item(t, t0 - 1.0, "1/3", (CXS[0], NUM_Y), NUM_SIZE, False)
        mp = ease_out(min(1.0, max(0.0, (t - prm["t_move"]) / 0.40)))
        bo = round(1.0 - min(1.0, max(0.0, (t - prm["t_move"]) / 0.25)), 2)
        if bo > 0:
            lay.alpha_composite(_static(("b", 1.0, bo), lambda: _bracket(1.0, bo)))
        x = BR_CX + (CXS[1] - BR_CX) * mp
        nums += num_item(t, t0 - 1.0, "2/3", (x, NUM_Y), NUM_SIZE, True)
    return lay, nums

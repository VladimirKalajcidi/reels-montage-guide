"""Сборка ролика. Речь не резана — резы только по картинке."""
import os, sys, math, subprocess, random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import graph as G
from storyboard import SHOTS, CAPS, SLOTS, DUR
from storyboard import layers as video_layers

BUILD = os.path.dirname(os.path.abspath(__file__))
SRC = "/Users/vladimirkalajcidi/reels_good2/videos/4/source.mov"
VIDEO_DIR = os.path.dirname(SRC)
A1 = f"{BUILD}/assets/aroll_4_A1.mp4"
A2 = f"{BUILD}/assets/aroll_4_A2.mp4"
OUT = "/Users/vladimirkalajcidi/reels_good2/videos/4/fourcolor_edit.mp4"

GRADE = ("eq=gamma=1.18:contrast=1.06:saturation=1.10:brightness=0.02,"
         "colorbalance=rm=0.04:gm=0.008:bm=-0.03,unsharp=5:5:0.30")
FRAMINGS = {"A1": (720, 1142, 0, 120), "A2": (560, 888, 80, 227)}

CX, CY, R_A_ = CARD_A[0], CARD_A[1], R_A
CW, CH = CARD_A[2], CARD_A[3]
NF = int(round(DUR * FPS))


# ---------------------------------------------------------------- подготовка
def prep_aroll():
    for name, out in (("A1", A1), ("A2", A2)):
        if os.path.exists(out):
            continue
        w, h, x, y = FRAMINGS[name]
        vf = f"hflip,crop={w}:{h}:{x}:{y},scale={CW}:{CH}:flags=lanczos,{GRADE}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14",
                        "-preset", "medium", "-pix_fmt", "yuv420p", out], check=True)
        print("готово:", out)


BX, BY, BW, BH = CARD_B
STOCK_DIR = f"{VIDEO_DIR}/stock"
STOCK_PREP_DIR = f"{STOCK_DIR}/prepared"
STOCK_CAP_CARD = (105, 388, 870, 196)


def prep_stock():
    """Каждая стоковая вставка → карточка B 858x620, кроп до 1.385:1, притемнение ~25–30%."""
    os.makedirs(STOCK_PREP_DIR, exist_ok=True)
    out = {}
    for i, (t0, t1, kind, prm) in enumerate(SHOTS):
        if kind != "stock":
            continue
        src = f"{STOCK_DIR}/stock_{prm['clip']}.mp4"
        dst = f"{STOCK_PREP_DIR}/sb4_{i}_{prm['clip']}.mp4"
        if not os.path.exists(dst):
            vf = ("crop='min(iw,ih*1.3839)':'min(ih,iw/1.3839)',"
                  f"scale={BW}:{BH}:flags=lanczos,"
                  "eq=brightness=-0.22:contrast=1.08:saturation=0.90,fps=30")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", str(prm.get("ss", 0)), "-t", f"{t1 - t0 + 0.2:.3f}",
                            "-i", src,
                            "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15",
                            "-preset", "medium", "-pix_fmt", "yuv420p", dst], check=True)
            print("готово:", dst)
        out[i] = dst
    return out


def ghost_masks():
    """Маска силуэта для приёма «слово за спикером» (R6). GrabCut на половинном разрешении."""
    segs = [(s[0], s[1]) for s in SHOTS if s[3].get("ghost")]
    need = set()
    for t0, t1 in segs:
        need |= set(range(int(t0 * FPS), int(math.ceil(t1 * FPS))))
    if not need:
        return {}
    cache = f"{BUILD}/assets/ghost_masks.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return {int(k): z[k] for k in z.files}
    cap = cv2.VideoCapture(A1)
    hw, hh = CW // 2, CH // 2
    masks, prev = {}, None
    f = -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        f += 1
        if f not in need:
            if prev is not None and (f - 1) in need:
                prev = None
            continue
        _, img = cap.retrieve()
        small = cv2.resize(img, (hw, hh), interpolation=cv2.INTER_AREA)
        bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
        if prev is None:
            m = np.zeros((hh, hw), np.uint8)
            rect = (int(hw * .06), int(hh * .22), int(hw * .90), int(hh * .78))
            cv2.grabCut(small, m, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        else:
            m = prev.copy()
            cv2.grabCut(small, m, None, bgd, fgd, 1, cv2.GC_INIT_WITH_MASK)
        prev = m
        b = np.where((m == 2) | (m == 0), 0, 255).astype(np.uint8)
        b = cv2.GaussianBlur(b, (0, 0), 2.0)
        masks[f] = b
        if len(masks) % 30 == 0:
            print("  маска", len(masks), "/", len(need))
    cap.release()
    np.savez_compressed(cache, **{str(k): v for k, v in masks.items()})
    return masks


# ---------------------------------------------------------------- текст
def line_items(runs, xy, anchor, reveal_chars=None, opacity=1.0):
    """runs: [(text, kind, size)] kind 'r'|'i'|'s'. Раскладка в одну строку, общая печать."""
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    fonts = [font(kmap[k], sz) for (_, k, sz) in runs]
    widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
    total = sum(widths)
    x, y, an = xy[0], xy[1], anchor
    if an[0] == "m":
        sx = x - total / 2
    elif an[0] == "r":
        sx = x - total
    else:
        sx = x
    items, cum, cx = [], 0, sx
    for i, (t, k, sz) in enumerate(runs):
        f = fonts[i]
        it = dict(text=t, font=f, xy=(cx, y), anchor="l" + an[1],
                  fill=WHITE, glow=WHITE,
                  glow_r=int(10 + sz * 0.10), glow_a=0.55 if k != "s" else 0.62,
                  opacity=opacity)
        if reveal_chars is not None:
            n = len(t)
            done = reveal_chars - cum
            if done <= 0:
                cum += n
                cx += widths[i]
                continue
            if done < n:
                it["reveal"] = ("wipe", cx + f.getlength(t[:int(done)]) + 2)
        cum += len(t)
        items.append(it)
        cx += widths[i] + 10
    return items


MAXW = 960  # 1080 - 2*60


def fit_runs(runs, maxw=MAXW):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    w = sum(font(kmap[k], sz).getlength(txt) for (txt, k, sz) in runs)
    if w <= maxw:
        return runs
    f = maxw / w
    return [(txt, k, max(22, int(sz * f))) for (txt, k, sz) in runs]


def _shot_kind_at(t):
    for t0, t1, kind, _ in SHOTS:
        if t0 <= t < t1:
            return kind
    return SHOTS[-1][2]


def _build_blocks():
    """Фразы собираются в блоки по 2-3. Внутри блока они НЕ исчезают, а копятся
    и разбрасываются по кадру; блок сбрасывается целиком на новой мысли."""
    blocks, cur = [], []
    for i, (t0, t1, runs, slot) in enumerate(CAPS):
        newblock = False
        if cur:
            pi = cur[-1]
            if _shot_kind_at(CAPS[pi][0]) != _shot_kind_at(t0):
                newblock = True
            elif t0 - CAPS[pi][1] > 0.30:
                newblock = True
            elif len(cur) >= 2:
                newblock = True
            elif type(CAPS[pi][3]) is not type(slot):
                newblock = True
        if newblock:
            blocks.append(cur); cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    info = {}
    for bid, b in enumerate(blocks):
        end = max(CAPS[i][1] for i in b)
        for k, i in enumerate(b):
            info[i] = (k, end, len(b), bid)
    return info


BLOCKS = _build_blocks()

# разброс внутри блока: (dx, dy, множитель кегля). Межстрочный шаг 62px.
SCATTER_A = [(-58, -62, 0.98), (55, 0, 1.08), (-30, 62, 0.94)]   # поверх лица
SCATTER_C = [(0, 0, 1.0), (18, 58, 1.04), (-24, 116, 0.96)]      # произвольная точка
# каждый третий блок — спокойный: ровная колонка без разброса и без игры кеглем
SIMPLE_A = [(0, -62, 1.0), (0, 0, 1.0), (0, 62, 1.0)]
SIMPLE_C = [(0, 0, 1.0), (0, 58, 1.0), (0, 116, 1.0)]
DIM = [1.0, 0.74, 0.58]                                          # чем старше — тем тише
PAD = 30                                                          # отступ от края карточки


def _eff_size(runs, ks=1.0):
    return max(r[2] for r in runs) * ks


def _block_layout():
    """Раскладка блока считается один раз. Шаг между строками зависит от их кегля,
    поэтому крупная строка не наезжает на предыдущую."""
    groups = {}
    for i, (k, bend, blen, bid) in BLOCKS.items():
        groups.setdefault(bid, []).append(i)
    lay = {}
    for bid, members in groups.items():
        members.sort()
        calm = (bid % 3 == 0)
        slot0 = CAPS[members[0]][3]
        flag = isinstance(slot0, str) and slot0 in ("T", "B")
        centered = isinstance(slot0, str) and not flag
        tbl = (SIMPLE_A if calm else SCATTER_A) if centered else (SIMPLE_C if calm else SCATTER_C)

        ys, prev, acc = [], None, 0.0
        for k, i in enumerate(members):
            sz = _eff_size(CAPS[i][2], 1.0 if flag else tbl[min(k, 2)][2])
            if k:
                acc += max(74.0, (prev + sz) * 0.92)
            ys.append(acc)
            prev = sz
        total = ys[-1]
        first_h = _eff_size(CAPS[members[0]][2], 1.0 if flag else tbl[0][2]) * 0.62
        last_h = _eff_size(CAPS[members[-1]][2], 1.0 if flag else tbl[min(len(members) - 1, 2)][2]) * 0.62

        for k, i in enumerate(members):
            ks = 1.0 if flag else tbl[min(k, 2)][2]
            dx = 0 if flag else tbl[min(k, 2)][0]
            dy = ys[k] - (total / 2 if centered else 0)
            lay[i] = dict(dx=dx, dy=dy, ks=ks, k=k,
                          top=-(total / 2 if centered else 0) - first_h,
                          bot=-(total / 2 if centered else 0) + total + last_h)
    return lay


def clamp_to_card(runs, x, y, an, card):
    """Строка обязана целиком лежать внутри карточки. Сначала двигаем, потом ужимаем."""
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    cx0, cx1 = card[0] + PAD, card[0] + card[2] - PAD
    cy0, cy1 = card[1] + PAD, card[1] + card[3] - PAD
    avail = cx1 - cx0
    w = sum(font(kmap[k], sz).getlength(txt) for (txt, k, sz) in runs)
    max_sz = max(sz for (_, _, sz) in runs)
    max_h = max_sz * 1.24
    height_avail = cy1 - cy0
    if w > avail or max_h > height_avail:
        f = min(avail / max(1, w), height_avail / max(1, max_h))
        runs = [(txt, k, max(22, int(sz * f))) for (txt, k, sz) in runs]
        w = sum(font(kmap[k], sz).getlength(txt) for (txt, k, sz) in runs)
    left = x - w / 2 if an[0] == "m" else (x - w if an[0] == "r" else x)
    left = min(max(left, cx0), cx1 - w)
    half = max(font(kmap[k], sz).size for (_, k, sz) in runs) * 0.62
    y = min(max(y, cy0 + half), cy1 - half)
    return runs, (left, y), "l" + an[1]


BLOCK_LAY = _block_layout()


def caption_items(t, shot_kind=None):
    graphic_no_caps = {
        "test99", "less1", "rare", "million", "sick100", "found99",
        "false1", "false9999", "sum", "total10098", "only99", "fraction",
        "final_less", "bayes_name", "base_rate", "repeat_check",
    }
    if shot_kind in graphic_no_caps:
        return []
    out = []
    if shot_kind == "stock":
        active = []
        for idx, (t0, t1, runs, slot) in enumerate(CAPS):
            k, bend, blen, bid = BLOCKS[idx]
            if t0 <= t < bend:
                active.append(idx)
        active = active[-2:]
        y0 = 462 if len(active) > 1 else 502
        for pos, idx in enumerate(active):
            t0, t1, runs, slot = CAPS[idx]
            newer = len(active) - pos - 1
            dim = DIM[min(newer, len(DIM) - 1)]
            runs = [(txt, kk, min(sz, 62 if kk != "s" else 72)) for (txt, kk, sz) in runs]
            runs, xy, an = clamp_to_card(runs, 540, y0 + pos * 70, "mm", STOCK_CAP_CARD)
            lt = t - t0
            nchars = sum(len(r[0]) for r in runs)
            type_dur = min(0.55, max(0.28, nchars * 0.030))
            rc = nchars if lt >= type_dur else nchars * (lt / type_dur)
            op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
            out += line_items(runs, xy, an, reveal_chars=rc, opacity=op)
        return out
    card = STOCK_CAP_CARD if shot_kind == "stock" else CARD_A
    for idx, (t0, t1, runs, slot) in enumerate(CAPS):
        k, bend, blen, bid = BLOCKS[idx]
        if not (t0 <= t < bend):
            continue
        # сколько фраз блока уже появилось ПОСЛЕ этой — настолько она приглушается
        start = idx - k
        newer = sum(1 for j in range(idx + 1, start + blen) if CAPS[j][0] <= t)
        dim = DIM[min(newer, len(DIM) - 1)]
        calm = (bid % 3 == 0)

        L = BLOCK_LAY[idx]
        if shot_kind == "stock":
            base_x, base_y, an = 540, 502, "mm"
        elif isinstance(slot, str) and slot in ("T", "B"):
            base_x, base_y, an = CARD_A[0] + 62, 372, "lm"
        elif isinstance(slot, str):
            base_x, base_y, an = SLOTS[slot][0], SLOTS[slot][1], SLOTS[slot][2]
        else:
            base_x, base_y, an = slot[0], slot[1], slot[2]

        # блок не влезает в карточку — двигаем его ЦЕЛИКОМ, а не отдельную строку
        lo, hi = card[1] + PAD, card[1] + card[3] - PAD
        shift = 0.0
        if base_y + L["bot"] > hi:
            shift = hi - (base_y + L["bot"])
        if base_y + L["top"] + shift < lo:
            shift = lo - (base_y + L["top"])
        xy = (base_x + L["dx"], base_y + L["dy"] + shift)
        ks = L["ks"]

        runs = [(txt, kk, max(24, int(sz * ks))) for (txt, kk, sz) in runs]
        if shot_kind == "stock":
            runs = [(txt, kk, min(sz, 62 if kk != "s" else 72)) for (txt, kk, sz) in runs]
        runs, xy, an = clamp_to_card(runs, xy[0], xy[1], an, card)
        lt = t - t0
        nchars = sum(len(r[0]) for r in runs)
        type_dur = min(0.55, max(0.28, nchars * 0.030))
        rc = nchars if lt >= type_dur else nchars * (lt / type_dur)
        op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
        out += line_items(runs, xy, an, reveal_chars=rc, opacity=op)
    return out


def num_items(t, t0, digits, xy, size, blue=True, prog_digits=True):
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


def pop_items(t, t0, text, xy, size, kind="s"):
    """R5a — белое число/слово: резкий поп за 2-3 кадра."""
    lt = t - t0
    if lt < 0:
        return []
    p = min(1.0, lt / (3 / FPS))
    sc = 0.6 + 0.4 * ease_out(p)
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    f = font(kmap[kind], max(8, int(size * sc)))
    return [dict(text=text, font=f, xy=xy, anchor="mm", fill=WHITE, glow=WHITE,
                 glow_r=int(size * 0.16), glow_a=0.7, opacity=min(1.0, lt / 0.08))]


def label_item(text, xy, size=50, kind="r", opacity=1.0, blue=False, anchor="mm"):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    col = BLUE if blue else WHITE
    gl = BLUE_GLOW if blue else WHITE
    return dict(text=text, font=font(kmap[kind], size), xy=xy, anchor=anchor,
                fill=col, glow=gl, glow_r=30 if blue else 14,
                glow_a=0.9 if blue else 0.55, opacity=opacity)


def progress_bar(x, y, w, h, p, color=BLUE, label=None):
    p = max(0.0, min(1.0, p))
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle([x, y, x + w, y + h], radius=18, outline=WHITE + (110,), width=3)
    if p > 0:
        d.rounded_rectangle([x, y, x + max(h, w * p), y + h], radius=18,
                            fill=color + (225,))
    if label:
        d.text((x + w / 2, y + h / 2), label, font=font("sans", 44),
               fill=WHITE + (230,), anchor="mm")
    return lay.filter(ImageFilter.GaussianBlur(0.15))


def pill_grid(total=80, active=1, blue_active=True, y0=745):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    cols, step, r = 10, 54, 15
    x0 = 540 - (cols - 1) * step / 2
    for i in range(total):
        row, col = divmod(i, cols)
        x, y = x0 + col * step, y0 + row * step
        if i < active:
            fill = BLUE + (255,) if blue_active else WHITE + (235,)
            outline = BLUE + (255,) if blue_active else WHITE + (220,)
        else:
            fill = WHITE + (38,)
            outline = WHITE + (78,)
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=2)
    return lay.filter(ImageFilter.GaussianBlur(0.1))


def equation_layer(items):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for txt, xy, size, col in items:
        color = BLUE if col == "blue" else WHITE
        d.text(xy, txt, font=font("sans", size), fill=color + (240,), anchor="mm")
    return lay.filter(ImageFilter.GaussianBlur(0.1))


_gfc = {}


def ghost_font(word, target=800):
    """Кегль под ширину полосы наверху кадра."""
    if word in _gfc:
        return _gfc[word]
    sz = 150
    for _ in range(40):
        w = font("sans", sz).getlength(word)
        if abs(w - target) < 12:
            break
        sz = max(60, min(300, int(sz * target / w)))
    _gfc[word] = font("sans", sz)
    return _gfc[word]


# ---------------------------------------------------------------- графика планов
# Ролик 4 («четыре краски и ошибка Кэмпе») рисует свою графику целиком в storyboard.py
# (модуль maps.py — карта Вороного, цепи Кэмпе, таймлайн). render.py её не дублирует,
# а просто делегирует storyboard.layers(), которая уже возвращает (L, T) в нужном формате.
def shot_layers(kind, prm, lt, dur, t):
    return video_layers(kind, prm, lt, dur, t)


def _unused_shot_layers(kind, prm, lt, dur, t):
    """Возвращает (list_of_RGBA_layers, list_of_text_items)."""
    L, T = [], []
    if kind == "test99":
        L.append(progress_bar(230, 860, 620, 72, ease_out(min(1, lt / 0.45)), BLUE, "точность теста"))
        T += num_items(t, t - lt + 0.12, "99%", (540, 1045), 190)
    elif kind == "less1":
        L.append(progress_bar(245, 1080, 590, 58, 0.01 * ease_out(min(1, lt / 0.5)), BLUE, None))
        T.append(label_item("на самом деле", (540, 760), 56))
        T += num_items(t, t - lt + 0.15, "< 1%", (540, 900), 205)
    elif kind == "rare":
        L.append(pill_grid(80, 1, True, 760))
        T += num_items(t, t - lt + 0.15, "1", (410, 1245), 150)
        T.append(label_item("из 10000", (585, 1255), 62, "s"))
    elif kind == "million":
        L.append(pill_grid(80, 80, False, 710))
        T += num_items(t, t - lt + 0.15, "1 000 000", (540, 1255), 140)
    elif kind == "sick100":
        L.append(pill_grid(80, 1, True, 720))
        T += num_items(t, t - lt + 0.15, "100", (540, 1050), 190)
        T.append(label_item("реально больных", (540, 1250), 58, "s"))
    elif kind == "found99":
        L.append(progress_bar(230, 870, 620, 72, ease_out(min(1, lt / 0.55)), BLUE, "из 100 больных"))
        T += num_items(t, t - lt + 0.18, "99", (540, 1080), 185)
        T.append(label_item("тест нашёл верно", (540, 1250), 54))
    elif kind == "false1":
        L.append(progress_bar(250, 920, 580, 58, 0.01 * ease_out(min(1, lt / 0.5)), BLUE, None))
        T += num_items(t, t - lt + 0.15, "1%", (540, 1090), 185)
        T.append(label_item("ошибка среди здоровых", (540, 1270), 50, "s"))
    elif kind == "false9999":
        L.append(pill_grid(80, 80, False, 680))
        T += num_items(t, t - lt + 0.15, "9999", (540, 1030), 185)
        T.append(label_item("ложных тревог", (540, 1240), 54, "r"))
    elif kind == "sum":
        T += num_items(t, t - lt + 0.10, "99", (540, 800), 158, True, False)
        T.append(label_item("реальные", (540, 918), 44, "s", opacity=0.86))
        T.append(label_item("+", (318, 1062), 100, "r", blue=False))
        T += num_items(t, t - lt + 0.22, "9999", (636, 1062), 124, True, False)
        T.append(label_item("ложные", (636, 1176), 44, "s", opacity=0.86))
        T.append(label_item("все положительные тесты", (540, 1326), 48, "r", opacity=0.92))
    elif kind == "total10098":
        T += num_items(t, t - lt + 0.15, "10098", (540, 980), 190)
        T.append(label_item("положительных тестов", (540, 1190), 56, "s"))
    elif kind == "only99":
        L.append(equation_layer([
            ("из них", (540, 760), 60, "white"),
            ("99", (540, 960), 190, "blue"),
            ("реально больны", (540, 1180), 62, "white"),
        ]))
    elif kind == "fraction":
        L.append(equation_layer([
            ("99", (540, 835), 165, "blue"),
            ("10098", (540, 1120), 165, "white"),
        ]))
        bar = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(bar).rounded_rectangle([335, 965, 745, 985], radius=10, fill=WHITE + (230,))
        L.append(bar)
    elif kind == "final_less":
        T += num_items(t, t - lt + 0.15, "< 1%", (540, 900), 220)
        T.append(label_item("а не 99%", (540, 1180), 72, "s"))
    elif kind == "bayes_name":
        T.append(label_item("теорема", (540, 820), 74))
        T += pop_items(t, t - lt + 0.28, "Байеса", (540, 1035), 150, "s")
    elif kind == "base_rate":
        L.append(pill_grid(80, 1, True, 720))
        T.append(label_item("важна редкость", (540, 1060), 76, "s"))
        T.append(label_item("а не только точность", (540, 1225), 52))
    elif kind == "repeat_check":
        T.append(label_item("повторная", (540, 860), 96, "s"))
        T.append(label_item("проверка", (540, 1040), 150, "r", blue=True))
        L.append(progress_bar(250, 1250, 580, 48, ease_out(min(1, lt / 0.5)), BLUE, None))
    if kind == "dots6":
        st = G.stagger(lt, 6, 0.24, 0.055)
        ns = {i: (st[i], "w") for i in range(6)}
        ed = [(i, j, "dim", ease_out(max(0, min(1, (lt - 0.55) / 0.4)))) for i, j in G.all_edges(6)]
        L.append(G.draw_graph(V6, ed, node_state=ns))
        T += num_items(t, 3.760, "6", (540, 1400), 210)
    elif kind == "tri_blue":
        p = ease_out(max(0, min(1, lt / 0.32)))
        ns = {i: (1.0, "dim") for i in range(6)}
        for i in (0, 2, 4): ns[i] = (1.1, "w")
        ed = DIM6 + [(0, 1, "blue", p), (2, 4, "blue", p), (0, 5, "blue", p)]
        L.append(G.draw_graph(V6, ed, tri=(0, 2, 4, "blue", max(0, min(1, (lt - 0.28) / 0.4))),
                              node_state=ns))
    elif kind == "tri_white":
        p = ease_out(max(0, min(1, lt / 0.32)))
        ns = {i: (1.0, "dim") for i in range(6)}
        for i in (1, 3, 5): ns[i] = (1.1, "w")
        ed = DIM6 + [(1, 3, "white", p), (3, 5, "white", p), (1, 5, "white", p)]
        L.append(G.draw_graph(V6, ed, tri=(1, 3, 5, "white", max(0, min(1, (lt - 0.28) / 0.4))),
                              node_state=ns))
    elif kind == "burst":
        col, tri = BURSTS[prm["seed"]]
        a, b, c, k = tri
        ed = [(i, j, col[(i, j)] if (i, j) in ((a, b), (a, c), (b, c)) else "dim", 1.0)
              for i, j in G.all_edges(6)]
        ns = {i: (1.0, "dim") for i in range(6)}
        for i in (a, b, c): ns[i] = (1.15, "b" if k == "blue" else "w")
        L.append(G.draw_graph(V6, ed, tri=(a, b, c, k, 1.0), node_state=ns))
    elif kind == "pick":
        ns = {i: (1.0, "w") for i in range(6)}
        ns[0] = (1.0 + 0.25 * ease_out(min(1, lt / 0.35)), "b")
        L.append(G.draw_graph(V6, DIM6, node_state=ns))
    elif kind == "fan":
        st = G.stagger(lt, 5, 0.26, 0.09)
        ed = [(0, i + 1, "grey", st[i]) for i in range(5)]
        ns = {i: (1.0, "w") for i in range(6)}; ns[0] = (1.25, "b")
        L.append(G.draw_graph(V6, ed, node_state=ns))
        T += num_items(t, 21.06, "5", (540, 1430), 190)
    elif kind in ("edge_b", "edge_w"):
        ns = {i: (1.0, "w") for i in range(6)}; ns[0] = (1.25, "b")
        base = [(0, i + 1, "grey", 1.0) for i in range(5)]
        if kind == "edge_b":
            ed = base + [(0, 1, "blue", ease_out(min(1, lt / 0.4)))]
        else:
            ed = base + [(0, 1, "blue", 1.0), (0, 2, "white", ease_out(min(1, lt / 0.4)))]
        L.append(G.draw_graph(V6, ed, node_state=ns))
    elif kind == "min3":
        ns = {i: (1.0, "w") for i in range(6)}; ns[0] = (1.25, "b")
        st = G.stagger(lt - 0.5, 3, 0.24, 0.09)
        ed = [(0, i + 1, "dim", 1.0) for i in range(5)]
        ed += [(0, 1, "blue", st[0]), (0, 3, "blue", st[1]), (0, 5, "blue", st[2])]
        L.append(G.draw_graph(V6, ed, node_state=ns))
        T += num_items(t, 29.50, "3", (540, 1430), 190)
    elif kind == "min3w":
        ns = {i: (1.0, "w") for i in range(6)}; ns[0] = (1.25, "b")
        st = G.stagger(lt - 0.4, 3, 0.24, 0.09)
        ed = [(0, i + 1, "dim", 1.0) for i in range(5)]
        ed += [(0, 1, "white", st[0]), (0, 3, "white", st[1]), (0, 5, "white", st[2])]
        L.append(G.draw_graph(V6, ed, node_state=ns))
        T += num_items(t, 34.52, "3", (540, 1430), 190)
    elif kind == "three":
        ns = {i: (1.0, "dim") for i in range(6)}
        ns[0] = (1.15, "b")
        p = ease_out(min(1, max(0, lt - 0.25) / 0.4))
        for i in (1, 3, 5): ns[i] = (1.0 + 0.25 * p, "w")
        ed = [(0, 1, "blue", 1.0), (0, 3, "blue", 1.0), (0, 5, "blue", 1.0)]
        ed += [(0, 2, "dim", 1.0), (0, 4, "dim", 1.0)]
        L.append(G.draw_graph(V6, ed, node_state=ns))
    elif kind == "caseA1":
        ns = {i: (1.0, "dim") for i in range(6)}
        ns[0] = (1.15, "b")
        for i in (1, 3, 5): ns[i] = (1.2, "w")
        ed = [(0, 1, "blue", 1.0), (0, 3, "blue", 1.0), (0, 5, "blue", 1.0)]
        ed += [(1, 5, "blue", ease_out(min(1, max(0, lt - 0.9) / 0.5)))]
        L.append(G.draw_graph(V6, ed, node_state=ns))
    elif kind == "caseA2":
        ns = {i: (1.0, "dim") for i in range(6)}
        ns[0] = (1.2, "b")
        for i in (1, 5): ns[i] = (1.2, "b")
        ns[3] = (1.0, "w")
        ed = [(0, 1, "blue", 1.0), (0, 3, "blue", 1.0), (0, 5, "blue", 1.0), (1, 5, "blue", 1.0)]
        L.append(G.draw_graph(V6, ed, tri=(0, 1, 5, "blue", ease_out(min(1, lt / 0.5))),
                              node_state=ns))
    elif kind == "caseB1":
        ns = {i: (1.0, "dim") for i in range(6)}
        ns[0] = (1.1, "b")
        for i in (1, 3, 5): ns[i] = (1.2, "w")
        st = G.stagger(lt - 0.2, 3, 0.26, 0.10)
        ed = [(0, 1, "blue", 1.0), (0, 3, "blue", 1.0), (0, 5, "blue", 1.0)]
        ed += [(1, 3, "white", st[0]), (3, 5, "white", st[1]), (1, 5, "white", st[2])]
        L.append(G.draw_graph(V6, ed, node_state=ns))
    elif kind == "caseB2":
        ns = {i: (1.0, "dim") for i in range(6)}
        for i in (1, 3, 5): ns[i] = (1.25, "w")
        ed = [(0, 1, "dim", 1.0), (0, 3, "dim", 1.0), (0, 5, "dim", 1.0),
              (1, 3, "white", 1.0), (3, 5, "white", 1.0), (1, 5, "white", 1.0)]
        L.append(G.draw_graph(V6, ed, tri=(1, 3, 5, "white", ease_out(min(1, lt / 0.5))),
                              node_state=ns))
    elif kind == "k5":
        st = G.stagger(lt, 5, 0.28, 0.075)
        ns = {i: (st[i], "w") for i in range(5)}
        ed = [(i, j, "dim", ease_out(max(0, min(1, (lt - 0.7) / 0.5)))) for i, j in G.all_edges(5)]
        L.append(G.draw_graph(V5, ed, node_state=ns))
        T += num_items(t, 59.52, "5", (540, 1400), 200)
    elif kind == "k5color":
        cyc = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
        star = [(0, 2), (1, 3), (2, 4), (0, 3), (1, 4)]
        sc = G.stagger(lt, 5, 0.22, 0.07)
        ss = G.stagger(lt - 0.55, 5, 0.22, 0.07)
        ed = [(a, b, "blue", sc[i]) for i, (a, b) in enumerate(cyc)]
        ed += [(a, b, "white", ss[i]) for i, (a, b) in enumerate(star)]
        L.append(G.draw_graph(V5, ed, node_state={i: (1.0, "w") for i in range(5)}))
    elif kind == "always":
        col, tri = BURSTS[2]
        a, b, c, k = tri
        ed = [(i, j, col[(i, j)], 1.0) for i, j in G.all_edges(6)]
        pr = ease_out(min(1, max(0, lt - 0.15) / 0.6))
        ns = {i: (1.0, "dim") for i in range(6)}
        for i in (a, b, c): ns[i] = (1.15, "w")
        L.append(G.draw_graph(V6, ed, tri=(a, b, c, k, max(0, min(1, (lt - 0.5) / 0.6))),
                              node_state=ns))
    elif kind == "resultA":
        pr = ease_out(min(1, lt / 0.45))
        ns = {i: (0.0, "dim") for i in range(6)}
        for i in (0, 1, 5): ns[i] = (1.25 * pr, "b")
        ed = [(0, 1, "blue", pr), (0, 5, "blue", pr), (1, 5, "blue", pr)]
        L.append(G.draw_graph(V6, ed, tri=(0, 1, 5, "blue", pr), node_state=ns))
    elif kind == "plain":
        if 26.020 <= t < 27.500:
            T += num_items(t, 26.34, "2", (540, 1000), 220)
        if 27.500 <= t < 28.780:
            T += num_items(t, 27.74, "5", (540, 1000), 220)
        if 43.540 <= t < 45.120:
            pass
        if 65.160 <= t < 68.560:
            T += num_items(t, 65.16, "6", (540, 1000), 230)
        if 55.340 <= t < 58.140:
            T += pop_items(t, 56.70, "Рамсея", (540, 1060), 140, "s")
    return L, T


# ---------------------------------------------------------------- главный цикл
def main():
    prep_aroll()
    stock_files = prep_stock()
    stock_caps = {i: cv2.VideoCapture(p) for i, p in stock_files.items()}
    stock_pos = {i: -1 for i in stock_files}

    cap1 = cv2.VideoCapture(A1)
    cap2 = cv2.VideoCapture(A2)
    maskA = rounded_mask(CW, CH, R_A_)
    maskB = rounded_mask(BW, BH, R_B)

    grid_cache = {}

    def grid_for(f):
        k = f // 3
        if k not in grid_cache:
            grid_cache.clear()
            grid_cache[k] = grid_canvas(CW, CH, phase=k * 0.06)
        return grid_cache[k]

    tmp = f"{BUILD}/assets/_video.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    si = 0
    for f in range(NF):
        t = f / FPS
        ok1 = cap1.grab(); ok2 = cap2.grab()
        while si < len(SHOTS) - 1 and t >= SHOTS[si][1]:
            si += 1
        t0, t1, kind, prm = SHOTS[si]
        shot_idx = si
        lt = t - t0

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        if kind in ("A1", "A2"):
            cap = cap1 if kind == "A1" else cap2
            okr, img = cap.retrieve()
            if not okr:
                img = np.zeros((CH, CW, 3), np.uint8)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            canvas.paste(pil, (CX, CY), maskA)
            tw = prm.get("topword")
            if tw:
                canvas.alpha_composite(text_layer((W, H), [
                    dict(text=tw, font=ghost_font(tw), xy=(540, 440), anchor="mm",
                         fill=WHITE, glow=WHITE, glow_r=24, glow_a=0.35,
                         opacity=0.62 * ease_out(min(1, lt / 0.30)))]))
        elif kind == "stock":
            v = stock_caps[shot_idx]
            want = int(round(lt * FPS))
            while stock_pos[shot_idx] < want:
                okg = v.grab()
                if not okg:
                    break
                stock_pos[shot_idx] += 1
            okr, img = v.retrieve()
            if okr:
                canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                             (BX, BY), maskB)
        else:
            canvas.paste(grid_for(f), (CX, CY), maskA)

        layers, titems = shot_layers(kind, prm, lt, t1 - t0, t)
        for L in layers:
            canvas.alpha_composite(L)
        items = titems + caption_items(t, kind)
        if items:
            canvas.alpha_composite(text_layer((W, H), items))

        ff.stdin.write(canvas.convert("RGB").tobytes())
        if f % 150 == 0:
            print(f"  кадр {f}/{NF}  ({t:5.1f}s)")

    ff.stdin.close(); ff.wait()
    cap1.release(); cap2.release()

    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", tmp, "-i", SRC,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy",
                    "-af", "highpass=f=70,loudnorm=I=-14:TP=-1.5:LRA=7",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", OUT], check=True)
    print("готово:", OUT)


if __name__ == "__main__":
    main()

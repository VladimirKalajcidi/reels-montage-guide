"""Сборка ролика 10 («число Данбара»).

Слойность: фон (лицо / сток / сетка) -> графика -> субтитры.
Аудио добавляет sfx10.py, исходная речь не режется.
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard10 import (SRC, DUR, SHOTS, CAPS, SLOTS, FACE_KINDS, GFX_ZONE,
                          GFX_X, OUT, shot_at, slot_for)
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_dunbar.mp4"
STOCK_DIR = "/Users/vladimirkalajcidi/reels_good/videos/10/stock"

GY0, GY1 = GFX_ZONE
GX0, GX1 = GFX_X
GCX = (GX0 + GX1) // 2
GCY = (GY0 + GY1) // 2
PAD = 30
GRID_CACHE = {}


def cv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def fit_bgr(fr, w, h):
    ih, iw = fr.shape[:2]
    target = w / h
    if iw / ih > target:
        nw = int(ih * target)
        x = (iw - nw) // 2
        fr = fr[:, x:x + nw]
    else:
        nh = int(iw / target)
        y = max(0, min((ih - nh) // 2, ih - nh))
        fr = fr[y:y + nh, :]
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)


def source_card(frame, kind):
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    if kind == "A2":
        w, h = 900, 1428
        x, y = (iw - w) // 2 + 8, 230
        fr = fr[y:y + h, x:x + w]
    else:
        w, h = 1000, 1585
        x, y = (iw - w) // 2 + 10, 125
        fr = fr[y:y + h, x:x + w]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_AREA)
    fr = cv2.convertScaleAbs(fr, alpha=1.06, beta=9)
    b, g, r = cv2.split(fr.astype(np.float32))
    r = np.clip(r * 1.04 + 3, 0, 255)
    b = np.clip(b * 0.96, 0, 255)
    fr = cv2.merge([b, g, r]).astype(np.uint8)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.08, 0, 255)
    fr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


def stock_card(fr):
    fr = fit_bgr(fr, CARD_B[2], CARD_B[3])
    fr = cv2.convertScaleAbs(fr, alpha=1.03, beta=-48)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.92, 0, 255)
    fr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


def clamp01(v):
    return max(0.0, min(1.0, v))


def gtext(text, xy, size=80, blue=False, serif=False, op=1.0, anchor="mm"):
    kind = "serif_it" if serif else "sans"
    return dict(text=text, font=font(kind, size), xy=xy, anchor=anchor,
                fill=BLUE if blue else WHITE,
                glow=BLUE_GLOW if blue else WHITE,
                glow_r=34 if blue else 15, glow_a=0.86 if blue else 0.48,
                opacity=op)


def pop(lt, dur=0.10):
    return ease_out(clamp01(lt / dur)) if lt >= 0 else 0.0


def blue_pop(lt, dur=0.38):
    if lt < 0:
        return 0.0, 1.0
    p = clamp01(lt / dur)
    e = ease_out(p)
    over = 1.0 + 0.03 * math.sin(p * math.pi)
    return e, over


def _layer():
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def text_layer(size, items):
    """Локальная быстрая версия style.text_layer.

    Размывает не весь холст под каждый текстовый элемент, а только bbox строки
    с запасом под glow. На длинном ролике это экономит минуты рендера.
    """
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    pd = ImageDraw.Draw(probe)
    for it in items:
        txt = it["text"]
        f = it["font"]
        xy = it["xy"]
        anchor = it.get("anchor", "mm")
        bbox = pd.textbbox(xy, txt, font=f, anchor=anchor)
        gr = int(it.get("glow_r", 14))
        pad = max(8, gr * 3)
        left = max(0, int(math.floor(bbox[0] - pad)))
        top = max(0, int(math.floor(bbox[1] - pad)))
        right = min(size[0], int(math.ceil(bbox[2] + pad)))
        bottom = min(size[1], int(math.ceil(bbox[3] + pad)))
        if right <= left or bottom <= top:
            continue
        sub = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
        d = ImageDraw.Draw(sub)
        local_xy = (xy[0] - left, xy[1] - top)
        d.text(local_xy, txt, font=f, fill=it["fill"] + (255,), anchor=anchor)
        if gr:
            a = sub.split()[3]
            blur = a.filter(ImageFilter.GaussianBlur(gr))
            gl = Image.new("RGBA", sub.size, it.get("glow", it["fill"]) + (0,))
            gl.putalpha(blur.point(lambda v: int(v * it.get("glow_a", 0.55))))
            comp = Image.new("RGBA", sub.size, (0, 0, 0, 0))
            comp.alpha_composite(gl)
            comp.alpha_composite(sub)
            sub = comp
        rev = it.get("reveal")
        if rev and rev[0] == "wipe":
            rx = int(rev[1] - left)
            if rx <= 0:
                continue
            if rx < sub.size[0]:
                m = Image.new("L", sub.size, 0)
                ImageDraw.Draw(m).rectangle([0, 0, rx, sub.size[1]], fill=255)
                m = m.filter(ImageFilter.GaussianBlur(1.2))
                sub.putalpha(Image.composite(sub.split()[3], Image.new("L", sub.size, 0), m))
        op = it.get("opacity", 1.0)
        if op < 1.0:
            sub.putalpha(sub.split()[3].point(lambda v: int(v * op)))
        layer.alpha_composite(sub, (left, top))
    return layer


def card_box(lay, xy, text, size=54, w=520, h=132, p=1.0):
    if p <= 0:
        return
    e = ease_out(clamp01(p))
    d = ImageDraw.Draw(lay)
    cw, ch = w * (0.72 + 0.28 * e), h * (0.72 + 0.28 * e)
    x, y = xy[0] - cw / 2, xy[1] - ch / 2
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([x, y, x + cw, y + ch], radius=24, fill=(255, 255, 255, int(65 * e)))
    lay.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(16)))
    d.rounded_rectangle([x, y, x + cw, y + ch], radius=24, fill=WHITE + (int(235 * e),))
    d.text(xy, text, font=font("sans", int(size * (0.74 + 0.26 * e))),
           fill=(0, 0, 0, int(250 * e)), anchor="mm")


def ring(lay, cx, cy, r, p, color=WHITE, width=5, alpha=220):
    p = clamp01(p)
    if p <= 0:
        return
    d = ImageDraw.Draw(lay)
    end = -90 + 360 * ease_out(p)
    d.arc([cx - r, cy - r, cx + r, cy + r], -90, end,
          fill=color + (alpha,), width=width)


def rings_base(lay, lt, full=False):
    specs = [(82, "5", 0.00), (145, "15", 0.32), (230, "50", 0.64), (320, "150", 0.96)]
    for r, label_txt, at in specs:
        p = 1.0 if full else clamp01((lt - at) / 0.55)
        color = BLUE if label_txt in {"5", "150"} else WHITE
        ring(lay, GCX, 1110, r, p, color=color, width=6 if color == BLUE else 4,
             alpha=240 if color == BLUE else 175)
        if p > 0.88:
            a = ease_out(clamp01((p - 0.88) / 0.12))
            lx = min(835, GCX + max(48, r - 42))
            lay.alpha_composite(text_layer((W, H), [
                gtext(label_txt, (lx, 1110), 44 if label_txt != "150" else 54,
                      blue=(label_txt in {"5", "150"}), op=a, anchor="mm")
            ]))


def g_limit(lay, lt):
    p = pop(lt, 0.18)
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle([250, 870, 830, 1340], radius=54, outline=WHITE + (170,), width=5)
    fill = int(430 * clamp01(lt / 1.10))
    d.rounded_rectangle([292, 1240 - fill, 388, 1240], radius=28, fill=BLUE + (235,))
    for x in [455, 530, 605, 680, 755]:
        d.ellipse([x - 20, 1010 - 20, x + 20, 1010 + 20], fill=WHITE + (170,))
    lay.alpha_composite(text_layer((W, H), [
        gtext("лимит", (540, 780), int(112 * (0.6 + 0.4 * p)), serif=True),
        gtext("связей", (540, 1418), 62, op=clamp01((lt - 0.45) / 0.35)),
    ]))


def g_num150(lay, lt):
    p, over = blue_pop(lt, 0.38)
    if p <= 0:
        return
    lay.alpha_composite(text_layer((W, H), [
        gtext("150", (GCX, GCY), int(210 * (0.55 + 0.45 * p) * over), blue=True),
        gtext("человек", (GCX, GCY + 170), 58, op=clamp01((lt - 0.30) / 0.30)),
    ]))


def g_neocortex(lay, lt):
    d = ImageDraw.Draw(lay)
    cx, cy = 365, 1030
    p = ease_out(clamp01(lt / 0.75))
    d.ellipse([cx - 125, cy - 95, cx + 125, cy + 95], outline=WHITE + (210,), width=5)
    for i in range(5):
        x0 = cx - 86 + i * 38
        d.arc([x0, cy - 50, x0 + 82, cy + 48], 195, 520, fill=WHITE + (145,), width=3)
    bar_x = 610
    d.line([(bar_x, 1340), (bar_x + 230, 1340)], fill=WHITE + (150,), width=4)
    d.rectangle([bar_x + 20, 1340 - 95 * p, bar_x + 70, 1340], fill=BLUE + (235,))
    d.rectangle([bar_x + 105, 1340 - 185 * p, bar_x + 155, 1340], fill=BLUE + (210,))
    d.rectangle([bar_x + 190, 1340 - 285 * p, bar_x + 240, 1340], fill=BLUE + (185,))
    lay.alpha_composite(text_layer((W, H), [
        gtext("неокортекс", (365, 1245), 52, serif=True),
        gtext("группа", (725, 950), 54, op=clamp01((lt - 0.55) / 0.35)),
    ]))


def g_formula(lay, lt):
    d = ImageDraw.Draw(lay)
    pts = [(230, 1280), (390, 1160), (540, 1060), (700, 960), (855, 840)]
    n = max(2, int(len(pts) * 8 * ease_out(clamp01(lt / 1.05))))
    curve = []
    for i in range(n):
        u = i / max(1, n - 1)
        j = min(len(pts) - 2, int(u * (len(pts) - 1)))
        v = u * (len(pts) - 1) - j
        x = pts[j][0] * (1 - v) + pts[j + 1][0] * v
        y = pts[j][1] * (1 - v) + pts[j + 1][1] * v
        curve.append((x, y))
    if len(curve) > 1:
        d.line(curve, fill=BLUE + (235,), width=7, joint="curve")
    lay.alpha_composite(text_layer((W, H), [
        gtext("формула", (365, 820), 54, op=clamp01((lt - 0.25) / 0.35)),
        gtext("человек", (735, 1375), 58, serif=True, op=clamp01((lt - 0.75) / 0.35)),
    ]))


def g_result150(lay, lt):
    g_num150(lay, lt)
    if lt > 1.05:
        card_box(lay, (GCX, 1325), "не просто теория", 50, 590, 124, clamp01((lt - 1.05) / 0.35))


def g_evidence(lay, lt):
    for i, txt in enumerate(["деревни", "армия", "открытки"]):
        card_box(lay, (GCX, 850 + i * 210), txt, 56, 560, 126, clamp01((lt - i * 0.42) / 0.36))


def g_rings_intro(lay, lt):
    rings_base(lay, lt)
    lay.alpha_composite(text_layer((W, H), [
        gtext("слои", (GCX, 805), 116, serif=True, op=pop(lt, 0.18))
    ]))


def g_ring5(lay, lt):
    rings_base(lay, lt, full=True)
    p, over = blue_pop(lt - 0.10)
    lay.alpha_composite(text_layer((W, H), [
        gtext("5", (GCX, 1110), int(175 * (0.55 + 0.45 * p) * over), blue=True),
        gtext("близкий круг", (GCX, 1360), 56, serif=True, op=clamp01((lt - 0.62) / 0.35)),
    ]))


def g_ring150(lay, lt):
    rings_base(lay, lt, full=True)
    p, over = blue_pop(lt - 0.08)
    lay.alpha_composite(text_layer((W, H), [
        gtext("150", (GCX, 1110), int(150 * (0.55 + 0.45 * p) * over), blue=True),
        gtext("предел", (GCX, 1395), 72, serif=True, op=clamp01((lt - 0.72) / 0.32)),
    ]))


def g_socials(lay, lt):
    p1, o1 = blue_pop(lt - 0.12)
    p2 = clamp01((lt - 1.25) / 0.35)
    lay.alpha_composite(text_layer((W, H), [
        gtext("1000", (GCX, 925), int(142 * (0.55 + 0.45 * p1) * o1), blue=True),
        gtext("подписчиков", (GCX, 1055), 54, op=clamp01((lt - 0.42) / 0.28)),
        gtext("150 связей", (GCX, 1285), int(70 * (0.65 + 0.35 * ease_out(p2))),
              blue=True, op=p2),
    ]))
    d = ImageDraw.Draw(lay)
    if lt > 2.2:
        p = ease_out(clamp01((lt - 2.2) / 0.45))
        d.line([(325, 1160), (755, 1160)], fill=WHITE + (210,), width=5)
        d.line([(755, 1160), (720, 1128)], fill=WHITE + (210,), width=5)


def g_rings_final(lay, lt):
    rings_base(lay, lt, full=False)
    if lt > 2.1:
        lay.alpha_composite(text_layer((W, H), [
            gtext("кольца данбара", (GCX, 790), 58, serif=True,
                  op=clamp01((lt - 2.1) / 0.35))
        ]))


GFX = {
    "limit": g_limit, "num150": g_num150, "neocortex": g_neocortex,
    "formula": g_formula, "result150": g_result150, "evidence": g_evidence,
    "rings_intro": g_rings_intro, "ring5": g_ring5, "ring150": g_ring150,
    "socials": g_socials, "rings_final": g_rings_final,
}


def graphics_layer(kind, lt):
    if kind not in GFX:
        return None
    lay = _layer()
    GFX[kind](lay, lt)
    return lay


def line_items(runs, xy, anchor, opacity=1.0, reveal_chars=None, max_w=None):
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    sizes = [sz for _, _, sz in runs]
    fonts = [font(kmap[k], sz) for (_, k, sz) in runs]
    widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
    total = sum(widths) + 8 * (len(runs) - 1)
    if max_w and total > max_w:
        sc = max_w / max(1, total)
        sizes = [max(22, int(sz * sc)) for sz in sizes]
        fonts = [font(kmap[runs[i][1]], sizes[i]) for i in range(len(runs))]
        widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
        total = sum(widths) + 8 * (len(runs) - 1)
    x, y = xy
    cx = x - total / 2 if anchor[0] == "m" else x
    out, cum = [], 0
    for i, (txt, k, _) in enumerate(runs):
        f = fonts[i]
        it = dict(text=txt, font=f, xy=(cx, y), anchor="l" + anchor[1],
                  fill=WHITE, glow=WHITE, glow_r=int(12 + sizes[i] * .08),
                  glow_a=0.60 if k == "s" else 0.48, opacity=opacity)
        if reveal_chars is not None:
            done = reveal_chars - cum
            if done <= 0:
                cx += widths[i] + 8
                cum += len(txt)
                continue
            if done < len(txt):
                it["reveal"] = ("wipe", cx + f.getlength(txt[:int(done)]) + 2)
        out.append(it)
        cx += widths[i] + 8
        cum += len(txt)
    return out


def build_blocks():
    blocks, cur = [], []
    for i, c in enumerate(CAPS):
        if cur:
            prev = CAPS[cur[-1]]
            same_shot = shot_at(c[0]) is shot_at(prev[0])
            if c[0] - prev[1] > 0.30 or len(cur) >= 2 or not same_shot:
                blocks.append(cur)
                cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    info = {}
    for bid, b in enumerate(blocks):
        end = max(CAPS[i][1] for i in b)
        for pos, i in enumerate(b):
            info[i] = (bid, pos, len(b), end)
    return info


BLOCKS = build_blocks()
DIM = [1.0, 0.74, 0.58]


def caption_layer(t):
    kind = shot_at(t)[2]
    slot = SLOTS[slot_for(kind)]
    active = [i for i, c in enumerate(CAPS) if c[0] <= t < BLOCKS[i][3]]
    items = []
    for idx in active[-3:]:
        t0, _, runs, _ = CAPS[idx]
        bid, pos, _, _ = BLOCKS[idx]
        newer = sum(1 for j in active if j > idx and BLOCKS[j][0] == bid)
        dim = DIM[min(newer, 2)]
        bx, by = slot["x"], slot["y"] + pos * slot["step"]
        if slot["scatter"]:
            calm = bid % 3 == 0
            bx += (0 if calm else [-48, 40, -22][min(pos, 2)])
        lt = t - t0
        nchars = sum(len(r[0]) for r in runs)
        type_dur = min(0.55, max(0.30, nchars * 0.030))
        reveal = nchars if lt >= type_dur else nchars * lt / type_dur
        op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
        max_w = slot["x_max"] - bx if slot["anchor"][0] == "l" else \
            2 * min(bx - slot["x_min"], slot["x_max"] - bx)
        items += line_items(runs, (bx, by), slot["anchor"], opacity=op,
                            reveal_chars=reveal, max_w=max_w)
    if not items:
        return None
    return text_layer((W, H), items)


def background(kind, prm, fr, t, stock_reader=None):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    if kind in FACE_KINDS:
        card = source_card(fr, kind)
        canvas.paste(card, (CARD_A[0], CARD_A[1]), rounded_mask(CARD_A[2], CARD_A[3], R_A))
    elif kind == "stock":
        sfr = stock_reader()
        card = stock_card(sfr)
        canvas.paste(card, (CARD_B[0], CARD_B[1]), rounded_mask(CARD_B[2], CARD_B[3], R_B))
    else:
        x, y, w, h = CARD_A
        key = int(t * 10)
        if key not in GRID_CACHE:
            if len(GRID_CACHE) > 8:
                GRID_CACHE.clear()
            GRID_CACHE[key] = grid_canvas(w, h, phase=(key / 10) * 0.7)
        g = GRID_CACHE[key]
        canvas.paste(g, (x, y), rounded_mask(w, h, R_A))
    return canvas


def main():
    cap = cv2.VideoCapture(SRC)
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    tmp = f"{BUILD}/assets/_video_dunbar_raw.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    nf = int(round(DUR * FPS))
    stock_cap, stock_key = None, None

    for fno in range(nf):
        ok, fr = cap.read()
        if not ok:
            break
        t = fno / FPS
        t0, t1, kind, prm = shot_at(t)
        lt = t - t0

        def read_stock():
            nonlocal stock_cap, stock_key
            key = (t0, prm["clip"])
            if key != stock_key:
                if stock_cap is not None:
                    stock_cap.release()
                stock_cap = cv2.VideoCapture(f"{STOCK_DIR}/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
                stock_key = key
            ok_s, sfr = stock_cap.read()
            if not ok_s:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok_s, sfr = stock_cap.read()
            return sfr

        canvas = background(kind, prm, fr, t, read_stock)
        gl = graphics_layer(kind, lt)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        wr.write(pil_to_cv(canvas.convert("RGB")))
        if fno and fno % 300 == 0:
            print(f"frame {fno}/{nf}")

    wr.release()
    cap.release()
    if stock_cap is not None:
        stock_cap.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp,
        "-c:v", "libx264", "-crf", "16", "-preset", "medium",
        "-pix_fmt", "yuv420p", VID,
    ], check=True)
    print("готово:", VID)


if __name__ == "__main__":
    main()

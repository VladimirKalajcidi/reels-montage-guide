"""Рендер ролика 7: паузы вырезаны, поверх — сток, сетка, субтитры."""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard7 import SRC_ORIG, SRC_CUT, CUTS, SHOTS, CAPS, SLOTS, DUR
from style import *

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_secretary.mp4"


def make_cut_source():
    if os.path.exists(SRC_CUT):
        return
    segs = []
    cur = 0.0
    for a, b in CUTS:
        segs.append((cur, a))
        cur = b
    segs.append((cur, 82.36))
    parts = []
    for i, (a, b) in enumerate(segs):
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
    chain = ";".join(parts) + ";" + "".join(f"[v{i}][a{i}]" for i in range(len(segs)))
    chain += f"concat=n={len(segs)}:v=1:a=1[v][a]"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC_ORIG,
        "-filter_complex", chain, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "14", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", SRC_CUT,
    ], check=True)


def cv_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def source_card(frame):
    fr = cv2.flip(frame, 1)
    ih, iw = fr.shape[:2]
    target = CARD_A[2] / CARD_A[3]
    if iw / ih > target:
        nw = int(ih * target)
        x = (iw - nw) // 2
        fr = fr[:, x:x + nw]
    else:
        nh = int(iw / target)
        y = max(0, (ih - nh) // 2 - 30)
        y = min(y, ih - nh)
        fr = fr[y:y + nh, :]
    fr = cv2.resize(fr, (CARD_A[2], CARD_A[3]), interpolation=cv2.INTER_AREA)
    fr = cv2.convertScaleAbs(fr, alpha=1.06, beta=8)
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.10
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    fr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv_to_pil(fr)


_stock_caps = {}


def stock_frame_from_bgr(fr):
    ih, iw = fr.shape[:2]
    target = CARD_B[2] / CARD_B[3]
    if iw / ih > target:
        nw = int(ih * target)
        x = (iw - nw) // 2
        fr = fr[:, x:x + nw]
    else:
        nh = int(iw / target)
        y = (ih - nh) // 2
        fr = fr[y:y + nh, :]
    fr = cv2.resize(fr, (CARD_B[2], CARD_B[3]), interpolation=cv2.INTER_AREA)
    fr = cv2.convertScaleAbs(fr, alpha=1.05, beta=-46)
    return cv_to_pil(fr)


def stock_frame(clip, ss, lt):
    path = f"{BUILD}/assets/stock_{clip}.mp4"
    cap = _stock_caps.get(path)
    if cap is None:
        cap = cv2.VideoCapture(path)
        _stock_caps[path] = cap
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, (ss + lt) * 1000))
    ok, fr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ok, fr = cap.read()
    return stock_frame_from_bgr(fr)


def draw_topword(canvas, word, color=WHITE):
    f = font("sans", 128)
    layer = text_layer((W, H), [dict(text=word, font=f, xy=(540, 450), anchor="mm",
                                    fill=color, glow=color, glow_r=24, glow_a=0.35,
                                    opacity=0.62)])
    canvas.alpha_composite(layer)


def draw_grid(canvas, rect=CARD_B, phase=0):
    x, y, w, h = rect
    g = grid_canvas(w, h, phase=phase)
    canvas.paste(g, (x, y), rounded_mask(w, h, R_B if rect == CARD_B else R_A))


def big_text(text, xy, size=150, blue=False, serif=False, op=1.0):
    kind = "serif_it" if serif else "sans"
    f = font(kind, size)
    color = BLUE if blue else WHITE
    return dict(text=text, font=f, xy=xy, anchor="mm", fill=color,
                glow=BLUE_GLOW if blue else WHITE,
                glow_r=38 if blue else 16, glow_a=0.68 if blue else 0.48, opacity=op)


def graphic(canvas, kind, prm, lt, t):
    draw_grid(canvas, CARD_B, phase=t * 0.7)
    items = []
    if kind == "rule37":
        items.append(big_text("37%", (540, 875), 178, blue=True))
        items.append(big_text("первые варианты", (540, 1005), 48))
        items.append(big_text("пропускаем", (540, 1065), 58, serif=True))
    elif kind == "grid_rule":
        for i, txt in enumerate(["1 наблюдай", "2 запомни лучшего", "3 выбирай дальше"]):
            y = 775 + i * 100
            items.append(big_text(txt, (540, y), 46 if i else 50, blue=i == 0))
    elif kind == "grid_pick":
        items.append(big_text("после отметки", (540, 805), 52))
        items.append(big_text("первый лучше всех", (540, 920), 64, serif=True))
        items.append(big_text("берём", (540, 1040), 84, blue=True))
    elif kind == "grid_early":
        items.append(big_text("слишком рано", (540, 820), 66, serif=True))
        items.append(big_text("уровень ещё", (540, 930), 52))
        items.append(big_text("не виден", (540, 1040), 82, blue=True))
    elif kind == "grid_late":
        items.append(big_text("слишком долго", (540, 805), 60, serif=True))
        items.append(big_text("лучший уже", (540, 920), 52))
        items.append(big_text("прошёл мимо", (540, 1035), 78, blue=True))
    elif kind == "balance":
        d = ImageDraw.Draw(canvas)
        d.line([(300, 1040), (780, 1040)], fill=WHITE + (210,), width=5)
        d.line([(540, 760), (540, 1085)], fill=BLUE + (230,), width=6)
        items.append(big_text("рано", (360, 905), 58))
        items.append(big_text("долго", (720, 905), 58))
        items.append(big_text("баланс", (540, 780), 72, serif=True))
    elif kind == "formula":
        items.append(big_text("1 / e", (540, 850), 168, blue=True))
        items.append(big_text("e ≈ 2,7", (540, 1010), 58))
    elif kind == "grid_max":
        items.append(big_text("37%", (360, 835), 108, blue=True))
        items.append(big_text("не 100%", (705, 835), 64))
        items.append(big_text("но это максимум", (540, 995), 58, serif=True))
    if items:
        canvas.alpha_composite(text_layer((W, H), items))


def line_items(runs, xy, anchor, opacity=1.0, reveal_chars=None):
    fonts = []
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    for _, k, sz in runs:
        fonts.append(font(kmap[k], sz))
    widths = [fonts[i].getlength(runs[i][0]) for i in range(len(runs))]
    total = sum(widths)
    x, y = xy
    sx = x - total / 2 if anchor[0] == "m" else x
    out, cx, cum = [], sx, 0
    for i, (txt, k, sz) in enumerate(runs):
        f = fonts[i]
        it = dict(text=txt, font=f, xy=(cx, y), anchor="l" + anchor[1],
                  fill=WHITE, glow=WHITE, glow_r=int(12 + sz * .08),
                  glow_a=0.58 if k == "s" else 0.48, opacity=opacity)
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
        if cur and (c[0] - CAPS[cur[-1]][1] > 0.30 or len(cur) >= 2 or c[3] != CAPS[cur[-1]][3]):
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


def caption_items(t, shot_kind):
    out = []
    active = [i for i, c in enumerate(CAPS) if c[0] <= t < BLOCKS[i][3]]
    for idx in active[-3:]:
        t0, _, runs, slot = CAPS[idx]
        bid, pos, blen, _ = BLOCKS[idx]
        newer = sum(1 for j in active if j > idx and BLOCKS[j][0] == bid)
        dim = DIM[min(newer, 2)]
        eff_slot = slot if shot_kind in ("A1", "A2") else "T"
        base_x, base_y, an = SLOTS[eff_slot]
        if eff_slot == "T":
            base_y = 710 + pos * 94
        else:
            calm = bid % 3 == 0
            dxs = [0, 0, 0] if calm else [-45, 38, -20]
            base_x += dxs[min(pos, 2)]
            base_y = 1245 + pos * 82
        lt = t - t0
        nchars = sum(len(r[0]) for r in runs)
        type_dur = min(0.55, max(0.30, nchars * 0.030))
        reveal = nchars if lt >= type_dur else nchars * lt / type_dur
        op = (0.35 + 0.65 * ease_out(min(1.0, lt / 0.40))) * dim
        out += line_items(runs, (base_x, base_y), an, opacity=op, reveal_chars=reveal)
    return out


def shot_at(t):
    for s in SHOTS:
        if s[0] <= t < s[1]:
            return s
    return SHOTS[-1]


def main():
    make_cut_source()
    cap = cv2.VideoCapture(SRC_CUT)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = f"{BUILD}/assets/_video_secretary_raw.mp4"
    wr = cv2.VideoWriter(tmp, fourcc, FPS, (W, H))
    nf = int(round(DUR * FPS))
    stock_cap = None
    stock_key = None
    cur_i = -1
    for fno in range(nf):
        ok, fr = cap.read()
        if not ok:
            break
        t = fno / FPS
        shot = shot_at(t)
        t0, t1, kind, prm = shot
        si = SHOTS.index(shot)
        lt = t - t0
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        if kind in ("A1", "A2"):
            card = source_card(fr)
            canvas.paste(card, (CARD_A[0], CARD_A[1]), rounded_mask(CARD_A[2], CARD_A[3], R_A))
            if prm.get("topword"):
                draw_topword(canvas, prm["topword"], BLUE if prm["topword"] == "37%" else WHITE)
        elif kind == "stock":
            key = (si, prm["clip"])
            if key != stock_key:
                if stock_cap is not None:
                    stock_cap.release()
                stock_cap = cv2.VideoCapture(f"{BUILD}/assets/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
                stock_key = key
            ok_s, sfr = stock_cap.read()
            if not ok_s:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok_s, sfr = stock_cap.read()
            card = stock_frame_from_bgr(sfr)
            canvas.paste(card, (CARD_B[0], CARD_B[1]), rounded_mask(CARD_B[2], CARD_B[3], R_B))
        else:
            graphic(canvas, kind, prm, lt, t)
        caps = caption_items(t, kind)
        if caps:
            canvas.alpha_composite(text_layer((W, H), caps))
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

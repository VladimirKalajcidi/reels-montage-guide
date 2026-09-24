"""Сборка ролика 6. Речь не резана — резы только по картинке."""
import os, sys, math, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import storyboard6 as SB
from storyboard6 import SHOTS, CAPS, SLOTS, DUR, FRAMINGS

BUILD = os.path.dirname(os.path.abspath(__file__))
SRC = SB.SRC
VIDEO_DIR = os.path.dirname(SRC)
A1 = f"{BUILD}/assets/aroll_6_A1.mp4"
A2 = f"{BUILD}/assets/aroll_6_A2.mp4"
OUT = SB.OUT

# без hflip: дубль уже не зеркальный (проверено по нашивке на футболке — свосh
# читается нормально, не задом наперёд).
GRADE = ("eq=gamma=1.18:contrast=1.06:saturation=1.10:brightness=0.02,"
         "colorbalance=rm=0.04:gm=0.008:bm=-0.03,unsharp=5:5:0.30")

CX, CY, R_A_ = CARD_A[0], CARD_A[1], R_A
CW, CH = CARD_A[2], CARD_A[3]
NF = int(round(DUR * FPS))


def prep_aroll():
    for name, out in (("A1", A1), ("A2", A2)):
        if os.path.exists(out):
            continue
        w, h, x, y = FRAMINGS[name]
        vf = f"crop={w}:{h}:{x}:{y},scale={CW}:{CH}:flags=lanczos,{GRADE}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14",
                        "-preset", "medium", "-pix_fmt", "yuv420p", out], check=True)
        print("готово:", out)


BX, BY, BW, BH = CARD_B
STOCK_DIR = f"{VIDEO_DIR}/stock"
STOCK_PREP_DIR = f"{STOCK_DIR}/prepared"


def prep_stock():
    os.makedirs(STOCK_PREP_DIR, exist_ok=True)
    out = {}
    for i, (t0, t1, kind, prm) in enumerate(SHOTS):
        if kind != "stock":
            continue
        src = f"{STOCK_DIR}/stock_{prm['clip']}.mp4"
        dst = f"{STOCK_PREP_DIR}/sb6_{i}_{prm['clip']}.mp4"
        if not os.path.exists(dst):
            vf = ("crop='min(iw,ih*1.3839)':'min(ih,iw/1.3839)',"
                  f"scale={BW}:{BH}:flags=lanczos,"
                  "eq=brightness=-0.24:contrast=1.05:saturation=0.92,fps=30")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", str(prm.get("ss", 0)), "-t", f"{t1 - t0 + 0.3:.3f}",
                            "-i", src,
                            "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15",
                            "-preset", "medium", "-pix_fmt", "yuv420p", dst], check=True)
            print("готово:", dst)
        out[i] = dst
    return out


# ---------------------------------------------------------------- текст
def line_items(runs, xy, anchor, reveal_chars=None, opacity=1.0):
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


MAXW = 960


def _shot_kind_at(t):
    for t0, t1, kind, _ in SHOTS:
        if t0 <= t < t1:
            return kind
    return SHOTS[-1][2]


def _build_blocks():
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

SCATTER_A = [(-58, -62, 0.98), (55, 0, 1.08), (-30, 62, 0.94)]
SCATTER_C = [(0, 0, 1.0), (18, 58, 1.04), (-24, 116, 0.96)]
SIMPLE_A = [(0, -62, 1.0), (0, 0, 1.0), (0, 62, 1.0)]
SIMPLE_C = [(0, 0, 1.0), (0, 58, 1.0), (0, 116, 1.0)]
DIM = [1.0, 0.74, 0.58]
PAD = 30


def _eff_size(runs, ks=1.0):
    return max(r[2] for r in runs) * ks


def _block_layout():
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
    out = []
    card = CARD_B if shot_kind == "stock" else CARD_A
    for idx, (t0, t1, runs, slot) in enumerate(CAPS):
        k, bend, blen, bid = BLOCKS[idx]
        if not (t0 <= t < bend):
            continue
        start = idx - k
        newer = sum(1 for j in range(idx + 1, start + blen) if CAPS[j][0] <= t)
        dim = DIM[min(newer, len(DIM) - 1)]

        L = BLOCK_LAY[idx]
        if shot_kind == "stock":
            # brand-kit §5: поверх стока — левым флагом от x136, первая строка y706,
            # внутри карточки B.
            base_x, base_y, an = 136, 706, "lm"
        elif isinstance(slot, str) and slot in ("T", "B"):
            base_x, base_y, an = CARD_A[0] + 62, 372, "lm"
        elif isinstance(slot, str):
            base_x, base_y, an = SLOTS[slot][0], SLOTS[slot][1], SLOTS[slot][2]
        else:
            base_x, base_y, an = slot[0], slot[1], slot[2]

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

    tmp = f"{BUILD}/assets/_video6.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    si = 0
    for f in range(NF):
        t = f / FPS
        cap1.grab(); cap2.grab()
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
                    dict(text=tw, font=SB.font("sans", _topword_size(tw)), xy=(540, 440), anchor="mm",
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

        layers_, titems = SB.layers(kind, prm, lt, t1 - t0, t)
        for L in layers_:
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


_gfc = {}


def _topword_size(word, target=800):
    if word in _gfc:
        return _gfc[word]
    sz = 150
    for _ in range(40):
        w = font("sans", sz).getlength(word)
        if abs(w - target) < 12:
            break
        sz = max(60, min(300, int(sz * target / w)))
    _gfc[word] = sz
    return sz


if __name__ == "__main__":
    main()

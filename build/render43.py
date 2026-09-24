"""Сборка ролика 10 («задача Монти Холла»). Речь не резана — резы только по картинке.
Слот 43 — см. docstring storyboard43.py и videos/10/status.md (слот 10 занят чужим
комплектом, подмена материала).

v2/v3 (2026-09-24): новый движок субтитров (см. блок «текст (v3)»), сток без притемнения.
Движок сетки/R5/R6 — из render40.py, два кропа лица — как в render42.py,
плюс: блок субтитров рвётся и на смене плана (brand-kit §5), сцена «три двери»
(doors43.py), кадр собирается функцией compose(), которую переиспользует qa43.py.
"""
import os, sys, subprocess, math
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from storyboard43 import SHOTS, CAPS, SLOTS, DUR, FACE_KINDS, GRID_KINDS
import doors43

BUILD = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/vladimirkalajcidi/reels_challenge"
SRC = f"{ROOT}/videos/10/source.mov"
VIDEO_DIR = os.path.dirname(SRC)
A1 = f"{BUILD}/assets/aroll_43_A1.mp4"
A2 = f"{BUILD}/assets/aroll_43_A2.mp4"
TMP = f"{BUILD}/assets/_video_monty.mp4"
OUT = f"{VIDEO_DIR}/monty_hall_edit.mp4"

GRADE = ("eq=gamma=1.18:contrast=1.06:saturation=1.10:brightness=0.02,"
         "colorbalance=rm=0.04:gm=0.008:bm=-0.03,unsharp=5:5:0.30")
# источник 720x1280, линия глаз ≈ y 540: A1 — вся ширина (глаза на 41% карточки),
# A2 — крупнее, тот же дубль (глаза на 42%)
FRAMINGS = {"A1": (720, 1142, 0, 68), "A2": (560, 888, 100, 167)}

CX, CY, R_A_ = CARD_A[0], CARD_A[1], R_A
CW, CH = CARD_A[2], CARD_A[3]
BX, BY, BW, BH = CARD_B
NF = int(round(DUR * FPS))


def prep_aroll():
    os.makedirs(f"{BUILD}/assets", exist_ok=True)
    for name, out in (("A1", A1), ("A2", A2)):
        if os.path.exists(out):
            continue
        w, h, x, y = FRAMINGS[name]
        vf = f"hflip,crop={w}:{h}:{x}:{y},scale={CW}:{CH}:flags=lanczos,{GRADE}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14",
                        "-preset", "medium", "-pix_fmt", "yuv420p", out], check=True)
        print("готово:", out)


STOCK_DIR = f"{VIDEO_DIR}/stock"
STOCK_PREP_DIR = f"{STOCK_DIR}/prepared"
STOCK_CAP_CARD = (BX, BY, BW, 230)   # верхняя треть карточки B
GRID_CAP_CARD = (137, 300, 838, 300)  # зона субтитров на сетке: y 300…600, флаг от x 167


def prep_stock():
    """Каждая вставка -> карточка B 858x620, кроп до 1.385:1, 30 fps (v2: без притемнения)."""
    os.makedirs(STOCK_PREP_DIR, exist_ok=True)
    out = {}
    for i, (t0, t1, kind, prm) in enumerate(SHOTS):
        if kind != "stock":
            continue
        src = f"{STOCK_DIR}/stock_{prm['clip']}.mp4"
        dst = f"{STOCK_PREP_DIR}/sb43v2_{i}_{prm['clip']}.mp4"
        if not os.path.exists(dst):
            # v2: без притемнения — вставки подобраны светлые, текст держит тень
            vf = ("crop='min(iw,ih*1.3839)':'min(ih,iw/1.3839)',"
                  f"scale={BW}:{BH}:flags=lanczos,fps=30")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", str(prm.get("ss", 0)), "-t", f"{t1 - t0 + 0.3:.3f}",
                            "-i", src,
                            "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15",
                            "-preset", "medium", "-pix_fmt", "yuv420p", dst], check=True)
            print("готово:", dst)
        out[i] = dst
    return out


# ---------------------------------------------------------------- текст (v3)
# v2 (2026-09-24): SF Pro Expanded Black капсом, одна фраза в 1–2 строки, тень вместо свечения,
# редкие слова цветом (RED/TEAL). v3 (2026-09-24, по ролику-образцу конкурента): кегль меньше
# (кап-высота ≈2.4% экрана), текст ТОЛЬКО внизу на всех планах, анимация — каждое слово
# всплывает снизу из размытия в момент, когда его произносят (≈7 кадров).
import json
from PIL import ImageFont, ImageFilter

SIZE = {"A": 62, "B": 62, "G": 62}
BASE = {"A": 1470, "B": 1160, "G": 1500}   # базовая линия ПОСЛЕДНЕЙ строки фразы — низ карточки
COLORS = {"r": WHITE, "s": WHITE, "i": WHITE, "red": (255, 59, 48), "teal": (45, 225, 194)}
SH_DX, SH_DY, SH_BLUR, SH_A = 3, 4, 2, 0.75
POP_DUR, POP_DY, POP_BLUR = 0.23, 28, 10    # всплытие слова: подъём 28px, блюр 10→0, 0→100%
PAD = 30
DIM = [1.0, 0.74, 0.58]
wide = doors43.wide
WORDS = json.load(open(f"{VIDEO_DIR}/words.json"))


def _shot_idx(t):
    for i, (t0, t1, _, _) in enumerate(SHOTS):
        if t0 <= t < t1:
            return i
    return len(SHOTS) - 1


def _build_blocks():
    """Границы жизни фразы: как в v1 (смена плана/зоны, пауза >0.30с, по 2 в блоке),
    но на экране всегда только последняя начавшаяся фраза блока."""
    blocks, cur = [], []
    for i, (t0, t1, runs, slot) in enumerate(CAPS):
        newblock = False
        if cur:
            pi = cur[-1]
            if CAPS[pi][3] != slot or _shot_idx(CAPS[pi][0]) != _shot_idx(t0):
                newblock = True
            elif t0 - CAPS[pi][1] > 0.30:
                newblock = True
            elif len(cur) >= 2:
                newblock = True
        if newblock:
            blocks.append(cur); cur = []
        cur.append(i)
    if cur:
        blocks.append(cur)
    info = {}
    for bid, b in enumerate(blocks):
        end = max(CAPS[i][1] for i in b)
        nxt = CAPS[blocks[bid + 1][0]][0] if bid + 1 < len(blocks) else DUR
        shot_end = SHOTS[_shot_idx(CAPS[b[0]][0])][1]
        end = min(max(end, min(nxt, end + 0.30)), shot_end)
        for k, i in enumerate(b):
            info[i] = (k, end, len(b), bid)
    return info


BLOCKS = _build_blocks()


def _words(runs):
    return [(w.upper(), COLORS[k]) for (txt, k, sz) in runs for w in txt.split()]


def _word_times(ci):
    """Моменты начала слов фразы по words.json (по порядку внутри [t_in, t_out))."""
    t0, t1, runs, slot = CAPS[ci]
    n = sum(len(txt.split()) for (txt, k, sz) in runs)
    ws = [w["start"] for w in WORDS if t0 - 0.06 <= w["start"] < t1]
    if len(ws) >= n:
        return [max(t0, x) for x in ws[:n]]
    step = (t1 - t0) / max(1, n)                 # страховка: равномерно по фразе
    return [t0 + k * step for k in range(n)]


WORD_T = {i: _word_times(i) for i in range(len(CAPS))}


def _wrap(words, f, avail):
    lines, cur = [], []
    sp = f.getlength(" ")
    for w in words:
        cand = cur + [w]
        wd = sum(f.getlength(x[0]) for x in cand) + sp * (len(cand) - 1)
        if cur and wd > avail:
            lines.append(cur); cur = [w]
        else:
            cur = cand
    return lines + [cur]


def _fit(words, sz0, avail):
    """Не больше 2 строк, самое длинное слово влезает в карточку; иначе кегль вниз."""
    sz = sz0
    while True:
        f = wide(sz)
        lines = _wrap(words, f, avail)
        sp = f.getlength(" ")
        widest = max(sum(f.getlength(x[0]) for x in ln) + sp * (len(ln) - 1) for ln in lines)
        if (len(lines) <= 2 and widest <= avail) or sz <= 30:
            return f, sz, lines
        sz -= 2


def _layout(ci):
    """Раскладка фразы: слова с конечными позициями (низ-центр карточки), без анимации."""
    t0, t1, runs, slot = CAPS[ci]
    card = CARD_B if slot == "B" else CARD_A
    avail = card[2] - 2 * PAD
    f, sz, lines = _fit(_words(runs), SIZE[slot], avail)
    strs = [" ".join(w for w, _ in ln) for ln in lines]
    bb = [f.getbbox(x, anchor="ls") for x in strs]
    pitch = max([bb[k][3] - bb[k + 1][1] + sz * 0.18 for k in range(len(lines) - 1)] or [0])
    base_last = BASE[slot]
    sp = f.getlength(" ")
    out, k_w = [], 0
    for k, ln in enumerate(lines):
        wline = sum(f.getlength(w) for w, _ in ln) + sp * (len(ln) - 1)
        x = card[0] + card[2] / 2 - wline / 2
        y = base_last - (len(lines) - 1 - k) * pitch
        for w, col in ln:
            out.append(dict(text=w, font=f, xy=(x, y), anchor="ls", fill=col, glow_r=0,
                            t_word=WORD_T[ci][k_w]))
            k_w += 1
            x += f.getlength(w) + sp
    return out


LAYOUT = {i: _layout(i) for i in range(len(CAPS))}


def _active(t):
    act = [i for i, c in enumerate(CAPS) if c[0] <= t < BLOCKS[i][1]]
    return act[-1] if act else None


def caption_items(t, shot_kind=None):
    """Слова текущей фразы в КОНЕЧНЫХ позициях (для проверки наложения строк)."""
    i = _active(t)
    if i is None:
        return []
    return [dict(it, opacity=1.0) for it in LAYOUT[i] if t >= it["t_word"]]


def _word_layer(it):
    sh = dict(it, fill=(0, 0, 0), xy=(it["xy"][0] + SH_DX, it["xy"][1] + SH_DY))
    lay = text_layer((W, H), [sh])
    lay.putalpha(lay.split()[3].filter(ImageFilter.GaussianBlur(SH_BLUR)).point(lambda v: int(v * SH_A)))
    lay.alpha_composite(text_layer((W, H), [it]))
    return lay


_WL = {}


def caption_layer(t):
    """Субтитры: тень, без свечения; каждое слово всплывает снизу из размытия."""
    i = _active(t)
    if i is None:
        return None
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for it in LAYOUT[i]:
        lt = t - it["t_word"]
        if lt < 0:
            continue
        key = (i, it["text"], it["xy"])
        if key not in _WL:
            if len(_WL) > 64:
                _WL.clear()
            _WL[key] = _word_layer({k: v for k, v in it.items() if k != "t_word"})
        lay = _WL[key]
        p = min(1.0, lt / POP_DUR)
        if p < 1.0:
            e = ease_out(p)
            lay = lay.transform(lay.size, Image.AFFINE, (1, 0, 0, 0, 1, -int(round(POP_DY * (1 - e)))))
            r = POP_BLUR * (1 - e)
            if r > 0.3:
                lay = lay.filter(ImageFilter.GaussianBlur(r))
            lay = lay.copy()
            lay.putalpha(lay.split()[3].point(lambda v: int(v * e)))
        out.alpha_composite(lay)
    return out


# ---------------------------------------------------------------- сетка и графика
_GRID = {}


def grid_bg(t):
    """Сетка R4: очень медленный дрейф — фаза квантуется, кадры кешируются."""
    key = round(t * 0.8 / 0.04) * 0.04
    if key not in _GRID:
        if len(_GRID) > 60:
            _GRID.clear()
        _GRID[key] = grid_canvas(CW, CH, phase=key)
    return _GRID[key]


def gfx(t, shot):
    """Слой графики переднего плана (без сетки) + текстовые items чисел.
    Возвращает (RGBA-слой или None, items). qa43.py сравнивает это с субтитрами."""
    t0, t1, kind, prm = shot
    if kind == "doors":
        return doors43.doors_gfx(t, shot)
    if kind == "num":
        tn = prm.get("t_num", t0)
        return None, doors43.num_item(t, tn, prm["digits"], (540, 1010), 200, prm["blue"], max_w=700)
    return None, []


_gfc = {}


def ghost_font(word, target=800):
    """R6: кегль под ширину полосы ≈800px."""
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


MASK_A = rounded_mask(CW, CH, R_A_)
MASK_B = rounded_mask(BW, BH, R_B)


def compose(t, shot, frame):
    """frame — BGR-кадр для лица/стока (или None). Возвращает RGBA-холст."""
    t0, t1, kind, prm = shot
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    items = []
    if kind in FACE_KINDS:
        img = frame if frame is not None else np.zeros((CH, CW, 3), np.uint8)
        canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (CX, CY), MASK_A)
        # v3: крупное слово R6 сверху убрано — текст только внизу
    elif kind == "stock":
        if frame is not None:
            canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (BX, BY), MASK_B)
    elif kind in GRID_KINDS:
        canvas.paste(grid_bg(t), (CX, CY), MASK_A)
        lay, nums = gfx(t, shot)
        if lay is not None:
            canvas.alpha_composite(lay)
        items += nums
    if items:
        canvas.alpha_composite(text_layer((W, H), items))
    cl = caption_layer(t)
    if cl is not None:
        canvas.alpha_composite(cl)
    return canvas


# ---------------------------------------------------------------- главный цикл
def main():
    prep_aroll()
    stock_files = prep_stock()
    stock_caps = {i: cv2.VideoCapture(p) for i, p in stock_files.items()}
    stock_pos = {i: -1 for i in stock_files}
    cap1, cap2 = cv2.VideoCapture(A1), cv2.VideoCapture(A2)
    # пишем во временный файл и переименовываем в конце: прерванный рендер
    # не оставит полуживой _video_monty.mp4 (так один раз и случилось)
    PART = TMP.replace(".mp4", ".part.mp4")

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", PART],
        stdin=subprocess.PIPE)

    for f in range(NF):
        t = f / FPS
        cap1.grab(); cap2.grab()
        si = _shot_idx(t)
        shot = SHOTS[si]
        t0, t1, kind, prm = shot
        frame = None
        if kind in FACE_KINDS:
            ok, frame = (cap1 if kind == "A1" else cap2).retrieve()
            frame = frame if ok else None
        elif kind == "stock":
            v = stock_caps[si]
            want = int(round((t - t0) * FPS))
            while stock_pos[si] < want:
                if not v.grab():
                    break
                stock_pos[si] += 1
            ok, frame = v.retrieve()
            frame = frame if ok else None
        canvas = compose(t, shot, frame)
        ff.stdin.write(canvas.convert("RGB").tobytes())
        if f % 150 == 0:
            print(f"  кадр {f}/{NF}  ({t:5.1f}s)", flush=True)

    ff.stdin.close()
    assert ff.wait() == 0, "ffmpeg упал"
    cap1.release(); cap2.release()
    os.replace(PART, TMP)
    print("готово (немое):", TMP)


if __name__ == "__main__":
    main()

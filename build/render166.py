"""Сборка ролика 66 («обезьяна и Гамлет»). Речь не резана — резы только по картинке.
Слот 166 = 100 + номер папки — см. docstring storyboard166.py и videos/66/status.md.
Копия render160.py (последний сданный тракт, урок 55) без изменений логики: пути, FRAMINGS, импорт графики (ape166),
cy в имени подготовленного стока (урок 57 v2). История тракта из render160:
Скопировано с render154.py без изменений логики (кроме FRAMINGS — новая точка съёмки); сверено diff'ом с render158.py
(последний сданный, урок 55) — совпадает, кроме путей, кропов и импорта графики. render154 — копия render150.py (тот же тракт 4K, та же комната), а тот — с render148.py (ролик 48, тот же тракт 4K, та же комната) без изменений логики: субтитры v3, сток без
притемнения, A-roll/сток через *.part.mp4, ss и cx в имени подготовленного стока, «/» — принудительный перенос.
Графика ролика 66 — ∞, печать фразы, числа на сетке (ape166.py). Фото (R11) в ролике нет.
"""
import os, sys, subprocess, math
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from storyboard166 import SHOTS, CAPS, DUR, FACE_KINDS, GRID_KINDS
import ape166
import photocrop145

BUILD = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/vladimirkalajcidi/reels_challenge"
SRC = f"{ROOT}/videos/66/source.mov"
VIDEO_DIR = os.path.dirname(SRC)
A1 = f"{BUILD}/assets/aroll_166_A1.mp4"
A2 = f"{BUILD}/assets/aroll_166_A2.mp4"
TMP = f"{BUILD}/assets/_video_ape166.mp4"
OUT = f"{VIDEO_DIR}/monkey_edit.mp4"

GRADE = ("eq=gamma=1.18:contrast=1.06:saturation=1.10:brightness=0.02,"
         "colorbalance=rm=0.04:gm=0.008:bm=-0.03,unsharp=5:5:0.30")
# правка автора к ролику 40 (2026-10-02, та же комната — белая стена слева): «кадры со мной потемнее, белый фон
# слева очень светлый» — вариант Г из grade140.py (videos/40/work/grade_before_after2.jpg, «ок»); для ролика 45 —
# та же комната и точка (стена слева 229–231 в исходнике, videos/45/work/grade145.jpg): гамма 1.18 → 1.10, мягкий спад светов,
# плавное затемнение левого края карточки (×0.72 у края → ×1.0 к 34% ширины). Стена 249 → 173, лицо 165 → 154.
# Ролик 48 — та же комната (серая стена, светлая полоса слева): грейд Г без изменений
# Ролик 50 — та же комната и точка, что у 48 (стена слева 171, p95 210 в масштабе 720): грейд Г без изменений
GRADE = (GRADE.replace("gamma=1.18", "gamma=1.10") + ",curves=all='0/0 0.6/0.59 0.85/0.80 1/0.90',"
         "geq=lum='lum(X,Y)*(0.72+0.28*min(1,X/(W*0.34)))':cb='cb(X,Y)':cr='cr(X,Y)'")
# кропы по детектору лица (Haar, 7 кадров, урок ролика 30 — заново, не из прошлого ролика):
# ролик 48 — та же комната, что 40–45, но точка другая (faces148.py: центр x≈348, глаза y≈663, ширина ≈385 —
# лицо крупнее и ниже, чем у 45: 368 / 618 / 346). Кропы считаны в масштабе 720 и ×3:
# A1 640×1014 от (28, 247) — глаза ≈41% карточки, лицо ≈60% ширины; A2 520×824 от (88, 325) — глаза ≈41%, ≈74%
# Ролик 50 (faces150.py, 7 кадров): центр x≈360, глаза y≈664, ширина ≈382 — как у 48, но на 12px правее:
# A1 640×1014 от (40, 247), A2 520×824 от (100, 325) — ×3
# Ролик 54 (faces154.py, 7 кадров): центр x≈358, глаза y≈664, ширина ≈376; стена слева 177 (p95 214), лицо 132 —
# та же комната и точка, что у 50: кропы и грейд Г без изменений
# Ролик 60 (faces160.py, 7 кадров): та же комната (стена слева 158, p95 224 < 230 — грейд Г без изменений), но камера
# ближе: центр x≈362, глаза y≈605, ширина лица ≈450 (у 54 — 376 / 664). Кропы заново в масштабе 720 и ×3:
# A1 720×1142 от (0, 125) — глаза 42% карточки, лицо ≈62% ширины; A2 580×920 от (72, 219) — глаза 42%, ≈78%
# Ролик 66 (faces166.py, 8 кадров): та же комната (стена слева 218, p95 224 < 230 — грейд Г без изменений), лицо
# центр x≈378, глаза y≈664, ширина ≈455 — глаза ниже, чем у 60 (605): A1 во всю ширину упирается в низ кадра (урок 30),
# сужен до 680×1079 от (38, 201) — глаза 43%, лицо ≈67% ширины; A2 580×920 от (88, 278) — глаза 42%, ≈78%. ×3.
# hflip нужен: в исходнике «dji» на петличке задом наперёд (videos/66/work/mic.jpg, урок 57)
FRAMINGS = {"A1": (2040, 3236, 114, 604), "A2": (1740, 2760, 264, 834)}

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
        # ролик 25: пишем во временный файл и переименовываем — убитый рендер оставил
        # дочерний ffmpeg дописывать A2, следующий запуск увидел файл и взял обрубок (A2 чёрные)
        part = out.replace(".mp4", ".part.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14",
                        "-preset", "medium", "-pix_fmt", "yuv420p", part], check=True)
        os.replace(part, out)
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
        cx = prm.get("cx", 0.5)   # ролик 26: сдвиг кропа по горизонтали (0 — левый край)
        dst = f"{STOCK_PREP_DIR}/sb166_{i}_{prm['clip']}_{prm.get('ss', 0)}_{cx}_{prm.get('cy', 0.5)}.mp4"
        if not os.path.exists(dst):
            # v2: без притемнения — вставки подобраны светлые, текст держит тень
            cy = prm.get("cy", 0.5)   # вертикальный исходник: центр полосы кропа по высоте
            vf = ("crop='min(iw,ih*1.3839)':'min(ih,iw/1.3839)':"
                  f"'(iw-ow)*{cx:.3f}':"
                  f"'clip(ih*{cy:.3f}-oh/2,0,ih-oh)',"
                  f"scale={BW}:{BH}:flags=lanczos,fps=30")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", str(prm.get("ss", 0)), "-t", f"{t1 - t0 + 0.3:.3f}",
                            "-i", src,
                            "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15",
                            "-preset", "medium", "-pix_fmt", "yuv420p", dst + ".part.mp4"], check=True)
            os.replace(dst + ".part.mp4", dst)
            print("готово:", dst)
        out[i] = dst
    return out


def prep_photos():
    """R11: подлинное фото -> карточка B 858x620, кроп до 1.385:1 (центр полосы по высоте cy),
    без притемнения. Статика допустима только здесь (shot-recipes R11)."""
    out = {}
    for i, (t0, t1, kind, prm) in enumerate(SHOTS):
        if kind != "photo":
            continue
        img = cv2.imread(f"{STOCK_DIR}/{prm['src']}")
        if prm.get("fit"):   # ролик 45: портрет целиком по высоте, поля — продолжение фона (photocrop145)
            tr, gm = prm.get("trim", 0), prm.get("gamma", 1.0)
            img = img[tr:img.shape[0] - tr, tr:img.shape[1] - tr]
            img = cv2.LUT(img, np.array([255 * (k / 255) ** (1 / gm) for k in range(256)], np.uint8))
            out[i] = photocrop145.fit_full(img, BW, BH)
            continue
        h, w = img.shape[:2]
        cw = int(min(w, h * 1.3839) * prm.get("zoom", 1.0))   # ролик 41: zoom (урок ролика 31)
        ch = int(cw / 1.3839)
        y0 = int(min(max(h * prm.get("cy", 0.5) - ch / 2, 0), h - ch))
        x0 = int((w - cw) * prm.get("cx", 0.5))   # ролик 45: сдвиг кропа (штемпель не по центру)
        # ролик 45: портреты Commons маленькие (294×326) — апскейл Ланцошем, не INTER_AREA
        out[i] = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (BW, BH),
                            interpolation=cv2.INTER_AREA if cw >= BW else cv2.INTER_LANCZOS4)
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
from root125 import wide
BR = "/"   # принудительный перенос строки внутри фразы
# токены-знаки препинания («–») из whisper — не слова
WORDS = [w for w in json.load(open(f"{VIDEO_DIR}/words.json"))
         if any(ch.isalnum() for ch in w["word"])]


def _shot_idx(t):
    for i, (t0, t1, _, _) in enumerate(SHOTS):
        if t0 <= t < t1:
            return i
    return len(SHOTS) - 1


def _build_blocks():
    """Границы жизни фразы: как в v1 (смена плана/зоны, пауза >0.30с, по 2 в блоке),
    но на экране всегда только последняя начавшаяся фраза блока."""
    blocks, cur = [], []
    for i, cap in enumerate(CAPS):
        t0, t1, runs, slot = cap[:4]
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
    # «π» — строчная греческая; upper() превратил бы её в «Π»
    return [(w.upper().replace("Π", "π"), COLORS[k]) for (txt, k, sz) in runs for w in txt.split()]


def _word_times_all():
    """Моменты начала слов каждой фразы. Слова words.json разбираются по порядку: фраза
    берёт следующие n ещё не занятых слов, начавшихся не раньше t_in−0.3с (слово могло
    начаться чуть до реза — появится на резе) и до t_out. Так два слова с одинаковым
    началом (whisper: «пол» 9.12–9.12, «случайным» 9.12) не достаются обеим фразам.
    Явные тайминги (5-й элемент CAPS) — где whisper склеил слова."""
    out, used = {}, -1
    for ci, cap in enumerate(CAPS):
        t0, t1, runs, slot = cap[:4]
        n = sum(len([w for w in txt.split() if w != BR]) for (txt, k, sz) in runs)
        if len(cap) > 4:
            out[ci] = [max(t0, x) for x in cap[4]]
            used = max([j for j, w in enumerate(WORDS) if w["start"] < t1] or [used])
            continue
        idx = [j for j, w in enumerate(WORDS) if j > used and t0 - 0.3 <= w["start"] < t1][:n]
        assert len(idx) == n, f"фраза {ci} «{runs}»: слов в words.json {len(idx)} из {n}"
        out[ci] = [max(t0, WORDS[j]["start"]) for j in idx]
        used = idx[-1]
    return out


WORD_T = _word_times_all()


# ролик 22: строка не кончается и трёхбуквенным предлогом («ИЗ ЕГЭ ПРО / ВКЛАДЫ»)
PREP = {"ПРО", "ДЛЯ", "БЕЗ", "ПОД", "НАД", "ПРИ", "ОБО", "ИЗО"}


def _wrap(words, f, avail):
    """Сбалансированный перенос в ≤2 строки: жадный давал висячее слово («ЭТО ОДНО И ТО / ЖЕ»).
    Из разбиений, где обе строки влезают, берётся самое ровное по ширине; строка не должна
    кончаться коротким словом (предлог/союз ≤2 букв). Не влезает ни одно — жадный перенос."""
    sp = f.getlength(" ")
    wl = lambda ln: sum(f.getlength(x[0]) for x in ln) + sp * (len(ln) - 1)
    if wl(words) <= avail:
        return [words]
    best = None
    for k in range(1, len(words)):
        a, b = words[:k], words[k:]
        if max(wl(a), wl(b)) > avail:
            continue
        cost = max(wl(a), wl(b)) + (1000 if (len(a[-1][0]) <= 2 or a[-1][0] in PREP) and len(a) > 1 else 0)
        if best is None or cost < best[0]:
            best = (cost, [a, b])
    if best is not None:
        return best[1]
    return _wrap_greedy(words, f, avail)


def _wrap_greedy(words, f, avail):
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
    """Не больше 2 строк, самое длинное слово влезает в карточку; иначе кегль вниз.
    Ролик 21: если перенос оставляет короткое слово (предлог/союз ≤2 букв) одно в конце
    строки или одно на строке («В / КОММЕНТАРИЯХ», «НЕ / ПРОЯВЛЯЕТСЯ»), а фраза влезает
    в одну строку кеглем до −6px — ставим одной строкой чуть мельче."""
    for sz in range(sz0, sz0 - 7, -2):
        f = wide(sz)
        sp = f.getlength(" ")
        if sum(f.getlength(x[0]) for x in words) + sp * (len(words) - 1) <= avail:
            if sz == sz0:
                break
            lines = _wrap(words, wide(sz0), avail)
            if any(len(ln[-1][0]) <= 2 or (len(ln) == 1 and len(ln[0][0]) <= 2) for ln in lines[:1]):
                return f, sz, [words]
            break
    # ролик 25: единственное разбиение оставляет предлог в конце строки («НАПИШИ В / КОММЕНТАРИЯХ»),
    # а одной строкой не влезает — кеглем до −6px ищем разбиение без висячего предлога
    hang = lambda ls: len(ls) == 2 and len(ls[0]) > 1 and (len(ls[0][-1][0]) <= 2 or ls[0][-1][0] in PREP)
    if hang(_wrap(words, wide(sz0), avail)):
        for sz in range(sz0 - 2, sz0 - 7, -2):
            ls = _wrap(words, wide(sz), avail)
            if len(ls) <= 2 and not hang(ls):
                return wide(sz), sz, ls
    sz = sz0
    while True:
        f = wide(sz)
        lines = _wrap(words, f, avail)
        sp = f.getlength(" ")
        widest = max(sum(f.getlength(x[0]) for x in ln) + sp * (len(ln) - 1) for ln in lines)
        if (len(lines) <= 2 and widest <= avail) or sz <= 30:
            return f, sz, lines
        sz -= 2


def _fit_forced(ws, sz0, avail):
    """Строки заданы «/» в тексте фразы; кегль вниз по 2px, пока самая широкая не влезет."""
    lines, cur = [], []
    for w in ws:
        if w[0] == BR:
            lines.append(cur); cur = []
        else:
            cur.append(w)
    lines.append(cur)
    sz = sz0
    while True:
        f = wide(sz)
        sp = f.getlength(" ")
        widest = max(sum(f.getlength(x[0]) for x in ln) + sp * (len(ln) - 1) for ln in lines)
        if widest <= avail or sz <= 30:
            return f, sz, lines
        sz -= 2


def _layout(ci):
    """Раскладка фразы: слова с конечными позициями (низ-центр карточки), без анимации."""
    t0, t1, runs, slot = CAPS[ci][:4]
    card = CARD_B if slot == "B" else CARD_A
    avail = card[2] - 2 * PAD
    ws = _words(runs)
    if any(w == BR for w, _ in ws):
        f, sz, lines = _fit_forced(ws, SIZE[slot], avail)
    else:
        f, sz, lines = _fit(ws, SIZE[slot], avail)
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
    Возвращает (RGBA-слой или None, items).
    qa166.py сравнивает это с субтитрами."""
    if shot[2] in GRID_KINDS:
        return ape166.gfx(t, shot)
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
    elif kind in ("stock", "photo"):
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
    photos = prep_photos()
    cap1, cap2 = cv2.VideoCapture(A1), cv2.VideoCapture(A2)
    # пишем во временный файл и переименовываем в конце: прерванный рендер
    # не оставит полуживой _video_ape166.mp4 (так один раз и случилось)
    PART = TMP.replace(".mp4", ".part.mp4")

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", PART],
        stdin=subprocess.PIPE)

    last_face = {}
    for f in range(NF):
        t = f / FPS
        cap1.grab(); cap2.grab()
        si = _shot_idx(t)
        shot = SHOTS[si]
        t0, t1, kind, prm = shot
        frame = None
        if kind in FACE_KINDS:
            ok, frame = (cap1 if kind == "A1" else cap2).retrieve()
            # ролик 50: в исходнике 1437 кадров, а NF = round(47.917·30) = 1438 — последний кадр A-roll
            # не читается и выходил чёрным (QA, проверка 4); держим последний прочитанный кадр
            if ok:
                last_face[kind] = frame
            else:
                frame = last_face.get(kind)
        elif kind == "photo":
            frame = photos[si]
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

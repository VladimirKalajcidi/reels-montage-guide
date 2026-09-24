"""Сборка ролика — базовый шаблон движка. Речь не резана — резы только по картинке.

Для нового ролика: скопируй в render<N>.py, поменяй N в импорте storyboard<N> и в путях
ниже (SRC/OUT/A1/A2/tmp-файлы), подставь SRC на свой videos/<N>/source.mov и FRAMINGS —
на кроп под конкретный дубль (см. README.md, «Первый запуск» — как подобрать числа кропа).
Скрипты конкретного ролика не переименовывать и не затирать под следующий — единственный
способ пересобрать уже сданный ролик (anti-patterns.md).

Собственная графика сведена к простым числам-ревилам (R5a, белые) — по anti-patterns.md
составная инфографика (R10) систематически ломает раскладку текст/графика там, где сток
закрывает смысл не хуже.

Субтитры — общий движок captions.py (brand-kit.md, «Субтитры v3»); вставки без притемнения
(подбираются светлые клипы, текст держит тень).
"""
import os, sys, subprocess
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from storyboard import SHOTS, CAPS, SLOTS, DUR, LABELS, NUMS

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_NUM = 1  # <- номер ролика (папка videos/<N>/) — подставь свой
SRC = f"{BUILD}/../videos/{VIDEO_NUM}/source.mov"
VIDEO_DIR = os.path.dirname(SRC)
A1 = f"{BUILD}/assets/aroll_{VIDEO_NUM}_A1.mp4"
A2 = f"{BUILD}/assets/aroll_{VIDEO_NUM}_A2.mp4"
OUT = f"{BUILD}/../videos/{VIDEO_NUM}/edit.mp4"

GRADE = ("eq=gamma=1.18:contrast=1.06:saturation=1.10:brightness=0.02,"
         "colorbalance=rm=0.04:gm=0.008:bm=-0.03,unsharp=5:5:0.30")
# источник 720x1280 (подставь свои размеры, если снято иначе): A1 — почти вся ширина,
# A2 — крупнее (уже кроп, тот же дубль). Числа подбираются под конкретное видео —
# см. README.md, там пример как проверить кроп кадром перед полным рендером.
FRAMINGS = {"A1": (720, 1142, 0, 120), "A2": (560, 888, 80, 227)}

CX, CY, R_A_ = CARD_A[0], CARD_A[1], R_A
CW, CH = CARD_A[2], CARD_A[3]
NF = int(round(DUR * FPS))


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


def prep_stock():
    """Каждая видео-вставка -> карточка B 858x620, кроп до 1.385:1, 30 fps. Без притемнения:
    клипы подбираются светлые, текст держит тень (brand-kit.md, «Субтитры v3»)."""
    os.makedirs(STOCK_PREP_DIR, exist_ok=True)
    out = {}
    for i, (t0, t1, kind, prm) in enumerate(SHOTS):
        if kind != "stock":
            continue
        src = f"{STOCK_DIR}/stock_{prm['clip']}.mp4"
        dst = f"{STOCK_PREP_DIR}/sb_{i}_{prm['clip']}.mp4"
        if not os.path.exists(dst):
            vf = ("crop='min(iw,ih*1.3839)':'min(ih,iw/1.3839)',"
                  f"scale={BW}:{BH}:flags=lanczos,fps=30")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                            "-ss", str(prm.get("ss", 0)), "-t", f"{t1 - t0 + 0.2:.3f}",
                            "-i", src,
                            "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15",
                            "-preset", "medium", "-pix_fmt", "yuv420p", dst], check=True)
            print("готово:", dst)
        out[i] = dst
    return out


# ---------------------------------------------------------------- текст
# Субтитры v3 (brand-kit.md, «Субтитры v3»): общий движок build/captions.py — SF Pro Expanded
# Black капсом, одна фраза в 1–2 строки только внизу, слова всплывают из размытия по words.json,
# тень вместо свечения. Тем же шрифтом без свечения — числа (num_item) и подписи (label_item).
from captions import Captions, num_item, label_item, shadowed_layer

CAP = Captions(CAPS, SHOTS, DUR, f"{VIDEO_DIR}/words.json")


def _is_insert(kind):
    return kind == "stock"


def label_for(t, kind):
    """R7 — служебная подпись под карточкой B (сток со счётным смыслом)."""
    for t0, t1, text in LABELS:
        if t0 <= t < t1 and _is_insert(kind):
            lt = t - t0
            op = min(1.0, lt / 0.30)
            return [label_item(text, (540, BY + BH + 46), 40, opacity=op)]
    return []


def numw_layer(t, t0):
    """Белое число-ревил (R5a) — плоская тёмная карточка A вместо сетки."""
    prm = NUMS[t0]
    lt = t - t0
    L, T = [], []
    card = Image.new("RGBA", (CW, CH), (8, 8, 10, 255))
    L.append(card)
    T += num_item(t, t0 + 0.05, prm["digits"], (CX + CW // 2, CY + CH // 2 - 30),
                  prm.get("size", 150), False, max_w=CW - 120)
    T.append(label_item(prm["sub"], (CX + CW // 2, CY + CH // 2 + 130), 50,
                        opacity=min(1.0, max(0.0, lt - 0.30) / 0.3)))
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

    tmp = f"{BUILD}/assets/_video_{VIDEO_NUM}.mp4"
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
        layers, titems = [], []

        if kind in ("A1", "A2"):
            cap = cap1 if kind == "A1" else cap2
            okr, img = cap.retrieve()
            if not okr:
                img = np.zeros((CH, CW, 3), np.uint8)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            canvas.paste(pil, (CX, CY), maskA)
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
        elif kind == "numw":
            layers, titems = numw_layer(t, t0)
            for L in layers:
                canvas.paste(L, (CX, CY)) if L.size == (CW, CH) else canvas.alpha_composite(L)
            layers = []

        for L in layers:
            canvas.alpha_composite(L)
        items = titems + label_for(t, kind)
        if items:
            canvas.alpha_composite(text_layer((W, H), items))
        cl = CAP.layer(t)                   # субтитры — последним слоем
        if cl is not None:
            canvas.alpha_composite(cl)

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

"""QA версий ролика 17 с другим хуком — чеклист delivery-specs §6.

    python3 qa17hook.py h1
"""
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS
import hook17 as H
import render17 as R
import storyboard17 as SB

key = sys.argv[1]
cfg = H.HOOKS[key]
OUT = cfg["out"]
ORIG = "/Users/vladimirkalajcidi/reels_good/videos/17/luhn_edit.mp4"
sped = bool(cfg.get("post_a"))
cropped = cfg.get("post_v", "").startswith("crop")


def rect_now(rect):
    """Прямоугольник карточки на КАДРЕ ВЕРСИИ.

    Кроп ×0.98 срезает по 1% с каждой стороны и возвращает холст 1080x1920,
    то есть двигает и растягивает вообще всё, включая карточку. Проверять
    её по координатам нескропленного холста бессмысленно — маска уедет
    и собственный светлый край карточки посчитается браком.
    """
    if not cropped:
        return rect
    x, y, w, h = rect
    sx, sy, s = 1080 * 0.01, 1920 * 0.01, 1 / 0.98
    return (int(round((x - sx) * s)), int(round((y - sy) * s)),
            int(round(w * s)), int(round(h * s)))


def probe(path, ent):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", ent, "-of", "csv=p=0", path],
                          capture_output=True, text=True).stdout.strip()


def shot_at_new(t):
    """План на новой таймлинии: сначала планы хука, потом тело со сдвигом."""
    dt = cfg["frames"] / FPS - H.CUT
    for a, b, k, p in cfg["shots"]:
        if a <= t < b:
            return k
    for a, b, k, p in SB.SHOTS:
        if a >= H.CUT and a + dt <= t < b + dt:
            return k
    return SB.SHOTS[-1][2]


def bright_outside():
    """Автопроверка 1: нет пикселей ярче 200 вне прямоугольника карточки."""
    cap = cv2.VideoCapture(OUT)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad, sample = 0, []
    for f in range(0, n, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        t = f / FPS * (1.02 if sped else 1.0)
        rect = CARD_B if shot_at_new(t) == "stock" else CARD_A
        mask = np.zeros(img.shape[:2], np.uint8)
        x, y, w, h = rect_now(rect)
        mask[y:y + h, x:x + w] = 1
        if np.any((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:6]


def line_overlaps():
    """Автопроверка 2: соседние строки блока хука не пересекаются."""
    caps = cfg["caps"]
    bad = 0
    for i in range(len(caps) - 1):
        if caps[i + 1][0] - caps[i][1] > 0.30:
            continue
        s1 = max(sz for _, _, sz in caps[i][2])
        s2 = max(sz for _, _, sz in caps[i + 1][2])
        slot = R.SLOTS[caps[i][3]]
        if slot["step"] < (s1 + s2) * 0.50:
            bad += 1
    return bad


def body_identical():
    """Тело обязано совпасть с уже сданной версией кадр в кадр."""
    if sped:
        return "пропущено (версия ускорена ×1.02, покадровое сравнение неприменимо)"
    a = cv2.VideoCapture(ORIG)
    b = cv2.VideoCapture(OUT)
    off = cfg["frames"]
    worst = 0.0
    for f in range(H.CUT_F + 5, 1736, 160):
        a.set(cv2.CAP_PROP_POS_FRAMES, f)
        b.set(cv2.CAP_PROP_POS_FRAMES, f - H.CUT_F + off)
        oka, ia = a.read()
        okb, ib = b.read()
        if not (oka and okb):
            break
        if cropped:                                      # версия обрезана на 2%
            x = int(1080 * 0.01)
            y = int(1920 * 0.01)
            ia = cv2.resize(ia[y:1920 - y, x:1080 - x], (1080, 1920),
                            interpolation=cv2.INTER_LINEAR)
        d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
        worst = max(worst, float((d > 26).mean()) * 100)
    a.release()
    b.release()
    return f"расхождение по пикселям не более {worst:.2f}%"


def seam():
    """На стыке хук/тело не должно быть щелчка."""
    wav = f"{H.BUILD}/assets/_seam_{key}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", OUT,
                    "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", wav], check=True)
    import wave
    w = wave.open(wav)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    w.close()
    os.remove(wav)
    i = int(cfg["frames"] / FPS / (1.02 if sped else 1.0) * 48000)
    jump = np.abs(np.diff(d[i - 240:i + 240])).max()
    ref = np.percentile(np.abs(np.diff(d)), 99.99)
    return jump, ref


def deformation():
    if cropped:
        # один и тот же объект в обеих версиях: строка цифр номера на сетке
        return "кроп ×0.98 -> ожидаем ×1.0204 на экране: " + measure_row()
    return "ускорение ×1.02: fps=%s, длительность %s" % (
        probe(OUT, "stream=r_frame_rate"), probe(OUT, "format=duration"))


def measure_row():
    """Ширина строки номера карты в оригинале и в версии."""
    def width(path, f):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        cap.release()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[860:940, :]
        xs = np.nonzero((g > 150).any(axis=0))[0]
        return xs.max() - xs.min()
    w0 = width(ORIG, 300)                                  # план card16 в оригинале
    w1 = width(OUT, 300 - H.CUT_F + cfg["frames"])
    return f"{w0}px -> {w1}px = ×{w1/w0:.4f}"


print(f"=== версия {cfg['version']} · {os.path.basename(OUT)} ===")
print("формат:", probe(OUT, "stream=width,height,r_frame_rate"),
      "· длительность", probe(OUT, "format=duration"))
b, bs = bright_outside()
print(f"bright_outside_card_frames={b} sample={bs}")
print(f"line_overlap_pairs={line_overlaps()}")
print("тело:", body_identical())
j, ref = seam()
print(f"стык: макс скачок между сэмплами {j:.0f}, внутри дорожки p99.99 = {ref:.0f}"
      f" -> {'норма' if j <= ref else 'ЩЕЛЧОК'}")
print("деформация:", deformation())

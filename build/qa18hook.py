"""QA версий ролика 18 с другим хуком — чеклист delivery-specs §6.

    python3 qa18hook.py h1
"""
import os
import subprocess
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS
import hook18 as H
import render18 as R
import storyboard18 as SB

key = sys.argv[1]
cfg = H.HOOKS[key]
OUT = cfg["out"]
ORIG = "/Users/vladimirkalajcidi/reels_good/videos/18/leap_edit.mp4"
CUT_F = cfg["cut_f"]
CUT = CUT_F / FPS
sped = bool(cfg.get("post_a"))
cropped = cfg.get("post_v", "").startswith("crop")


def rect_now(rect):
    """Прямоугольник карточки на КАДРЕ ВЕРСИИ.

    Кроп ×0.98 срезает по 1% с каждой стороны и возвращает холст 1080x1920,
    то есть двигает и растягивает вообще всё, включая карточку. Проверять
    её по координатам нескропленного холста бессмысленно — маска уедет
    и собственный светлый край карточки посчитается браком.

    Границы после кропа попадают на дробный пиксель (правый край карточки A
    уезжает на 984.9), поэтому начало берём floor, конец ceil и добавляем
    1px на растушёвку bilinear. Без этого собственный край карточки
    считается «текстом за карточкой» — это артефакт ресемпла, а не брак кадра.
    """
    if not cropped:
        return rect
    x, y, w, h = rect
    sx, sy, s = 1080 * 0.01, 1920 * 0.01, 1 / 0.98
    x0 = int(np.floor((x - sx) * s)) - 1
    y0 = int(np.floor((y - sy) * s)) - 1
    x1 = int(np.ceil((x + w - sx) * s)) + 1
    y1 = int(np.ceil((y + h - sy) * s)) + 1
    return (max(0, x0), max(0, y0), x1 - max(0, x0), y1 - max(0, y0))


def probe(path, ent):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                           "-show_entries", ent, "-of", "csv=p=0", path],
                          capture_output=True, text=True).stdout.strip()


def shot_at_new(t):
    """План на новой таймлинии: сначала планы хука, потом тело со сдвигом."""
    dt = cfg["frames"] / FPS - CUT
    for a, b, k, p in cfg["shots"]:
        if a <= t < b:
            return k
    for a, b, k, p in SB.SHOTS:
        if a >= CUT and a + dt <= t < b + dt:
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


def gfx_text_overlap():
    """Графика и субтитры блока хука не имеют общих пикселей (альфа >40)."""
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()
    lts = H.lts_for(cfg)
    bad, sample = 0, []
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        t0, t1, kind, _p = R.shot_at(t)
        gl = R.graphics_layer(kind, (t - t0) * lts.get(kind, 1.0))
        cl = R.caption_layer(t)
        if gl is None or cl is None:
            continue
        n = int(np.count_nonzero((np.array(gl.split()[3]) > 40)
                                 & (np.array(cl.split()[3]) > 40)))
        if n:
            bad += 1
            sample.append((round(t, 2), kind, n))
    return bad, sample[:6]


def text_in_card():
    """Субтитры блока хука целиком внутри своей карточки с отступом 30px."""
    bad, sample = 0, []
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        kind = R.shot_at(t)[2]
        cl = R.caption_layer(t)
        if cl is None:
            continue
        ys, xs = np.nonzero(np.array(cl.split()[3]) > 120)
        if len(ys) == 0:
            continue
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        if (xs.min() < x + 30 or xs.max() > x + w - 30
                or ys.min() < y + 30 or ys.max() > y + h - 30):
            bad += 1
            sample.append((round(t, 2), kind))
    return bad, sample[:6]


def line_overlaps():
    """Соседние строки блока хука не пересекаются."""
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


def old_hook_gone():
    """От старого начала не должно остаться ни кадра: тело склеено с cut_f.

    Проверяем не конфиг, а сам кадр: cut_f обязан быть ПЕРВЫМ кадром своего
    плана. Границы планов стоят на дробных кадрах (8.08с = кадр 242.4), и
    round вместо ceil оставляет в сборке последний кадр выброшенного плана —
    ровно этот дефект был у h1 в первой сборке.
    """
    prev = SB.shot_at((CUT_F - 1) / FPS)
    cur = SB.shot_at(CUT_F / FPS)
    ok = prev is not cur
    idx = SB.SHOTS.index(cur)            # первый СОХРАНЁННЫЙ план
    edge = SB.SHOTS[idx][0]
    return (f"выброшены планы 1–{idx} "
            f"({SB.SHOTS[0][0]:.2f}–{edge:.2f}с) и все их субтитры "
            f"({sum(1 for c in SB.CAPS if c[0] < edge)} фраз); "
            f"кадр {CUT_F - 1} -> `{prev[2]}`, кадр {CUT_F} -> `{cur[2]}` "
            f"-> {'рез ровно по границе плана' if ok else 'ХВОСТ СТАРОГО ПЛАНА'}")


def body_identical():
    """Тело обязано совпасть с уже сданной версией кадр в кадр."""
    if sped:
        return "пропущено (версия ускорена ×1.02, покадровое сравнение неприменимо)"
    a = cv2.VideoCapture(ORIG)
    b = cv2.VideoCapture(OUT)
    off = cfg["frames"]
    n = int(a.get(cv2.CAP_PROP_FRAME_COUNT))
    worst = 0.0
    for f in range(CUT_F + 5, n - 2, 120):
        a.set(cv2.CAP_PROP_POS_FRAMES, f)
        b.set(cv2.CAP_PROP_POS_FRAMES, f - CUT_F + off)
        oka, ia = a.read()
        okb, ib = b.read()
        if not (oka and okb):
            break
        if cropped:                                      # версия обрезана на 2%
            x, y = int(1080 * 0.01), int(1920 * 0.01)
            ia = cv2.resize(ia[y:1920 - y, x:1080 - x], (1080, 1920),
                            interpolation=cv2.INTER_LINEAR)
        d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
        worst = max(worst, float((d > 26).mean()) * 100)
    a.release()
    b.release()
    return f"расхождение по пикселям не более {worst:.2f}%"


def seam():
    """На стыке хук/тело не должно быть щелчка."""
    wav = f"{H.BUILD}/assets/_seam18_{key}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", OUT,
                    "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", wav], check=True)
    w = wave.open(wav)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    w.close()
    os.remove(wav)
    i = int(cfg["frames"] / FPS / (1.02 if sped else 1.0) * 48000)
    jump = np.abs(np.diff(d[i - 240:i + 240])).max()
    ref = np.percentile(np.abs(np.diff(d)), 99.99)
    return jump, ref


def card_width(path, f):
    """Ширина карточки A на кадре с лицом: карточка сильно светлее чёрного поля."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    cap.release()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[900:1000, :]
    xs = np.nonzero((g > 60).any(axis=0))[0]
    return int(xs.max() - xs.min() + 1)


def deformation():
    """Деформацию меряем машиной: один и тот же объект в обеих версиях."""
    if cropped:
        f_orig = int(round(46.0 * FPS))                  # план A2, лицо в карточке
        w0 = card_width(ORIG, f_orig)
        w1 = card_width(OUT, f_orig - CUT_F + cfg["frames"])
        return (f"кроп ×0.98 -> ожидаем ×1.0204: ширина карточки A "
                f"{w0}px -> {w1}px = ×{w1 / w0:.4f}")
    return ("ускорение ×1.02: fps=%s (норма 30/1), длительность %s, "
            "аудио %s" % (probe(OUT, "stream=r_frame_rate"),
                          probe(OUT, "format=duration"),
                          subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                          "-show_entries", "stream=duration", "-of", "csv=p=0",
                                          OUT], capture_output=True, text=True).stdout.strip()))


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr.splitlines()
    tail = [l.strip() for l in r[-20:]]
    g = lambda k: next((l.split()[-2] for l in tail if l.startswith(k)), "?")
    return g("I:"), g("LRA:"), g("Peak:")


print(f"=== версия {cfg['version']} · {os.path.basename(OUT)} ===")
print("формат:", probe(OUT, "stream=width,height,r_frame_rate"),
      "· длительность", probe(OUT, "format=duration"))
print("старый хук:", old_hook_gone())
b, bs = bright_outside()
print(f"bright_outside_card_frames={b} sample={bs}")
o, os_ = gfx_text_overlap()
print(f"gfx_text_overlap_frames={o} sample={os_}")
tc, tcs = text_in_card()
print(f"text_out_of_card_frames={tc} sample={tcs}")
print(f"line_overlap_pairs={line_overlaps()}")
print("тело:", body_identical())
j, ref = seam()
print(f"стык: макс скачок между сэмплами {j:.0f}, внутри дорожки p99.99 = {ref:.0f}"
      f" -> {'норма' if j <= ref else 'ЩЕЛЧОК'}")
print("деформация:", deformation())
print("громкость версии:", loudness(OUT), "· оригинал:", loudness(ORIG))

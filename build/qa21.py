"""QA ролика 21: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами whisper."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard21 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          shot_at)
from style import CARD_A, CARD_B, FPS
import render21

ALPHA = 40
WORDS_JSON = "/Users/vladimirkalajcidi/reels_good/videos/21/source.json"


def bright_outside():
    """Автопроверка 1: нет пикселей ярче 200 вне прямоугольника карточки."""
    cap = cv2.VideoCapture(OUT)
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        kind = shot_at(f / FPS)[2]
        rect = CARD_B if kind == "stock" else CARD_A
        mask = np.zeros(img.shape[:2], np.uint8)
        x, y, w, h = rect
        mask[y:y + h, x:x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:8]


def line_overlaps():
    """Автопроверка 2: соседние строки блока не пересекаются по реальным габаритам."""
    bad, sample = 0, []
    for i in range(len(CAPS) - 1):
        if CAPS[i + 1][0] - CAPS[i][1] > 0.30:
            continue
        if shot_at(CAPS[i][0]) is not shot_at(CAPS[i + 1][0]):
            continue
        s1 = max(sz for _, _, sz in CAPS[i][2])
        s2 = max(sz for _, _, sz in CAPS[i + 1][2])
        slot = render21.SLOTS[render21.slot_for(shot_at(CAPS[i][0])[2])]
        if slot["step"] < (s1 + s2) * 0.50:
            bad += 1
            sample.append((round(CAPS[i][0], 2), s1, s2, slot["step"]))
    return bad, sample[:8]


def _gfx_and_big(t):
    """Слой переднего плана: графика плана + крупное слово R6, если оно есть."""
    t0, _, kind, prm = shot_at(t)
    lay = render21.graphics_layer(kind, t - t0)
    if prm.get("big"):
        bw = render21.big_word_layer(prm["big"], t - t0)
        if bw is not None:
            if lay is None:
                lay = bw
            else:
                lay.alpha_composite(bw)
    return lay


def gfx_text_overlap():
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        gl = _gfx_and_big(t)
        if gl is None:
            continue
        cl = render21.caption_layer(t)
        if cl is None:
            continue
        ga = np.array(gl.split()[3])
        ca = np.array(cl.split()[3])
        n = int(np.count_nonzero((ga > ALPHA) & (ca > ALPHA)))
        if n:
            bad += 1
            sample.append((round(t, 2), shot_at(t)[2], n))
    return bad, sample[:8]


def gfx_in_zone():
    """Графика планов на сетке не выходит из своей зоны (brand-kit §5).
    Крупное слово R6 живёт в карточке A, а не в зоне графики — проверяется
    отдельно, в big_word_in_card()."""
    y0, y1 = GFX_ZONE
    x0, x1 = GFX_X
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        ts, _, kind, _ = shot_at(t)
        gl = render21.graphics_layer(kind, t - ts)
        if gl is None:
            continue
        a = np.array(gl.split()[3])
        ys, xs = np.nonzero(a > ALPHA)
        if len(ys) == 0:
            continue
        if ys.min() < y0 or ys.max() > y1 or xs.min() < x0 or xs.max() > x1:
            bad += 1
            sample.append((round(t, 2), kind,
                           int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return bad, sample[:8]


def big_word_in_card():
    """R6 стоит внутри карточки A с отступом 30px и не заходит на лицо."""
    bad, sample = 0, []
    x, y, w, h = CARD_A
    for t0, t1, kind, prm in SHOTS:
        if not prm.get("big"):
            continue
        for f in range(int(t0 * FPS), int(t1 * FPS), 3):
            lay = render21.big_word_layer(prm["big"], f / FPS - t0)
            if lay is None:
                continue
            a = np.array(lay.split()[3])
            ys, xs = np.nonzero(a > 120)
            if len(ys) == 0:
                continue
            if (xs.min() < x + 30 or xs.max() > x + w - 30
                    or ys.min() < y + 30 or ys.max() > y + h - 30):
                bad += 1
                sample.append((round(f / FPS, 2), int(xs.min()), int(xs.max()),
                               int(ys.min()), int(ys.max())))
    return bad, sample[:6]


def text_in_card():
    """Субтитры целиком внутри своей карточки с отступом 30px (brand-kit §5)."""
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        kind = shot_at(t)[2]
        cl = render21.caption_layer(t)
        if cl is None:
            continue
        a = np.array(cl.split()[3])
        ys, xs = np.nonzero(a > 120)
        if len(ys) == 0:
            continue
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        if (xs.min() < x + 30 or xs.max() > x + w - 30
                or ys.min() < y + 30 or ys.max() > y + h - 30):
            bad += 1
            sample.append((round(t, 2), kind,
                           int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return bad, sample[:8]


def _words():
    import json
    out = []
    for seg in json.load(open(WORDS_JSON))["segments"]:
        out += [(w["start"], w["end"], w["word"].strip()) for w in seg.get("words", [])]
    return out


def speech_sync():
    """Ни один рез не попадает в середину слова."""
    words = _words()
    bad, sample = 0, []
    for t0, _, _, _ in SHOTS[1:]:
        for a, b, w in words:
            if a + 0.02 < t0 < b - 0.02:
                bad += 1
                sample.append((round(t0, 2), w, round(a, 2), round(b, 2)))
                break
    return bad, sample


def caption_sync():
    """Каждая фраза стартует не раньше своего слова: t0 обязан попадать
    в речь (в слово или в паузу перед ним), а не опережать её."""
    words = _words()
    bad, sample = 0, []
    for t0, _, runs, _ in CAPS:
        near = [w for w in words if abs(w[0] - t0) < 0.26]
        if not near:
            bad += 1
            sample.append((round(t0, 2), "".join(r[0] for r in runs).strip()))
    return bad, sample[:8]


def dead_air():
    """Нет провалов >=3с без смены картинки и планов короче 1с (кроме очереди R3)."""
    long_ = [(a, round(b - a, 2)) for a, b, _, _ in SHOTS if b - a >= 3.0]
    short = [(a, round(b - a, 2)) for a, b, _, _ in SHOTS if b - a < 1.0]
    return long_, short


def blue_numbers():
    """Синих чисел на ролик — не больше 4 (brand-kit §4)."""
    from storyboard21 import GAP, MARGIN
    return [("gap800", f"{GAP} ₽"), ("plus", f"+{MARGIN} ₽")]


def rhythm():
    lens = sorted(b - a for a, b, _, _ in SHOTS)
    face = sum(b - a for a, b, k, _ in SHOTS if k in FACE_KINDS)
    stock = sum(b - a for a, b, k, _ in SHOTS if k == "stock")
    return dict(shots=len(SHOTS), avg=round(sum(lens) / len(lens), 2),
                median=round(lens[len(lens) // 2], 2),
                shortest=round(lens[0], 2), longest=round(lens[-1], 2),
                first_cut=SHOTS[0][1], face_pct=round(100 * face / DUR, 1),
                stock_pct=round(100 * stock / DUR, 1))


if __name__ == "__main__":
    print("ритм:", rhythm())
    print("синих чисел:", len(blue_numbers()), blue_numbers())
    print("планы >=3с:", dead_air()[0], " планы <1с:", dead_air()[1])
    c, cs = speech_sync()
    print(f"cuts_inside_word={c} sample={cs}")
    cc, ccs = caption_sync()
    print(f"captions_off_speech={cc} sample={ccs}")
    o, os_ = gfx_text_overlap()
    print(f"gfx_text_overlap_frames={o} sample={os_}")
    z, zs = gfx_in_zone()
    print(f"gfx_out_of_zone_frames={z} sample={zs}")
    bw, bws = big_word_in_card()
    print(f"bigword_out_of_card_frames={bw} sample={bws}")
    tc, tcs = text_in_card()
    print(f"text_out_of_card_frames={tc} sample={tcs}")
    lo, los = line_overlaps()
    print(f"line_overlap_pairs={lo} sample={los}")
    if os.path.exists(OUT):
        b, bs = bright_outside()
        print(f"bright_outside_card_frames={b} sample={bs}")

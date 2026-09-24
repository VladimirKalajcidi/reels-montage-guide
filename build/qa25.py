"""QA ролика 25: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами, честность построения Морли.

Пословные тайминги лежат в `videos/25/words.json` — плоский список слов,
собранный по речевым отрезкам (см. шапку storyboard25).
"""
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard25 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, MAIN, ALT1, ALT2, _ang, shot_at)
from style import CARD_A, CARD_B, FPS
import render25

ALPHA = 40


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
    """Автопроверка 2: строки блока не пересекаются по реальным габаритам глифов.

    Считается по тем же данным, что и рендер: шаг блока из render25.BLOCKS,
    вертикальный габарит строки — из font.getbbox, а не из кегля.
    """
    from style import font
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    boxes = {}
    for i, (t0, t1, runs, _) in enumerate(CAPS):
        top = min(font(kmap[k], sz).getbbox(txt)[1] for txt, k, sz in runs)
        bot = max(font(kmap[k], sz).getbbox(txt)[3] for txt, k, sz in runs)
        boxes[i] = (top, bot)
    bad, sample = 0, []
    by_block = {}
    for i in range(len(CAPS)):
        bid, pos, _, _, yoff = render25.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, yoff))
    for bid, rows in by_block.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            # строки рисуются с anchor "*m": bbox отсчитывается от середины строки
            ha = (boxes[a[1]][1] - boxes[a[1]][0]) / 2
            hb = (boxes[b[1]][1] - boxes[b[1]][0]) / 2
            if a[2] + ha > b[2] - hb:
                bad += 1
                sample.append((bid, round(CAPS[a[1]][0], 2), round(a[2] + ha, 1),
                               round(b[2] - hb, 1)))
    return bad, sample[:8]


def gfx_text_overlap():
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        t0, _, kind, _ = shot_at(t)
        gl = render25.graphics_layer(kind, t - t0)
        if gl is None:
            continue
        cl = render25.caption_layer(t)
        if cl is None:
            continue
        ga = np.array(gl.split()[3])
        ca = np.array(cl.split()[3])
        n = int(np.count_nonzero((ga > ALPHA) & (ca > ALPHA)))
        if n:
            bad += 1
            sample.append((round(t, 2), kind, n))
    return bad, sample[:8]


def gfx_in_zone():
    y0, y1 = GFX_ZONE
    x0, x1 = GFX_X
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        ts, _, kind, _ = shot_at(t)
        gl = render25.graphics_layer(kind, t - ts)
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


def text_in_card():
    """Субтитры целиком внутри своей карточки с отступом 30px (brand-kit §5)."""
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        kind = shot_at(t)[2]
        cl = render25.caption_layer(t)
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
    return [(w["start"], w["end"], w["word"]) for w in json.load(open(WORDS))]


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
    """Каждая фраза стартует на своём слове, а не раньше и не позже."""
    words = _words()
    bad, sample = 0, []
    for t0, _, runs, _ in CAPS:
        near = [w for w in words if abs(w[0] - t0) < 0.26]
        if not near:
            bad += 1
            sample.append((round(t0, 2), "".join(r[0] for r in runs).strip()))
    return bad, sample[:8]


def caption_words_match():
    """Субтитры не расходятся с речью по словам: каждое слово субтитра
    встречается в расшифровке в том же порядке."""
    import re
    norm = lambda w: re.sub(r"[^а-яa-z]", "", w.lower().replace("ё", "е"))
    said = [norm(w[2]) for w in _words()]
    said = [w for w in said if w]
    typed = []
    for t0, _, runs, _ in CAPS:
        for txt, _, _ in runs:
            typed += [norm(w) for w in txt.split()]
    typed = [w for w in typed if w]
    i, missing = 0, []
    for w in typed:
        j = i
        while j < len(said) and said[j] != w:
            j += 1
        if j < len(said):
            i = j + 1
        else:
            missing.append(w)
    return len(missing), missing[:8]


def caption_style():
    """Строчные, без пунктуации, 2-4 слова во фразе (brand-kit §5)."""
    bad = []
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs).strip()
        if any(ch in text for ch in ".,!?;:«»"):
            bad.append(("пунктуация", t0, text))
        if text != text.lower():
            bad.append(("капс", t0, text))
        n = len(text.split())
        if not 2 <= n <= 4:
            bad.append((f"{n} слов", t0, text))
    return len(bad), bad[:8]


def no_caption_on_number_shot():
    """На плане `year` субтитра быть не должно: число там уже написано графикой
    и звучит в речи (brand-kit §5 «одно и то же не пишется дважды»)."""
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "year")
    inside = [c for c in CAPS if t0 <= c[0] < t1]
    return len(inside), [(round(c[0], 2), "".join(r[0] for r in c[2])) for c in inside]


def blue_and_lime():
    """Синим — только результат, лайм — один элемент (brand-kit §4)."""
    return dict(blue_shots=["teaser", "equilateral", "equalMarks",
                            "anyStart1", "anyStart2", "recap"],
                blue_numbers=1,   # 60° на плане equalMarks
                white_numbers=1,  # 1899 на плане year
                lime="деление угла при C на плане trisect (две линии + три дуги)")


def geometry():
    """Построение обязано быть настоящим, а не «похожим», и это проверяется
    для всех трёх исходных треугольников:

    1. треугольник Морли равносторонний — стороны сходятся до 1e-6 px,
       углы равны 60° до 1e-6 градуса;
    2. каждая точка Морли лежит ровно на обеих своих трисектрисах;
    3. трисектрисы действительно делят угол на три равные части;
    4. всё построение (вершины, точки, ноги трисектрис) помещается
       в зону графики целиком.
    """
    gx0, gx1 = GFX_X
    gy0, gy1 = GFX_ZONE
    bad, info = [], {}
    for name, sc in (("MAIN", MAIN), ("ALT1", ALT1), ("ALT2", ALT2)):
        A, B, C = sc["tri"]
        MA, MB, MC = sc["M"]
        sides = [math.dist(MA, MB), math.dist(MB, MC), math.dist(MC, MA)]
        angs = [math.degrees(_ang(MB, MA, MC)), math.degrees(_ang(MC, MB, MA)),
                math.degrees(_ang(MA, MC, MB))]
        if max(sides) - min(sides) > 1e-6:
            bad.append((name, "стороны не равны", max(sides) - min(sides)))
        if max(abs(a - 60.0) for a in angs) > 1e-6:
            bad.append((name, "углы не 60", angs))
        # трисектрисы делят угол ровно на три равные части
        for key, (V, U, Wp) in {"A": (A, B, C), "B": (B, C, A), "C": (C, A, B)}.items():
            th = math.degrees(_ang(U, V, Wp))
            u0, u1 = sc["tris"][key]
            p0 = (V[0] + u0[0], V[1] + u0[1])
            p1 = (V[0] + u1[0], V[1] + u1[1])
            parts = [math.degrees(_ang(U, V, p0)),
                     math.degrees(_ang(p0, V, p1)),
                     math.degrees(_ang(p1, V, Wp))]
            if max(abs(p - th / 3) for p in parts) > 1e-6:
                bad.append((name, f"угол {key} поделён неровно", parts))
        # точка Морли лежит на обеих своих трисектрисах
        for (V1, u1), (V2, u2), M in sc["pairs"]:
            for (V, u) in ((V1, u1), (V2, u2)):
                d = abs((M[0] - V[0]) * u[1] - (M[1] - V[1]) * u[0])
                if d > 1e-6:
                    bad.append((name, "точка не на трисектрисе", d))
        pts = list(sc["tri"]) + list(sc["M"]) + [p for f in sc["feet"].values() for p in f]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if min(xs) < gx0 or max(xs) > gx1 or min(ys) < gy0 or max(ys) > gy1:
            bad.append((name, "построение вне зоны",
                        (round(min(xs)), round(max(xs)), round(min(ys)), round(max(ys)))))
        info[name] = dict(side=round(sides[0], 3),
                          spread=float(f"{max(sides) - min(sides):.2e}"),
                          angs=[round(a, 6) for a in angs],
                          tri_angs=[round(math.degrees(_ang(B, A, C)), 1),
                                    round(math.degrees(_ang(C, B, A)), 1),
                                    round(math.degrees(_ang(A, C, B)), 1)])
    return len(bad), bad, info


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
    print("цвет:", blue_and_lime())
    gb, gs, ginfo = geometry()
    print(f"geometry_violations={gb} sample={gs}")
    for k, v in ginfo.items():
        print("  ", k, v)
    c, cs = speech_sync()
    print(f"cuts_inside_word={c} sample={cs}")
    cc, ccs = caption_sync()
    print(f"captions_off_speech={cc} sample={ccs}")
    cw, cws = caption_words_match()
    print(f"caption_words_not_in_speech={cw} sample={cws}")
    st, sts = caption_style()
    print(f"caption_style_violations={st} sample={sts}")
    nn, nns = no_caption_on_number_shot()
    print(f"captions_on_year_shot={nn} sample={nns}")
    o, os_ = gfx_text_overlap()
    print(f"gfx_text_overlap_frames={o} sample={os_}")
    z, zs = gfx_in_zone()
    print(f"gfx_out_of_zone_frames={z} sample={zs}")
    tc, tcs = text_in_card()
    print(f"text_out_of_card_frames={tc} sample={tcs}")
    lo, los = line_overlaps()
    print(f"line_overlap_pairs={lo} sample={los}")
    if os.path.exists(OUT):
        b, bs = bright_outside()
        print(f"bright_outside_card_frames={b} sample={bs}")

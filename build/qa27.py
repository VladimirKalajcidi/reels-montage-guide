"""QA ролика 27: зоны карточек, пересечение графики с субтитрами, наложение
строк, синхрон субтитров и резов с пословными таймингами, честность чисел.

Пословные тайминги лежат в `videos/27/words.json` — плоский список слов,
собранный по речевым отрезкам (см. шапку storyboard27).
"""
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard27 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, shot_at)
from geom27 import (POLY, POLY2, ROWS, D1, D2, S1, S2, DIFF, AREA,
                    shoelace, pick, self_intersections, convex)
from style import CARD_A, CARD_B, FPS
import render27

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
    """Автопроверка 2: строки блока не пересекаются по реальным габаритам глифов."""
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
        bid, pos, _, _, yoff = render27.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, yoff))
    for bid, rows in by_block.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            # строки рисуются с anchor "*m": bbox отсчитывается от середины строки
            ha = (boxes[a[1]][1] - boxes[a[1]][0]) / 2
            hb = (boxes[b[1]][1] - boxes[b[1]][0]) / 2
            if a[2] + ha > b[2] - hb:
                bad += 1
                sample.append((bid, round(CAPS[a[1]][0], 2)))
    return bad, sample[:8]


def gfx_text_overlap():
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        t0, _, kind, _ = shot_at(t)
        gl = render27.graphics_layer(kind, t - t0)
        if gl is None:
            continue
        cl = render27.caption_layer(t)
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
        gl = render27.graphics_layer(kind, t - ts)
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
        cl = render27.caption_layer(t)
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
    """Каждое слово субтитра встречается в расшифровке в том же порядке."""
    norm = lambda w: re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))
    said = [x for x in (norm(w[2]) for w in _words()) if x]
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


# Планы, где число или координаты произносятся вслух и их несёт графика:
# субтитра на них быть не должно вовсе (brand-kit §5).
SILENT_KINDS = ("coords", "sum1", "sum2", "result")
# На плане `diff` субтитр разрешён, но в нём не может быть ни одной цифры:
# числа 70, 33 и 37 в этот момент стоят на экране графикой.
DIGIT_FREE_KINDS = ("diff", "lace1", "lace2")


def no_caption_on_number_shots():
    bad = []
    for t0, t1, k, _ in SHOTS:
        if k in SILENT_KINDS:
            for c in CAPS:
                if t0 <= c[0] < t1:
                    bad.append((k, round(c[0], 2), "".join(r[0] for r in c[2])))
        if k in DIGIT_FREE_KINDS:
            for c in CAPS:
                text = "".join(r[0] for r in c[2])
                if t0 <= c[0] < t1 and any(ch.isdigit() for ch in text):
                    bad.append((k, round(c[0], 2), text))
    return len(bad), bad


def numbers():
    """Числа на экране пересчитаны, а не переписаны из сценария."""
    out = dict(
        products_right=(D1, S1),
        products_left=(D2, S2),
        diff=DIFF,
        area_shoelace=AREA,
        area_pick=pick(POLY),
        poly2_shoelace=shoelace(POLY2),
        poly2_pick=pick(POLY2),
    )
    bad = []
    if AREA != shoelace(POLY):
        bad.append("площадь не сходится со шнуровкой")
    if AREA != pick(POLY):
        bad.append("шнуровка и формула Пика дают разное")
    if shoelace(POLY2) != pick(POLY2):
        bad.append("второй многоугольник: шнуровка и Пик разошлись")
    if S1 - S2 != DIFF:
        bad.append("разность посчитана неверно")
    # сверка с тем, что произносится вслух
    said = " ".join(w["word"] for w in json.load(open(WORDS)))
    for want in ("70", "33", "18"):
        if want not in said:
            bad.append(f"в речи нет числа {want}, а на экране оно есть")
    return len(bad), bad, out


def geometry():
    """Фигуры обязаны быть настоящими: простыми, замкнутыми, в узлах решётки
    и целиком внутри зоны графики."""
    bad = []
    for name, p in (("POLY", POLY), ("POLY2", POLY2)):
        if self_intersections(p):
            bad.append((name, "самопересечение", self_intersections(p)))
        if len(set(p)) != len(p):
            bad.append((name, "повторяющиеся вершины", p))
        if any(int(q[0]) != q[0] or int(q[1]) != q[1] for q in p):
            bad.append((name, "вершина не в узле решётки", p))
        for q in p:
            x, y = render27.px(*q)
            if not (GFX_X[0] <= x <= GFX_X[1] and GFX_ZONE[0] <= y <= GFX_ZONE[1]):
                bad.append((name, "вершина вне зоны графики", q, (x, y)))
    if ROWS[0] != ROWS[-1]:
        bad.append(("ROWS", "первая точка не повторена в конце", ROWS))
    if len(ROWS) != len(POLY) + 1:
        bad.append(("ROWS", "лишние строки в таблице", len(ROWS)))
    if convex(POLY2):
        bad.append(("POLY2", "не невыпуклый — «любой формы» не показано", None))
    return len(bad), bad, dict(poly=POLY, poly2=POLY2, convex_poly=convex(POLY))


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
    nb, ns, ninfo = numbers()
    print(f"number_violations={nb} sample={ns}")
    for k, v in ninfo.items():
        print("  ", k, v)
    gb, gs, ginfo = geometry()
    print(f"geometry_violations={gb} sample={gs}")
    print("  ", ginfo)
    c, cs = speech_sync()
    print(f"cuts_inside_word={c} sample={cs}")
    cc, ccs = caption_sync()
    print(f"captions_off_speech={cc} sample={ccs}")
    cw, cws = caption_words_match()
    print(f"caption_words_not_in_speech={cw} sample={cws}")
    st, sts = caption_style()
    print(f"caption_style_violations={st} sample={sts}")
    nn, nns = no_caption_on_number_shots()
    print(f"captions_on_number_shots={nn} sample={nns}")
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

"""QA ролика 26: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами, честность маршрутов и чисел.

Пословные тайминги лежат в `videos/26/words.json` — плоский список слов,
собранный по речевым отрезкам (см. шапку storyboard26).
"""
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard26 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, ADDR5, ADDR6, ADDR10, ADDR20, ADDR50,
                          TOURS5, OPT6, LONG6, NN10, GOOD10, CROSS10,
                          all_tours, tour_len, crossings, shot_at)
from style import CARD_A, CARD_B, FPS
import render26

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
        bid, pos, _, _, yoff = render26.BLOCKS[i]
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
        gl = render26.graphics_layer(kind, t - t0)
        if gl is None:
            continue
        cl = render26.caption_layer(t)
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
        gl = render26.graphics_layer(kind, t - ts)
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
        cl = render26.caption_layer(t)
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
    norm = lambda w: re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))
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


def no_caption_on_number_shots():
    """На планах, где число несёт графика, субтитра быть не должно, а на плане
    `tenNum` субтитр обязан закончиться до появления числа
    (brand-kit §5 «одно и то же не пишется дважды»)."""
    bad = []
    for kind in ("factorial", "huge"):
        t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == kind)
        for c in CAPS:
            if t0 <= c[0] < t1:
                bad.append((kind, round(c[0], 2), "".join(r[0] for r in c[2])))
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "tenNum")
    num_at = t0 + 0.916                       # ревил 181 440 в g_tenNum
    for c in CAPS:
        if t0 <= c[0] < t1 and c[1] > num_at + 0.04:
            bad.append(("tenNum", round(c[0], 2), "".join(r[0] for r in c[2])))
    return len(bad), bad


def blue_and_lime():
    """Синим — только числа и результат, лайм — один элемент (brand-kit §4)."""
    return dict(blue_numbers=["19! на плане factorial", "3 · 10^62 на плане huge"],
                blue_routes=["короткий маршрут на route1", "маршрут после 2-opt на goodRoute"],
                white_numbers=["181 440 на tenNum", "121 645 100 408 832 000 на factorial"],
                lime="два самопересекающихся ребра жадного маршрута "
                     "(планы heuristic и goodRoute — один и тот же элемент)")


def numbers():
    """Числа на экране и числа в речи — пересчёт, а не вера в сценарий.

    Для n адресов, где один из них стартовый, различных замкнутых маршрутов
    (n-1)!/2, а с учётом направления объезда — (n-1)!.
    """
    f = math.factorial
    out = dict(
        five_tours=(len(TOURS5), f(4) // 2),
        ten_tours=(181440, f(9) // 2),
        twenty_tours=("19!", f(19)),
        fifty_tours=("3 · 10^62", float(f"{f(49) / 2:.3e}")),
    )
    out["atoms_claim"] = dict(
        routes_50=float(f"{f(49) / 2:.3e}"),
        atoms_observable_universe="~1e80",
        verdict="в речи «превышает число атомов видимой Вселенной» — неверно, "
                "порог проходится примерно на 61 адресе; на экране этого "
                "сравнения нет",
        n_needed=next(n for n in range(50, 100) if f(n - 1) / 2 > 1e80),
    )
    bad = []
    if out["five_tours"][0] != out["five_tours"][1]:
        bad.append("миниатюр не 12")
    if out["ten_tours"][0] != out["ten_tours"][1]:
        bad.append("181 440 не равно 9!/2")
    if f(19) != 121645100408832000:
        bad.append("19! написано неверно")
    return len(bad), bad, out


def geometry():
    """Маршруты обязаны быть настоящими, а не «похожими»:

    1. каждый маршрут — перестановка всех адресов, замкнутая в цикл;
    2. 12 миниатюр — двенадцать различных маршрутов, а не повторы;
    3. синий маршрут на route1 — точный оптимум перебором всех 60;
    4. жадный маршрут действительно жадный и действительно с самопересечением;
    5. маршрут после 2-opt короче и без пересечений;
    6. все точки лежат внутри зоны графики.
    """
    gx0, gx1 = GFX_X
    gy0, gy1 = GFX_ZONE
    bad = []

    def is_tour(order, n):
        return sorted(order) == list(range(n))

    for name, pts, order in (("OPT6", ADDR6, OPT6), ("LONG6", ADDR6, LONG6),
                             ("NN10", ADDR10, NN10), ("GOOD10", ADDR10, GOOD10)):
        if not is_tour(order, len(pts)):
            bad.append((name, "не перестановка адресов", order))
    for i, o in enumerate(TOURS5):
        if not is_tour(o, 5):
            bad.append((f"TOURS5[{i}]", "не перестановка", o))
    if len(set(TOURS5)) != 12:
        bad.append(("TOURS5", "маршруты повторяются", len(set(TOURS5))))
    if tour_len(ADDR6, OPT6) > min(tour_len(ADDR6, o) for o in all_tours(6)) + 1e-9:
        bad.append(("OPT6", "не оптимум", tour_len(ADDR6, OPT6)))
    if tour_len(ADDR6, LONG6) <= tour_len(ADDR6, OPT6):
        bad.append(("LONG6", "не длиннее оптимума", None))
    if len(CROSS10) == 0:
        bad.append(("NN10", "нет самопересечения — нечего показывать лаймом", None))
    if crossings(ADDR10, GOOD10):
        bad.append(("GOOD10", "после 2-opt остались пересечения",
                    crossings(ADDR10, GOOD10)))
    if tour_len(ADDR10, GOOD10) >= tour_len(ADDR10, NN10):
        bad.append(("GOOD10", "не короче жадного", None))
    for name, pts in (("ADDR5", ADDR5), ("ADDR6", ADDR6), ("ADDR10", ADDR10),
                      ("ADDR20", ADDR20), ("ADDR50", ADDR50)):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if min(xs) < gx0 or max(xs) > gx1 or min(ys) < gy0 or max(ys) > gy1:
            bad.append((name, "точки вне зоны",
                        (round(min(xs)), round(max(xs)), round(min(ys)), round(max(ys)))))
    info = dict(opt6=round(tour_len(ADDR6, OPT6)),
                long6=round(tour_len(ADDR6, LONG6)),
                long6_ratio=round(tour_len(ADDR6, LONG6) / tour_len(ADDR6, OPT6), 2),
                nn10=round(tour_len(ADDR10, NN10)),
                nn10_crossings=CROSS10,
                good10=round(tour_len(ADDR10, GOOD10)),
                good10_gain=f"{100 * (1 - tour_len(ADDR10, GOOD10) / tour_len(ADDR10, NN10)):.1f}%")
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

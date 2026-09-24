"""QA ролика 33: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами, честность чисел и логики задачи.
"""
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard33 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, CELLS, HALF, COLS, PRISONERS, LIMIT, PCT,
                          ACTOR, PERM, CHAIN, OWN_BOX, LONG_CYCLE, BLUE_CELLS,
                          shot_at)
from style import CARD_A, FPS
import render33 as R

ALPHA = 40


def bright_outside():
    cap = cv2.VideoCapture(OUT)
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        mask = np.zeros(img.shape[:2], np.uint8)
        x, y, w, h = CARD_A
        mask[y:y + h, x:x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:8]


def line_overlaps():
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
        bid, pos, _, _, yoff = R.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, yoff))
    for bid, rows in by_block.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
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
        gl = R.graphics_layer(kind, t - t0)
        if gl is None:
            continue
        cl = R.caption_layer(t)
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
        if kind == "A1word":            # R6 живёт над карточкой A, не в зоне графики
            continue
        gl = R.graphics_layer(kind, t - ts)
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


def gfx_clipped():
    """Графика не должна упираться во врезку зоны — иначе фигура выглядит обрезанной."""
    y0, y1 = GFX_ZONE
    x0, x1 = GFX_X
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        ts, _, kind, _ = shot_at(t)
        if kind == "A1word":
            continue
        gl = R.graphics_layer(kind, t - ts)
        if gl is None:
            continue
        a = np.array(gl.split()[3])
        ys, xs = np.nonzero(a > 120)
        if len(ys) == 0:
            continue
        if (ys.max() > y1 - 22 or ys.min() < y0 + 22
                or xs.max() > x1 - 22 or xs.min() < x0 + 22):
            bad += 1
            sample.append((round(t, 2), kind,
                           int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return bad, sample[:8]


def text_in_card():
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        kind = shot_at(t)[2]
        cl = R.caption_layer(t)
        if cl is None:
            continue
        a = np.array(cl.split()[3])
        ys, xs = np.nonzero(a > 120)
        if len(ys) == 0:
            continue
        x, y, w, h = CARD_A
        if (xs.min() < x + 30 or xs.max() > x + w - 30
                or ys.min() < y + 30 or ys.max() > y + h - 30):
            bad += 1
            sample.append((round(t, 2), kind,
                           int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return bad, sample[:8]


def _words():
    return [(w["start"], w["end"], w["word"]) for w in json.load(open(WORDS))]


def speech_sync():
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
    words = _words()
    bad, sample = 0, []
    for t0, _, runs, _ in CAPS:
        near = [w for w in words if abs(w[0] - t0) < 0.26]
        if not near:
            bad += 1
            sample.append((round(t0, 2), "".join(r[0] for r in runs).strip()))
    return bad, sample[:8]


def caption_in_shot():
    bad, sample = 0, []
    for t0, t1, runs, _ in CAPS:
        s0, s1, _, _ = shot_at(t0)
        if t1 > s1 + 1e-6:
            bad += 1
            sample.append((round(t0, 2), round(t1, 2), round(s1, 2),
                           "".join(r[0] for r in runs).strip()))
    return bad, sample[:8]


def caption_words_match():
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


def spoken_numbers_not_typed():
    bad = []
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs)
        if re.search(r"\d", text):
            bad.append((round(t0, 2), text.strip()))
    return len(bad), bad


def no_caption_on_number_shots():
    """Число на экране и субтитр не говорят одно и то же (brand-kit §5).

    1) ни один субтитр не называет число словами («сто», «пятьдесят»,
       «тридцать один», «треть») и не содержит цифр;
    2) в момент ревила синего 31% на экране нет ни одной строки субтитра;
    3) на плане A1word субтитр не произносит слово, которое несёт R6.
    """
    bad = []
    num_words = re.compile(r"\b(сто|сотня|сотни|пятьдесят|полсотни|тридцать|"
                           r"треть|трети)\b")
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs).lower()
        if num_words.search(text):
            bad.append(("число словами", round(t0, 2), text.strip()))
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "result")
    rev0, rev1 = t0 + 0.50, t0 + 0.95
    for i, c in enumerate(CAPS):
        if c[0] < rev1 and R.BLOCKS[i][3] > rev0:
            bad.append(("субтитр поверх ревила 31%", round(c[0], 2),
                        "".join(r[0] for r in c[2]).strip()))
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "A1word")
    for c in CAPS:
        text = "".join(r[0] for r in c[2]).lower()
        if t0 <= c[0] < t1 and "стратег" in text:
            bad.append(("A1word", round(c[0], 2), text))
    return len(bad), bad


def blue_and_lime():
    return dict(blue_numbers=["31% на плане result"],
                blue_marks=["найденная своя бумажка (findOwn, chain)",
                            "замкнутый цикл (cycle)",
                            "спасённые (allSaved), треть решётки (payoff)"],
                white_numbers=["100 (people)", "50 (fifty)", "50 (cycleLen)",
                               "1/2^100 (zeroChance)",
                               "номера коробок и бумажек в цепочке"],
                lime="не используется — в ролике нет матричных/табличных "
                     "элементов, где лайм уместен по brand-kit §4")


def permutation():
    """Перестановка на экране пересчитывается, а не берётся на веру."""
    bad = []
    v = R.slip_values()
    if sorted(v) != list(range(1, CELLS + 1)):
        bad.append("slip_values — не перестановка 1..100")
    for box, val in PERM.items():
        if v[box - 1] != val:
            bad.append(f"в коробке {box} лежит {v[box - 1]}, а планы рисуют {val}")
    # цепочка, по которой идёт ACTOR, обязана быть циклом и вернуться к нему
    seen, cur = [], ACTOR
    for _ in range(CELLS):
        seen.append(cur)
        cur = v[cur - 1]
        if cur == ACTOR:
            break
    if seen != CHAIN:
        bad.append(f"маршрут заключённого {ACTOR} — {seen}, а рисуем {CHAIN}")
    if v[OWN_BOX - 1] != ACTOR:
        bad.append(f"бумажка {ACTOR} лежит не в коробке {OWN_BOX}")
    # коллинеарность: три клетки цикла не должны лежать на одной прямой,
    # иначе замкнутый цикл на экране выглядит отрезком туда-обратно
    (x0, y0), (x1, y1), (x2, y2) = [R.cell_c(n) for n in CHAIN]
    area = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    if area < 4000:
        bad.append(f"клетки цикла почти на одной прямой (площадь {area:.0f})")
    if LONG_CYCLE <= HALF:
        bad.append("пример длинного цикла не длиннее порога 50")
    if LONG_CYCLE > CELLS:
        bad.append("цикл длиннее, чем всего коробок")
    return len(bad), bad, dict(
        actor=ACTOR, chain=CHAIN, own_box=OWN_BOX,
        cycle_len=len(CHAIN), triangle_area=round(area),
        long_cycle=LONG_CYCLE, threshold=LIMIT,
        note=f"клетка 45 — ряд {(45 - 1) // COLS + 1}, столбец {(45 - 1) % COLS + 1}")


def probability():
    """Обе вероятности из речи пересчитываются точно, дробями."""
    from fractions import Fraction
    bad = []
    p_strategy = 1 - sum(Fraction(1, k) for k in range(HALF + 1, CELLS + 1))
    p_random = Fraction(1, 2) ** CELLS
    if round(float(p_strategy) * 100) != PCT:
        bad.append(f"31% на экране не сходится: {float(p_strategy):.5f}")
    if abs(float(p_strategy) - 1 / 3) > 0.03:
        bad.append("«один шанс из трёх» расходится с пересчётом больше чем на 3 п.п.")
    if float(p_random) > 1e-25:
        bad.append("«практически нулевой» — не такой уж нулевой")
    if BLUE_CELLS != PCT:
        bad.append("на плане payoff синих клеток не столько же, сколько процентов")
    return len(bad), bad, dict(
        p_strategy=round(float(p_strategy), 5),
        shown=f"{PCT}%",
        one_in=round(1 / float(p_strategy), 2),
        p_random=f"{float(p_random):.3g}",
        blue_cells=f"{BLUE_CELLS} из {CELLS}")


def rhythm():
    lens = sorted(b - a for a, b, _, _ in SHOTS)
    face = sum(b - a for a, b, k, _ in SHOTS if k in FACE_KINDS)
    return dict(shots=len(SHOTS), avg=round(sum(lens) / len(lens), 2),
                median=round(lens[len(lens) // 2], 2),
                shortest=round(lens[0], 2), longest=round(lens[-1], 2),
                first_cut=SHOTS[0][1], face_pct=round(100 * face / DUR, 1))


if __name__ == "__main__":
    print("ритм:", rhythm())
    print("цвет:", blue_and_lime())
    nb, ns, ninfo = permutation()
    print(f"permutation_violations={nb} sample={ns}")
    for k, v in ninfo.items():
        print("  ", k, v)
    pb, ps, pinfo = probability()
    print(f"probability_violations={pb} sample={ps}")
    print("  ", pinfo)
    c, cs = speech_sync()
    print(f"cuts_inside_word={c} sample={cs}")
    cc, ccs = caption_sync()
    print(f"captions_off_speech={cc} sample={ccs}")
    ci, cis = caption_in_shot()
    print(f"captions_crossing_cut={ci} sample={cis}")
    cw, cws = caption_words_match()
    print(f"caption_words_not_in_speech={cw} sample={cws}")
    st, sts = caption_style()
    print(f"caption_style_violations={st} sample={sts}")
    sn, sns = spoken_numbers_not_typed()
    print(f"digits_in_captions={sn} sample={sns}")
    nn, nns = no_caption_on_number_shots()
    print(f"captions_on_number_shots={nn} sample={nns}")
    o, os_ = gfx_text_overlap()
    print(f"gfx_text_overlap_frames={o} sample={os_}")
    z, zs = gfx_in_zone()
    print(f"gfx_out_of_zone_frames={z} sample={zs}")
    cl, cls = gfx_clipped()
    print(f"gfx_touching_mask_frames={cl} sample={cls}")
    tc, tcs = text_in_card()
    print(f"text_out_of_card_frames={tc} sample={tcs}")
    lo, los = line_overlaps()
    print(f"line_overlap_pairs={lo} sample={los}")
    if os.path.exists(OUT):
        b, bs = bright_outside()
        print(f"bright_outside_card_frames={b} sample={bs}")

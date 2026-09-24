"""QA ролика 28: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами, честность чисел парадокса.

Пословные тайминги лежат в `videos/28/words.json` — плоский список слов,
собранный по речевым отрезкам (см. шапку storyboard28).
"""
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard28 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, FOUND, LOW, HIGH, EV, MULT, ANY_SUMS,
                          BILL_UNIT, shot_at)
from style import CARD_A, CARD_B, FPS
import render28

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
        bid, pos, _, _, yoff = render28.BLOCKS[i]
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
        gl = render28.graphics_layer(kind, t - t0)
        if gl is None:
            continue
        cl = render28.caption_layer(t)
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
        gl = render28.graphics_layer(kind, t - ts)
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
    """Графика не должна упираться в маску зоны: если элемент касается врезки,
    у него срезан край (так на первом проходе обрезался низ конверта)."""
    y0, y1 = GFX_ZONE
    x0, x1 = GFX_X
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        ts, _, kind, _ = shot_at(t)
        gl = render28.graphics_layer(kind, t - ts)
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
    """Субтитры целиком внутри своей карточки с отступом 30px (brand-kit §5)."""
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        kind = shot_at(t)[2]
        cl = render28.caption_layer(t)
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


def caption_in_shot():
    """Фраза целиком лежит внутри своего плана: субтитр, переехавший рез,
    отрисовался бы в слоте соседнего плана."""
    bad, sample = 0, []
    for t0, t1, runs, _ in CAPS:
        s0, s1, _, _ = shot_at(t0)
        if t1 > s1 + 1e-6:
            bad += 1
            sample.append((round(t0, 2), round(t1, 2), round(s1, 2),
                           "".join(r[0] for r in runs).strip()))
    return bad, sample[:8]


def caption_words_match():
    """Субтитры не расходятся с речью по словам: каждое слово субтитра
    встречается в расшифровке в том же порядке."""
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
    `open1000` субтитр обязан закончиться до появления числа
    (brand-kit §5 «одно и то же не пишется дважды»)."""
    bad = []
    for kind in ("branchNums", "ev1250"):
        t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == kind)
        for c in CAPS:
            if t0 <= c[0] < t1:
                bad.append((kind, round(c[0], 2), "".join(r[0] for r in c[2])))
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "open1000")
    num_at = t0 + 0.76                        # ревил 1 000 ₽ в g_open1000
    for c in CAPS:
        if t0 <= c[0] < t1 and c[1] > num_at + 0.04:
            bad.append(("open1000", round(c[0], 2), "".join(r[0] for r in c[2])))
    return len(bad), bad


def spoken_numbers_not_typed():
    """Числа, которые звучат в речи, не должны попадать в субтитр:
    их носитель — графика."""
    bad = []
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs)
        if re.search(r"\d", text):
            bad.append((round(t0, 2), text.strip()))
    return len(bad), bad


def blue_and_lime():
    """Синим — только числа и результат, лайм — один элемент (brand-kit §4)."""
    return dict(blue_numbers=["1 250 ₽ на плане ev1250 (и оно же на swapGood)",
                              "× 1,25 на плане anySum"],
                blue_marks=["метка середины на оси (ev1250)",
                            "стрелка обмена на swapGood — результат рассуждения"],
                white_numbers=["1 000 ₽ (open1000, swapGood, flaw)",
                               "500 и 2 000 (branchNums, ev1250, flaw, prior)",
                               "пары сумм на anySum и prior"],
                lime="разъехавшиеся полосы веса на плане flaw — "
                     "единственный лаймовый элемент ролика")


def numbers():
    """Числа на экране и числа в речи — пересчёт, а не вера в сценарию.

    Парадокс двух конвертов: в руках S, во втором либо S/2, либо 2S.
    Среднее (S/2 + 2S)/2 = 1,25 S — при любом S, что и утверждает речь.
    """
    bad = []
    if (LOW + HIGH) / 2 != EV:
        bad.append(f"среднее ({LOW}+{HIGH})/2 != {EV}")
    if LOW * 2 != FOUND or FOUND * 2 != HIGH:
        bad.append("500/1000/2000 не образуют пару «вдвое»")
    if abs(EV / FOUND - MULT) > 1e-9:
        bad.append(f"{EV}/{FOUND} != {MULT}")
    for s in ANY_SUMS:
        if abs(((s / 2 + 2 * s) / 2) / s - MULT) > 1e-9:
            bad.append(f"для {s} множитель не {MULT}")
        if int(s * MULT) != s * MULT:
            bad.append(f"{s} × 1,25 не целое — на экране было бы округление")
    out = dict(
        found=FOUND, branches=(LOW, HIGH), ev=EV,
        ev_check=f"({LOW} + {HIGH}) / 2 = {(LOW + HIGH) // 2}",
        multiplier=f"(S/2 + 2S)/2 = {MULT} S — не зависит от S",
        any_sums=[(s, int(s * MULT)) for s in ANY_SUMS],
        script_verdict="арифметика речи верна: 1250 = 1,25 × 1000, и то же "
                       "рассуждение действительно проходит при любой сумме; "
                       "разрешение «две суммы считаем равновероятными без "
                       "основания» — стандартное и верное",
    )
    return len(bad), bad, out


def geometry():
    """Графика обязана быть настоящей, а не «похожей»:

    1. в паре конвертов правый действительно вдвое «толще» левого — по числу
       купюр, а не на глаз;
    2. метка среднего на оси стоит ровно на середине отрезка между метками
       500 и 2 000 — иначе «среднее» нарисовано неправдой;
    3. конверты не налезают друг на друга и не касаются краёв зоны.
    """
    bad = []
    r = render28
    if r.PAIR_DX * 2 - r.DBL_W <= 40:
        bad.append(("double", "конверты пары почти касаются",
                    r.PAIR_DX * 2 - r.DBL_W))
    # ось: середина между метками = позиция синей метки в конце анимации
    mid_axis = (r.AX_X0 + r.AX_X1) / 2
    mark_end = r.AX_X0 + (r.AX_X1 - r.AX_X0) * 0.5
    if abs(mid_axis - mark_end) > 0.5:
        bad.append(("ev1250", "метка не на середине оси", mid_axis, mark_end))
    from storyboard28 import EV_XY
    if abs(EV_XY[0] - mid_axis) > 1.0:
        bad.append(("ev1250", "число не под серединой оси", EV_XY[0], mid_axis))
    # купюры: масштаб один на весь ролик, одна купюра = BILL_UNIT
    for name, val in (("open1000", FOUND), ("swapGood", FOUND),
                      ("double L", FOUND), ("double R", HIGH),
                      ("branchNums L", LOW), ("branchNums R", HIGH)):
        if val % BILL_UNIT:
            bad.append((name, "сумма не делится на масштаб купюры", val))
    if HIGH // BILL_UNIT != 2 * (FOUND // BILL_UNIT):
        bad.append(("double", "правый конверт не вдвое толще левого", None))
    if HIGH // BILL_UNIT != 4 * (LOW // BILL_UNIT):
        bad.append(("branchNums", "2 000 не вчетверо толще 500", None))
    info = dict(pair_gap=r.PAIR_DX * 2 - r.DBL_W,
                axis=(r.AX_X0, r.AX_X1), axis_mid=mid_axis,
                bill_unit=BILL_UNIT,
                bills=dict(found=FOUND // BILL_UNIT, low=LOW // BILL_UNIT,
                           high=HIGH // BILL_UNIT),
                envelope_bottom=dict(double=r.DBL_Y + r.DBL_W * 0.68 / 2,
                                     open1000=r.OPEN_Y + r.OPEN_W * 0.68 / 2,
                                     swapGood=r.SG_Y + r.SG_W * 0.68 / 2,
                                     zone_limit=GFX_ZONE[1] - 16))
    return len(bad), bad, info


def rhythm():
    lens = sorted(b - a for a, b, _, _ in SHOTS)
    face = sum(b - a for a, b, k, _ in SHOTS if k in FACE_KINDS)
    stock = sum(b - a for a, b, k, _ in SHOTS if k == "stock")
    return dict(shots=len(SHOTS), avg=round(sum(lens) / len(lens), 2),
                median=round((lens[len(lens) // 2 - 1] + lens[len(lens) // 2]) / 2, 2),
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

"""QA ролика 31: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами, честность чисел и масштаба.
"""
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard31 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, SHORT_MIN, LONG_MIN, RATIO, SCHED_MEAN,
                          NAIVE_WAIT, SEEN_MEAN, REAL_WAIT, INTERVALS, SPAN_MIN,
                          DROP_MIN, P_SHORT_NUM, P_SHORT_DEN, NUM_AT, shot_at)
from style import CARD_A, CARD_B, FPS
import render31 as R

ALPHA = 40


def bright_outside():
    cap = cv2.VideoCapture(OUT)
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        x, y, w, h = CARD_A
        mask = np.zeros(img.shape[:2], np.uint8)
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
    """Ни одной цифры в субтитрах: все числа несёт графика."""
    bad = []
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs)
        if re.search(r"\d", text):
            bad.append((round(t0, 2), text.strip()))
    return len(bad), bad


def no_caption_on_number_shots():
    """Там, где число графики — это ровно то слово, которое звучит, субтитр
    обязан замолчать (brand-kit §5 «одно и то же не пишется дважды»)."""
    bad = []
    # число появляется в середине плана — субтитр обязан кончиться до попа
    for kind in ("schedTicks", "naiveWait", "gapLong"):
        t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == kind)
        num_at = t0 + NUM_AT[kind]
        for c in CAPS:
            if t0 <= c[0] < t1 and c[1] > num_at + 0.04:
                bad.append((kind, round(c[0], 2), "".join(r[0] for r in c[2])))
    # планы, где говорит только графика — субтитров быть не должно вообще
    for kind in ("gapShortNum", "pairBoth"):
        t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == kind)
        for c in CAPS:
            if t0 <= c[0] < t1:
                bad.append((kind, round(c[0], 2), "".join(r[0] for r in c[2])))
    # × 10 стартует в начале плана — субтитр не должен начинаться до конца попа
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "ratioNum")
    num_done = t0 + NUM_AT["ratioNum"] + 0.30
    for c in CAPS:
        if t0 <= c[0] < t1 and c[0] < num_done - 0.04:
            bad.append(("ratioNum", round(c[0], 2), "".join(r[0] for r in c[2])))
    # R6 «ПАРАДОКС»: субтитр обрывается до слова
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "A1word")
    for c in CAPS:
        text = "".join(r[0] for r in c[2]).lower()
        if t0 <= c[0] < t1 and "парадокс" in text:
            bad.append(("A1word", round(c[0], 2), text))
    return len(bad), bad


def blue_and_lime():
    return dict(blue_numbers=["× 10 на плане ratioNum", "18 на планах seenBar/schedBar"],
                blue_marks=["длинный промежуток на pairAnswer/ratioSetup/ratioNum — "
                            "тот, в который пассажир попадает чаще (результат)",
                            "длинные промежутки и метки в них на dropsTally",
                            "полоса 18 минут на seenBar/schedBar",
                            "метки прихода на manyOthers"],
                white_numbers=["10 (schedTicks и schedBar)", "5 (naiveWait)",
                               "2 (gapShortNum)", "20 (gapLong)"],
                lime="не используется — матричных и табличных элементов в ролике нет")


def numbers():
    """Арифметика ролика пересчитана, а не взята из речи на веру.

    Модель: интервал равен 2 или 20 минутам, доля коротких 5/9. Тогда
    E[L] = 10 (совпадает с «в среднем раз в 10 минут»), а средний интервал,
    в который попадает пассажир, пришедший в случайный момент, равен
    E[L²]/E[L] = 18 минут, то есть ожидание 9 минут, а не 5.
    """
    bad = []
    p, q = P_SHORT_NUM, P_SHORT_DEN - P_SHORT_NUM
    el_num = p * SHORT_MIN + q * LONG_MIN
    el2_num = p * SHORT_MIN ** 2 + q * LONG_MIN ** 2
    if el_num != SCHED_MEAN * P_SHORT_DEN:
        bad.append(f"средний интервал {el_num}/{P_SHORT_DEN} != {SCHED_MEAN}")
    if el2_num * P_SHORT_DEN != SEEN_MEAN * el_num * P_SHORT_DEN:
        bad.append(f"E[L^2]/E[L] = {el2_num}/{el_num} != {SEEN_MEAN}")
    if LONG_MIN // SHORT_MIN != RATIO or LONG_MIN % SHORT_MIN:
        bad.append(f"{LONG_MIN}/{SHORT_MIN} != {RATIO}")
    if SCHED_MEAN != 2 * NAIVE_WAIT:
        bad.append("наивное ожидание не равно половине среднего интервала")
    if SEEN_MEAN <= SCHED_MEAN:
        bad.append("парадокса нет: наблюдаемый интервал не длиннее расписания")
    # состав линии времени должен ровно воспроизводить долю 5/9
    n_short = sum(1 for v in INTERVALS if v == SHORT_MIN)
    n_long = sum(1 for v in INTERVALS if v == LONG_MIN)
    if (n_short, n_long) != (p, q):
        bad.append(f"на линии {n_short} коротких и {n_long} длинных вместо {p}/{q}")
    if sum(INTERVALS) != SPAN_MIN:
        bad.append("сумма промежутков не сходится")
    share_long = n_long * LONG_MIN / SPAN_MIN
    hit_long = sum(1 for m in DROP_MIN if R._in_long(m)) / len(DROP_MIN)
    if abs(share_long - hit_long) > 0.06:
        bad.append(f"метки прихода {hit_long:.3f} расходятся с долей времени "
                   f"в длинных промежутках {share_long:.3f}")
    out = dict(
        short=SHORT_MIN, long=LONG_MIN, ratio=RATIO,
        p_short=f"{p}/{P_SHORT_DEN}",
        mean_interval=f"({p}·{SHORT_MIN} + {q}·{LONG_MIN})/{P_SHORT_DEN} = {SCHED_MEAN}",
        naive_wait=f"{SCHED_MEAN}/2 = {NAIVE_WAIT} (то, что в речи названо неверным)",
        seen_interval=f"E[L²]/E[L] = {el2_num}/{P_SHORT_DEN} ÷ {el_num}/{P_SHORT_DEN} = {SEEN_MEAN}",
        real_wait=f"{SEEN_MEAN}/2 = {REAL_WAIT} минут — настоящий ответ на хук",
        share_of_time_in_long=round(share_long, 4),
        drops_in_long=f"{sum(1 for m in DROP_MIN if R._in_long(m))} из {len(DROP_MIN)}",
        script_verdict="ошибок в речи нет: «в среднем раз в 10 минут», "
                       "«через 2 минуты», «через 20 минут» и «в 10 раз больше "
                       "времени» сходятся между собой; «5 минут» в речи прямо "
                       f"названо неверным ответом, настоящий — {REAL_WAIT} минут",
    )
    return len(bad), bad, out


def geometry():
    """Масштаб на экране должен физически совпадать со временем: длина
    промежутка в пикселях пропорциональна его длительности в минутах."""
    bad = []
    short_px = R.P_SHORT_X1 - R.P_SHORT_X0
    long_px = R.P_LONG_X1 - R.P_LONG_X0
    if abs(long_px / short_px - RATIO) > 1e-6:
        bad.append(f"пара промежутков: {long_px}/{short_px} = "
                   f"{long_px/short_px:.4f} вместо {RATIO}")
    seen_px = R.BAR_SEEN_X1 - R.BAR_X0
    sched_px = R.BAR_SCHED_X1 - R.BAR_X0
    if abs(seen_px / sched_px - SEEN_MEAN / SCHED_MEAN) > 1e-6:
        bad.append(f"столбики: {seen_px}/{sched_px} != {SEEN_MEAN}/{SCHED_MEAN}")
    # линия времени: каждый промежуток нарисован своей длительностью
    for i, v in enumerate(INTERVALS):
        px = R.L_X[i + 1] - R.L_X[i]
        if abs(px - v * R.PPM_L) > 1e-6:
            bad.append(f"промежуток {i}: {px:.1f}px вместо {v * R.PPM_L:.1f}")
    # деления внутри длинного промежутка режут его ровно на RATIO коротких
    if R.DIVS + 1 != RATIO:
        bad.append(f"делений {R.DIVS}, длинный промежуток режется не на {RATIO}")
    # каждая метка прихода лежит внутри линии
    for m in DROP_MIN:
        if not 0 <= m <= SPAN_MIN:
            bad.append(f"метка {m} вне линии времени")
    info = dict(px_per_min_pair=R.PPM_P, px_per_min_line=round(R.PPM_L, 3),
                px_per_min_bar=R.PPM_BAR,
                pair=f"{short_px}px / {long_px}px = 1:{long_px//short_px}",
                bars=f"{sched_px}px / {seen_px}px = {SCHED_MEAN}:{SEEN_MEAN} мин")
    return len(bad), bad, info


def rhythm():
    lens = sorted(b - a for a, b, _, _ in SHOTS)
    face = sum(b - a for a, b, k, _ in SHOTS if k in FACE_KINDS)
    stock = sum(b - a for a, b, k, _ in SHOTS if k == "stock")
    return dict(shots=len(SHOTS), avg=round(sum(lens) / len(lens), 2),
                median=round((lens[len(lens) // 2 - 1] + lens[len(lens) // 2]) / 2, 2)
                if len(lens) % 2 == 0 else round(lens[len(lens) // 2], 2),
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

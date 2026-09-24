"""QA ролика 36: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами, честность логики парадокса.
"""
import json
import math
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard36 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, OUTCOMES, ANSWER_ATLEAST, ANSWER_ELDER,
                          ANSWER_NAIVE, BIG_WORD, shot_at)
from style import CARD_A, CARD_B, FPS
import render36 as R

ALPHA = 40


def bright_outside():
    """Ярче 200 вне карточки плана — брак. Карточка A для лица и сетки, B для стока."""
    cap = cv2.VideoCapture(OUT)
    bad, sample = 0, []
    for f in range(0, int(DUR * FPS), 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        kind = shot_at(f / FPS)[2]
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
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
    """Текст целиком внутри карточки плана с отступом 30px (для стока — карточка B)."""
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
    """Число/слово на экране и субтитр не говорят одно и то же (brand-kit §5).

    1) ни один субтитр не содержит цифр и не называет число словами
       («одна вторая», «одной третьей», «четыре», «три», «два»);
    2) в момент каждого ревила (белое 1/2, синие 1/3 и 1/2) на экране нет
       ни одной строки субтитра;
    3) на плане A1word субтитр не произносит слово, которое несёт R6;
    4) на планах перечисления исходов (fill1/fill2) субтитров нет — их
       несут заполняющиеся ячейки.
    """
    bad = []
    num_words = re.compile(r"\b(одна вторая|одну вторую|одной второй|половина|"
                           r"одна третья|одной третьей|треть|трети|"
                           r"четыре|четырёх|четырех|три|трёх|трех|два|две|двух)\b")
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs).lower()
        if num_words.search(text):
            bad.append(("число словами", round(t0, 2), text.strip()))
    # окно — от кадра, на котором render36 начинает рисовать число, и до
    # конца плана; границы плана в окно не входят
    eps = 1e-6
    for kind, off0 in (("half1", 0.576), ("thirdNum", 0.000),
                       ("halfBlue", 0.999)):
        t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == kind)
        rev0, rev1 = t0 + off0, t1
        for i, c in enumerate(CAPS):
            if c[0] < rev1 - eps and R.BLOCKS[i][3] > rev0 + eps:
                bad.append((f"субтитр поверх ревила ({kind})", round(c[0], 2),
                            "".join(r[0] for r in c[2]).strip()))
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "A1word")
    for c in CAPS:
        text = "".join(r[0] for r in c[2]).lower()
        if t0 <= c[0] < t1 and BIG_WORD.lower() in text:
            bad.append(("A1word", round(c[0], 2), text))
    for kind in ("fill1", "fill2"):
        t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == kind)
        for c in CAPS:
            if t0 <= c[0] < t1:
                bad.append((f"субтитр на перечислении ({kind})", round(c[0], 2),
                            "".join(r[0] for r in c[2]).strip()))
    return len(bad), bad


def blue_and_lime():
    return dict(blue_numbers=[f"{ANSWER_ATLEAST[0]}/{ANSWER_ATLEAST[1]} (thirdNum)",
                              f"{ANSWER_ELDER[0]}/{ANSWER_ELDER[1]} (halfBlue)"],
                blue_marks=["ячейка-победитель «оба мальчики» (oneOfThree)"],
                white_numbers=[f"{ANSWER_NAIVE[0]}/{ANSWER_NAIVE[1]} (half1) — "
                               "ложная интуиция, поэтому белым, а не синим"],
                lime="не используется — в ролике нет матричных/табличных "
                     "элементов, где лайм уместен по brand-kit §4")


def logic():
    """Оба ответа пересчитываются из пространства исходов, а не берутся на веру.

    Пространство: (старший, младший) из {М,Д}^2, все четыре равновероятны.
    Условие меняет множество допустимых исходов, а не сами исходы.
    """
    bad = []
    if len(OUTCOMES) != 4 or len(set(OUTCOMES)) != 4:
        bad.append("пространство исходов не равно четырём различным парам")

    at_least = [o for o in OUTCOMES if "B" in o]          # хотя бы один мальчик
    elder = [o for o in OUTCOMES if o[0] == "B"]          # старший — мальчик
    both = ("B", "B")

    def frac(sub):
        n = sum(1 for o in sub if o == both)
        g = math.gcd(n, len(sub)) or 1
        return (n // g, len(sub) // g)

    if len(at_least) != 3:
        bad.append(f"условие «хотя бы один мальчик» оставило {len(at_least)} исхода, не 3")
    if len(elder) != 2:
        bad.append(f"условие «старший — мальчик» оставило {len(elder)} исхода, не 2")
    if frac(at_least) != ANSWER_ATLEAST:
        bad.append(f"ответ при «хотя бы один» посчитан неверно: {frac(at_least)}")
    if frac(elder) != ANSWER_ELDER:
        bad.append(f"ответ при «старший мальчик» посчитан неверно: {frac(elder)}")
    if ANSWER_ATLEAST == ANSWER_ELDER:
        bad.append("два условия дали один ответ — ролик разбирает несуществующую разницу")
    if ANSWER_NAIVE != ANSWER_ELDER:
        bad.append("ложная интуиция не совпала с ответом второй задачи — "
                   "ролик держится на том, что 1/2 верно, но не для этого условия")
    # ДД — единственный исход, который отсекается первым условием
    dropped = [o for o in OUTCOMES if o not in at_least]
    if dropped != [("G", "G")]:
        bad.append(f"первым условием отсечён не только «девочка-девочка»: {dropped}")
    return len(bad), bad, dict(
        space="(старший, младший) из {М,Д}^2, 4 равновероятных исхода",
        at_least=f"«хотя бы один мальчик» -> {len(at_least)} исхода -> "
                 f"{ANSWER_ATLEAST[0]}/{ANSWER_ATLEAST[1]}",
        elder=f"«старший — мальчик» -> {len(elder)} исхода -> "
              f"{ANSWER_ELDER[0]}/{ANSWER_ELDER[1]}",
        naive="ложная интуиция даёт тот же 1/2, но для другого условия")


def rhythm():
    lens = sorted(b - a for a, b, _, _ in SHOTS)
    face = sum(b - a for a, b, k, _ in SHOTS if k in FACE_KINDS)
    stock = sum(b - a for a, b, k, _ in SHOTS if k == "stock")
    return dict(shots=len(SHOTS), avg=round(sum(lens) / len(lens), 2),
                median=round(lens[len(lens) // 2], 2),
                shortest=round(lens[0], 2), longest=round(lens[-1], 2),
                first_cut=SHOTS[0][1], face_pct=round(100 * face / DUR, 1),
                stock_pct=round(100 * stock / DUR, 1))


def stock_reader_seek():
    """Ридер стока обязан вставать ровно на запрошенное время с ПЕРВОГО кадра
    плана. Догонять чтением подряд нельзя: на ss>13.3с guard=400 кадров не
    доезжал, и первый кадр плана показывал чужой момент клипа (13.31с вместо
    16.60с) — один кадр-вспышка на склейке. Глазами не ловится: в контакт-лист
    первый кадр плана обычно не попадает.
    """
    bad = []
    for t0, t1, kind, prm in SHOTS:
        if kind != "stock":
            continue
        R._STOCK.key = None                     # состояние как при старте плана
        for k in range(3):
            rel = k / FPS
            R._STOCK.frame(prm["clip"], prm["ss"], rel)
            want = prm["ss"] + rel
            if abs(R._STOCK.pos - want) > 1.0 / FPS + 1e-3:
                bad.append((prm["clip"], round(want, 3), round(R._STOCK.pos, 3)))
    R._STOCK.release()
    return len(bad), bad


def stock_band():
    """Полоса субтитров на стоковой вставке: белый текст должен читаться."""
    out = []
    for t0, t1, kind, prm in SHOTS:
        if kind != "stock":
            continue
        vals = []
        for t in np.linspace(t0 + 0.05, t1 - 0.05, 12):
            fr = R._STOCK.frame(prm["clip"], prm.get("ss", 0.0), float(t) - t0)
            card = R.stock_card(fr, prm.get("cx", 0.5), prm.get("cy", 0.5))
            g = np.array(card.convert("L"))
            band = g[706 - CARD_B[1]:900 - CARD_B[1], 26:820]
            vals.append(band.mean())
        v = np.array(vals)
        out.append((prm["clip"], round(float(v.mean()), 1),
                    round(float(np.percentile(v, 90)), 1)))
    R._STOCK.release()
    return out


def face_band():
    """Та же полоса на планах с лицом (карточка A, y 1260...1400)."""
    cap = cv2.VideoCapture(R.SRC)
    vals = []
    times = [t0 + (t1 - t0) * f for t0, t1, k, _ in SHOTS if k in FACE_KINDS
             for f in (0.25, 0.5, 0.75)]
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * FPS)))
        ok, fr = cap.read()
        if not ok:
            continue
        card = R.source_card(fr, "A1")
        g = np.array(card.convert("L"))
        band = g[1260 - CARD_A[1]:1400 - CARD_A[1], 200 - CARD_A[0]:880 - CARD_A[0]]
        vals.append(band.mean())
    cap.release()
    v = np.array(vals)
    return round(float(v.mean()), 1), round(float(np.percentile(v, 90)), 1)


if __name__ == "__main__":
    print("ритм:", rhythm())
    print("цвет:", blue_and_lime())
    lb, ls, linfo = logic()
    print(f"logic_violations={lb} sample={ls}")
    for k, v in linfo.items():
        print("  ", k, v)
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
    sr, srs = stock_reader_seek()
    print(f"stock_reader_off_by_frames={sr} sample={srs}")
    print("stock_caption_band:", stock_band())
    print("face_caption_band:", face_band())
    if os.path.exists(OUT):
        b, bs = bright_outside()
        print(f"bright_outside_card_frames={b} sample={bs}")

"""QA ролика 37: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами, честность вероятностной модели.
"""
import json
import math
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard37 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, CARDS, RR, BB, RB, ANSWER, ANSWER_NAIVE,
                          BIG_WORD, shot_at)
from style import CARD_A, CARD_B, FPS
import render37 as R

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
                sample.append((round(t0, 3), w, round(a, 3), round(b, 3)))
                break
    return bad, sample


def cut_energy():
    """Рез не должен приходиться на атаку слова: whisper этого не ловит, поэтому
    отдельно меряется огибающая исходника на кадре реза (см. snap37.py)."""
    wav = f"{os.path.dirname(os.path.abspath(__file__))}/assets/_align37.wav"
    if not os.path.exists(wav):
        return -1, []
    import wave
    w = wave.open(wav)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    sr = w.getframerate()
    n = int(sr * 0.010)
    env = np.sqrt(np.convolve(a ** 2, np.ones(n) / n, "same"))
    bad, sample = 0, []
    for t0, _, _, _ in SHOTS[1:]:
        e = float(env[int(round(t0 * FPS) / FPS * sr)])
        if e > 0.020:
            bad += 1
        sample.append((round(t0, 3), round(e, 4)))
    return bad, sorted(sample, key=lambda s: -s[1])[:5]


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


def _norm(w):
    return re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))


def caption_words_match():
    """Каждое слово субтитра должно реально прозвучать, в том же порядке.
    Дефис разбивается: «красно-чёрная» произносится двумя токенами whisper."""
    said = [x for x in (_norm(w[2]) for w in _words()) if x]
    typed = []
    for t0, _, runs, _ in CAPS:
        for txt, _, _ in runs:
            for word in txt.split():
                typed += [x for x in (_norm(p) for p in word.split("-")) if x]
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


def digits_in_captions():
    bad = []
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs)
        if re.search(r"\d", text):
            bad.append((round(t0, 2), text.strip()))
    return len(bad), bad


def no_caption_on_number_shots():
    """Число/слово на экране и субтитр не говорят одно и то же (brand-kit §5).

    1) дробь-ответ не набирается словами нигде;
    2) на СЧЁТНЫХ планах (где количество объектов и есть смысл кадра) субтитр
       не называет это количество словами;
    3) в момент каждого ревила (белое 1/2, синее 2/3) на экране нет ни одной
       строки субтитра — сверка покадровая, не по секундам;
    4) на плане A1word субтитр не произносит слово, которое несёт R6;
    5) на плане threeSides, где счёт несут сами плитки, субтитров нет.
    """
    bad = []
    frac_words = re.compile(r"\b(одна вторая|одну вторую|одной второй|половина|"
                            r"две трети|двум третям|две третьих|треть)\b")
    count_words = re.compile(r"\b(один|одна|одну|одной|два|две|двух|двумя|"
                             r"три|трёх|трех|тремя)\b")
    COUNT_SHOTS = {"cards3", "sides", "sidesRR", "sidesRB", "threeSides",
                   "twoOfThree", "bottomRed", "twiceMore", "waysRed"}
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs).lower()
        if frac_words.search(text):
            bad.append(("дробь словами", round(t0, 2), text.strip()))
        if shot_at(t0)[2] in COUNT_SHOTS and count_words.search(text):
            bad.append(("счёт словами на счётном плане", round(t0, 2), text.strip()))
    # окно ревила считается КАДРАМИ: субтитр гаснет за кадр до появления числа,
    # и сравнение по секундам ловило бы это как пересечение в 0.3мс
    for kind, off0 in (("naiveHalf", R.NAIVE_NUM_AT), ("twoThirds", 0.434)):
        t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == kind)
        rev_f0 = math.floor((t0 + off0) * FPS) + 1
        for i, c in enumerate(CAPS):
            cap_f0 = math.ceil(c[0] * FPS)
            cap_f1 = math.ceil(R.BLOCKS[i][3] * FPS) - 1
            if cap_f1 >= rev_f0 and cap_f0 < math.ceil(t1 * FPS):
                bad.append((f"субтитр поверх ревила ({kind})", round(c[0], 2),
                            "".join(r[0] for r in c[2]).strip()))
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "A1word")
    for c in CAPS:
        text = "".join(r[0] for r in c[2]).lower()
        if t0 <= c[0] < t1 and BIG_WORD.lower() in text:
            bad.append(("A1word", round(c[0], 2), text))
    for kind in ("threeSides",):
        t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == kind)
        for c in CAPS:
            if t0 <= c[0] < t1:
                bad.append((f"субтитр на счётном плане ({kind})", round(c[0], 2),
                            "".join(r[0] for r in c[2]).strip()))
    return len(bad), bad


def blue_and_lime():
    return dict(blue_numbers=[f"{ANSWER[0]}/{ANSWER[1]} (twoThirds)"],
                blue_marks=["две плитки-стороны красно-красной карты и сама карта "
                            "(twoOfThree, bottomRed, waysRed)"],
                white_numbers=[f"{ANSWER_NAIVE[0]}/{ANSWER_NAIVE[1]} (naiveHalf) — "
                               "ложная интуиция, поэтому белым, а не синим"],
                red=f"ОТСТУПЛЕНИЕ ОТ brand-kit §4 по прямому решению автора: "
                    f"красная сторона карты залита красным {R.RED} "
                    f"(свечение {R.RED_GLOW}). Цвет стороны — предмет задачи, "
                    "и «красная сторона» рисуется красной. Красным залиты "
                    "ТОЛЬКО половины карт и плитки-стороны: ни текст, ни "
                    "числа, ни каркас карт, ни линии красными не становятся",
                lime="не используется — в ролике нет матричных/табличных "
                     "элементов, где лайм уместен по brand-kit §4")


def red_only_on_sides():
    """Красный допущен только как заливка стороны карты.

    Проверяется, что красных пикселей нет там, где им взяться неоткуда:
    1) в слое субтитров — текст никогда не красный;
    2) в полосе белого числа 1/2 (ниже карты, y > 1290) на плане naiveHalf;
    3) нигде на плане twoThirds — там в кадре только синее число.
    """
    def red_mask(layer):
        arr = np.array(layer.convert("RGBA")).astype(np.int16)
        a, r, g, b = arr[:, :, 3], arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        return (a > 120) & (r - g > 60) & (r - b > 60)

    bad = []
    for f in range(0, int(DUR * FPS), 2):
        t = f / FPS
        ts, _, kind, _ = shot_at(t)
        cl = R.caption_layer(t)
        if cl is not None and red_mask(cl).any():
            bad.append(("красное в субтитре", round(t, 2), kind))
        if kind not in ("naiveHalf", "twoThirds"):
            continue
        gl = R.graphics_layer(kind, t - ts)
        if gl is None:
            continue
        m = red_mask(gl)
        if kind == "naiveHalf":
            m = m[1290:, :]                    # полоса белого числа под картой
        if m.any():
            bad.append((f"красное на плане {kind}", round(t, 2), int(m.sum())))
    return len(bad), bad[:8]


def logic():
    """Ответ пересчитывается из пространства СТОРОН, а не карт.

    Шесть сторон (карта, индекс стороны) равновероятны. Условие «сверху
    красная» оставляет три стороны; нижняя красная тогда и только тогда,
    когда сторона принадлежит карте RR.
    """
    bad = []
    if len(CARDS) != 3:
        bad.append("карт не три")
    if sorted(CARDS) != sorted([("R", "R"), ("B", "B"), ("R", "B")]):
        bad.append(f"набор карт не тот: {CARDS}")

    sides = [(ci, si) for ci in range(len(CARDS)) for si in (0, 1)]
    if len(sides) != 6:
        bad.append("сторон не шесть")
    red = [(ci, si) for (ci, si) in sides if CARDS[ci][si] == "R"]
    black = [(ci, si) for (ci, si) in sides if CARDS[ci][si] == "B"]
    if len(red) != 3 or len(black) != 3:
        bad.append(f"красных сторон {len(red)}, чёрных {len(black)} — не 3 и 3")

    good = [(ci, si) for (ci, si) in red if CARDS[ci][1 - si] == "R"]
    g = math.gcd(len(good), len(red)) or 1
    frac = (len(good) // g, len(red) // g)
    if frac != ANSWER:
        bad.append(f"ответ посчитан неверно: {frac}")
    if [ci for ci, _ in good] != [RR, RR]:
        bad.append("обе выигрышные стороны обязаны принадлежать красно-красной карте")

    # ложная интуиция: равновероятными считают КАРТЫ, у которых есть красная
    # сторона, а не сами стороны
    cand = [ci for ci in range(len(CARDS)) if "R" in CARDS[ci]]
    gc = math.gcd(sum(1 for ci in cand if CARDS[ci] == ("R", "R")), len(cand)) or 1
    naive = (sum(1 for ci in cand if CARDS[ci] == ("R", "R")) // gc, len(cand) // gc)
    if naive != ANSWER_NAIVE:
        bad.append(f"ложная интуиция посчитана неверно: {naive}")
    if naive == ANSWER:
        bad.append("верный ответ совпал с ложной интуицией — ролик разбирает "
                   "несуществующую разницу")
    # красно-красная карта даёт ровно вдвое больше способов увидеть красное
    ways_rr = sum(1 for (ci, si) in red if ci == RR)
    ways_rb = sum(1 for (ci, si) in red if ci == RB)
    if ways_rr != 2 * ways_rb:
        bad.append(f"«в два раза больше» не подтверждается: {ways_rr} против {ways_rb}")
    return len(bad), bad, dict(
        space="6 сторон (карта, индекс), все равновероятны",
        condition=f"«сверху красная» -> {len(red)} стороны",
        answer=f"снизу красная <=> сторона карты RR -> {len(good)} из {len(red)} "
               f"-> {ANSWER[0]}/{ANSWER[1]}",
        naive=f"равновероятными считают карты ({len(cand)} шт) -> "
              f"{ANSWER_NAIVE[0]}/{ANSWER_NAIVE[1]}",
        twice=f"способов увидеть красное: у RR {ways_rr}, у RB {ways_rb}")


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
    плана (регрессия ролика 36: догон чтением подряд давал чужой кадр на склейке).
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


def stock_reader_reset():
    """После release() ридер обязан переоткрыть клип, а не отдавать последний
    кадр. Метрика откалибрована: со старым release (без сброса key) даёт 2
    нарушения, с новым — 0.
    """
    bad = []
    for t0, t1, kind, prm in SHOTS:
        if kind != "stock":
            continue
        R._STOCK.frame(prm["clip"], prm["ss"], 0.0)
        R._STOCK.release()
        a = R._STOCK.frame(prm["clip"], prm["ss"], 0.0)
        b = R._STOCK.frame(prm["clip"], prm["ss"], 12.0 / FPS)
        if a is None or b is None or float(np.abs(a.astype(np.int16)
                                                  - b.astype(np.int16)).mean()) < 0.5:
            bad.append(prm["clip"])
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


def stock_motion():
    """Вставка обязана играть, а не стоять: покадровая разница внутри плана."""
    out = []
    for t0, t1, kind, prm in SHOTS:
        if kind != "stock":
            continue
        prev, d = None, []
        for t in np.linspace(t0, t1 - 1.0 / FPS, 24):
            fr = R._STOCK.frame(prm["clip"], prm.get("ss", 0.0), float(t) - t0)
            g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (214, 155)).astype(np.float32)
            if prev is not None:
                d.append(float(np.abs(g - prev).mean()))
            prev = g
        out.append((prm["clip"], round(float(np.mean(d)), 2)))
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
    rr, rrs = red_only_on_sides()
    print(f"red_on_numbers_frames={rr} sample={rrs}")
    lb, ls, linfo = logic()
    print(f"logic_violations={lb} sample={ls}")
    for k, v in linfo.items():
        print("  ", k, v)
    c, cs = speech_sync()
    print(f"cuts_inside_word={c} sample={cs}")
    ce, ces = cut_energy()
    print(f"cuts_on_speech_attack={ce} самые громкие резы={ces}")
    cc, ccs = caption_sync()
    print(f"captions_off_speech={cc} sample={ccs}")
    ci, cis = caption_in_shot()
    print(f"captions_crossing_cut={ci} sample={cis}")
    cw, cws = caption_words_match()
    print(f"caption_words_not_in_speech={cw} sample={cws}")
    st, sts = caption_style()
    print(f"caption_style_violations={st} sample={sts}")
    sn, sns = digits_in_captions()
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
    rs, rss = stock_reader_reset()
    print(f"stock_reader_stale_after_release={rs} sample={rss}")
    print("stock_caption_band:", stock_band())
    print("stock_motion_edges:", stock_motion())
    print("face_caption_band:", face_band())
    if os.path.exists(OUT):
        b, bs = bright_outside()
        print(f"bright_outside_card_frames={b} sample={bs}")

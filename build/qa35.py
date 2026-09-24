"""QA ролика 35: зоны карточек, пересечение графики с субтитрами, наложение строк,
синхрон субтитров и резов с пословными таймингами, честность логики парадокса.
"""
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard35 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X, FACE_KINDS,
                          WORDS, BRANCH_A, BRANCH_B, shot_at)
from style import CARD_A, CARD_B, FPS
import render35 as R

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


def digits_in_captions():
    bad = []
    for t0, _, runs, _ in CAPS:
        text = "".join(r[0] for r in runs)
        if re.search(r"\d", text):
            bad.append((round(t0, 2), text.strip()))
    return len(bad), bad


def no_caption_duplicates_graphics():
    """Субтитр не повторяет то, что несёт графика (brand-kit §5).

    1) на плане A1word субтитр не произносит слово, которое несёт R6;
    2) на стоковой вставке субтитр не называет парикмахера — его несёт кадр;
    3) слово «парадокс» в субтитрах не набирается ни разу.
    """
    bad = []
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "A1word")
    for c in CAPS:
        text = "".join(r[0] for r in c[2]).lower()
        if t0 <= c[0] < t1 and "парадокс" in text:
            bad.append(("A1word", round(c[0], 2), text))
    t0, t1 = next((a, b) for a, b, k, _ in SHOTS if k == "stock")
    for c in CAPS:
        text = "".join(r[0] for r in c[2]).lower()
        if t0 <= c[0] < t1 and "парикмахер" in text:
            bad.append(("сток", round(c[0], 2), text))
    for c in CAPS:
        text = "".join(r[0] for r in c[2]).lower()
        if "парадокс" in text:
            bad.append(("парадокс словом", round(c[0], 2), text))
    return len(bad), bad


def blue_and_lime():
    return dict(blue_numbers=[],
                blue_marks=["крест «R нарушает своё условие» (caseA2)",
                            "контур R как вывод «не содержит себя» (caseA3)",
                            "галочка «R подходит под условие» (caseB1)",
                            "вынужденная копия R внутри R (caseB2)",
                            "замкнутая петля состояний (loop)",
                            "петля и «?» у парикмахера (barber3)",
                            "крест на коробке из произвольного свойства (prop2)",
                            "галочка на множестве, стоящем на аксиомах (axioms)"],
                white_numbers=[],
                lime="не используется — в ролике нет матричных/табличных "
                     "элементов, где лайм уместен по brand-kit §4")


def logic():
    """Обе ветки парадокса пересчитываются, а не берутся на веру.

    R = { x : x не содержит x }.  Условие приёма в R: «не содержит сам себя».
    fits_condition = not contains_self;  must_contain = fits_condition.
    Парадокс = каждая ветка отрицает свою же посылку.
    """
    bad = []
    for name, br in (("A", BRANCH_A), ("B", BRANCH_B)):
        fits = (not br["contains_self"])
        if fits != br["fits_condition"]:
            bad.append(f"ветка {name}: выполнение условия посчитано неверно")
        must = fits
        if must != br["must_contain"]:
            bad.append(f"ветка {name}: обязанность лежать в R посчитана неверно")
        if must == br["contains_self"]:
            bad.append(f"ветка {name}: посылка и вывод совпали — противоречия нет")
    if BRANCH_A["must_contain"] != BRANCH_B["contains_self"]:
        bad.append("вывод ветки A не ведёт в посылку ветки B")
    if BRANCH_B["must_contain"] != BRANCH_A["contains_self"]:
        bad.append("вывод ветки B не ведёт в посылку ветки A")
    return len(bad), bad, dict(
        branch_A="R содержит R -> условие нарушено -> R не должен лежать в R",
        branch_B="R не содержит R -> условие выполнено -> R должен лежать в R",
        cycle="A -> B -> A, замкнут",
        barber="то же самое: брадобрей бреет ровно тех, кто не бреет себя")


def rhythm():
    lens = sorted(b - a for a, b, _, _ in SHOTS)
    face = sum(b - a for a, b, k, _ in SHOTS if k in FACE_KINDS)
    stock = sum(b - a for a, b, k, _ in SHOTS if k == "stock")
    return dict(shots=len(SHOTS), avg=round(sum(lens) / len(lens), 2),
                median=round(lens[len(lens) // 2], 2),
                shortest=round(lens[0], 2), longest=round(lens[-1], 2),
                first_cut=SHOTS[0][1], face_pct=round(100 * face / DUR, 1),
                stock_pct=round(100 * stock / DUR, 1))


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
    """Вставка обязана играть, а не быть слайдом (editing-taste §5).

    Считается по кадрам проекта (30 fps): средняя покадровая разница яркости
    внутри карточки B и плотность контурных пикселей (объект в кадре, а не
    пустая сцена). Опора — принятый клип ролика 34: motion 4.19, edges 0.006.
    """
    out = []
    for t0, t1, kind, prm in SHOTS:
        if kind != "stock":
            continue
        prev, mo, ed = None, [], []
        for k in range(int((t1 - t0) * FPS)):
            fr = R._STOCK.frame(prm["clip"], prm.get("ss", 0.0), k / FPS)
            c = np.array(R.stock_card(fr, prm.get("cx", 0.5),
                                      prm.get("cy", 0.5)).convert("L"),
                         dtype=np.float32)
            if prev is not None:
                mo.append(float(np.abs(c - prev).mean()))
            prev = c
            ed.append(float(np.mean(np.abs(cv2.Sobel(c, cv2.CV_32F, 1, 1, ksize=3)) > 40)))
        out.append((prm["clip"], round(float(np.mean(mo)), 2),
                    round(float(np.mean(ed)), 4)))
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
    sn, sns = digits_in_captions()
    print(f"digits_in_captions={sn} sample={sns}")
    nn, nns = no_caption_duplicates_graphics()
    print(f"captions_duplicating_graphics={nn} sample={nns}")
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
    print("stock_caption_band:", stock_band())
    print("stock_motion_edges:", stock_motion())
    print("face_caption_band:", face_band())
    if os.path.exists(OUT):
        b, bs = bright_outside()
        print(f"bright_outside_card_frames={b} sample={bs}")

"""QA версий ролика 35 с альтернативными хуками.

Проверяется только блок хука и стык с телом: тело — те же кадры, что в
сданном `russell_edit.mp4`, и оно уже проверено в qa35.py.
"""
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS
import render35 as R
import storyboard35 as SB
import hook35 as HK

ALPHA = 40
BODY = HK.BODY


def _words(cfg):
    return [(w["start"], w["end"], w["word"]) for w in json.load(open(cfg["words"]))]


def bind(cfg):
    shots = HK.shots_of(cfg)
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, s=shots: next((x for x in s if x[0] <= t < x[1]), s[-1])
    R.BLOCKS = R.build_blocks()
    return shots


def cuts_inside_word(cfg, shots):
    bad = []
    for t0, _, _, _ in shots[1:]:
        for a, b, w in _words(cfg):
            if a + 0.02 < t0 < b - 0.02:
                bad.append((round(t0, 3), w, round(a, 3), round(b, 3)))
                break
    return len(bad), bad


def captions_off_speech(cfg):
    words = _words(cfg)
    bad = [(round(t0, 3), "".join(r[0] for r in runs).strip())
           for t0, _, runs, _ in cfg["caps"]
           if not any(abs(w[0] - t0) < 0.26 for w in words)]
    return len(bad), bad


def captions_crossing_cut(cfg, shots):
    bad = []
    for t0, t1, runs, _ in cfg["caps"]:
        s = next((x for x in shots if x[0] <= t0 < x[1]), shots[-1])
        if t1 > s[1] + 1e-6:
            bad.append((round(t0, 3), round(t1, 3), round(s[1], 3),
                        "".join(r[0] for r in runs).strip()))
    return len(bad), bad


def caption_words_match(cfg):
    norm = lambda w: re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))
    said = [w for w in (norm(x[2]) for x in _words(cfg)) if w]
    typed = [w for t0, _, runs, _ in cfg["caps"] for txt, _, _ in runs
             for w in (norm(x) for x in txt.split()) if w]
    i, missing = 0, []
    for w in typed:
        j = i
        while j < len(said) and said[j] != w:
            j += 1
        if j < len(said):
            i = j + 1
        else:
            missing.append(w)
    return len(missing), missing


def caption_style(cfg):
    bad = []
    for t0, _, runs, _ in cfg["caps"]:
        text = "".join(r[0] for r in runs).strip()
        if any(ch in text for ch in ".,!?;:«»"):
            bad.append(("пунктуация", t0, text))
        if text != text.lower():
            bad.append(("капс", t0, text))
        n = len(text.split())
        if not 2 <= n <= 4:
            bad.append((f"{n} слов", t0, text))
    return len(bad), bad


def layers(cfg, shots):
    """Пересечение графики с текстом, выход графики из зоны, текст за карточкой."""
    y0, y1 = SB.GFX_ZONE
    x0, x1 = SB.GFX_X
    over = zone = clip = card = 0
    sample = []
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        ts, _, kind, _ = R.shot_at(t)
        gl = R.graphics_layer(kind, t - ts)
        cl = R.caption_layer(t)
        if gl is not None and cl is not None:
            ga, ca = np.array(gl.split()[3]), np.array(cl.split()[3])
            if np.any((ga > ALPHA) & (ca > ALPHA)):
                over += 1
                sample.append(("overlap", round(t, 2), kind))
        if gl is not None:
            a = np.array(gl.split()[3])
            ys, xs = np.nonzero(a > ALPHA)
            if len(ys) and (ys.min() < y0 or ys.max() > y1 or xs.min() < x0 or xs.max() > x1):
                zone += 1
            ys, xs = np.nonzero(a > 120)
            if len(ys) and (ys.max() > y1 - 22 or ys.min() < y0 + 22
                            or xs.max() > x1 - 22 or xs.min() < x0 + 22):
                clip += 1
        if cl is not None:
            a = np.array(cl.split()[3])
            ys, xs = np.nonzero(a > 120)
            x, y, w, h = CARD_A
            if len(ys) and (xs.min() < x + 30 or xs.max() > x + w - 30
                            or ys.min() < y + 30 or ys.max() > y + h - 30):
                card += 1
                sample.append(("card", round(t, 2), kind))
    return over, zone, clip, card, sample[:6]


def line_overlaps(cfg):
    from style import font
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    boxes = {}
    for i, (t0, t1, runs, _) in enumerate(cfg["caps"]):
        top = min(font(kmap[k], sz).getbbox(txt)[1] for txt, k, sz in runs)
        bot = max(font(kmap[k], sz).getbbox(txt)[3] for txt, k, sz in runs)
        boxes[i] = (top, bot)
    by_block = {}
    for i in range(len(cfg["caps"])):
        bid, pos, _, _, yoff = R.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, yoff))
    bad = []
    for bid, rows in by_block.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            ha = (boxes[a[1]][1] - boxes[a[1]][0]) / 2
            hb = (boxes[b[1]][1] - boxes[b[1]][0]) / 2
            if a[2] + ha > b[2] - hb:
                bad.append((bid, round(cfg["caps"][a[1]][0], 2)))
    return len(bad), bad


def _rect_after_crop(rect, k=0.98):
    """Кроп x0.98 с возвратом холста растягивает кадр от центра (540, 960):
    прямоугольник карточки в версии лежит не там, где в оригинале, и маску
    проверки надо пересчитать — иначе рамка карточки сама попадёт «наружу»."""
    x, y, w, h = rect
    fx = lambda v: 540 + (v - 540) / k
    fy = lambda v: 960 + (v - 960) / k
    # +1px по каждой стороне: bilinear при возврате холста размазывает край
    # карточки на соседний пиксель (замер по кадру — карточка 97..984 при
    # расчётных 96.1..983.9), и без запаса её собственный край читался бы
    # как «яркое вне карточки»
    x0, y0, x1, y1 = fx(x) - 1, fy(y) - 1, fx(x + w) + 1, fy(y + h) + 1
    return (int(np.floor(x0)), int(np.floor(y0)),
            int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0)))


def bright_outside(cfg):
    """Ярче 200 вне карточки плана — брак.

    Карточка B — только на стоковом плане тела; на границе плана берётся
    объединение A и B, потому что деформация времени сдвигает кадр на доли
    кадра и у самой склейки неясно, какой из планов уже на экране."""
    cap = cv2.VideoCapture(cfg["out"])
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    speed = 1.02 if cfg.get("post_a") else 1.0
    crop = 0.98 if "crop" in (cfg.get("post_v") or "") else 1.0
    dt = cfg["frames"] / FPS - HK.CUT_V
    bounds = [t for t, _, _, _ in SB.SHOTS] + [SB.DUR]
    bad, sample = 0, []
    for f in range(0, nf, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        t = f / FPS * speed                      # время в несжатой версии
        mask = np.zeros(img.shape[:2], np.uint8)
        rects = [CARD_A]
        if t >= cfg["frames"] / FPS:
            tb = t - dt
            kinds = {SB.shot_at(tb)[2]}
            if any(abs(tb - b) < 1.5 / FPS for b in bounds):   # у самой склейки
                kinds |= {SB.shot_at(max(0.0, tb - 1.5 / FPS))[2],
                          SB.shot_at(min(SB.DUR - 1e-3, tb + 1.5 / FPS))[2]}
            rects = [CARD_B if k == "stock" else CARD_A for k in kinds]
        for r in rects:
            x, y, w, h = _rect_after_crop(r, crop) if crop != 1.0 else r
            mask[max(0, y):y + h, max(0, x):x + w] = 1
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((g > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:8]


def body_identical(cfg):
    """Тело версии должно кадр в кадр совпадать со сданным роликом.

    Сравнивается немое видео ДО деформации: кроп и ускорение накладываются
    отдельным проходом на уже готовые кадры, и кадры при этом не
    перерисовываются — проверять надо именно склейку."""
    a = cv2.VideoCapture(f"{HK.BUILD}/assets/_video_{SB.TAG}{key_of(cfg)}.mp4")
    b = cv2.VideoCapture(BODY)
    a.set(cv2.CAP_PROP_POS_FRAMES, cfg["frames"])
    b.set(cv2.CAP_PROP_POS_FRAMES, HK.CUT_F)
    worst, share, n = 0.0, 0.0, 0
    while True:
        oa, fa = a.read()
        ob, fb = b.read()
        if not (oa and ob):
            break
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        worst = max(worst, float(d.mean()))
        share = max(share, float(np.mean(d > 8)))
        n += 1
    a.release()
    b.release()
    return n, round(worst, 3), round(100 * share, 3)


def key_of(cfg):
    return next(k for k, v in HK.HOOKS.items() if v is cfg)


def seam_click(cfg):
    """Скачок между соседними сэмплами на шве не должен выделяться из дорожки."""
    import wave
    mix = f"{HK.BUILD}/assets/_mix_{SB.TAG}{key_of(cfg)}.wav"
    w = wave.open(mix)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    a = a.reshape(-1, 2).mean(axis=1)
    sr = w.getframerate()
    i = int(cfg["frames"] / FPS * sr)
    d = np.abs(np.diff(a))
    return round(float(d[i - 3:i + 3].max()), 5), round(float(np.percentile(d, 99.9)), 5)


def deform(cfg):
    """Деформация версии измеряется, а не заявляется.

    Кроп x0.98 обязан дать x1.0204 на экране; ускорение x1.02 — ровно 30 fps
    и длительность в 1.02 раза меньше несжатой."""
    out = cfg["out"]
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=r_frame_rate,nb_frames",
                         "-show_entries", "format=duration", "-of", "json", out],
                        capture_output=True, text=True)
    j = json.loads(pr.stdout)
    fr = j["streams"][0]["r_frame_rate"]
    nb = int(j["streams"][0]["nb_frames"])
    dur = float(j["format"]["duration"])
    plain = cfg["frames"] + n_body_frames()
    res = dict(r_frame_rate=fr, frames=nb, duration=round(dur, 3),
               frames_before=plain)
    if "crop" in (cfg.get("post_v") or ""):
        res["measured_scale"] = round(measure_scale(cfg), 4)
    if "setpts" in (cfg.get("post_v") or ""):
        res["measured_speed"] = round(plain / nb, 4)
    return res


def n_body_frames():
    c = cv2.VideoCapture(BODY)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return n - HK.CUT_F


def measure_scale(cfg):
    """Ширина карточки A на кадре тела: в версии с кропом должна вырасти x1.0204."""
    def card_w(path, f):
        c = cv2.VideoCapture(path)
        c.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = c.read()
        c.release()
        row = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[925]
        xs = np.nonzero(row > 24)[0]
        return xs.max() - xs.min() + 1
    f_ver = cfg["frames"] + 40
    w_ver = card_w(cfg["out"], f_ver)
    w_org = card_w(SB.OUT, HK.CUT_F + 40)
    return w_ver / w_org


if __name__ == "__main__":
    for key in sys.argv[1:] or ["h1", "h2"]:
        cfg = HK.HOOKS[key]
        print(f"\n===== {key}: {cfg['title']}")
        shots = bind(cfg)
        lens = [round(b - a, 3) for a, b, _, _ in shots]
        print("блок хука:", [(n, l) for (_, _, n, _), l in zip(shots, lens)],
              "· первый рез %.3fс" % shots[0][1])
        c, cs = cuts_inside_word(cfg, shots)
        print(f"cuts_inside_word={c} sample={cs}")
        c, cs = captions_off_speech(cfg)
        print(f"captions_off_speech={c} sample={cs}")
        c, cs = captions_crossing_cut(cfg, shots)
        print(f"captions_crossing_cut={c} sample={cs}")
        c, cs = caption_words_match(cfg)
        print(f"caption_words_not_in_speech={c} sample={cs}")
        c, cs = caption_style(cfg)
        print(f"caption_style_violations={c} sample={cs}")
        o, z, cl, cd, sm = layers(cfg, shots)
        print(f"gfx_text_overlap={o} gfx_out_of_zone={z} gfx_touching_mask={cl} "
              f"text_out_of_card={cd} sample={sm}")
        c, cs = line_overlaps(cfg)
        print(f"line_overlap_pairs={c} sample={cs}")
        print("пауза хук->тело: %.3fс (в оригинале на этом стыке 0.543с)"
              % (cfg["frames"] / FPS - cfg["speech_end"] + (4.806 - HK.CUT_A)))
        print("средний RGB карточки A: хук", HK.card_rgb(cfg["src"], 1.0),
              "· тело", HK.card_rgb(SB.SRC, 20.0))
        bi = body_identical(cfg)
        if bi:
            print("тело кадр в кадр: %d кадров, макс. средняя разница %.3f/255, "
                  "пикселей с разницей >8: %.3f%%" % bi)
        print("шов: скачок %.5f при p99.9 дорожки %.5f" % seam_click(cfg))
        print("деформация:", deform(cfg))
        b, bs = bright_outside(cfg)
        print(f"bright_outside_card_frames={b} sample={bs}")

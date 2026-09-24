"""QA версий с другим хуком (ролик 32).

Проверяется то, что специфично для версий: от старого хука не осталось кадров,
тело совпадает кадр в кадр с версией 1, пауза на шве, отсутствие щелчка,
деформация версии измерена по объекту в кадре, fps после ускорения, длительности
видео и звука сходятся, громкость совпадает с оригиналом.
"""
import json
import math
import os
import re
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import storyboard32 as SB
import render32 as R
import hook32 as HK
from style import FPS, CARD_A

VD = "/Users/vladimirkalajcidi/reels_good/videos/32"
ORIG = SB.OUT


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate,nb_frames,width,height",
                        "-of", "json", path], capture_output=True, text=True)
    v = json.loads(r.stdout)["streams"][0]
    r2 = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                         "stream=codec_type,duration", "-of", "json", path],
                        capture_output=True, text=True)
    d = {s["codec_type"]: float(s.get("duration", 0)) for s in json.loads(r2.stdout)["streams"]}
    return dict(fps=v["r_frame_rate"], frames=int(v["nb_frames"]),
                w=int(v["width"]), h=int(v["height"]),
                v_dur=d.get("video", 0), a_dur=d.get("audio", 0))


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[-1400:]
    g = lambda pat: float(re.search(pat, tail).group(1))
    return g(r"I:\s*(-?\d+\.\d+) LUFS"), g(r"Peak:\s*(-?\d+\.\d+) dBFS"), g(r"LRA:\s*(-?\d+\.\d+) LU")


def body_identical(path, key, step=17):
    """Тело версии должно совпадать с версией 1 кадр в кадр.

    Для версии с деформацией кадра сравнение идёт после обратного приведения
    к общей геометрии, поэтому допуск — уровень шума кодека, а не 0.
    """
    cfg = HK.HOOKS[key]
    a = cv2.VideoCapture(ORIG)
    b = cv2.VideoCapture(path)
    off = cfg["frames"]
    rate = 1.02 if cfg.get("post_a") else 1.0
    worst, checked = 0.0, 0
    for f in range(HK.CUT_F, 1531, step):
        a.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok1, x = a.read()
        j = int(round((off + (f - HK.CUT_F)) / rate))
        b.set(cv2.CAP_PROP_POS_FRAMES, j)
        ok2, y = b.read()
        if not (ok1 and ok2):
            continue
        if y.shape != x.shape:
            y = cv2.resize(y, (x.shape[1], x.shape[0]))
        d = float(np.mean(np.abs(x.astype(np.int16) - y.astype(np.int16))))
        worst = max(worst, d)
        checked += 1
    a.release()
    b.release()
    return checked, round(worst, 2)


def no_old_hook(path, key):
    """От старого хука не должно остаться ни кадра.

    `шов` — первый кадр тела версии против кадра CUT_F оригинала: обязаны
    совпасть (тело склеено готовыми кадрами). `начало` — кадр 10 версии против
    кадра 10 оригинала: обязаны РАЗОЙТИСЬ, там другой дубль. `хвост хука` —
    последний кадр блока хука против кадра CUT_F-1 оригинала: тоже обязаны
    разойтись, иначе в версию затесался кадр старого начала.
    """
    cfg = HK.HOOKS[key]
    rate = 1.02 if cfg.get("post_a") else 1.0
    a = cv2.VideoCapture(ORIG)
    b = cv2.VideoCapture(path)
    out = {}
    pairs = (("шов (должны совпасть)", HK.CUT_F, int(round(cfg["frames"] / rate))),
             ("начало (должны разойтись)", 10, int(round(10 / rate))),
             ("хвост хука (должны разойтись)", HK.CUT_F - 1,
              int(round(cfg["frames"] / rate)) - 1))
    for name, fo, fv in pairs:
        a.set(cv2.CAP_PROP_POS_FRAMES, fo)
        ok1, x = a.read()
        b.set(cv2.CAP_PROP_POS_FRAMES, fv)
        ok2, y = b.read()
        if not (ok1 and ok2):
            continue
        if y.shape != x.shape:
            y = cv2.resize(y, (x.shape[1], x.shape[0]))
        out[name] = round(float(np.mean(np.abs(x.astype(np.int16) - y.astype(np.int16)))), 2)
    a.release()
    b.release()
    return out


def seam_click(path, key):
    """Скачок между соседними сэмплами на шве не должен выделяться на фоне
    типичного скачка внутри дорожки."""
    cfg = HK.HOOKS[key]
    wav = f"{HK.BUILD}/assets/_qa32seam.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                    "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", wav], check=True)
    import wave
    w = wave.open(wav, "rb")
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    os.remove(wav)
    rate = 1.02 if cfg.get("post_a") else 1.0
    i = int(cfg["frames"] / FPS / rate * 48000)
    local = float(np.abs(np.diff(x[i - 3:i + 4])).max())
    typical = float(np.percentile(np.abs(np.diff(x)), 99.9))
    return round(local, 5), round(typical, 5)


def pause_at_seam(key):
    """Пауза «хук -> тело» = хвост тишины хука + собственная пауза тела."""
    cfg = HK.HOOKS[key]
    hw = json.load(open(f"{VD}/words_hook{cfg['version'] - 1}.json"))
    last = max(w["end"] for w in hw)
    tail = cfg["frames"] / FPS - last
    body = json.load(open(SB.WORDS))
    first = min(w["start"] for w in body if w["start"] >= HK.CUT)
    return round(tail + (first - HK.CUT), 3), round(tail, 3), round(first - HK.CUT, 3)


def deform(path, key):
    """Деформацию мерим по объекту в кадре: ширина ряда фигур в теле версии
    против той же ширины в версии 1."""
    cfg = HK.HOOKS[key]
    t_orig = 29.30                     # план parityB, ряд стоит целиком
    rate = 1.02 if cfg.get("post_a") else 1.0

    def row_width(p, f):
        c = cv2.VideoCapture(p)
        c.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = c.read()
        c.release()
        if not ok:
            return None
        band = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = img.shape[0]
        y = int(round(R.BODY_Y * h / 1920))
        strip = band[y - 2:y + 3, :].max(axis=0)
        xs = np.nonzero(strip > 170)[0]
        return int(xs.max() - xs.min()) if len(xs) else None

    f_orig = int(round(t_orig * FPS))
    f_ver = int(round((cfg["frames"] + (f_orig - HK.CUT_F)) / rate))
    a = row_width(ORIG, f_orig)
    b = row_width(path, f_ver)
    out = dict(row_px_v1=a, row_px_ver=b,
               scale=(round(b / a, 4) if a and b else None))
    if cfg.get("post_a"):                       # версия с ускорением
        pr = probe(path)
        want = (cfg["frames"] + (1531 - HK.CUT_F)) / FPS / rate
        out.update(kind="ускорение x1.02", fps=pr["fps"],
                   dur=round(pr["v_dur"], 3), dur_expected=round(want, 3),
                   av_delta=round(abs(pr["v_dur"] - pr["a_dur"]), 3))
    else:
        out.update(kind="кроп x0.98", scale_expected=round(1 / 0.98, 4))
    return out


def caption_words_match(key):
    """Слова субтитров хука обязаны быть в расшифровке этого дубля."""
    cfg = HK.HOOKS[key]
    norm = lambda w: re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))
    said = [norm(w["word"]) for w in
            json.load(open(f"{VD}/words_hook{cfg['version'] - 1}.json"))]
    said = [w for w in said if w]
    typed = [norm(w) for _t0, _t1, runs, _s in cfg["caps"]
             for txt, _k, _sz in runs for w in txt.split()]
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
    return len(missing), missing


def caption_checks(key):
    cfg = HK.HOOKS[key]
    hw = json.load(open(f"{VD}/words_hook{cfg['version'] - 1}.json"))
    bad = []
    for t0, t1, runs, _s in cfg["caps"]:
        text = "".join(r[0] for r in runs).strip()
        if any(ch in text for ch in ".,!?;:«»"):
            bad.append(("пунктуация", t0, text))
        if text != text.lower():
            bad.append(("капс", t0, text))
        n = len(text.split())
        if not 2 <= n <= 4:
            bad.append((f"{n} слов", t0, text))
        if re.search(r"\d", text):
            bad.append(("цифра в субтитре", t0, text))
        if not any(abs(w["start"] - t0) < 0.26 for w in hw):
            bad.append(("мимо речи", t0, text))
        s = next((s for s in cfg["shots"] if s[0] <= t0 < s[1]), None)
        if s and t1 > s[1] + 1e-6:
            bad.append(("субтитр через рез", t0, text))
    return len(bad), bad


def cuts_inside_word(key):
    cfg = HK.HOOKS[key]
    hw = json.load(open(f"{VD}/words_hook{cfg['version'] - 1}.json"))
    bad = []
    for t0, _t1, _k, _p in cfg["shots"][1:]:
        for w in hw:
            if w["start"] + 0.02 < t0 < w["end"] - 0.02:
                bad.append((round(t0, 3), w["word"]))
    return len(bad), bad


def rhythm(key, path):
    cfg = HK.HOOKS[key]
    hook_shots = [(a, b, k) for a, b, k, _ in cfg["shots"]]
    body_shots = [(max(a, HK.CUT), b, k) for a, b, k, _ in SB.SHOTS if b > HK.CUT]
    dt = cfg["frames"] / FPS - HK.CUT
    all_len = [b - a for a, b, _ in hook_shots] + [b - a for a, b, _ in body_shots]
    face = (sum(b - a for a, b, k in hook_shots if k in SB.FACE_KINDS)
            + sum(b - a for a, b, k in body_shots if k in SB.FACE_KINDS))
    dur = probe(path)["v_dur"]
    rate = 1.02 if cfg.get("post_a") else 1.0
    all_len = [x / rate for x in all_len]
    face /= rate
    s = sorted(all_len)
    return dict(shots=len(s), avg=round(sum(s) / len(s), 2), median=round(s[len(s) // 2], 2),
                shortest=round(s[0], 2), longest=round(s[-1], 2),
                first_cut=round(hook_shots[0][1] / rate, 2),
                face_pct=round(100 * face / dur, 1), dur=round(dur, 2))


def pixel_checks(key):
    """Те же машинные проверки, что у версии 1, но по кадрам блока хука:
    текст не выходит за карточку, графика не пересекает субтитры и не выходит
    из зоны, строки блока не налезают друг на друга."""
    from style import font
    cfg = HK.HOOKS[key]
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()
    scale = cfg.get("gfx_scale", {})
    y0, y1 = SB.GFX_ZONE
    x0, x1 = SB.GFX_X
    cx, cy, cw, chh = CARD_A
    out = dict(text_out_of_card=0, gfx_text_overlap=0, gfx_out_of_zone=0)
    samples = []
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        t0, _t1, kind, _p = R.shot_at(t)
        gl = R.graphics_layer(kind, (t - t0) * scale.get(kind, 1.0))
        cl = R.caption_layer(t)
        if cl is not None:
            a = np.array(cl.split()[3])
            ys, xs = np.nonzero(a > 120)
            if len(ys) and (xs.min() < cx + 30 or xs.max() > cx + cw - 30
                            or ys.min() < cy + 30 or ys.max() > cy + chh - 30):
                out["text_out_of_card"] += 1
                samples.append(("текст за карточкой", round(t, 2), kind))
        if gl is not None:
            ga = np.array(gl.split()[3])
            ys, xs = np.nonzero(ga > 40)
            if len(ys) and (ys.min() < y0 or ys.max() > y1
                            or xs.min() < x0 or xs.max() > x1):
                out["gfx_out_of_zone"] += 1
                samples.append(("графика вне зоны", round(t, 2), kind))
            if cl is not None:
                ca = np.array(cl.split()[3])
                if int(np.count_nonzero((ga > 40) & (ca > 40))):
                    out["gfx_text_overlap"] += 1
                    samples.append(("графика на тексте", round(t, 2), kind))
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    boxes = {}
    for i, (_t0, _t1, runs, _s) in enumerate(cfg["caps"]):
        top = min(font(kmap[k], sz).getbbox(txt)[1] for txt, k, sz in runs)
        bot = max(font(kmap[k], sz).getbbox(txt)[3] for txt, k, sz in runs)
        boxes[i] = (top, bot)
    by_block = {}
    for i in range(len(cfg["caps"])):
        bid, pos, _n, _e, yoff = R.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, yoff))
    pairs = 0
    for _bid, rows in by_block.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            ha = (boxes[a[1]][1] - boxes[a[1]][0]) / 2
            hb = (boxes[b[1]][1] - boxes[b[1]][0]) / 2
            if a[2] + ha > b[2] - hb:
                pairs += 1
    out["line_overlap_pairs"] = pairs
    return out, samples[:6]


if __name__ == "__main__":
    print("версия 1:", probe(ORIG), loudness(ORIG))
    for key in ("h1", "h2"):
        cfg = HK.HOOKS[key]
        p = cfg["out"]
        print(f"\n===== {key} -> {os.path.basename(p)} (версия {cfg['version']}) =====")
        print("формат:", probe(p))
        i, tp, lra = loudness(p)
        print(f"громкость: I={i} LUFS  TP={tp} dBTP  LRA={lra} LU  "
              f"(версия 1: {loudness(ORIG)[0]} LUFS)")
        print("ритм:", rhythm(key, p))
        print("шов (средняя разница пикселей):", no_old_hook(p, key))
        ch, worst = body_identical(p, key)
        print(f"тело кадр в кадр: сверено {ch} кадров, худшая средняя разница {worst}")
        loc, typ = seam_click(p, key)
        print(f"щелчок на шве: локальный скачок {loc} против типичного {typ} (p99.9)")
        print("пауза хук->тело:", pause_at_seam(key), "(всего, хвост хука, пауза тела)")
        print("деформация (ширина ряда px):", deform(p, key))
        print("субтитры мимо расшифровки:", caption_words_match(key))
        print("субтитры (стиль/тайминг):", caption_checks(key))
        print("резы внутри слова:", cuts_inside_word(key))
        print("пиксельные проверки блока хука:", pixel_checks(key))

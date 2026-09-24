"""QA версий ролика 31 с другими хуками.

    python3 qa31hook.py h1
    python3 qa31hook.py h2

Проверяется блок хука (субтитры, графика, зоны) плюс всё, что относится
к версии целиком: от старого начала не осталось ни кадра и ни субтитра,
тело совпадает с версией №1 кадр в кадр, деформация версии измерена,
на шве нет щелчка, громкость сведена к громкости оригинала.
"""
import json
import math
import os
import re
import subprocess
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, FPS, W, H
import render31 as R
import storyboard31 as SB
import hook31 as HK

BUILD = os.path.dirname(os.path.abspath(__file__))
ALPHA = 40
V1 = SB.OUT
BODY = f"{BUILD}/assets/_video_{SB.TAG}.mp4"


def setup(key):
    cfg = HK.HOOKS[key]
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()
    R.GFX.update(HK.HOOK_GFX)
    return cfg


def words_of(key):
    p = f"/Users/vladimirkalajcidi/reels_good/videos/31/words_hook{key[-1]}.json"
    return [(w["start"], w["end"], w["word"]) for w in json.load(open(p))]


# --- субтитры хука ----------------------------------------------------------
def caption_words_match(cfg, key):
    norm = lambda w: re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))
    said = [w for w in (norm(x[2]) for x in words_of(key)) if w]
    typed = [w for c in cfg["caps"] for txt, _, _ in c[2]
             for w in (norm(t) for t in txt.split()) if w]
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
        if re.search(r"\d", text):
            bad.append(("цифра", t0, text))
    return len(bad), bad[:8]


def caption_sync(cfg, key):
    words = words_of(key)
    bad = []
    for t0, _, runs, _ in cfg["caps"]:
        if not any(abs(w[0] - t0) < 0.26 for w in words):
            bad.append((round(t0, 2), "".join(r[0] for r in runs).strip()))
    return len(bad), bad[:8]


def caption_in_shot(cfg):
    bad = []
    for t0, t1, runs, _ in cfg["caps"]:
        s = R.shot_at(t0)
        if t1 > s[1] + 1e-6:
            bad.append((round(t0, 2), round(t1, 2), round(s[1], 2),
                        "".join(r[0] for r in runs).strip()))
    return len(bad), bad[:8]


def cuts_inside_word(cfg, key):
    words = words_of(key)
    bad = []
    for t0 in [s[0] for s in cfg["shots"][1:]]:
        for a, b, w in words:
            if a + 0.02 < t0 < b - 0.02:
                bad.append((round(t0, 2), w, round(a, 2), round(b, 2)))
                break
    return len(bad), bad


def captions_on_number_shots(cfg):
    """Число графики и субтитр не звучат одновременно."""
    bad = []
    for t0, t1, kind, _ in cfg["shots"]:
        num = None
        if kind == "schedTen":
            num = t0 + 1.082
        elif kind == "waitFive":
            num = t0 + 1.810
        if num is None:
            continue
        for c in cfg["caps"]:
            if t0 <= c[0] < t1 and c[1] > num + 0.04:
                bad.append((kind, round(c[0], 2), "".join(r[0] for r in c[2])))
    return len(bad), bad


# --- пиксельные проверки блока хука -----------------------------------------
def _layers(cfg, step=2):
    for f in range(0, cfg["frames"], step):
        t = f / FPS
        t0, _, kind, _ = R.shot_at(t)
        yield t, kind, R.graphics_layer(kind, t - t0), R.caption_layer(t)


def gfx_text_overlap(cfg):
    bad, sample = 0, []
    for t, kind, gl, cl in _layers(cfg):
        if gl is None or cl is None:
            continue
        n = int(np.count_nonzero((np.array(gl.split()[3]) > ALPHA)
                                 & (np.array(cl.split()[3]) > ALPHA)))
        if n:
            bad += 1
            sample.append((round(t, 2), kind, n))
    return bad, sample[:8]


def gfx_in_zone(cfg):
    y0, y1 = SB.GFX_ZONE
    x0, x1 = SB.GFX_X
    bad, sample = 0, []
    for t, kind, gl, _ in _layers(cfg):
        if gl is None:
            continue
        ys, xs = np.nonzero(np.array(gl.split()[3]) > ALPHA)
        if len(ys) == 0:
            continue
        if ys.min() < y0 or ys.max() > y1 or xs.min() < x0 or xs.max() > x1:
            bad += 1
            sample.append((round(t, 2), kind, int(xs.min()), int(xs.max()),
                           int(ys.min()), int(ys.max())))
    return bad, sample[:8]


def gfx_clipped(cfg):
    y0, y1 = SB.GFX_ZONE
    x0, x1 = SB.GFX_X
    bad, sample = 0, []
    for t, kind, gl, _ in _layers(cfg):
        if gl is None:
            continue
        ys, xs = np.nonzero(np.array(gl.split()[3]) > 120)
        if len(ys) == 0:
            continue
        if (ys.max() > y1 - 22 or ys.min() < y0 + 22
                or xs.max() > x1 - 22 or xs.min() < x0 + 22):
            bad += 1
            sample.append((round(t, 2), kind, int(xs.min()), int(xs.max()),
                           int(ys.min()), int(ys.max())))
    return bad, sample[:8]


def text_in_card(cfg):
    bad, sample = 0, []
    x, y, w, h = CARD_A
    for t, kind, _, cl in _layers(cfg):
        if cl is None:
            continue
        ys, xs = np.nonzero(np.array(cl.split()[3]) > 120)
        if len(ys) == 0:
            continue
        if (xs.min() < x + 30 or xs.max() > x + w - 30
                or ys.min() < y + 30 or ys.max() > y + h - 30):
            bad += 1
            sample.append((round(t, 2), kind, int(xs.min()), int(xs.max()),
                           int(ys.min()), int(ys.max())))
    return bad, sample[:8]


def line_overlaps(cfg):
    from style import font
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    boxes = {}
    for i, (t0, t1, runs, _) in enumerate(cfg["caps"]):
        boxes[i] = (min(font(kmap[k], sz).getbbox(txt)[1] for txt, k, sz in runs),
                    max(font(kmap[k], sz).getbbox(txt)[3] for txt, k, sz in runs))
    by_block = {}
    for i in range(len(cfg["caps"])):
        bid, pos, _, _, yoff = R.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, yoff))
    bad, sample = 0, []
    for bid, rows in by_block.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            ha = (boxes[a[1]][1] - boxes[a[1]][0]) / 2
            hb = (boxes[b[1]][1] - boxes[b[1]][0]) / 2
            if a[2] + ha > b[2] - hb:
                bad += 1
                sample.append((bid, round(cfg["caps"][a[1]][0], 2)))
    return bad, sample[:8]


# --- версия целиком ---------------------------------------------------------
SLACK = 2          # запас на билинейный ресемпл кропа, не на текст


def card_rect(cfg):
    """Габарит карточки A с учётом деформации версии.

    Для версии с кропом считается по фактической геометрии ffmpeg
    (crop=iw*0.98 -> 1058x1881 со смещением 11/19, затем scale обратно),
    а не по идеальному 1/0.98: разница доходит до полутора пикселей.
    Запас SLACK=2px — на билинейный ресемпл края карточки."""
    x, y, w, h = CARD_A
    if (cfg.get("post_v") or "").startswith("crop=iw*0.98"):
        cw, ch = int(W * 0.98), int(H * 0.98)
        xo, yo = (W - cw) // 2, (H - ch) // 2
        sx, sy = W / cw, H / ch
        x0, x1 = (x - xo) * sx, (x + w - 1 - xo) * sx
        y0, y1 = (y - yo) * sy, (y + h - 1 - yo) * sy
        x, y = math.floor(x0), math.floor(y0)
        w, h = math.ceil(x1) - x + 1, math.ceil(y1) - y + 1
    return (x - SLACK, y - SLACK, w + 2 * SLACK, h + 2 * SLACK)


def bright_outside(cfg):
    cap = cv2.VideoCapture(cfg["out"])
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x, y, w, h = card_rect(cfg)
    bad, sample = 0, []
    for f in range(0, nf, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[max(0, y):y + h, max(0, x):x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:8]


def _frames(path, idxs):
    c = cv2.VideoCapture(path)
    out, i, mx, want = {}, 0, max(idxs), set(idxs)
    while i <= mx:
        ok, f = c.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    c.release()
    return out


def body_identical(cfg, key):
    """Тело версии — те же кадры, что в версии №1 (сверяется до деформации)."""
    off = cfg["frames"] - HK.CUT_F
    idxs = [HK.CUT_F, 300, 700, 1100, 1410]
    a = _frames(BODY, idxs)
    b = _frames(f"{BUILD}/assets/_video_{SB.TAG}{key}.mp4", [i + off for i in idxs])
    return [(n, round(float(np.abs(a[n].astype(int) - b[n + off].astype(int)).mean()), 3))
            for n in idxs]


def no_old_hook_frame(cfg, key):
    """Первый кадр тела в версии — это кадр CUT_F, а не более ранний."""
    off = cfg["frames"] - HK.CUT_F
    a = _frames(BODY, [HK.CUT_F - 2, HK.CUT_F - 1, HK.CUT_F])
    b = _frames(f"{BUILD}/assets/_video_{SB.TAG}{key}.mp4", [cfg["frames"]])[cfg["frames"]]
    return {n: round(float(np.abs(a[n].astype(int) - b.astype(int)).mean()), 2)
            for n in a}


def no_old_hook_caption():
    """Ни один субтитр старого начала не доживает до точки реза."""
    alive = [c for i, c in enumerate(SB.CAPS)
             if c[0] < HK.CUT_T <= _block_end(i)]
    return len(alive), [(round(c[0], 2), "".join(r[0] for r in c[2])) for c in alive]


def _block_end(i):
    import importlib
    m = importlib.import_module("render31")
    return m.build_blocks()[i][3] if False else _BODY_BLOCKS[i][3]


def deform(cfg, key):
    """Деформация версии измеряется по одному и тому же объекту."""
    off = cfg["frames"] - HK.CUT_F
    n1 = 945
    if cfg.get("post_a") == "atempo=1.02":
        n2 = int(round((n1 + off) / 1.02))
    else:
        n2 = n1 + off
    a = _frames(V1, [n1])[n1]
    b = _frames(cfg["out"], [n2])[n2]

    def wide(img, yy=900):
        row = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[yy].astype(int)
        xs = np.nonzero(row > 28)[0]
        return int(xs.max() - xs.min() + 1)

    w1, w2 = wide(a), wide(b)
    return w1, w2, round(w2 / w1, 4)


def speed_check(cfg):
    """Ускорение версии измеряется по кадрам и по звуку, а не по флагу."""
    if not (cfg.get("post_a") or "").startswith("atempo="):
        return None
    k = float(cfg["post_a"].split("=")[1])
    src_frames = cfg["frames"] + (1411 - HK.CUT_F)
    a = av(cfg["out"])
    out_frames = int(a["frames"])
    return dict(nominal=k,
                frames=f"{src_frames} -> {out_frames} = x{src_frames/out_frames:.4f}",
                fps=a["fps"],
                v_dur=a["v_dur"], a_dur=a["a_dur"],
                av_delta=round(abs(float(a["v_dur"]) - float(a["a_dur"])), 4))


def seam_click(key, cfg):
    p = f"{BUILD}/assets/_mix_{SB.TAG}{key}.wav"
    w = wave.open(p)
    sr = w.getframerate()
    d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    d = d.reshape(-1, 2).mean(1)
    i = int(cfg["frames"] / FPS * sr)
    return (round(float(np.abs(np.diff(d[i - 5:i + 5])).max()), 5),
            round(float(np.percentile(np.abs(np.diff(d)), 99.9)), 5))


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr
    m = json.loads(s[s.rindex("{"):s.rindex("}") + 1])
    return float(m["input_i"]), float(m["input_tp"]), float(m["input_lra"])


def av(path):
    def q(sel, ent):
        return subprocess.run(["ffprobe", "-v", "error", "-select_streams", sel,
                               "-show_entries", ent, "-of", "csv=p=0", path],
                              capture_output=True, text=True).stdout.strip()
    return dict(fps=q("v:0", "stream=r_frame_rate"),
                frames=q("v:0", "stream=nb_frames"),
                v_dur=q("v:0", "stream=duration"),
                a_dur=q("a:0", "stream=duration"))


_BODY_BLOCKS = R.build_blocks()


if __name__ == "__main__":
    key = sys.argv[1]
    cfg = setup(key)
    print(f"=== версия {cfg['version']} ({key}) — {os.path.basename(cfg['out'])} ===")
    print("хук:", cfg["title"])
    print("планов в блоке хука:", len(cfg["shots"]),
          "· кадров:", cfg["frames"], f"({cfg['frames']/FPS:.3f}с)")
    print("cuts_inside_word=%d %s" % cuts_inside_word(cfg, key))
    print("captions_off_speech=%d %s" % caption_sync(cfg, key))
    print("captions_crossing_cut=%d %s" % caption_in_shot(cfg))
    print("caption_words_not_in_speech=%d %s" % caption_words_match(cfg, key))
    print("caption_style_violations=%d %s" % caption_style(cfg))
    print("captions_on_number_shots=%d %s" % captions_on_number_shots(cfg))
    print("line_overlap_pairs=%d %s" % line_overlaps(cfg))
    print("gfx_text_overlap_frames=%d %s" % gfx_text_overlap(cfg))
    print("gfx_out_of_zone_frames=%d %s" % gfx_in_zone(cfg))
    print("gfx_touching_mask_frames=%d %s" % gfx_clipped(cfg))
    print("text_out_of_card_frames=%d %s" % text_in_card(cfg))
    n, s = no_old_hook_caption()
    print(f"old_hook_captions_alive_at_cut={n} {s}")
    print("первый кадр тела vs соседние кадры версии №1:", no_old_hook_frame(cfg, key))
    print("тело кадр в кадр (средняя |Δ| по кадру, до деформации):",
          body_identical(cfg, key))
    w1, w2, k = deform(cfg, key)
    print(f"деформация кадра: карточка A {w1}px -> {w2}px = x{k}")
    sp = speed_check(cfg)
    if sp:
        print("деформация темпа:", sp)
    j, ref = seam_click(key, cfg)
    print(f"щелчок на шве: скачок {j} · 99.9-й перцентиль дорожки {ref} "
          f"-> {'ок' if j <= ref else 'ЩЕЛЧОК'}")
    print("формат:", av(cfg["out"]))
    i, tp, lra = loudness(cfg["out"])
    print(f"громкость: {i} LUFS · TP {tp} dBTP · LRA {lra} LU "
          f"(оригинал {HK.TARGET_LUFS} LUFS)")
    print("bright_outside_card_frames=%d %s" % bright_outside(cfg))

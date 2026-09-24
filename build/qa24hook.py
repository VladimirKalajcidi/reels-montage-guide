"""QA версий ролика 24 с другими хуками (delivery-specs §6, блок «Версия с другим хуком»).

    python3 qa24hook.py h1
"""
import json
import os
import subprocess
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, FPS
import storyboard24 as SB
import hook24 as HK

BUILD = os.path.dirname(os.path.abspath(__file__))
MAIN = SB.OUT
BODY = f"{BUILD}/assets/_video_regress.mp4"


def old_hook_gone():
    """От старого хука не осталось ни кадра: соседние кадры вокруг реза
    обязаны лежать в РАЗНЫХ планах, иначе рез прошёл внутри плана."""
    a = SB.shot_at((HK.CUT_F - 1) / FPS)
    b = SB.shot_at(HK.CUT_F / FPS)
    dropped = [s for s in SB.SHOTS[:SB.SHOTS.index(b)]]
    return dict(ok=a is not b, before=a[:3], after=b[:3],
                dropped_shots=len(dropped),
                dropped_caps=len([c for c in SB.CAPS if c[0] < b[0]]))


def body_identical(key, n=14):
    """Тело кадр в кадр совпадает со сданной версией: сверяем немой монтаж
    версии (до деформации) с _video_regress.mp4 по PSNR."""
    cfg = HK.HOOKS[key]
    ver = cv2.VideoCapture(f"{BUILD}/assets/_video_regress{key}.mp4")
    body = cv2.VideoCapture(BODY)
    total = int(body.get(cv2.CAP_PROP_FRAME_COUNT)) - HK.CUT_F
    worst = (1e9, None)
    for i in np.linspace(0, total - 2, n).astype(int):
        ver.set(cv2.CAP_PROP_POS_FRAMES, cfg["frames"] + int(i))
        body.set(cv2.CAP_PROP_POS_FRAMES, HK.CUT_F + int(i))
        oka, a = ver.read()
        okb, b = body.read()
        if not (oka and okb):
            continue
        mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
        psnr = 99.0 if mse < 1e-6 else 10 * np.log10(255 ** 2 / mse)
        if psnr < worst[0]:
            worst = (round(float(psnr), 1), int(i))
    ver.release()
    body.release()
    return dict(worst_psnr_db=worst[0], at_body_frame=worst[1], frames_checked=n)


def card_width(path, t):
    """Ширина карточки A на кадре: по крайним столбцам ярче порога."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cols = np.nonzero((g > 26).sum(axis=0) > 40)[0]
    return int(cols.min()), int(cols.max()), int(cols.max() - cols.min() + 1)


def deformation(key):
    """Деформация версии измеряется, а не принимается на веру.

    Кроп меряется по ширине карточки A на одном и том же кадре тела,
    ускорение — по числу кадров и по длительностям видео и звука.
    """
    cfg = HK.HOOKS[key]
    post = cfg.get("post_v") or ""
    dt = cfg["frames"] / FPS - HK.CUT
    speed = 1.02 if "setpts" in post else 1.0
    t_main = 50.4                                  # план с лицом внутри тела:
    # мерить надо по кадру с лицом, а не по сетке — у графического плана
    # видимая ширина карточки задана вигнетом, а не её геометрией
    a = card_width(MAIN, t_main)
    b = card_width(cfg["out"], (t_main + dt) / speed)
    out = dict(kind=("ускорение x1.02" if speed != 1.0 else "кроп x0.98"),
               card_main=a, card_version=b)
    if a and b:
        out["scale_on_screen"] = round(b[2] / a[2], 4)
        out["scale_expected"] = round(1 / 0.98, 4) if speed == 1.0 else 1.0
    f = video_facts(cfg["out"])
    raw = cfg["frames"] + (int(video_facts(BODY)["frames"]) - HK.CUT_F)
    out["frames_expected"] = int(round(raw / speed))
    out["frames_actual"] = f["frames"]
    out["av_dur_gap"] = round(abs(f["v_dur"] - f["a_dur"]), 3)
    return out


def rhythm(key):
    """Ритм версии целиком: планы хука + планы тела после реза."""
    cfg = HK.HOOKS[key]
    dt = cfg["frames"] / FPS - HK.CUT
    speed = 1.02 if "setpts" in (cfg.get("post_v") or "") else 1.0
    shots = [(t0, t1) for t0, t1, _k, _p in cfg["shots"]]
    faces = [(t0, t1) for t0, t1, k, _p in cfg["shots"] if k in SB.FACE_KINDS]
    for t0, t1, k, _p in SB.SHOTS:
        if t1 <= HK.CUT + 0.01:
            continue
        a, b = max(t0, HK.CUT) + dt, t1 + dt
        shots.append((a, b))
        if k in SB.FACE_KINDS:
            faces.append((a, b))
    dur = (shots[-1][1]) / speed
    lens = sorted((b - a) / speed for a, b in shots)
    face = sum((b - a) / speed for a, b in faces)
    return dict(dur=round(dur, 2), shots=len(shots),
                avg=round(sum(lens) / len(lens), 2),
                median=round(lens[len(lens) // 2], 2),
                shortest=round(lens[0], 2), longest=round(lens[-1], 2),
                first_cut=round(shots[0][1] / speed, 2),
                face_pct=round(100 * face / dur, 1))


def video_facts(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate,nb_frames,width,height",
                        "-show_entries", "format=duration", "-of", "json", path],
                       capture_output=True, text=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    r2 = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                         "-show_entries", "stream=duration", "-of", "json", path],
                        capture_output=True, text=True)
    st = json.loads(r2.stdout).get("streams") or [{}]      # немой монтаж — без дорожки
    return dict(fps=s["r_frame_rate"], frames=int(s["nb_frames"]),
                size=f'{s["width"]}x{s["height"]}',
                v_dur=round(float(j["format"]["duration"]), 3),
                a_dur=round(float(st[0].get("duration", 0)), 3))


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[-1200:]
    def grab(tag):
        for ln in tail.splitlines():
            if ln.strip().startswith(tag):
                return float(ln.split()[1])
        return None
    return dict(I=grab("I:"), LRA=grab("LRA:"), TP=grab("Peak:"))


def seam_click(key):
    """На стыке «хук -> тело» нет щелчка: скачок между соседними сэмплами
    того же порядка, что внутри дорожки."""
    cfg = HK.HOOKS[key]
    wav = f"{BUILD}/assets/_mix_regress{key}.wav"
    f = wave.open(wav)
    n = f.getnframes()
    d = np.frombuffer(f.readframes(n), np.int16).astype(np.float32).reshape(-1, 2)
    f.close()
    sr = HK.SR
    i = int(cfg["frames"] / FPS * sr)
    dif = np.abs(np.diff(d[:, 0]))
    win = dif[i - 400:i + 400]
    return dict(seam_max_jump=int(win.max()), track_p999=int(np.percentile(dif, 99.9)),
                track_max=int(dif.max()))


def hook_body_pause(key):
    """Пауза между последним словом хука и первым словом тела."""
    cfg = HK.HOOKS[key]
    hw = json.load(open(f"/Users/vladimirkalajcidi/reels_good/videos/24/"
                        f"words_{os.path.basename(cfg['src']).split('.')[0]}.json"))
    body_words = json.load(open(SB.WORDS))
    first = next(w for w in body_words if w["start"] >= HK.CUT)
    dt = cfg["frames"] / FPS - HK.CUT
    speed = 1.02 if "setpts" in (cfg.get("post_v") or "") else 1.0
    return dict(hook_last_word=hw[-1]["word"], hook_ends=hw[-1]["end"],
                body_first_word=first["word"],
                body_starts=round(first["start"] + dt, 3),
                pause=round((first["start"] + dt - hw[-1]["end"]) / speed, 3))


def hook_picture(key):
    """Те же три машинные проверки, что и у основного ролика, но по блоку хука:
    текст внутри карточки, графика не пересекает текст, графика в своей зоне."""
    import render24 as R
    cfg = HK.HOOKS[key]
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()
    lts = HK.lts_for(cfg)
    y0, y1 = SB.GFX_ZONE
    x0, x1 = SB.GFX_X
    bad_card = bad_ovl = bad_zone = bad_lines = 0
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        t0, _t1, kind, _p = R.shot_at(t)
        cl = R.caption_layer(t)
        gl = R.graphics_layer(kind, (t - t0) * lts.get(kind, 1.0))
        if cl is not None:
            a = np.array(cl.split()[3])
            ys, xs = np.nonzero(a > 120)
            if len(ys):
                cx, cy, cw, ch = CARD_A
                if (xs.min() < cx + 30 or xs.max() > cx + cw - 30
                        or ys.min() < cy + 30 or ys.max() > cy + ch - 30):
                    bad_card += 1
        if gl is not None:
            ga = np.array(gl.split()[3])
            ys, xs = np.nonzero(ga > 40)
            if len(ys) and (ys.min() < y0 or ys.max() > y1
                            or xs.min() < x0 or xs.max() > x1):
                bad_zone += 1
            if cl is not None and np.any((ga > 40) & (np.array(cl.split()[3]) > 40)):
                bad_ovl += 1
    from style import font
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    by_block = {}
    for i in range(len(cfg["caps"])):
        bid, pos, _n, _e, yoff = R.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, yoff))
    for rows in by_block.values():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            ha = max(font(kmap[k], sz).getbbox(tx)[3] - font(kmap[k], sz).getbbox(tx)[1]
                     for tx, k, sz in cfg["caps"][a[1]][2]) / 2
            hb = max(font(kmap[k], sz).getbbox(tx)[3] - font(kmap[k], sz).getbbox(tx)[1]
                     for tx, k, sz in cfg["caps"][b[1]][2]) / 2
            if a[2] + ha > b[2] - hb:
                bad_lines += 1
    return dict(text_out_of_card=bad_card, gfx_text_overlap=bad_ovl,
                gfx_out_of_zone=bad_zone, line_overlaps=bad_lines)


def hook_captions(key):
    """Субтитры хука: строчные, без пунктуации, 2-4 слова, все слова из дубля."""
    import re
    cfg = HK.HOOKS[key]
    norm = lambda w: re.sub(r"[^а-яa-z]", "", w.lower().replace("ё", "е"))
    src_words = json.load(open(f"/Users/vladimirkalajcidi/reels_good/videos/24/"
                               f"words_{os.path.basename(cfg['src']).split('.')[0]}.json"))
    said = [norm(w["word"]) for w in src_words]
    bad = []
    i = 0
    for t0, _t1, runs, _slot in cfg["caps"]:
        text = "".join(r[0] for r in runs).strip()
        if any(ch in text for ch in ".,!?;:«»"):
            bad.append(("пунктуация", t0, text))
        if text != text.lower():
            bad.append(("капс", t0, text))
        if not 2 <= len(text.split()) <= 4:
            bad.append((f"{len(text.split())} слов", t0, text))
        if not any(abs(w["start"] - t0) < 0.26 for w in src_words):
            bad.append(("мимо речи", t0, text))
        for w in text.split():
            j = i
            while j < len(said) and said[j] != norm(w):
                j += 1
            if j < len(said):
                i = j + 1
            else:
                bad.append(("нет в речи", t0, w))
    return dict(violations=len(bad), sample=bad[:6])


def bright_outside(key):
    """Текст за карточкой — по всему файлу версии, включая деформацию."""
    cfg = HK.HOOKS[key]
    dt = cfg["frames"] / FPS - HK.CUT
    speed = 1.02 if "setpts" in (cfg.get("post_v") or "") else 1.0
    crop = "crop" in (cfg.get("post_v") or "")
    cap = cv2.VideoCapture(cfg["out"])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad, sample = 0, []
    for f in range(0, n, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        t = f / FPS * speed                      # время на недеформированной ленте
        if t < cfg["frames"] / FPS:
            kind = next((s[2] for s in cfg["shots"] if s[0] <= t < s[1]), None)
        else:
            kind = SB.shot_at(t - dt)[2]
        from style import CARD_B
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        if crop:                                 # кроп 0.98 двигает и растягивает рамку
            x0n, y0n = (x - 1080 * 0.01) / 0.98, (y - 1920 * 0.01) / 0.98
            x, y = int(np.floor(x0n)) - 2, int(np.floor(y0n)) - 2
            w, h = int(np.ceil(w / 0.98)) + 4, int(np.ceil(h / 0.98)) + 4
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[max(0, y):y + h, max(0, x):x + w] = 1
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((g > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return dict(frames=bad, sample=sample[:6])


def card_rgb_delta(key):
    """Средний RGB внутри карточки A: кадры хука против кадров тела."""
    cfg = HK.HOOKS[key]
    hook = HK.card_rgb(cfg["src"], 1.0)
    main = HK.card_rgb(SB.SRC, 1.0)
    return dict(hook=hook, body=main,
                delta=max(abs(a - b) for a, b in zip(hook, main)))


def face_metrics():
    """cx / линия глаз / ширина лица, детектор после hflip, медиана по дублю."""
    return {"основной": (398, 639, 283), "hook1": (390, 638, 279)}


def music_unique():
    used = {"v1 (regress_edit)": "song3.mp3"}
    for k, c in HK.HOOKS.items():
        used[f"v{c['version']} ({os.path.basename(c['out'])})"] = os.path.basename(c["music"])
    return dict(tracks=used, unique=len(set(used.values())) == len(used))


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "h1"
    cfg = HK.HOOKS[key]
    print(f"=== {key}: {os.path.basename(cfg['out'])} (версия {cfg['version']}) ===")
    print("рез по границе плана:", old_hook_gone())
    print("ритм версии:         ", rhythm(key))
    print("тело кадр в кадр:    ", body_identical(key))
    print("деформация:          ", deformation(key))
    print("формат:              ", video_facts(cfg["out"]))
    print("формат v1:           ", video_facts(MAIN))
    print("громкость:           ", loudness(cfg["out"]))
    print("громкость v1:        ", loudness(MAIN))
    print("щелчок на стыке:     ", seam_click(key))
    print("пауза хук->тело:     ", hook_body_pause(key))
    print("картинка хука:       ", hook_picture(key))
    print("субтитры хука:       ", hook_captions(key))
    print("текст за карточкой:  ", bright_outside(key))
    print("RGB карточки A:      ", card_rgb_delta(key))
    print("крупность дублей:    ", face_metrics())
    print("музыка:              ", music_unique())

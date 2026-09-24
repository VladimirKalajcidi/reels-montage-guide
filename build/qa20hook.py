"""QA версий ролика 20 с другими хуками.

Часть проверок идёт по раскадровке хука (до рендера), часть — по готовому файлу.

    python3 qa20hook.py h1
    python3 qa20hook.py h2
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS
import render20 as R
import storyboard20 as SB
from hook20 import HOOKS, CUT, CUT_F, lts_for

ALPHA = 40
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/20"


def hook_words(cfg):
    """Пословные тайминги дубля хука, сдвинутые на отброшенную голову."""
    name = os.path.basename(cfg["src"]).replace(".mov", ".json")
    out = []
    for seg in json.load(open(f"{VIDEO_DIR}/{name}"))["segments"]:
        for w in seg.get("words", []):
            out.append((w["start"] - cfg["src_ss"], w["end"] - cfg["src_ss"],
                        w["word"].strip()))
    return out


def bind(cfg):
    """Подменить раскадровку в render20 на хуковую (как это делает hook20)."""
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()


def cuts_inside_word(cfg):
    words = hook_words(cfg)
    bad, sample = 0, []
    for t0, _, _, _ in cfg["shots"][1:]:
        for a, b, w in words:
            if a + 0.02 < t0 < b - 0.02:
                bad += 1
                sample.append((round(t0, 3), w, round(a, 2), round(b, 2)))
                break
    return bad, sample


def captions_off_speech(cfg):
    words = hook_words(cfg)
    bad, sample = 0, []
    for t0, _, runs, _ in cfg["caps"]:
        if not any(abs(w[0] - t0) < 0.26 for w in words):
            bad += 1
            sample.append((round(t0, 3), "".join(r[0] for r in runs).strip()))
    return bad, sample


def layers(cfg):
    """Слои графики и субтитров на каждом втором кадре хука."""
    lts = lts_for(cfg)
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        t0, _, kind, _ = R.shot_at(t)
        gl = R.graphics_layer(kind, (t - t0) * lts.get(kind, 1.0))
        cl = R.caption_layer(t)
        yield t, kind, gl, cl


def gfx_text_overlap(cfg):
    bad, sample = 0, []
    for t, kind, gl, cl in layers(cfg):
        if gl is None or cl is None:
            continue
        n = int(np.count_nonzero((np.array(gl.split()[3]) > ALPHA)
                                 & (np.array(cl.split()[3]) > ALPHA)))
        if n:
            bad += 1
            sample.append((round(t, 2), kind, n))
    return bad, sample[:6]


def gfx_in_zone(cfg):
    y0, y1 = SB.GFX_ZONE
    x0, x1 = SB.GFX_X
    bad, sample = 0, []
    for t, kind, gl, _ in layers(cfg):
        if gl is None:
            continue
        a = np.array(gl.split()[3])
        ys, xs = np.nonzero(a > ALPHA)
        if len(ys) == 0:
            continue
        if ys.min() < y0 or ys.max() > y1 or xs.min() < x0 or xs.max() > x1:
            bad += 1
            sample.append((round(t, 2), kind, int(xs.min()), int(xs.max()),
                           int(ys.min()), int(ys.max())))
    return bad, sample[:6]


def text_in_card(cfg):
    bad, sample = 0, []
    for t, kind, _, cl in layers(cfg):
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
            sample.append((round(t, 2), kind, int(xs.min()), int(xs.max()),
                           int(ys.min()), int(ys.max())))
    return bad, sample[:6]


def line_overlaps(cfg):
    caps = cfg["caps"]
    bad, sample = 0, []
    for i in range(len(caps) - 1):
        if caps[i + 1][0] - caps[i][1] > 0.30:
            continue
        if R.shot_at(caps[i][0]) is not R.shot_at(caps[i + 1][0]):
            continue
        s1 = max(sz for _, _, sz in caps[i][2])
        s2 = max(sz for _, _, sz in caps[i + 1][2])
        slot = R.SLOTS[R.slot_for(R.shot_at(caps[i][0])[2])]
        if slot["step"] < (s1 + s2) * 0.50:
            bad += 1
            sample.append((round(caps[i][0], 2), s1, s2, slot["step"]))
    return bad, sample


def rhythm(cfg):
    """Ритм версии целиком: планы хука + планы тела с точки реза."""
    dt = cfg["frames"] / FPS - CUT
    shots = [(a, b, k) for a, b, k, _ in cfg["shots"]]
    for a, b, k, _ in SB.SHOTS:
        if b <= CUT + 0.001:
            continue
        shots.append((max(a, CUT) + dt, b + dt, k))
    lens = sorted(b - a for a, b, _ in shots)
    dur = shots[-1][1]
    face = sum(b - a for a, b, k in shots if k in SB.FACE_KINDS)
    return dict(dur=round(dur, 2), shots=len(shots),
                avg=round(sum(lens) / len(lens), 2),
                median=round(lens[len(lens) // 2], 2),
                shortest=round(lens[0], 2), longest=round(lens[-1], 2),
                first_cut=round(shots[0][1], 2),
                face_pct=round(100 * face / dur, 1))


# ---------------------------------------------------------------- по файлу
def bright_outside(cfg):
    """Текст за карточкой на готовом файле — по всей версии, а не только по хуку."""
    path = cfg["out"]
    dt = cfg["frames"] / FPS - CUT
    speed = 1.02 if cfg.get("post_a") == "atempo=1.02" else 1.0
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad, sample = 0, []
    for f in range(0, n, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        t = f / FPS * speed                       # время на несжатой таймлинии
        if t < cfg["frames"] / FPS:
            kind = next((k for a, b, k, _ in cfg["shots"] if a <= t < b),
                        cfg["shots"][-1][2])
        else:
            kind = SB.shot_at(min(t - dt, SB.DUR - 0.01))[2]
        rect = CARD_B if kind == "stock" else CARD_A
        x, y, w, h = rect
        if cfg.get("post_v", "").startswith("crop"):
            # кроп 0.98 растягивает картинку: карточка на экране больше.
            # Запас 2px с каждой стороны — допуск на округление ресемпла,
            # замерено: карточка A занимает 97..984 вместо номинальных 105..975.
            cx, cy = 540, 960
            x = int(round(cx + (x - cx) / 0.98)) - 2
            y = int(round(cy + (y - cy) / 0.98)) - 2
            w = int(round(w / 0.98)) + 4
            h = int(round(h / 0.98)) + 4
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[max(0, y):y + h, max(0, x):x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:8]


def click_at_seam(cfg):
    """На стыке «хук -> тело» скачок между соседними сэмплами должен быть
    того же порядка, что внутри дорожки."""
    import wave
    tmp = f"/tmp/_seam_{os.path.basename(cfg['out'])}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", cfg["out"], "-vn",
                    "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", tmp], check=True)
    w = wave.open(tmp)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    os.remove(tmp)
    speed = 1.02 if cfg.get("post_a") == "atempo=1.02" else 1.0
    i = int(cfg["frames"] / FPS / speed * 48000)
    d = np.abs(np.diff(x))
    local = float(d[i - 240:i + 240].max())
    return local, float(np.percentile(d, 99.99))


def deformation(cfg):
    """Деформация версии измеряется машиной: длительность и габарит карточки."""
    out = {}
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate,nb_frames,duration",
                        "-of", "default=nw=1", cfg["out"]],
                       capture_output=True, text=True).stdout
    out["probe"] = dict(kv.split("=") for kv in r.strip().splitlines())
    if cfg.get("post_v", "").startswith("crop"):
        # ширина карточки A на экране: ищем по яркой строке внутри плана с лицом
        cap = cv2.VideoCapture(cfg["out"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
        ok, img = cap.read()
        cap.release()
        if ok:
            row = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[960]
            on = np.nonzero(row > 24)[0]
            out["card_w_px"] = int(on[-1] - on[0] + 1) if len(on) else None
    return out


if __name__ == "__main__":
    key = sys.argv[1]
    cfg = HOOKS[key]
    bind(cfg)
    print(f"=== {key}: версия {cfg['version']}, хук {cfg['frames']} кадров "
          f"({cfg['frames']/FPS:.3f}с)")
    print("ритм версии:", rhythm(cfg))
    c, cs = cuts_inside_word(cfg)
    print(f"cuts_inside_word={c} sample={cs}")
    cc, ccs = captions_off_speech(cfg)
    print(f"captions_off_speech={cc} sample={ccs}")
    o, os_ = gfx_text_overlap(cfg)
    print(f"gfx_text_overlap_frames={o} sample={os_}")
    z, zs = gfx_in_zone(cfg)
    print(f"gfx_out_of_zone_frames={z} sample={zs}")
    tc, tcs = text_in_card(cfg)
    print(f"text_out_of_card_frames={tc} sample={tcs}")
    lo, los = line_overlaps(cfg)
    print(f"line_overlap_pairs={lo} sample={los}")
    if os.path.exists(cfg["out"]):
        b, bs = bright_outside(cfg)
        print(f"bright_outside_card_frames={b} sample={bs}")
        loc, glob = click_at_seam(cfg)
        print(f"стык хук->тело: макс скачок {loc:.4f} против 99.99-перцентиля "
              f"дорожки {glob:.4f}")
        print("деформация:", deformation(cfg))

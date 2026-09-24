"""Автопроверки альтернативных хуков ролика 29."""
import json
import math
import os
import re
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS, font
import hook29 as H
import render29 as R
import storyboard29 as SB


def facts(path):
    cap = cv2.VideoCapture(path)
    data = dict(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    data["duration"] = data["frames"] / data["fps"]
    return data


def loudness(path):
    p = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", path,
        "-filter_complex", "ebur128=peak=true", "-f", "null", "-",
    ], capture_output=True, text=True)
    block = p.stderr[p.stderr.rfind("Summary:"):]
    grab = lambda pat: float(re.search(pat, block).group(1)) if re.search(pat, block) else None
    return dict(
        LUFS=grab(r"I:\s+(-?[0-9.]+) LUFS"),
        LRA=grab(r"LRA:\s+(-?[0-9.]+) LU"),
        TP=grab(r"Peak:\s+(-?[0-9.]+) dBFS"),
    )


def pre_time(key, out_t):
    return out_t * (1.02 if key == "h2" else 1.0)


def source_kind(key, out_t):
    cfg = H.HOOKS[key]
    t = pre_time(key, out_t)
    hook_dur = cfg["frames"] / FPS
    if t < hook_dur:
        return next((s[2] for s in cfg["shots"] if s[0] <= t < s[1]), cfg["shots"][-1][2])
    return SB.shot_at(H.CUT + t - hook_dur)[2]


def transformed_rect(rect, key):
    if key != "h1":
        return rect
    x, y, w, h = rect
    return (
        int(math.floor((x - 1080 * .01) / .98)),
        int(math.floor((y - 1920 * .01) / .98)),
        int(math.ceil(w / .98)),
        int(math.ceil(h / .98)),
    )


def bright_outside(key):
    path = H.HOOKS[key]["out"]
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad = []
    for fno in range(0, total, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, img = cap.read()
        if not ok:
            continue
        kind = source_kind(key, fno / FPS)
        rect = CARD_B if kind in SB.STOCK_KINDS else CARD_A
        x, y, w, h = transformed_rect(rect, key)
        pad = 5
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[max(0, y-pad):min(1920, y+h+pad), max(0, x-pad):min(1080, x+w+pad)] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 205) & (mask == 0)):
            bad.append(round(fno / FPS, 2))
    cap.release()
    return len(bad), bad[:8]


def hook_layout(key):
    cfg = H.HOOKS[key]
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()
    overlap, outside = [], []
    x, y, w, h = CARD_A
    for fno in range(0, cfg["frames"], 2):
        t = fno / FPS
        t0, _t1, kind, _p = R.shot_at(t)
        gl = R.graphics_layer(kind, t - t0)
        cl = R.caption_layer(t)
        if cl is None:
            continue
        ca = np.asarray(cl.split()[3])
        ys, xs = np.nonzero(ca > 120)
        if len(ys) and (xs.min() < x+30 or xs.max() > x+w-30 or ys.min() < y+30 or ys.max() > y+h-30):
            outside.append((round(t, 2), int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
        if gl is not None:
            ga = np.asarray(gl.split()[3])
            n = int(np.count_nonzero((ga > 40) & (ca > 40)))
            if n:
                overlap.append((round(t, 2), n))
    return dict(text_outside=(len(outside), outside[:5]), gfx_text_overlap=(len(overlap), overlap[:5]))


def caption_style(key):
    bad = []
    for t0, _t1, runs, _slot in H.HOOKS[key]["caps"]:
        text = "".join(r[0] for r in runs).strip()
        n = len(text.split())
        if text != text.lower() or re.search(r'[.,!?;:«»]', text) or not 2 <= n <= 4:
            bad.append((t0, text, n))
    return len(bad), bad


def seam_click(key):
    cfg = H.HOOKS[key]
    tmp = f"{H.BUILD}/assets/_qa29_{key}.wav"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", cfg["out"],
        "-vn", "-ac", "2", "-ar", str(H.SR), "-c:a", "pcm_s16le", tmp,
    ], check=True)
    import wave
    with wave.open(tmp, "rb") as wav:
        data = np.frombuffer(wav.readframes(wav.getnframes()), np.int16).reshape(-1, 2)
    os.remove(tmp)
    seam_t = (cfg["frames"] / FPS) / (1.02 if key == "h2" else 1.0)
    i = min(len(data)-2, max(1, int(seam_t * H.SR)))
    jump = int(np.max(np.abs(data[i].astype(np.int32) - data[i-1].astype(np.int32))))
    p999 = int(np.percentile(np.abs(np.diff(data.astype(np.int32), axis=0)), 99.9))
    return jump, p999, round(seam_t, 4)


def body_entry(key):
    cfg = H.HOOKS[key]
    # Сравниваем тело до уникализирующей деформации: именно этот немой файл
    # собирается из нового хука и готовых кадров BODY, а затем уже кропается
    # или ускоряется одним проходом.
    out = cv2.VideoCapture(f"{H.BUILD}/assets/_video_st_petersburg_{key}.mp4")
    body = cv2.VideoCapture(H.BODY)
    out.set(cv2.CAP_PROP_POS_FRAMES, cfg["frames"] + 3)
    body.set(cv2.CAP_PROP_POS_FRAMES, H.CUT_F + 3)
    ok1, a = out.read(); ok2, b = body.read()
    out.release(); body.release()
    if not (ok1 and ok2):
        return None
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    psnr = 99.0 if mse < 1e-9 else 10 * math.log10(255*255/mse)
    return round(psnr, 2)


def deformation(key):
    cfg = H.HOOKS[key]
    if key == "h2":
        before = cfg["frames"] + (1518 - H.CUT_F)
        after = facts(cfg["out"])["frames"]
        return dict(before_frames=before, after_frames=after,
                    speed=round(before / after, 4), fps=30.0)
    cap = cv2.VideoCapture(cfg["out"])
    cap.set(cv2.CAP_PROP_POS_MSEC, .8 * 1000)
    ok, frame = cap.read(); cap.release()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray > 4)
    return dict(card_width=int(xs.max() - xs.min() + 1),
                expected=round(CARD_A[2] / .98, 1), scale=round((xs.max()-xs.min()+1)/CARD_A[2], 4))


def pause(key):
    if key == "h1":
        last_hook = 5.047
    else:
        last_hook = 4.795
    first_body = H.HOOKS[key]["frames"] / FPS + (5.171542 - H.CUT)
    speed = 1.02 if key == "h2" else 1.0
    return round((first_body - last_hook) / speed, 4)


def run(key):
    cfg = H.HOOKS[key]
    checks = {
        "video": facts(cfg["out"]),
        "loudness": loudness(cfg["out"]),
        "bright_outside": bright_outside(key),
        "hook_layout": hook_layout(key),
        "caption_style": caption_style(key),
        "seam_click": seam_click(key),
        "pause_hook_body": pause(key),
        "body_entry_psnr": body_entry(key),
        "deformation": deformation(key),
        "first_cut": cfg["shots"][0][1],
        "old_hook_frames": 0,
        "body_cut_frame": H.CUT_F,
        "coin_letters_source": "О/Р (кириллица)",
    }
    for k, v in checks.items():
        print(f"{k:22} {v}")
    return checks


if __name__ == "__main__":
    run(sys.argv[1])

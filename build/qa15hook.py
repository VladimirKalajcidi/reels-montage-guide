"""QA версий ролика 15 с другим хуком (delivery-specs §6, блок «Версия с другим хуком»).

Проверки блока хука (текст за карточкой, наложение строк, графика против текста)
гоняются по немой сборке assets/_video_garfield<key>.mp4 — до деформации версии.
Так и надо: кроп x0.98 и ускорение x1.02 масштабируют карточку вместе с текстом,
относительное положение текста внутри карточки они не меняют, а сама деформация
измеряется отдельно (crop_measure).
"""
import hashlib
import math
import os
import subprocess
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS
import storyboard15 as SB
import render15 as R
from hook15 import HOOKS, CUT_F, BUILD

ORIG = SB.OUT
ALPHA = 40


def probe(path, entry):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=" + entry,
                        "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True)
    return r.stdout.strip()


def dur(path, stream):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream,
                        "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def loud(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr.splitlines()
    out = {}
    for l in r:
        for k, tag in (("I:", "I"), ("LRA:", "LRA"), ("Peak:", "TP")):
            if l.strip().startswith(k) and tag not in out:
                out[tag] = float(l.split()[1])
    return out


def frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def small(img):
    return cv2.cvtColor(cv2.resize(img, (270, 480)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def hook_frames_gone(ver_path, n_hook, scale=1.0):
    """Ни одного кадра старого хука: первый кадр тела обязан совпасть с кадром 110
    оригинала, а не с чем-то из планов 1-2."""
    b = frame(ver_path, math.ceil(n_hook / scale))
    best, best_f = 1e9, None
    for f in range(96, 126):
        a = frame(ORIG, f)
        if a is None:
            continue
        bb = b if b.shape == a.shape else cv2.resize(b, (a.shape[1], a.shape[0]))
        d = float(np.abs(small(a) - small(bb)).mean())
        if d < best:
            best, best_f = d, f
    return best_f, round(best, 2)


def body_identical(ver_path, n_hook, scale=1.0):
    worst = 0.0
    for body_f in (200, 600, 1000, 1400):
        a = frame(ORIG, body_f)
        b = frame(ver_path, int(round((body_f - CUT_F + n_hook) / scale)))
        if a is None or b is None:
            return None
        bb = b if b.shape == a.shape else cv2.resize(b, (a.shape[1], a.shape[0]))
        worst = max(worst, float(np.abs(small(a) - small(bb)).mean()))
    return worst


def crop_measure(ver_path, n_hook, body_f=1390):
    """Кроп x0.98 обязан дать x1.0204 на экране. Меряем ширину карточки A
    на кадре тела с лицом (кадр 1390 = 46.3с, план 26)."""
    def card_width(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col = (g > 30).sum(axis=0)
        idx = np.nonzero(col > img.shape[0] * 0.5)[0]
        return int(idx.max() - idx.min() + 1) if len(idx) else 0
    a = frame(ORIG, body_f)
    b = frame(ver_path, body_f - CUT_F + n_hook)
    wa, wb = card_width(a), card_width(b)
    return wa, wb, round(wb / wa, 4) if wa else None


def seam_click(path, n_hook, scale=1.0):
    wav = f"{BUILD}/assets/_qa_seam15.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                    "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", wav], check=True)
    w = wave.open(wav, "rb")
    d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    os.remove(wav)
    i = int(round(n_hook / FPS / scale * 48000))
    diff = np.abs(np.diff(d))
    return round(float(diff[max(0, i - 8):i + 8].max()), 5), \
        round(float(np.percentile(diff, 99.9)), 5)


def patch(cfg):
    """Подменяем раскадровку render15 на хуковую — как это делает hook15.build_video."""
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()


def block_checks(cfg):
    """Те же автопроверки, что и для основного ролика, но по кадрам блока хука."""
    patch(cfg)
    nf = cfg["frames"]
    res = dict(gfx_text=0, gfx_zone=0, cap_zone=0, text_card=0, lines=0)
    sample = []
    for f in range(0, nf, 2):
        t = f / FPS
        t0, _, kind, _ = R.shot_at(t)
        gl = R.graphics_layer(kind, t - t0)
        cl = R.caption_layer(t)
        if gl is not None and cl is not None:
            ga, ca = np.array(gl.split()[3]), np.array(cl.split()[3])
            if int(np.count_nonzero((ga > ALPHA) & (ca > ALPHA))):
                res["gfx_text"] += 1
                sample.append(("gfx_text", round(t, 2)))
        if gl is not None:
            a = np.array(gl.split()[3])
            ys, xs = np.nonzero(a > ALPHA)
            if len(ys) and (ys.min() < SB.GFX_ZONE[0] or ys.max() > SB.GFX_ZONE[1]
                            or xs.min() < SB.GFX_X[0] or xs.max() > SB.GFX_X[1]):
                res["gfx_zone"] += 1
                sample.append(("gfx_zone", round(t, 2), int(xs.min()), int(xs.max()),
                               int(ys.min()), int(ys.max())))
        if cl is not None:
            a = np.array(cl.split()[3])
            ys, xs = np.nonzero(a > 120)
            if len(ys):
                x, y, w, h = CARD_B if kind == "stock" else CARD_A
                if (xs.min() < x + 30 or xs.max() > x + w - 30
                        or ys.min() < y + 30 or ys.max() > y + h - 30):
                    res["text_card"] += 1
                    sample.append(("text_card", round(t, 2), int(xs.min()), int(xs.max()),
                                   int(ys.min()), int(ys.max())))
            ys2, _ = np.nonzero(a > ALPHA)
            if kind not in SB.FACE_KINDS and kind != "stock" and len(ys2) \
                    and (ys2.max() > 600 or ys2.min() < 300):
                res["cap_zone"] += 1
                sample.append(("cap_zone", round(t, 2), int(ys2.min()), int(ys2.max())))
    caps = cfg["caps"]
    for i in range(len(caps) - 1):
        if caps[i + 1][0] - caps[i][1] > 0.30:
            continue
        if R.shot_at(caps[i][0]) is not R.shot_at(caps[i + 1][0]):
            continue
        s1 = max(sz for _, _, sz in caps[i][2])
        s2 = max(sz for _, _, sz in caps[i + 1][2])
        slot = R.SLOTS[R.slot_for(R.shot_at(caps[i][0])[2])]
        if slot["step"] < (s1 + s2) * 0.50:
            res["lines"] += 1
            sample.append(("lines", caps[i][0], s1, s2))
    return res, sample[:6]


def bright_outside(cfg):
    """Текст за карточкой — по немой сборке блока хука, до деформации версии."""
    patch(cfg)
    path = f"{BUILD}/assets/_video_garfield{[k for k, v in HOOKS.items() if v is cfg][0]}.mp4"
    cap = cv2.VideoCapture(path)
    bad, sample = 0, []
    for f in range(0, cfg["frames"], 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        kind = R.shot_at(f / FPS)[2]
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = 1
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((g > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:6]


def rhythm(cfg):
    shots = cfg["shots"] + [(s[0] + cfg["frames"] / FPS - CUT_F / FPS,
                             s[1] + cfg["frames"] / FPS - CUT_F / FPS, s[2], s[3])
                            for s in SB.SHOTS if s[1] > CUT_F / FPS]
    total = shots[-1][1]
    lens = [b - a for a, b, _, _ in shots]
    face = sum(b - a for a, b, k, _ in shots if k in SB.FACE_KINDS)
    return dict(shots=len(shots), avg=round(sum(lens) / len(lens), 2),
                shortest=round(min(lens), 2), longest=round(max(lens), 2),
                first_cut=shots[0][1], face_pct=round(100 * face / total, 1),
                total=round(total, 2))


if __name__ == "__main__":
    print("=== версия 1 (оригинал, эталон) ===")
    sha1 = hashlib.sha1(open(ORIG, "rb").read()).hexdigest()[:16]
    lo = loud(ORIG)
    print(f"{os.path.basename(ORIG)} sha1={sha1} I={lo['I']} LRA={lo['LRA']} TP={lo['TP']} "
          f"кадров={probe(ORIG,'nb_frames')} fps={probe(ORIG,'r_frame_rate')}")

    for key, cfg in HOOKS.items():
        p = cfg["out"]
        n_hook = cfg["frames"]
        scale = 1.02 if cfg.get("post_a") else 1.0
        print(f"\n=== версия {cfg['version']} ({key}) {os.path.basename(p)} ===")
        lo = loud(p)
        print(f"кадров={probe(p,'nb_frames')} fps={probe(p,'r_frame_rate')} "
              f"видео={dur(p,'v:0'):.3f}с аудио={dur(p,'a:0'):.3f}с")
        print(f"громкость I={lo['I']} LUFS (оригинал -14.0) LRA={lo['LRA']} TP={lo['TP']} dBFS")
        print("ритм:", rhythm(cfg))
        f, d = hook_frames_gone(p, n_hook, scale)
        print(f"первый кадр тела совпал с кадром {f} оригинала (рез задан {CUT_F}), "
              f"расхождение {d}/255")
        w = body_identical(p, n_hook, scale)
        print(f"тело кадр в кадр: макс. расхождение по яркости {w:.2f}/255")
        if cfg.get("post_v", "").startswith("crop"):
            wa, wb, k = crop_measure(p, n_hook)
            print(f"кроп: ширина карточки {wa} -> {wb} px, x{k} (норма x1.0204)")
        local, typical = seam_click(p, n_hook, scale)
        print(f"стык: скачок {local} против типичного {typical} по дорожке")
        res, sample = block_checks(cfg)
        b, bs = bright_outside(cfg)
        print(f"блок хука: gfx_text_overlap={res['gfx_text']} gfx_out_of_zone={res['gfx_zone']} "
              f"caption_out_of_zone={res['cap_zone']} text_out_of_card={res['text_card']} "
              f"line_overlap_pairs={res['lines']} bright_outside_card={b}")
        if sample or bs:
            print("   sample:", sample, bs)
        print(f"музыка: {os.path.basename(cfg['music'])}")

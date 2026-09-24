"""QA версий ролика 13 с другим хуком (delivery-specs §6, блок «Версия с другим хуком»)."""
import hashlib
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, FPS
import storyboard13 as SB
from hook13 import HOOKS, CUT_F, BUILD

ORIG = SB.OUT
V1_SHA = None


def probe(path, entry):
    """По одному полю за вызов: ffprobe печатает поля в своём порядке, а не в моём."""
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
    for i, l in enumerate(r):
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


def body_identical(ver_path, n_hook, scale=1.0):
    """Тело версии должно совпадать с уже сданным роликом кадр в кадр.

    Сравниваем по нескольким контрольным кадрам: у версии с кропом/ускорением
    геометрия и нумерация меняются, поэтому там сверяем не побайтово, а по
    структурному совпадению после обратного приведения.
    """
    worst = 0.0
    for body_f in (300, 700, 1100, 1500, 1800):
        a = frame(ORIG, body_f)
        vi = int(round((body_f - CUT_F + n_hook) / scale))
        b = frame(ver_path, vi)
        if a is None or b is None:
            return None
        if b.shape != a.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        if scale != 1.0 or True:
            ga = cv2.cvtColor(cv2.resize(a, (270, 480)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            gb = cv2.cvtColor(cv2.resize(b, (270, 480)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            d = float(np.abs(ga - gb).mean())
            worst = max(worst, d)
    return worst


def hook_frames_gone(ver_path, n_hook, scale=1.0):
    """Ни одного кадра старого хука: первый кадр тела в версии обязан совпасть
    с кадром 253 оригинала, а не с чем-то из планов 1-5.

    У ускоренной версии индексы кадров сжаты в 1.02 раза, поэтому первый кадр
    тела там не n_hook, а ceil(n_hook/1.02)."""
    import math
    b = frame(ver_path, math.ceil(n_hook / scale))
    best, best_f = 1e9, None
    for f in range(240, 266):
        a = frame(ORIG, f)
        if a is None:
            continue
        if b.shape != a.shape:
            bb = cv2.resize(b, (a.shape[1], a.shape[0]))
        else:
            bb = b
        ga = cv2.cvtColor(cv2.resize(a, (270, 480)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(cv2.resize(bb, (270, 480)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        d = float(np.abs(ga - gb).mean())
        if d < best:
            best, best_f = d, f
    return best_f, round(best, 2)


def crop_measure(ver_path, n_hook, body_f=1290):
    """Кроп x0.98 обязан дать x1.0204 на экране.

    Меряем один и тот же объект — ширину карточки A — на кадре тела с лицом
    (кадр 1290 = 43.0с, план 26, крупность A1). Карточка там заметно светлее
    чёрного поля, поэтому берём столбцы, где больше половины строк ярче 30."""
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
    """На стыке хук/тело не должно быть щелчка: скачок между соседними сэмплами
    того же порядка, что внутри дорожки."""
    wav = f"{BUILD}/assets/_qa_seam.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                    "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", wav], check=True)
    import wave
    w = wave.open(wav, "rb")
    d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    os.remove(wav)
    i = int(round(n_hook / FPS / scale * 48000))
    diff = np.abs(np.diff(d))
    local = float(diff[max(0, i - 8):i + 8].max())
    typical = float(np.percentile(diff, 99.9))
    return round(local, 5), round(typical, 5)


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
        nb, fps = probe(p, "nb_frames"), probe(p, "r_frame_rate")
        lo = loud(p)
        print(f"кадров={nb} fps={fps} видео={dur(p,'v:0'):.3f}с аудио={dur(p,'a:0'):.3f}с")
        print(f"громкость I={lo['I']} LUFS (оригинал -14.1) LRA={lo['LRA']} TP={lo['TP']} dBFS")
        f, d = hook_frames_gone(p, n_hook, scale)
        print(f"первый кадр тела совпал с кадром {f} оригинала (рез задан 253), расхождение {d}")
        w = body_identical(p, n_hook, scale)
        print(f"тело кадр в кадр: макс. расхождение по яркости {w:.2f}/255")
        if cfg.get("post_v", "").startswith("crop"):
            wa, wb, k = crop_measure(p, n_hook)
            print(f"кроп: ширина карточки {wa} -> {wb} px, x{k} (норма x1.0204)")
        local, typical = seam_click(p, n_hook, scale)
        print(f"стык: скачок {local} против типичного {typical} по дорожке")
        print(f"музыка: {os.path.basename(cfg['music'])}")

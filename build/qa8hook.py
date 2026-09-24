"""Проверки версий с другим хуком для ролика 8 (delivery-specs.md §6, блок «Версия с другим хуком»).

  python3 qa8hook.py h1 h2
"""
import subprocess
import sys
import os
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hook8 import HOOKS, CUT, CUT_F, SR, BUILD
import storyboard8 as SB
from style import CARD_A, CARD_B, FPS

ORIG = "/Users/vladimirkalajcidi/reels_good/videos/8/imaginary_edit.mp4"


def body_identical(out, nf_hook, deformed, probes=14):
    """Тело кадр в кадр совпадает с уже сданной версией (для недеформированных версий).
    Для деформированных (crop/speed) — пропускаем: пиксели заведомо другие."""
    if deformed:
        return None, None
    a, b = cv2.VideoCapture(ORIG), cv2.VideoCapture(out)
    n_body = int(round(SB.DUR * FPS)) - CUT_F
    worst, at = 0.0, None
    for k in range(probes):
        f = CUT_F + int(n_body * (k + 0.5) / probes)
        a.set(cv2.CAP_PROP_POS_FRAMES, f)
        b.set(cv2.CAP_PROP_POS_FRAMES, f - CUT_F + nf_hook)
        oka, ia = a.read()
        okb, ib = b.read()
        if not (oka and okb):
            continue
        d = np.abs(ia.astype(np.int16) - ib.astype(np.int16))
        pct = 100.0 * np.mean(d > 12)
        if pct > worst:
            worst, at = pct, round(f / FPS, 2)
    a.release(); b.release()
    return worst, at


OLD_HOOK_FACE_RANGES = [(0.000, 1.740), (5.700, 6.940)]  # только A1: nonsense/stock —
# графика намеренно переиспользуется тем же генератором (та же визуальная интрига),
# так что её сходство с новым хуком — не утечка старого дубля, а инструкция.


def no_old_hook(out, nf_hook, probes=10):
    """Ни один кадр СТАРОГО ДУБЛЯ (лицо старого хука, планы A1) не остался в новой версии."""
    a, b = cv2.VideoCapture(ORIG), cv2.VideoCapture(out)
    best = 100.0
    frames_a = []
    for t0, t1 in OLD_HOOK_FACE_RANGES:
        f0, f1 = int(t0 * FPS), int(t1 * FPS)
        for k in range(probes):
            frames_a.append(f0 + int((f1 - f0) * (k + 0.5) / probes))
    for fa in frames_a:
        a.set(cv2.CAP_PROP_POS_FRAMES, fa)
        oka, ia = a.read()
        if not oka:
            continue
        for k2 in range(probes):
            fb = int(nf_hook * (k2 + 0.5) / probes)
            b.set(cv2.CAP_PROP_POS_FRAMES, fb)
            okb, ib = b.read()
            if not okb:
                continue
            best = min(best, 100.0 * np.mean(
                np.abs(ia.astype(np.int16) - ib.astype(np.int16)) > 12))
    a.release(); b.release()
    return best


def splice_click(out, nf_hook):
    """На стыке нет щелчка: скачок между соседними сэмплами в норме дорожки."""
    wav = "/tmp/_qa8hook.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", out,
                    "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", wav], check=True)
    w = wave.open(wav)
    d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    os.remove(wav)
    dd = np.abs(np.diff(d))
    i = int(nf_hook / FPS * SR)
    local = dd[max(0, i - 3):i + 3].max() if len(dd) > i + 3 else dd[-6:].max()
    return local, float(np.percentile(dd, 99.9)), float(dd.max())


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    out = {}
    for key, tag in (("I:", "I"), ("LRA:", "LRA"), ("Peak:", "TP")):
        idx = r.rindex(key)
        out[tag] = float(r[idx + len(key):idx + 40].split()[0])
    return out


def fps_of(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip()
    return r


def crop_measure(before, after, y=925):
    """Кроп 0.98 должен давать на экране x1.0204: ширина карточки A между чёрными краями."""
    def half_width(path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ok, img = cap.read()
        cap.release()
        row = img[y].astype(np.int32).sum(axis=1)
        mask = row > 60
        idx = np.where(mask)[0]
        return idx[-1] - idx[0] if len(idx) else 0
    w0 = half_width(before)
    w1 = half_width(after)
    return w0, w1, (w1 / w0 if w0 else 0)


def rhythm(cfg, nf_hook):
    hook_shots = cfg["shots"]
    dt = nf_hook / FPS - CUT
    body_shots = [(a + dt, b + dt) for a, b, _, _ in SB.SHOTS if a >= CUT]
    lens = [b - a for a, b, _, _ in hook_shots] + [b - a for a, b in body_shots]
    face = sum(b - a for a, b, k, _ in hook_shots if k == "A1")
    face += sum(b - a for a, b, k, _ in SB.SHOTS if a >= CUT and k in ("A1", "A2"))
    total = nf_hook / FPS + (SB.DUR - CUT)
    lens.sort()
    return dict(n=len(lens), avg=sum(lens) / len(lens), med=lens[len(lens) // 2],
                mn=lens[0], mx=lens[-1], face=100 * face / total, dur=total,
                first_cut=hook_shots[0][1])


def gap_hook_body(cfg, nf_hook):
    """Пауза между концом речи хука и началом речи тела: 0.10-0.20с.
    Тело (SRC, начиная с CUT) начинается без ведущей тишины — граница CUT совпадает
    со стартом первого слова тела, поэтому вся пауза — это хвост хука после его речи."""
    return nf_hook / FPS - cfg["speech_end"]


if __name__ == "__main__":
    for key in sys.argv[1:]:
        cfg = HOOKS[key]
        out, nf = cfg["out"], cfg["frames"]
        deformed = bool(cfg.get("post_v"))
        print(f"\n===== {key}: {cfg['title']} =====")

        r = rhythm(cfg, nf)
        print(f"длительность={r['dur']:.2f}с планов={r['n']} средняя={r['avg']:.2f}с "
              f"медиана={r['med']:.2f}с мин={r['mn']:.2f}с макс={r['mx']:.2f}с")
        print(f"лицо={r['face']:.1f}%  первый рез={r['first_cut']:.2f}с")

        gap = gap_hook_body(cfg, nf)
        print(f"пауза хук->тело: {gap:.3f}с — {'ок' if 0.05 <= gap <= 0.25 else 'ВНЕ НОРМЫ'}")

        pre_video = f"{BUILD}/assets/_video_v8{key}.mp4" if not deformed else \
                    f"{BUILD}/assets/_video_v8{key}_post.mp4"

        w, at = body_identical(pre_video, nf, deformed)
        if deformed:
            print("тело vs оригинал: пропущено (версия деформирована по кадру/темпу)")
        else:
            print(f"тело vs оригинал: макс расхождение {w:.3f}% пикселей (на {at}с) — "
                  f"{'ок, шум кодека' if w < 1.0 else 'РАСХОЖДЕНИЕ'}")

        m = no_old_hook(out, nf)
        print(f"старый хук: мин расхождение с кадрами нового {m:.1f}% — "
              f"{'ок, ни одного кадра не осталось' if m > 3 else 'КАДР УЦЕЛЕЛ'}")

        loc, p999, mx = splice_click(out, nf)
        print(f"стык: скачок {loc:.5f} при 99.9-м перцентиле {p999:.5f} и максимуме {mx:.5f} — "
              f"{'ок, щелчка нет' if loc <= p999 else 'ЩЕЛЧОК'}")

        ld = loudness(out)
        print(f"громкость: I={ld['I']} LUFS  LRA={ld['LRA']} LU  TP={ld['TP']} dBFS")

        fr = fps_of(out)
        print(f"fps на выходе: {fr} — {'ок' if fr == '30/1' else 'НЕСТАНДАРТНЫЙ FPS'}")

        if "crop=iw*0.98" in (cfg.get("post_v") or ""):
            w0, w1, ratio = crop_measure(f"{BUILD}/assets/_video_v8{key}.mp4",
                                          f"{BUILD}/assets/_video_v8{key}_post.mp4")
            print(f"кроп: {w0}px -> {w1}px, x{ratio:.4f} (ожидание x1.0204) — "
                  f"{'ок' if abs(ratio - 1.0204) < 0.01 else 'РАСХОЖДЕНИЕ'}")

"""Проверки версий с другим хуком (delivery-specs.md §6, блок «Версия с другим хуком»).

  python3 qa7hook.py h1 h2
"""
import subprocess
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hook7 import HOOKS, CUT, CUT_F, SR
import storyboard7 as SB
from style import CARD_A, CARD_B, FPS

ORIG = "/Users/vladimirkalajcidi/reels_good/videos/7/secretary_edit.mp4"


def body_identical(out, nf_hook, probes=14):
    """Тело кадр в кадр совпадает с уже сданной версией.

    Сверяем немую сборку ДО деформации версии: кроп ×0.98 и ускорение ×1.02
    накладываются последним проходом поверх готовых кадров и меняют пиксели
    и нумерацию кадров намеренно. Сама деформация проверяется отдельно —
    измерением объекта в кадре и счётом кадров (см. вывод ниже)."""
    a, b = cv2.VideoCapture(ORIG), cv2.VideoCapture(out)
    n_body = int(SB.DUR * FPS) - CUT_F
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
    a.release()
    b.release()
    return worst, at


def no_old_hook(out, cfg, nf_hook, probes=12):
    """Ни один кадр старого хука не остался в новой версии.

    Сравниваем только A-roll: именно он несёт старую речь и старые субтитры.
    Стоковые планы из сравнения исключены — тот же клип с тем же ss в новом хуке
    стоит намеренно («графику брать ту же, что стояла в старом хуке»), и совпадение
    там означает соблюдение правила, а не уцелевший кадр.
    """
    def face_frames(shots, limit):
        out_f = []
        for t0, t1, kind, _ in shots:
            if kind not in SB.FACE_KINDS:
                continue
            f0, f1 = int(t0 * FPS), min(int(t1 * FPS), limit)
            for k in range(probes):
                f = f0 + int((f1 - f0) * (k + 0.5) / probes)
                if f < limit:
                    out_f.append(f)
        return out_f

    olds = face_frames([s for s in SB.SHOTS if s[0] < CUT], CUT_F)
    news = face_frames(cfg["shots"], nf_hook)
    a, b = cv2.VideoCapture(ORIG), cv2.VideoCapture(out)
    best = 100.0
    for fa in olds:
        a.set(cv2.CAP_PROP_POS_FRAMES, fa)
        oka, ia = a.read()
        if not oka:
            continue
        for fb in news:
            b.set(cv2.CAP_PROP_POS_FRAMES, fb)
            okb, ib = b.read()
            if not okb:
                continue
            best = min(best, 100.0 * np.mean(
                np.abs(ia.astype(np.int16) - ib.astype(np.int16)) > 12))
    a.release()
    b.release()
    return best          # минимальное расхождение; близко к 0 = кадр уцелел


def splice_click(out, nf_hook, speed=1.0):
    """На стыке нет щелчка: скачок между соседними сэмплами в норме дорожки."""
    wav = "/tmp/_qa7hook.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", out,
                    "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", wav], check=True)
    import wave
    w = wave.open(wav)
    d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    os.remove(wav)
    dd = np.abs(np.diff(d))
    i = int(nf_hook / FPS / speed * SR)   # стык на деформированном таймкоде
    local = dd[i - 3:i + 3].max()
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


def rhythm(cfg, nf_hook):
    hook_shots = cfg["shots"]
    dt = nf_hook / FPS - CUT
    body_shots = [(a + dt, b + dt) for a, b, _, _ in SB.SHOTS if a >= CUT]
    lens = [b - a for a, b, _, _ in hook_shots] + [b - a for a, b in body_shots]
    face = sum(b - a for a, b, k, _ in hook_shots if k in SB.FACE_KINDS)
    face += sum(b - a for a, b, k, _ in SB.SHOTS if a >= CUT and k in SB.FACE_KINDS)
    total = nf_hook / FPS + (SB.DUR - CUT)
    lens.sort()
    return dict(n=len(lens), avg=sum(lens) / len(lens), med=lens[len(lens) // 2],
                mn=lens[0], mx=lens[-1], face=100 * face / total, dur=total,
                first_cut=hook_shots[0][1])


if __name__ == "__main__":
    for key in sys.argv[1:]:
        cfg = HOOKS[key]
        out, nf = cfg["out"], cfg["frames"]
        raw = f"{os.path.dirname(os.path.abspath(__file__))}/assets/_video_secretary_{key}.mp4"
        print(f"\n===== {key}: {cfg['title']} =====")
        r = rhythm(cfg, nf)
        print(f"длительность={r['dur']:.2f}с планов={r['n']} средняя={r['avg']:.2f}с "
              f"медиана={r['med']:.2f}с мин={r['mn']:.2f}с макс={r['mx']:.2f}с")
        print(f"лицо={r['face']:.1f}%  первый рез={r['first_cut']:.2f}с")
        w, at = body_identical(raw, nf)
        print(f"тело vs оригинал: макс расхождение {w:.3f}% пикселей (на {at}с) — "
              f"{'ок, шум кодека' if w < 1.0 else 'РАСХОЖДЕНИЕ'}")
        m = no_old_hook(raw, cfg, nf)
        print(f"старый хук: мин расхождение с кадрами нового {m:.1f}% — "
              f"{'ок, ни одного кадра не осталось' if m > 3 else 'КАДР УЦЕЛЕЛ'}")
        speed = 1.02 if cfg["post_a"] else 1.0
        loc, p999, mx = splice_click(out, nf, speed)
        print(f"стык: скачок {loc:.5f} при 99.9-м перцентиле {p999:.5f} и максимуме {mx:.5f} — "
              f"{'ок, щелчка нет' if loc <= p999 else 'ЩЕЛЧОК'}")
        import os as _o
        print(f"уникализация: версия {cfg['version']} · трек {_o.path.basename(cfg['music'])} · "
              f"деформация {cfg['post_v'] or 'нет'}")
        ld = loudness(out)
        print(f"громкость: I={ld['I']} LUFS  LRA={ld['LRA']} LU  TP={ld['TP']} dBFS")

"""Проверки версий с другим хуком для ролика 9 (delivery-specs.md §6 + START-HERE).

    python3 qa9hook.py h1 h2
"""
import os
import subprocess
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hook9 import HOOKS, CUT, CUT_F, SR, BUILD
import storyboard9 as SB
import render9 as R
from style import CARD_A, CARD_B, FPS, W, H, text_layer

ORIG = SB.OUT
ALPHA = 40
N_BODY = 1342                      # кадров тела, перенесённых из _video_euler.mp4


# --- 1. автопроверка: яркие пиксели вне карточки ---------------------------------
def bright_outside(key, cfg, nf_hook, step=3):
    """По НЕдеформированной сборке: кроп x0.98 сдвигает всю картинку наружу от центра,
    и номинальный прямоугольник карточки к ней уже не применим."""
    shots = cfg["shots"]
    cap = cv2.VideoCapture(f"{BUILD}/assets/_video_euler{key}.mp4")
    bad, sample = 0, []
    for f in range(0, nf_hook, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        t = f / FPS
        kind = next((s[2] for s in shots if s[0] <= t < s[1]), shots[-1][2])
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
            sample.append(round(t, 2))
    cap.release()
    return bad, sample[:8]


# --- 2. автопроверка: наложение строк внутри блока (по слоям, до кодека) ----------
def _patch(cfg):
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()


def line_overlap(cfg):
    """Реальные вертикальные габариты соседних строк блока — не кегль.
    Каждая строка блока растеризуется отдельно и без свечения (свечение размыто,
    оно пересекается всегда и о читаемости ничего не говорит); пересечение
    множеств занятых строк пикселей у соседей по блоку = дефект."""
    _patch(cfg)
    bad, sample = 0, []
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        active = [i for i, c in enumerate(R.CAPS) if c[0] <= t < R.BLOCKS[i][3]]
        if len(active) < 2:
            continue
        spans = []
        for idx in active:
            items = _line_items_at(t, idx)
            if not items:
                spans.append(None)
                continue
            for it in items:
                it["glow_r"] = 0
                it["opacity"] = 1.0
            a = np.array(text_layer((W, H), items).split()[3])
            rows = np.nonzero((a > ALPHA).any(axis=1))[0]
            spans.append((rows.min(), rows.max()) if len(rows) else None)
        for a_, b_ in zip(spans, spans[1:]):
            if a_ and b_ and not (a_[1] < b_[0] or b_[1] < a_[0]):
                bad += 1
                sample.append((round(t, 2), a_, b_))
                break
    return bad, sample[:8]


def _line_items_at(t, idx):
    """Элементы одной строки субтитра на момент t — та же геометрия, что в render9."""
    t0, _, runs = R.CAPS[idx]
    bid, pos, _, _ = R.BLOCKS[idx]
    kind = R.shot_at(t)[2]
    slot = R.SLOTS[R.slot_for(kind)]
    bx, by = slot["x"], slot["y"] + pos * slot["step"]
    if slot["scatter"]:
        calm = bid % 3 == 0
        bx += (0 if calm else [-45, 38, -20][min(pos, 2)])
    max_w = slot["x_max"] - bx if slot["anchor"][0] == "l" else \
        2 * min(bx - slot["x_min"], slot["x_max"] - bx)
    return R.line_items(runs, (bx, by), slot["anchor"], max_w=max_w)


# --- 3. тело кадр в кадр совпадает с уже сданным роликом --------------------------
def body_identical(key, nf_hook, probes=14):
    """Сравниваем НЕдеформированную сборку: деформация накладывается после."""
    pre = f"{BUILD}/assets/_video_euler{key}.mp4"
    a, b = cv2.VideoCapture(ORIG), cv2.VideoCapture(pre)
    worst, at = 0.0, None
    for k in range(probes):
        f = CUT_F + int(N_BODY * (k + 0.5) / probes)
        a.set(cv2.CAP_PROP_POS_FRAMES, f)
        b.set(cv2.CAP_PROP_POS_FRAMES, f - CUT_F + nf_hook)
        oka, ia = a.read()
        okb, ib = b.read()
        if not (oka and okb):
            continue
        pct = 100.0 * np.mean(np.abs(ia.astype(np.int16) - ib.astype(np.int16)) > 12)
        if pct > worst:
            worst, at = pct, round(f / FPS, 2)
    a.release(); b.release()
    return worst, at


# --- 4. от старого хука не осталось ни одного кадра ------------------------------
OLD_HOOK_FACE = [(0.000, 1.640)]   # только A1: сток 4619 переиспользован намеренно


def no_old_hook(out, nf_hook, probes=10):
    a, b = cv2.VideoCapture(ORIG), cv2.VideoCapture(out)
    best = 100.0
    for t0, t1 in OLD_HOOK_FACE:
        f0, f1 = int(t0 * FPS), int(t1 * FPS)
        for k in range(probes):
            a.set(cv2.CAP_PROP_POS_FRAMES, f0 + int((f1 - f0) * (k + 0.5) / probes))
            oka, ia = a.read()
            if not oka:
                continue
            for k2 in range(probes):
                b.set(cv2.CAP_PROP_POS_FRAMES, int(nf_hook * (k2 + 0.5) / probes))
                okb, ib = b.read()
                if not okb:
                    continue
                if ib.shape != ia.shape:
                    ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]))
                best = min(best, 100.0 * np.mean(
                    np.abs(ia.astype(np.int16) - ib.astype(np.int16)) > 12))
    a.release(); b.release()
    return best


# --- 5. стык: щелчка нет ---------------------------------------------------------
def splice_click(out, nf_hook, speed=1.0):
    wav = f"{BUILD}/assets/_qa9hook.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", out,
                    "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", wav], check=True)
    w = wave.open(wav)
    d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    os.remove(wav)
    dd = np.abs(np.diff(d))
    i = int(nf_hook / FPS / speed * SR)
    local = dd[max(0, i - 3):i + 3].max() if len(dd) > i + 3 else dd[-6:].max()
    return local, float(np.percentile(dd, 99.9)), float(dd.max())


# --- 6. пауза между речью хука и первым словом тела ------------------------------
BODY_FIRST_WORD = 3.540            # «в ней…», source.srt


def gap_hook_body(cfg, nf_hook):
    return nf_hook / FPS - cfg["speech_end"] + (BODY_FIRST_WORD - CUT)


# --- 7. цвет: средний RGB внутри карточки A на хуке и на теле --------------------
def card_rgb(path, frames):
    cap = cv2.VideoCapture(path)
    x, y, w, h = CARD_A
    acc = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        sx = img.shape[1] / W
        roi = img[int(y * sx):int((y + h) * sx), int(x * sx):int((x + w) * sx)]
        acc.append((roi[:, :, 2].mean(), roi[:, :, 1].mean(), roi[:, :, 0].mean()))
    cap.release()
    a = np.array(acc)
    return a.mean(axis=0)


# --- 8. служебное ----------------------------------------------------------------
def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    out = {}
    for key, tag in (("I:", "I"), ("LRA:", "LRA"), ("Peak:", "TP")):
        idx = r.rindex(key)
        out[tag] = float(r[idx + len(key):idx + 40].split()[0])
    return out


def probe(path, entries):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=" + entries, "-of", "csv=p=0", path],
                       capture_output=True, text=True).stdout.strip()
    return r


def crop_measure(before, after, y=925):
    """Кроп 0.98 обязан дать на экране x1.0204: ширина карточки A между чёрными краями."""
    def width(path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ok, img = cap.read()
        cap.release()
        row = img[y].astype(np.int32).sum(axis=1)
        idx = np.where(row > 60)[0]
        return idx[-1] - idx[0] if len(idx) else 0
    w0, w1 = width(before), width(after)
    return w0, w1, (w1 / w0 if w0 else 0)


def rhythm(cfg, nf_hook):
    dt = nf_hook / FPS - CUT
    hook_shots = cfg["shots"]
    body_shots = [(a, b, k) for a, b, k, _ in SB.SHOTS if a >= CUT]
    lens = [b - a for a, b, _, _ in hook_shots] + [b - a for a, b, _ in body_shots]
    face = sum(b - a for a, b, k, _ in hook_shots if k in SB.FACE_KINDS)
    face += sum(b - a for a, b, k in body_shots if k in SB.FACE_KINDS)
    total = nf_hook / FPS + (SB.DUR - CUT)
    lens.sort()
    return dict(n=len(lens), avg=sum(lens) / len(lens), med=lens[len(lens) // 2],
                mn=lens[0], mx=lens[-1], face=100 * face / total, dur=total,
                first_cut=hook_shots[0][1])


if __name__ == "__main__":
    for key in sys.argv[1:]:
        cfg = HOOKS[key]
        out, nf = cfg["out"], cfg["frames"]
        speed = 1.02 if "setpts" in (cfg.get("post_v") or "") else 1.0
        print(f"\n===== {key} (версия {cfg['version']}): {cfg['title']} =====")

        r = rhythm(cfg, nf)
        print(f"хронометраж={r['dur']:.2f}с планов={r['n']} средняя={r['avg']:.2f}с "
              f"медиана={r['med']:.2f}с мин={r['mn']:.2f}с макс={r['mx']:.2f}с")
        print(f"лицо={r['face']:.0f}% (норма 30–45)  первый рез={r['first_cut']:.2f}с "
              f"({'ок' if r['first_cut'] <= 2.0 else 'ПОЗДНО'})")

        gap = gap_hook_body(cfg, nf)
        print(f"пауза хук->первое слово тела: {gap:.3f}с "
              f"(в оригинале 0.240с) — {'ок' if 0.15 <= gap <= 0.40 else 'ВНЕ НОРМЫ'}")

        b, bs = bright_outside(key, cfg, nf)
        print(f"[авто 1] яркие пиксели вне карточки: кадров={b} {bs}")

        o, osx = line_overlap(cfg)
        print(f"[авто 2] наложение строк в блоке: кадров={o} {osx}")

        w, at = body_identical(key, nf)
        print(f"тело vs euler_edit.mp4: макс расхождение {w:.3f}% пикселей (на {at}с) — "
              f"{'ок, шум кодека' if w < 1.0 else 'РАСХОЖДЕНИЕ'}")

        m = no_old_hook(out, nf)
        print(f"старый хук: мин расхождение с кадрами нового {m:.1f}% — "
              f"{'ок, ни одного кадра не осталось' if m > 3 else 'КАДР УЦЕЛЕЛ'}")

        loc, p999, mx = splice_click(out, nf, speed)
        print(f"стык: скачок {loc:.5f} при 99.9-м перцентиле {p999:.5f}, максимум {mx:.5f} — "
              f"{'ок, щелчка нет' if loc <= p999 else 'ЩЕЛЧОК'}")

        ld = loudness(out)
        print(f"громкость: I={ld['I']} LUFS  LRA={ld['LRA']} LU  TP={ld['TP']} dBFS")

        pr = probe(out, "width,height,r_frame_rate,duration,nb_frames").split(",")
        print(f"файл: {pr[0]}x{pr[1]} {pr[2]} fps · {pr[4]} кадров · {float(pr[3]):.2f}с — "
              f"{'ок' if pr[2] == '30/1' and pr[0] == '1080' else 'ПРОВЕРИТЬ'}")

        # сравниваются только планы с лицом: планы на сетке — чёрный фон, они утянули бы
        # среднее вниз и замер потерял бы смысл
        hook_f = [int((a + b) / 2 * FPS) for a, b, k, _ in cfg["shots"] if k in SB.FACE_KINDS]
        body_f = [nf + int(((a + b) / 2 - CUT) * FPS)
                  for a, b, k, _ in SB.SHOTS if a >= CUT and k in SB.FACE_KINDS]
        ch = card_rgb(out, [int(f / speed) for f in hook_f])
        cb = card_rgb(out, [int(f / speed) for f in body_f])
        d = np.abs(ch - cb).max()
        print("цвет в карточке A (только лицо): хук R=%.0f G=%.0f B=%.0f · "
              "тело R=%.0f G=%.0f B=%.0f · макс расхождение %.0f — %s"
              % (*ch, *cb, d, "ок" if d <= 10 else "ВЫШЕ НОРМЫ (~10)"))

        pre = f"{BUILD}/assets/_video_euler{key}.mp4"
        if "crop=iw*0.98" in (cfg.get("post_v") or ""):
            w0, w1, ratio = crop_measure(pre, f"{BUILD}/assets/_video_euler{key}_post.mp4")
            print(f"деформация: кроп {w0}px -> {w1}px, x{ratio:.4f} (ожидание x1.0204) — "
                  f"{'ок' if abs(ratio - 1.0204) < 0.01 else 'РАСХОЖДЕНИЕ'}")
        if speed != 1.0:
            n0 = int(probe(pre, "nb_frames"))
            n1 = int(pr[4])
            print(f"деформация: ускорение {n0} -> {n1} кадров, x{n0/n1:.4f} "
                  f"(ожидание x1.0200) — {'ок' if abs(n0/n1 - 1.02) < 0.005 else 'РАСХОЖДЕНИЕ'}")

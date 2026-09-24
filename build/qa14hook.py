"""QA версии ролика 14 с хуком 1."""
import os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS, W, H
import render14 as R
import storyboard14 as SB
import hook14 as HK

cfg = HK.HOOKS["h1"]
OUT = cfg["out"]
NF_HOOK = cfg["frames"]
CUT_F = HK.CUT_F
ALPHA = 40

# карточки в геометрии версии 2: кроп 0.98 + возврат холста
K = 1 / 0.98
TOL = 2      # кроп 0.98 пересобирает кадр билинейно и размазывает край карточки
             # на соседний пиксель: замер даёт карточку A x 97..984 против
             # расчётных 96..983. Допуск 2px закрывает ресемпл и не в состоянии
             # спрятать вылет текста — тот измеряется десятками пикселей.
def defo(rect):
    x, y, w, h = rect
    return (int(round((x - W * 0.01) * K)) - TOL, int(round((y - H * 0.01) * K)) - TOL,
            int(round(w * K)) + 2 * TOL, int(round(h * K)) + 2 * TOL)
CARD_A2, CARD_B2 = defo(CARD_A), defo(CARD_B)


def hook_shot_at(t):
    return next((s for s in cfg["shots"] if s[0] <= t < s[1]), cfg["shots"][-1])


def _prep():
    R.CAPS = cfg["caps"]
    R.shot_at = hook_shot_at
    R.BLOCKS = R.build_blocks()


def gfx_text_overlap():
    _prep(); bad, smp = 0, []
    for f in range(0, NF_HOOK, 2):
        t = f / FPS
        t0, _, kind, _ = hook_shot_at(t)
        gl = R.graphics_layer(kind, t - t0)
        cl = R.caption_layer(t)
        if gl is None or cl is None: continue
        n = int(np.count_nonzero((np.array(gl.split()[3]) > ALPHA) & (np.array(cl.split()[3]) > ALPHA)))
        if n: bad += 1; smp.append((round(t, 2), kind, n))
    return bad, smp[:6]


def gfx_in_zone():
    _prep(); y0, y1 = SB.GFX_ZONE; x0, x1 = SB.GFX_X
    bad, smp = 0, []
    for f in range(0, NF_HOOK, 2):
        t = f / FPS
        ts, _, kind, _ = hook_shot_at(t)
        gl = R.graphics_layer(kind, t - ts)
        if gl is None: continue
        a = np.array(gl.split()[3]); ys, xs = np.nonzero(a > ALPHA)
        if len(ys) == 0: continue
        if ys.min() < y0 or ys.max() > y1 or xs.min() < x0 or xs.max() > x1:
            bad += 1; smp.append((round(t, 2), kind, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return bad, smp[:6]


def text_in_card():
    _prep(); bad, smp = 0, []
    for f in range(0, NF_HOOK, 2):
        t = f / FPS
        kind = hook_shot_at(t)[2]
        cl = R.caption_layer(t)
        if cl is None: continue
        a = np.array(cl.split()[3]); ys, xs = np.nonzero(a > 120)
        if len(ys) == 0: continue
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        if (xs.min() < x + 30 or xs.max() > x + w - 30 or ys.min() < y + 30 or ys.max() > y + h - 30):
            bad += 1; smp.append((round(t, 2), kind, int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))
    return bad, smp[:6]


def caption_zone():
    _prep(); bad, smp = 0, []
    for f in range(0, NF_HOOK, 2):
        t = f / FPS
        kind = hook_shot_at(t)[2]
        if kind in SB.FACE_KINDS: continue
        cl = R.caption_layer(t)
        if cl is None: continue
        ys, _ = np.nonzero(np.array(cl.split()[3]) > ALPHA)
        if len(ys) and (ys.max() > 600 or ys.min() < 300):
            bad += 1; smp.append((round(t, 2), kind, int(ys.min()), int(ys.max())))
    return bad, smp[:6]


def line_overlaps():
    caps = cfg["caps"]; bad, smp = 0, []
    for i in range(len(caps) - 1):
        if caps[i + 1][0] - caps[i][1] > 0.30: continue
        if hook_shot_at(caps[i][0]) is not hook_shot_at(caps[i + 1][0]): continue
        s1 = max(sz for _, _, sz in caps[i][2]); s2 = max(sz for _, _, sz in caps[i + 1][2])
        step = R.SLOTS[R.slot_for(hook_shot_at(caps[i][0])[2])]["step"]
        if step < (s1 + s2) * 0.50:
            bad += 1; smp.append((round(caps[i][0], 2), s1, s2, step))
    return bad, smp[:6]


def bright_outside():
    """Готовый файл: ничего ярче 200 вне карточки (в геометрии версии 2)."""
    cap = cv2.VideoCapture(OUT); bad, smp = 0, []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in range(0, total, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
        if not ok: continue
        if f < NF_HOOK:
            kind = hook_shot_at(f / FPS)[2]
        else:
            kind = SB.shot_at((f - NF_HOOK + CUT_F) / FPS)[2]
        x, y, w, h = CARD_B2 if kind == "stock" else CARD_A2
        m = np.zeros(img.shape[:2], np.uint8); m[max(0,y):y+h, max(0,x):x+w] = 1
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((g > 200) & (m == 0)):
            bad += 1; smp.append(round(f / FPS, 2))
    cap.release()
    return bad, smp[:8]


def rhythm():
    """Ритм всей версии: планы хука + планы тела после реза."""
    shots = [(a, b) for a, b, _, _ in cfg["shots"]]
    dt = NF_HOOK / FPS - HK.CUT
    for a, b, _, _ in SB.SHOTS:
        if b <= HK.CUT: continue
        shots.append((max(a, HK.CUT) + dt, b + dt))
    lens = [b - a for a, b in shots]
    dur = NF_HOOK / FPS + (SB.DUR - HK.CUT)
    face = sum(b - a for a, b, k, _ in cfg["shots"] if k in SB.FACE_KINDS)
    face += sum(b - max(a, HK.CUT) for a, b, k, _ in SB.SHOTS
                if k in SB.FACE_KINDS and b > HK.CUT)
    return dict(shots=len(shots), avg=round(sum(lens)/len(lens), 2),
                median=round(sorted(lens)[len(lens)//2], 2),
                shortest=round(min(lens), 2), longest=round(max(lens), 2),
                first_cut=round(shots[0][1], 2), dur=round(dur, 2),
                face_pct=round(100*face/dur, 1))


if __name__ == "__main__":
    print("ритм версии:", rhythm())
    for name, fn in (("gfx_text_overlap", gfx_text_overlap), ("gfx_out_of_zone", gfx_in_zone),
                     ("caption_out_of_zone", caption_zone), ("text_out_of_card", text_in_card),
                     ("line_overlap_pairs", line_overlaps)):
        n, s = fn(); print(f"{name}={n} sample={s}")
    if os.path.exists(OUT):
        n, s = bright_outside(); print(f"bright_outside_card_frames={n} sample={s}")

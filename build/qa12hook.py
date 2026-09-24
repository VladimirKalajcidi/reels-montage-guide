"""QA версий ролика 12 с другими хуками.

Проверяет то же, что qa12, но на новой таймлинии: блок хука рисуется по своей
раскадровке, тело идёт готовыми кадрами со сдвигом. Отдельно учитываются
деформации версий: у версии 2 карточка на экране увеличена кропом x0.98,
у версии 3 сдвинуты номера кадров из-за ускорения x1.02.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import storyboard12 as SB
import render12 as R
import hook12 as HK
from style import CARD_A, CARD_B, FPS

ALPHA = 40


def timeline(key):
    """Возвращает (kind_at(t), длительность) для новой таймлинии версии."""
    cfg = HK.HOOKS[key]
    nf = cfg["frames"]
    dt = nf / FPS - HK.CUT
    body_n = 1186 - HK.CUT_F

    def kind_at(t):
        if t < nf / FPS:
            return next((s for s in cfg["shots"] if s[0] <= t < s[1]),
                        cfg["shots"][-1])[2]
        return SB.shot_at(t - dt)[2]

    return kind_at, (nf + body_n) / FPS


def crop_rect(rect, scale):
    """Куда уезжает прямоугольник карточки после кропа x`scale` с возвратом холста.

    Запас в 1px с каждой стороны — это край карточки после ресемпла: ffmpeg округляет
    размер кропа до целых пикселей, и граница попадает на доли пикселя. Замер по
    готовому файлу: карточка A занимает 888x1409 при расчётных 887.7x1408.2.
    Убежавший за карточку текст — это десятки пикселей, так что проверка не слепнет.
    """
    if scale == 1.0:
        return rect
    x, y, w, h = rect
    ox, oy = 1080 * (1 - scale) / 2, 1920 * (1 - scale) / 2
    x2 = int(np.floor((x - ox) / scale)) - 1
    y2 = int(np.floor((y - oy) / scale)) - 1
    return (x2, y2, int(np.ceil(w / scale)) + 2, int(np.ceil(h / scale)) + 2)


def bright_outside(key):
    cfg = HK.HOOKS[key]
    kind_at, dur = timeline(key)
    scale = 0.98 if "crop" in (cfg.get("post_v") or "") else 1.0
    tempo = 1.02 if "setpts" in (cfg.get("post_v") or "") else 1.0
    cap = cv2.VideoCapture(cfg["out"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad, sample = 0, []
    for f in range(0, total, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        t = f * tempo / FPS
        if t >= dur:
            break
        rect = CARD_B if kind_at(t) == "stock" else CARD_A
        x, y, w, h = crop_rect(rect, scale)
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[max(0, y):y + h, max(0, x):x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
            sample.append((round(t, 2), kind_at(t)))
    cap.release()
    return bad, sample[:8]


def hook_line_overlaps(key):
    """Строки блока хука не пересекаются: шаг >= (кегль1+кегль2)*0.70."""
    cfg = HK.HOOKS[key]
    caps = cfg["caps"]
    shot = lambda t: next((s for s in cfg["shots"] if s[0] <= t < s[1]), cfg["shots"][-1])
    bad, sample = 0, []
    for i in range(len(caps) - 1):
        if caps[i + 1][0] - caps[i][1] > 0.30:
            continue
        if shot(caps[i][0]) is not shot(caps[i + 1][0]):
            continue
        s1 = max(sz for _, _, sz in caps[i][2])
        s2 = max(sz for _, _, sz in caps[i + 1][2])
        slot = SB.SLOTS[SB.slot_for(shot(caps[i][0])[2])]
        need = max(60, (s1 + s2) * 0.70)
        if slot["step"] < need:
            bad += 1
            sample.append((caps[i][0], s1, s2, slot["step"], round(need, 1)))
    return bad, sample


def hook_layers(key):
    """Графика хука не пересекает субтитры, лежит в зоне, текст внутри карточки."""
    cfg = HK.HOOKS[key]
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()
    y0, y1 = SB.GFX_ZONE
    gx0, gx1 = SB.GFX_X
    ov = zone = out_card = 0
    smp = []
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        t0, _t1, kind, prm = R.shot_at(t)
        gl = R.graphics_layer(kind, (t - t0) * prm.get("lts", 1.0))
        cl = R.caption_layer(t)
        if gl is not None:
            a = np.array(gl.split()[3])
            ys, xs = np.nonzero(a > ALPHA)
            if len(ys) and (ys.min() < y0 or ys.max() > y1
                            or xs.min() < gx0 or xs.max() > gx1):
                zone += 1
                smp.append(("zone", round(t, 2), int(xs.min()), int(xs.max()),
                            int(ys.min()), int(ys.max())))
            if cl is not None:
                ca = np.array(cl.split()[3])
                if np.any((a > ALPHA) & (ca > ALPHA)):
                    ov += 1
        if cl is not None:
            rect = CARD_B if kind == "stock" else CARD_A
            x, y, w, h = rect
            ca = np.array(cl.split()[3])
            ys, xs = np.nonzero(ca > ALPHA)
            if len(ys) and (xs.min() < x + 30 or xs.max() > x + w - 30
                            or ys.min() < y + 30 or ys.max() > y + h - 30):
                out_card += 1
                smp.append(("card", round(t, 2), int(xs.min()), int(xs.max()),
                            int(ys.min()), int(ys.max())))
    return ov, zone, out_card, smp[:6]


def rhythm(key):
    cfg = HK.HOOKS[key]
    nf = cfg["frames"]
    dt = nf / FPS - HK.CUT
    lens = [b - a for a, b, _, _ in cfg["shots"]]
    face = sum(b - a for a, b, k, _ in cfg["shots"] if k in SB.FACE_KINDS)
    body_shots = [(max(a, HK.CUT) + dt, b + dt, k) for a, b, k, _ in SB.SHOTS if b > HK.CUT]
    lens += [b - a for a, b, _ in body_shots]
    face += sum(b - a for a, b, k in body_shots if k in SB.FACE_KINDS)
    dur = nf / FPS + (SB.DUR - HK.CUT)
    srt = sorted(lens)
    return dict(shots=len(lens), avg=round(sum(lens) / len(lens), 2),
                median=round(srt[len(srt) // 2], 2), shortest=round(min(lens), 2),
                longest=round(max(lens), 2), first_cut=cfg["shots"][0][1],
                dur=round(dur, 2), face_pct=round(100 * face / dur, 1))


if __name__ == "__main__":
    for key in ("h1", "h2"):
        cfg = HK.HOOKS[key]
        print(f"=== версия {cfg['version']} · {os.path.basename(cfg['out'])}")
        print("   ритм:", rhythm(key))
        lo, los = hook_line_overlaps(key)
        print(f"   line_overlap_pairs={lo} {los}")
        ov, zone, oc, smp = hook_layers(key)
        print(f"   gfx_text_overlap={ov} gfx_out_of_zone={zone} caption_out_of_card={oc} {smp}")
        b, bs = bright_outside(key)
        print(f"   bright_outside_card_frames={b} sample={bs}")

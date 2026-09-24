"""Автопроверки версий ролика 16 с другими хуками.

Проверяется то, что перечислено в delivery-specs §6 в блоке «Версия с другим хуком»,
плюс обе штатные картиночные проверки на всей длине версии.
"""
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS
import storyboard16 as SB
import hook16 as HK
import render16 as R

V1 = "/Users/vladimirkalajcidi/reels_good/videos/16/a4_edit.mp4"


def shot_kind_at(cfg, t):
    """Тип плана на новой таймлинии: сначала хук, потом сдвинутое тело."""
    hook_len = cfg["frames"] / FPS
    if t < hook_len:
        return next((s[2] for s in cfg["shots"] if s[0] <= t < s[1]), cfg["shots"][-1][2])
    return SB.shot_at(t - hook_len + HK.CUT)[2]


def deform_rect(rect, cfg):
    """Кроп 0.98 двигает и саму карточку — проверять надо по её новому месту.

    crop=iw*0.98:ih*0.98 + scale обратно: точка (x,y) -> ((x-10.8)/0.98, (y-19.2)/0.98).
    """
    x, y, w, h = rect
    if "crop=iw*0.98" not in (cfg.get("post_v") or ""):
        return rect
    dx, dy, k = 0.01 * 1080, 0.01 * 1920, 0.98
    x0, y0 = (x - dx) / k, (y - dy) / k
    x1, y1 = (x + w - dx) / k, (y + h - dy) / k
    # запас 1px: 10.8/0.98 не целое, край карточки после билинейного апскейла
    # ложится на дробную координату и размазывается на соседний столбец.
    # Без запаса проверка ловит саму кромку карточки (ровно x=984), а не текст.
    return (int(np.floor(x0)) - 1, int(np.floor(y0)) - 1,
            int(np.ceil(x1 - x0)) + 2, int(np.ceil(y1 - y0)) + 2)


def card_bbox(path, frame):
    """Габарит карточки на кадре: всё, что не чёрное поле."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(g > 28)
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def bright_outside(path, cfg, scale=1.0):
    """Нет пикселей ярче 200 вне прямоугольника карточки."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad, sample = 0, []
    for f in range(0, n, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        kind = shot_kind_at(cfg, f / FPS * scale)
        x, y, w, h = deform_rect(CARD_B if kind == "stock" else CARD_A, cfg)
        x, y = max(0, x), max(0, y)
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:8]


def hook_layers(cfg):
    """Графика и субтитры хука не пересекаются; текст внутри карточки."""
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()
    ov, oz, oc = 0, 0, 0
    for f in range(0, cfg["frames"], 2):
        t = f / FPS
        t0, _t1, kind, prm = R.shot_at(t)
        gl = R.graphics_layer(kind, (t - t0) * prm.get("lts", 1.0) + prm.get("lt0", 0.0))
        cl = R.caption_layer(t)
        if gl is not None:
            a = np.array(gl.split()[3])
            ys, xs = np.nonzero(a > 40)
            if len(ys) and (ys.min() < SB.GFX_ZONE[0] or ys.max() > SB.GFX_ZONE[1]
                            or xs.min() < SB.GFX_X[0] or xs.max() > SB.GFX_X[1]):
                oz += 1
            if cl is not None:
                if np.count_nonzero((a > 40) & (np.array(cl.split()[3]) > 40)):
                    ov += 1
        if cl is not None:
            a = np.array(cl.split()[3])
            ys, xs = np.nonzero(a > 120)
            if len(ys):
                x, y, w, h = CARD_A
                if (xs.min() < x + 30 or xs.max() > x + w - 30
                        or ys.min() < y + 30 or ys.max() > y + h - 30):
                    oc += 1
    return ov, oz, oc


def body_identical(path, cfg, scale=1.0):
    """Тело версии кадр в кадр совпадает с версией №1.

    Сравнивается НЕмое видео до деформации: кроп 0.98 и ускорение 1.02 меняют
    геометрию и тайминг по построению, и сравнивать после них бессмысленно.
    """
    hook_n = cfg["frames"]
    a = cv2.VideoCapture(V1)
    b = cv2.VideoCapture(path)
    scale = 1.0
    worst, checked = 0.0, 0
    for off in (10, 200, 500, 900, 1180):
        fa, fb = HK.CUT_F + off, int(round((hook_n + off) / scale))
        a.set(cv2.CAP_PROP_POS_FRAMES, fa)
        b.set(cv2.CAP_PROP_POS_FRAMES, fb)
        oka, ia = a.read()
        okb, ib = b.read()
        if not (oka and okb):
            continue
        if ia.shape != ib.shape:
            ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]))
        mse = ((ia.astype(np.float64) - ib.astype(np.float64)) ** 2).mean()
        worst = max(worst, mse)
        checked += 1
    a.release()
    b.release()
    return checked, round(10 * np.log10(255 ** 2 / max(worst, 1e-9)), 1)


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate,width,height,nb_frames",
                        "-of", "default=nw=1", path], capture_output=True, text=True)
    d = dict(l.split("=") for l in r.stdout.strip().splitlines())
    va = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                         "stream=codec_type,duration", "-of", "csv=p=0", path],
                        capture_output=True, text=True).stdout.strip().splitlines()
    d["durations"] = [x.split(",")[1] for x in va if "," in x]
    return d


if __name__ == "__main__":
    for key in ("h1", "h2"):
        cfg = HK.HOOKS[key]
        path = cfg["out"]
        scale = 1.02 if cfg.get("post_a") else 1.0
        print(f"\n===== версия {cfg['version']} · {os.path.basename(path)}")
        print(" ", probe(path))
        ov, oz, oc = hook_layers(cfg)
        print(f"  графика x субтитры хука: {ov} кадров")
        print(f"  графика хука вне зоны:   {oz} кадров")
        print(f"  текст хука вне карточки: {oc} кадров")
        b, s = bright_outside(path, cfg, scale)
        print(f"  ярче 200 вне карточки:   {b} кадров {s}")
        n, psnr = body_identical(f"{HK.BUILD}/assets/_video_a4{key}.mp4", cfg)
        print(f"  тело против версии №1:   {n} проб, худший PSNR {psnr} dB")
        # деформация версии — измеряем, а не верим флагу
        w1, h1 = card_bbox(V1, HK.CUT_F + 379)
        w2, h2 = card_bbox(path, int(round((cfg["frames"] + 379) / scale)))
        print(f"  габарит карточки: версия 1 {w1}x{h1} -> версия {cfg['version']} "
              f"{w2}x{h2}  (x{w2/w1:.4f} по ширине, x{h2/h1:.4f} по высоте)")

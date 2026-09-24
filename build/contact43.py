"""Контакт-лист ключевых кадров ролика 10 (слот 43) — смысловая проверка глазами.
Кадры берутся из готового ролика (по умолчанию) или собираются compose() до рендера (--pre)."""
import os, sys
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import FPS, font
from storyboard43 import SHOTS
import render43 as R

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/contact43.jpg"
PRE = "--pre" in sys.argv


def times():
    ts = []
    for t0, t1, kind, prm in SHOTS:
        ts += [t0 + 0.05, (t0 + t1) / 2]
        for k in ("t_pick", "t_num", "t_move", "t_word"):
            if k in prm and t0 <= prm[k] < t1:
                ts.append(min(t1 - 0.04, prm[k] + 0.45))
        ts.append(t1 - 0.05)
    return sorted(set(round(t, 2) for t in ts))


def frame_pre(t):
    si = R._shot_idx(t); shot = SHOTS[si]; t0, t1, kind, prm = shot
    fr = None
    if kind in R.FACE_KINDS:
        c = cv2.VideoCapture(R.A1 if kind == "A1" else R.A2)
        c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * FPS))); ok, fr = c.read()
    elif kind == "stock":
        c = cv2.VideoCapture(f"{R.STOCK_PREP_DIR}/sb43v2_{si}_{prm['clip']}.mp4")
        c.set(cv2.CAP_PROP_POS_FRAMES, int(round((t - t0) * FPS))); ok, fr = c.read()
    return R.compose(t, shot, fr).convert("RGB")


def main():
    ts = times()
    cap = None if PRE else cv2.VideoCapture(R.OUT)
    tiles = []
    for t in ts:
        if PRE:
            im = frame_pre(t)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * FPS)))
            ok, fr = cap.read()
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        im = im.resize((216, 384), Image.LANCZOS)
        ImageDraw.Draw(im).text((6, 4), f"{t:.2f}", font=font("sans", 22), fill=(255, 255, 0))
        tiles.append(im)
    cols = 12
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (216 * cols, 384 * rows), (40, 40, 40))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * 216, (i // cols) * 384))
    sheet.save(OUT, quality=85)
    print("кадров:", len(tiles), "->", OUT)


if __name__ == "__main__":
    main()

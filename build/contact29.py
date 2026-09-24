"""Контакт-лист ключевых кадров ролика 29 для ручной смысловой проверки."""
import os
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard29 import OUT, shot_at

DEST = "/Users/vladimirkalajcidi/reels_good/videos/29/contact-sheet.jpg"
TIMES = [
    0.9, 2.15, 4.0, 5.75, 7.95, 9.85, 11.55, 13.35, 15.05,
    16.85, 18.85, 20.55, 21.85, 23.75, 25.95, 28.55, 31.90,
    34.15, 36.95, 38.65, 40.15, 41.70, 44.20, 47.35, 49.45,
]


def main():
    cap = cv2.VideoCapture(OUT)
    tiles = []
    fnt = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 24)
    for t in TIMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(fr).resize((216, 384), Image.LANCZOS)
        kind = shot_at(t)[2]
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([5, 5, 130, 36], radius=8, fill=(0, 0, 0, 185))
        d.text((12, 9), f"{t:04.1f}  {kind}", font=fnt, fill="white")
        tiles.append(im)
    cap.release()
    cols = 5; rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * 216, rows * 384), "black")
    for i, im in enumerate(tiles):
        out.paste(im, ((i % cols) * 216, (i // cols) * 384))
    out.save(DEST, quality=94)
    print(DEST)


if __name__ == "__main__":
    main()

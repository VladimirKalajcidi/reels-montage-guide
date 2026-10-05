"""Контакт-лист ролика 66 (слот 166): середина каждого плана + графика ape166 — в полёте и на полном
раскрытии (∞ с точкой, печать TO BE OR NOT TO BE, 29 септиллионов, 130 000, 100%), вспышка молнии."""
import subprocess, sys, io
from PIL import Image, ImageDraw, ImageFont
from storyboard166 import SHOTS
OUT_MP4 = "/Users/vladimirkalajcidi/reels_challenge/videos/66/monkey_edit.mp4"
TS = sorted(set([round((t0 + t1) / 2, 2) for t0, t1, _, _ in SHOTS] +
                [2.2, 2.6, 3.3, 12.4, 13.35, 13.7, 14.3, 17.3, 18.1, 19.1, 19.4, 20.7, 21.8, 29.9, 30.2, 30.8, 31.4, 32.1,
                 39.10]))
f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 26)
tiles = []
for t in TS:
    png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", OUT_MP4, "-frames:v", "1",
                          "-vf", "scale=270:480", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True).stdout
    im = Image.open(io.BytesIO(png)).convert("RGB")
    ImageDraw.Draw(im).text((6, 4), f"{t:.2f}", font=f, fill=(255, 220, 0))
    tiles.append(im)
cols = 10
sheet = Image.new("RGB", (cols * 270, ((len(tiles) + cols - 1) // cols) * 480))
for k, im in enumerate(tiles):
    sheet.paste(im, ((k % cols) * 270, (k // cols) * 480))
sheet.save(sys.argv[1], quality=88)
print(sys.argv[1], len(tiles), "кадров")

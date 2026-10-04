"""Контакт-лист ролика 57 v2 (слот 157): середина каждого плана + графика conv157 — в полёте и на полном
раскрытии (конверты 2X/X, 1 000 ₽, 500/2 000, 1 250 ₽, петля +25%, дерево 50%?)."""
import subprocess, sys, io
from PIL import Image, ImageDraw, ImageFont
from storyboard157 import SHOTS
OUT_MP4 = "/Users/vladimirkalajcidi/reels_challenge/videos/57/paradox_edit.mp4"
TS = sorted(set([round((t0 + t1) / 2, 2) for t0, t1, _, _ in SHOTS] +
                [6.64, 7.40, 9.0, 11.48, 12.40, 15.64, 16.00, 17.10, 20.42, 20.62, 21.30, 23.20, 30.10, 30.60, 31.10,
                 31.80, 32.60, 33.20, 36.55, 37.30, 38.10, 39.10, 40.20, 52.80]))
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

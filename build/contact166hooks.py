"""Контакт-лист версии с хуком ролика 66 (monkey_hook1.mp4): кадры блока хука, шов и начало тела;
копия contact166.py. середина каждого плана + графика ape166 — в полёте и на полном
раскрытии (∞ с точкой, печать TO BE OR NOT TO BE, 29 септиллионов, 130 000, 100%), вспышка молнии."""
import subprocess, sys, io
from PIL import Image, ImageDraw, ImageFont
from storyboard166 import SHOTS
OUT_MP4 = "/Users/vladimirkalajcidi/reels_challenge/videos/66/monkey_hook1.mp4"
TS = [0.3, 0.7, 1.0, 1.3, 1.40, 1.6, 1.9, 2.2, 2.35, 2.45, 2.8, 3.3, 3.9, 4.45, 4.9, 5.25, 5.30, 5.37, 5.45, 5.6]
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

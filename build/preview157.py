"""Превью графики ролика 57 (слот 157) до рендера: кадры каждого графического плана в моменты ревилов
(холст целиком, с сеткой и субтитрами) -> work/preview157.jpg; check_zone по всем кадрам сетки."""
import sys
from PIL import Image
import render157 as r
import conv157 as g
from storyboard157 import SHOTS, GRID_KINDS

T = g.T
PICK = {"env": [6.62, 7.40, 8.70], "g1000": [11.50, 12.40, 13.30], "g500": [15.70, 16.30, 17.50],
        "g1250": [20.45, 21.20, 23.80], "loop": [30.60, 31.80, 32.75, 33.40], "g5050": [37.10, 38.00, 39.20, 40.20]}
tiles = []
for sh in SHOTS:
    if sh[2] not in GRID_KINDS:
        continue
    for t in PICK[sh[2]]:
        im = r.compose(t, sh, None).convert("RGB").crop((60, 200, 1020, 1660)).resize((320, 487))
        tiles.append(im)
cols = 7
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 320, rows * 487), "black")
for k, im in enumerate(tiles):
    sheet.paste(im, ((k % cols) * 320, (k // cols) * 487))
sheet.save("/Users/vladimirkalajcidi/reels_challenge/videos/57/work/preview157.jpg", quality=85)
if "zone" in sys.argv:
    print("вне зоны кадров:", g.check_zone([s for s in SHOTS if s[2] in GRID_KINDS]))

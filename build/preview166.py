"""Превью графики ролика 66: кадры сцен ape166 целиком (сетка + графика + субтитр) через render166.compose,
рамка зоны графики красным -> videos/66/work/preview166.jpg; плюс check_zone по всем кадрам графических планов."""
from PIL import Image, ImageDraw
import ape166 as P
import render166 as R
from storyboard166 import SHOTS
FR = [("inf", 2.10), ("inf", 3.30), ("tobe", 12.50), ("tobe", 13.45), ("tobe", 14.30), ("odds", 17.20),
      ("odds", 18.00), ("odds", 19.40), ("k130", 21.00), ("k130", 22.00), ("p100", 30.80), ("p100", 32.00)]
tiles = []
for kind, t in FR:
    sh = next(s for s in SHOTS if s[2] == kind)
    im = R.compose(t, sh, None)
    d = ImageDraw.Draw(im)
    d.rectangle((P.GX0, P.GY0, P.GX1, P.GY1), outline=(255, 0, 0, 255), width=2)
    d.text((130, 250), f"{kind} {t}", fill=(255, 255, 0, 255))
    tiles.append(im.crop((105, 238, 975, 1618)).convert("RGB").resize((348, 552)))
C = 6
sheet = Image.new("RGB", (348 * C, 552 * ((len(tiles) + C - 1) // C)))
for k, tl in enumerate(tiles):
    sheet.paste(tl, ((k % C) * 348, (k // C) * 552))
sheet.save("/Users/vladimirkalajcidi/reels_challenge/videos/66/work/preview166.jpg", quality=85)
if __name__ == "__main__":
    print("вне зоны кадров:", P.check_zone([s for s in SHOTS if s[2] in R.GRID_KINDS]))

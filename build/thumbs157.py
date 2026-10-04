"""Контакт-лист превью Pixabay из search157.log по запросам; id из used_ids.txt помечены USED.
python3 thumbs157.py <папка превью> <запрос> ... -> <папка>/sheet_<запрос>.jpg"""
import sys, os
from PIL import Image, ImageDraw, ImageFont
D = sys.argv[1]
used = {l.split("\t")[0].strip() for l in open("/Users/vladimirkalajcidi/reels_challenge/videos/57/used_ids.txt")}
rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "search157.log"))]
f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 20)
W, H, C = 320, 180, 4
for q in sys.argv[2:]:
    hits = list(dict.fromkeys((r[1], r[2], r[3]) for r in rows if r[0] == q))
    sheet = Image.new("RGB", (W * C, H * ((len(hits) + C - 1) // C)), "black")
    for k, (src, i, d) in enumerate(hits):
        try:
            im = Image.open(f"{D}/{src}_{i}.jpg").convert("RGB").resize((W, H))
        except Exception:
            continue
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, 200, 26), fill="black")
        dr.text((4, 2), f"{i} {d}{' USED' if i in used else ''}", fill="red" if i in used else "white", font=f)
        sheet.paste(im, ((k % C) * W, (k // C) * H))
    sheet.save(f"{D}/sheet_{q.replace(' ', '_')}.jpg", quality=75)
    print(q, len(hits))

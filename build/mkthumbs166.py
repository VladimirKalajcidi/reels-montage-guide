"""Mixkit (без API): превью клипов со страницы поиска -> контакт-лист; id уже использованных
в роликах 1–47 помечены USED. python3 mkthumbs166.py <page.html> <out.jpg>"""
import sys, os, re, subprocess
from PIL import Image, ImageDraw, ImageFont
page, out = sys.argv[1], sys.argv[2]
d = os.path.dirname(os.path.abspath(page))   # урок ролика 37
used = {l.split("\t")[0].strip() for l in open("/Users/vladimirkalajcidi/reels_challenge/videos/66/used_ids.txt")}
html = open(page, encoding="utf-8", errors="ignore").read()
ids = list(dict.fromkeys(re.findall(r"assets\.mixkit\.co/videos/(\d+)/\1-thumb-360-0\.jpg", html)))
f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 22)
W, H, C = 320, 180, 4
sheet = Image.new("RGB", (W * C, H * ((len(ids) + C - 1) // C)), "black")
for k, i in enumerate(ids):
    p = f"{d}/mk_{i}.jpg"
    if not os.path.exists(p):
        subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "-o", p, f"https://assets.mixkit.co/videos/{i}/{i}-thumb-360-0.jpg"])
    try:
        im = Image.open(p).convert("RGB").resize((W, H))
    except Exception:
        continue
    dr = ImageDraw.Draw(im)
    dr.rectangle((0, 0, 170, 28), fill="black")
    dr.text((4, 2), f"{i}{' USED' if i in used else ''}", fill="red" if i in used else "white", font=f)
    sheet.paste(im, ((k % C) * W, (k // C) * H))
sheet.save(out, quality=80)
print(os.path.basename(page), len(ids))

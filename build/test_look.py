import subprocess, sys, os
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *

SRC = "/Users/vladimirkalajcidi/reels_good/videos/new/source.mov"
OUT = "/Users/vladimirkalajcidi/reels_good/build/test"

# кадрирование A-roll: hflip + crop + scale в карточку 870x1379
# A1 (базовая, scale 0.895):  crop 972x1541 @ (75, 354) после hflip
# A2 (крупнее, scale 1.0):    crop 870x1379 @ (125, 361) после hflip
FRAMINGS = {"A1": (972, 1541, 75, 354), "A2": (870, 1379, 125, 361)}


def grab(t, framing="A1"):
    w, h, x, y = FRAMINGS[framing]
    p = f"/tmp/_g{os.getpid()}.png"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t),
                    "-i", SRC, "-frames:v", "1",
                    "-vf", f"hflip,crop={w}:{h}:{x}:{y},scale=870:1379:flags=lanczos", p],
                   check=True)
    im = Image.open(p).convert("RGB")
    os.remove(p)
    return im


def base_canvas():
    return Image.new("RGBA", (W, H), (0, 0, 0, 255))


# ---------- тест 1: A-roll + субтитр ----------
c = base_canvas()
x, y, w, h = CARD_A
c.paste(grab(12, "A1"), (x, y), rounded_mask(w, h, R_A))
f = font("sans", 56)
c.alpha_composite(text_layer((W, H), [
    dict(text="точно пять связей", font=f, xy=(540, 1305), anchor="mm",
         fill=WHITE, glow=WHITE, glow_r=14, glow_a=0.5),
]))
c.convert("RGB").save(f"{OUT}/t1_aroll.jpg", quality=92)

# ---------- тест 2: сетка + число ----------
c = base_canvas()
g = grid_canvas(w, h, phase=1.0)
c.paste(g, (x, y), rounded_mask(w, h, R_A))
items = [
    dict(text="ровно из", font=font("sans", 52), xy=(540, 800), anchor="mm",
         fill=WHITE, glow=WHITE, glow_r=12, glow_a=0.5),
    dict(text="6", font=font("sans", 300), xy=(540, 990), anchor="mm",
         fill=BLUE, glow=BLUE_GLOW, glow_r=38, glow_a=0.95),
    dict(text="человек", font=font("sans", 52), xy=(540, 1160), anchor="mm",
         fill=WHITE, glow=WHITE, glow_r=12, glow_a=0.5),
]
c.alpha_composite(text_layer((W, H), items))
c.convert("RGB").save(f"{OUT}/t2_grid_num.jpg", quality=92)

# ---------- тест 3: смесь начертаний + serif italic ----------
c = base_canvas()
c.paste(grid_canvas(w, h, phase=3.0), (x, y), rounded_mask(w, h, R_A))
sans = font("sans", 58)
ser = font("serif_it", 76)
tw = sans.getlength("это работает ")
items = [
    dict(text="это работает ", font=sans, xy=(540 - tw / 2, 900), anchor="lm",
         fill=WHITE, glow=WHITE, glow_r=12, glow_a=0.5),
    dict(text="всегда", font=ser, xy=(540 + tw / 2, 900), anchor="lm",
         fill=WHITE, glow=WHITE, glow_r=16, glow_a=0.6),
]
c.alpha_composite(text_layer((W, H), items))
c.convert("RGB").save(f"{OUT}/t3_mixed.jpg", quality=92)

# ---------- тест 4: карточка B на сетке ----------
c = base_canvas()
bx, by, bw, bh = CARD_B
c.paste(grid_canvas(bw, bh, phase=2.0), (bx, by), rounded_mask(bw, bh, R_B))
c.alpha_composite(text_layer((W, H), [
    dict(text="теорема", font=font("sans", 60), xy=(540, 520), anchor="mm",
         fill=WHITE, glow=WHITE, glow_r=14, glow_a=0.5),
    dict(text="Рамсея", font=font("serif_it", 130), xy=(540, 930), anchor="mm",
         fill=WHITE, glow=WHITE, glow_r=26, glow_a=0.7),
]))
c.convert("RGB").save(f"{OUT}/t4_cardb.jpg", quality=92)

print("ok")

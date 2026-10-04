"""Превью ролика 56 до рендера: на каждый план — кадры начала/середины/конца (+ события графики),
кадр с лицом — через тот же кроп и грейд, что prep_aroll. -> videos/56/work/preview156_<n>.jpg"""
import subprocess, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import render156 as r
from style import W, H

OUT = "/Users/vladimirkalajcidi/reels_challenge/videos/56/work"
f = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 26)
_face = {}


def face(kind, t):
    w, h, x, y = r.FRAMINGS[kind]
    vf = f"hflip,crop={w}:{h}:{x}:{y},scale={r.CW}:{r.CH}:flags=lanczos,{r.GRADE}"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", r.SRC, "-frames:v", "1", "-vf", vf,
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(r.CH, r.CW, 3)


def stock(si, t):
    p = r.prep_stock()[si]
    c = cv2.VideoCapture(p)
    c.set(cv2.CAP_PROP_POS_FRAMES, int((t - r.SHOTS[si][0]) * 30))
    ok, fr = c.read()
    return fr


def frames(times):
    out = []
    for t in times:
        si = r._shot_idx(t)
        s = r.SHOTS[si]
        fr = face(s[2], t) if s[2] in r.FACE_KINDS else stock(si, t) if s[2] == "stock" else None
        im = r.compose(t, s, fr).convert("RGB").resize((360, 640), Image.LANCZOS)
        ImageDraw.Draw(im).text((8, 6), f"{t:.2f} {s[2]}", fill=(255, 255, 0), font=f)
        out.append(im)
    return out


def sheet(ims, path, cols=8):
    sh = Image.new("RGB", (cols * 360, ((len(ims) + cols - 1) // cols) * 640), "black")
    for k, im in enumerate(ims):
        sh.paste(im, ((k % cols) * 360, (k // cols) * 640))
    sh.save(path, quality=85)


if __name__ == "__main__":
    ts = [float(x) for x in sys.argv[2:]]
    sheet(frames(ts), f"{OUT}/preview156_{sys.argv[1]}.jpg")
    print("ok", len(ts))

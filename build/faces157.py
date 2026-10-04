"""Лицо в исходнике ролика 57 (после hflip, масштаб 720×1280 — исходник 4K): Haar на 8 кадрах —
сверка с FRAMINGS (урок ролика 30: новая съёмка — кропы заново)."""
import cv2, numpy as np
SRC = "/Users/vladimirkalajcidi/reels_challenge/videos/57/source.mov"
OUT = "/Users/vladimirkalajcidi/reels_challenge/videos/57/work"
casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
cap = cv2.VideoCapture(SRC)
nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for n, f in enumerate(np.linspace(10, nf - 10, 8).astype(int)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read()
    fr = cv2.resize(cv2.flip(fr, 1), (720, 1280), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    det = casc.detectMultiScale(g, 1.15, 6, minSize=(80, 80))
    if n in (0, 3, 6):
        cv2.imwrite(f"{OUT}/f_{f//30}.jpg", fr)
    if len(det) == 0:
        print(f, "нет лица"); continue
    x, y, w, h = max(det, key=lambda r: r[2] * r[3])
    ee = eyes.detectMultiScale(g[y:y + int(h * 0.6), x:x + w], 1.12, 8, minSize=(18, 18))
    ey = y + int(np.mean([e[1] + e[3] / 2 for e in ee])) if len(ee) >= 2 else y + int(h * 0.42)
    print(f"{f/30:5.1f}s лицо x {x}…{x+w} (центр {x+w//2}) ширина {w} верх {y} низ {y+h} глаза y {ey}")

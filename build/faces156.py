"""Ролик 56: детектор лица (Haar) на 7 кадрах копии 720×1280 (исходник 2160×3840 — урок ролика 40),
кадр уже отзеркален (hflip, как в тракте A-roll). Печатает центр/ширину лица, линию глаз и яркость
стены слева / лица — для FRAMINGS и грейда."""
import subprocess, cv2, numpy as np
SRC = "/Users/vladimirkalajcidi/reels_challenge/videos/56/source.mov"
fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ec = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
rows = []
for t in (1, 7, 17, 30, 40, 51, 56):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", SRC, "-frames:v", "1",
                          "-vf", "hflip,scale=720:1280", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True).stdout
    img = np.frombuffer(raw, np.uint8).reshape(1280, 720, 3)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = sorted(fc.detectMultiScale(g, 1.1, 6, minSize=(150, 150)), key=lambda r: -r[2])
    if not len(f):
        print(t, "лица нет"); continue
    x, y, w, h = f[0]
    eyes = ec.detectMultiScale(g[y:y + h // 2, x:x + w], 1.1, 6)
    ey = y + (np.mean([e[1] + e[3] / 2 for e in eyes]) if len(eyes) else 0.42 * h)
    wall = g[100:600, 0:120].mean()
    face = g[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4].mean()
    rows.append((x + w / 2, ey, w))
    print(f"t={t:>2}  центр x={x + w/2:.0f}  глаза y={ey:.0f}  ширина={w}  низ лица={y+h}  стена={wall:.0f}  лицо={face:.0f}")
    if t == 17:
        cv2.imwrite("/Users/vladimirkalajcidi/reels_challenge/videos/56/work/face17.jpg", img)
a = np.array(rows)
print("медиана: x=%.0f глаза=%.0f ширина=%.0f" % tuple(np.median(a, 0)))

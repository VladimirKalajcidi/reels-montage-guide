"""Контакт-лист клипа ролика 57 в кропе карточки B с шагом 1с (урок ролика 37: ss выбирать по объекту в кадре,
а не по скану): python3 clipsheet157.py <id> <cx> <out.jpg> [cy]. Рамка — полоса субтитров карточки B."""
import sys, cv2, numpy as np
D = "/Users/vladimirkalajcidi/reels_challenge/videos/57/stock"
clip, cx, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cy = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
c = cv2.VideoCapture(f"{D}/stock_{clip}.mp4")
dur = c.get(cv2.CAP_PROP_FRAME_COUNT) / c.get(cv2.CAP_PROP_FPS)
tiles = []
step = max(1.0, round(dur / 15))
for s in np.arange(0, dur - 0.2, step):
    c.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
    ok, f = c.read()
    if not ok:
        break
    h, w = f.shape[:2]
    cw, ch = min(w, int(h * 1.3839)), min(h, int(w / 1.3839))
    x0 = int(round((w - cw) * cx)); y0 = int(min(max(h * cy - ch / 2, 0), h - ch))
    f = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (286, 207))
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(f, (50, 147), (236, 187), (0, 0, 255), 1)
    cv2.putText(f, f"{s:.0f}s {g.mean():.0f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    tiles.append(f)
while len(tiles) % 5:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]), [cv2.IMWRITE_JPEG_QUALITY, 75])

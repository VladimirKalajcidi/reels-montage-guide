"""Контакт-лист стока ролика 57 (слот 157): каждый сток-план в кропе карточки B (с учётом cx) —
кадры начала, середины и конца отрезка + полоса субтитров. python3 stockframes145.py <out.jpg>"""
import sys, cv2, numpy as np
from storyboard157 import SHOTS
D = "/Users/vladimirkalajcidi/reels_challenge/videos/57/stock"
rows = []
for t0, t1, kind, prm in SHOTS:
    if kind != "stock":
        continue
    c = cv2.VideoCapture(f"{D}/stock_{prm['clip']}.mp4")
    tiles = []
    for k in (0.02, 0.5, 0.98):
        c.set(cv2.CAP_PROP_POS_MSEC, (prm["ss"] + (t1 - t0) * k) * 1000)
        ok, f = c.read()
        h, w = f.shape[:2]
        cw, ch = min(w, int(h * 1.3839)), min(h, int(w / 1.3839))
        x0 = int(round((w - cw) * prm.get("cx", 0.5)))
        y0 = int(min(max(h * prm.get("cy", 0.5) - ch / 2, 0), h - ch))
        f = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (286, 207))
        cv2.rectangle(f, (50, 147), (236, 187), (0, 0, 255), 1)
        tiles.append(f)
    lab = np.zeros((207, 154, 3), np.uint8)
    cv2.putText(lab, prm["clip"], (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(lab, f"{t0:.2f}", (5, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    rows.append(np.hstack([lab] + tiles))
half = (len(rows) + 1) // 2
L, R = rows[:half], rows[half:] + [np.zeros_like(rows[0])] * (2 * half - len(rows))
cv2.imwrite(sys.argv[1], np.hstack([np.vstack(L), np.vstack(R)]), [cv2.IMWRITE_JPEG_QUALITY, 80])

"""Контакт-лист клипа ролика 66: кадры с шагом 1с в кропе карточки B (1.385:1, cx), подпись — секунда,
красная рамка — полоса субтитров. python3 stocksheet166.py <id>[:cx] ... -> videos/66/work/cs/c_<id>.jpg"""
import cv2, numpy as np, sys
D = "/Users/vladimirkalajcidi/reels_challenge/videos/66/stock"
OUT = "/Users/vladimirkalajcidi/reels_challenge/videos/66/work/cs"
for arg in sys.argv[1:]:
    clip, cx = (arg.split(":") + ["0.5"])[:2]
    cap = cv2.VideoCapture(f"{D}/stock_{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for s in range(0, min(32, int(n / fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps)); ok, fr = cap.read()
        if not ok: break
        h, w = fr.shape[:2]; cw = int(min(w, h * 1.3839)); ch = int(cw / 1.3839)
        x0 = int((w - cw) * float(cx)); y0 = (h - ch) // 2
        im = cv2.resize(fr[y0:y0 + ch, x0:x0 + cw], (277, 200))
        cv2.rectangle(im, (48, 152), (229, 181), (0, 0, 255), 1)
        cv2.putText(im, str(s), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{OUT}/c_{clip}.jpg", np.vstack([np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)]))
    print(clip, len(tiles))

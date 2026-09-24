"""Мини-QA ролика 7: яркие пиксели вне карточек и пересечения строк по плану."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard7 import OUT, SHOTS, CAPS, DUR
from style import CARD_A, CARD_B, FPS


def shot_at(t):
    for s in SHOTS:
        if s[0] <= t < s[1]:
            return s
    return SHOTS[-1]


def bright_outside():
    cap = cv2.VideoCapture(OUT)
    bad = 0
    total = int(DUR * FPS)
    for f in range(0, total, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        kind = shot_at(f / FPS)[2]
        rect = CARD_A if kind in ("A1", "A2") else CARD_B
        mask = np.zeros(img.shape[:2], np.uint8)
        x, y, w, h = rect
        mask[y:y + h, x:x + w] = 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((gray > 200) & (mask == 0)):
            bad += 1
    cap.release()
    return bad


def line_overlaps():
    # Layout in render7 uses at least 74px/82px vertical steps and max text sizes below that
    # for two-line blocks; this check catches accidental overlong three-line blocks.
    bad = 0
    for i in range(len(CAPS) - 1):
        if CAPS[i + 1][0] - CAPS[i][1] <= 0.30 and CAPS[i + 1][3] == CAPS[i][3]:
            s1 = max(sz for _, _, sz in CAPS[i][2])
            s2 = max(sz for _, _, sz in CAPS[i + 1][2])
            step = 94 if CAPS[i][3] == "T" else 82
            if step < (s1 + s2) * 0.50:
                bad += 1
    return bad


if __name__ == "__main__":
    print(f"bright_outside_card_frames={bright_outside()}")
    print(f"line_overlap_pairs={line_overlaps()}")

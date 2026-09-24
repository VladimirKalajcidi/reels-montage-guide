"""Автопроверки ролика 7 (START-HERE.md + delivery-specs.md §6).

1. Яркие пиксели вне карточки — скан готового mp4 через каждые 3 кадра.
2. Пересечение строк внутри блока — по реальным габаритам глифов (getbbox).
3. Общие пиксели слоя графики и слоя субтитров — рендер слоёв по отдельности.
Все три должны дать ноль.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render7 as R
from storyboard7 import OUT, SHOTS, CAPS, DUR, FACE_KINDS, shot_at, slot_for
from style import CARD_A, CARD_B, FPS, font


def bright_outside():
    cap = cv2.VideoCapture(OUT)
    bad, sample = 0, []
    total = int(DUR * FPS)
    for f in range(0, total, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        kind = shot_at(f / FPS)[2]
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.ones(gray.shape, bool)
        mask[y:y + h, x:x + w] = False
        if np.any((gray > 200) & mask):
            bad += 1
            if len(sample) < 8:
                sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample


def line_overlaps():
    """Реальные вертикальные габариты соседних строк блока не должны пересекаться."""
    bad, sample = 0, []
    by_block = {}
    for i in range(len(CAPS)):
        bid, pos, _, _, off, tot = R.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, off))
    for bid, rows in by_block.items():
        rows.sort()
        kind = shot_at(CAPS[rows[0][1]][0])[2]
        base = R.SLOTS[slot_for(kind)]["y"]
        spans = []
        for pos, i, off in rows:
            top, bot = 1e9, -1e9
            for txt, k, sz in CAPS[i][2]:
                b = font(R.KMAP[k], sz).getbbox(txt)
                # anchor "*m": глиф центрируется по вертикали относительно базы
                mid = (b[1] + b[3]) / 2
                top = min(top, base + off + b[1] - mid)
                bot = max(bot, base + off + b[3] - mid)
            spans.append((top, bot, i))
        for a in range(len(spans) - 1):
            if spans[a][1] > spans[a + 1][0]:
                bad += 1
                if len(sample) < 8:
                    sample.append((CAPS[spans[a][2]][2][0][0],
                                   CAPS[spans[a + 1][2]][2][0][0]))
    return bad, sample


def gfx_vs_text():
    """Слой графики и слой субтитров не должны иметь общих пикселей (альфа > 40)."""
    bad, sample = 0, []
    t = 0.0
    while t < DUR:
        t0, t1, kind, prm = shot_at(t)
        gl = R.graphics_layer(kind, prm, t - t0)
        if gl is not None:
            cl = R.caption_layer(t)
            if cl is not None:
                ga = np.array(gl.split()[3]) > 40
                ca = np.array(cl.split()[3]) > 40
                if np.any(ga & ca):
                    bad += 1
                    if len(sample) < 8:
                        sample.append(round(t, 2))
        t += 0.10
    return bad, sample


def caption_inside_card():
    """Буквы субтитра — внутри карточки с отступом 30px; свечение — внутри карточки.

    Возвращает (кадров с буквами за отступом, кадров со свечением за карточкой).
    """
    bad_in, bad_out, sample = 0, 0, []
    t = 0.0
    while t < DUR:
        kind = shot_at(t)[2]
        x, y, w, h = CARD_B if kind == "stock" else CARD_A
        cl = R.caption_layer(t)
        if cl is not None:
            a = np.array(cl.split()[3])
            inset = np.ones(a.shape, bool)
            inset[y + 30:y + h - 30, x + 30:x + w - 30] = False
            card = np.ones(a.shape, bool)
            card[y:y + h, x:x + w] = False
            if np.any((a > 128) & inset):
                bad_in += 1
                if len(sample) < 8:
                    sample.append(round(t, 2))
            if np.any((a > 20) & card):
                bad_out += 1
        t += 0.10
    return bad_in, bad_out, sample


def rhythm():
    lens = sorted(t1 - t0 for t0, t1, _, _ in SHOTS)
    face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k in FACE_KINDS)
    return dict(shots=len(SHOTS), avg=sum(lens) / len(lens),
                median=lens[len(lens) // 2], mn=lens[0], mx=lens[-1],
                face_pct=100 * face / DUR,
                first_cut=SHOTS[0][1])


if __name__ == "__main__":
    r = rhythm()
    print(f"планов={r['shots']} средняя={r['avg']:.2f}с медиана={r['median']:.2f}с "
          f"мин={r['mn']:.2f}с макс={r['mx']:.2f}с")
    print(f"лицо={r['face_pct']:.1f}%  первый рез={r['first_cut']:.2f}с")
    b, s = line_overlaps()
    print(f"line_overlap_pairs={b} sample={s}")
    b, s = gfx_vs_text()
    print(f"gfx_text_shared_frames={b} sample={s}")
    bi, bo, s = caption_inside_card()
    print(f"caption_past_inset_frames={bi} caption_past_card_frames={bo} sample={s}")
    if os.path.exists(OUT):
        b, s = bright_outside()
        print(f"bright_outside_card_frames={b} sample={s}")
    else:
        print("bright_outside_card_frames=(нет файла, запусти после рендера)")

"""Речевые автопроверки ролика 16: резы по границам слов + синхрон субтитров.

delivery-specs.md §6: «ни один рез не попал в середину слова» и
«субтитры совпадают с речью кадр в кадр» — обе проверяются по пословным
таймингам Whisper (`videos/16/source.json`), а не на слух.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard16 import SHOTS, CAPS, DUR

WORDS = []
for seg in json.load(open("/Users/vladimirkalajcidi/reels_good/videos/16/source.json"))["segments"]:
    for w in seg["words"]:
        WORDS.append((w["start"], w["end"], w["word"].strip()))

EPS = 0.02          # четверть кадра — допуск на округление таймингов


def cuts_inside_words():
    bad = []
    for t0, _, _, _ in SHOTS[1:]:
        for ws, we, txt in WORDS:
            if ws + EPS < t0 < we - EPS:
                bad.append((round(t0, 2), txt, round(ws, 2), round(we, 2)))
    return bad


def caption_drift():
    """Начало каждой фразы должно совпасть с началом какого-то слова."""
    starts = [w[0] for w in WORDS]
    bad = []
    for t0, t1, runs, _ in CAPS:
        d = min(abs(t0 - s) for s in starts)
        if d > 0.06:
            bad.append((round(t0, 2), "".join(r[0] for r in runs).strip(), round(d, 3)))
    return bad


def caption_gaps():
    """Речь без субтитра дольше 0.8с — либо осознанный план, либо дыра."""
    covered = [(a, b) for a, b, _, _ in CAPS]
    holes = []
    for ws, we, txt in WORDS:
        if not any(a - 0.05 <= ws and we <= b + 0.05 for a, b in covered):
            holes.append((round(ws, 2), txt))
    return holes


def caption_bounds():
    bad = []
    for t0, t1, runs, _ in CAPS:
        n = sum(len(r[0].split()) for r in runs)
        txt = "".join(r[0] for r in runs).strip()
        if n > 4 or n < 1:
            bad.append(("слов", txt, n))
        if any(ch in txt for ch in ".,!?«»:;"):
            bad.append(("пунктуация", txt, ""))
        if txt != txt.lower():
            bad.append(("капс", txt, ""))
    return bad


if __name__ == "__main__":
    c = cuts_inside_words()
    print(f"резов внутри слова: {len(c)} {c[:6]}")
    d = caption_drift()
    print(f"фраз не по началу слова: {len(d)} {d[:6]}")
    b = caption_bounds()
    print(f"нарушений «2-4 слова / строчные / без пунктуации»: {len(b)} {b[:6]}")
    h = caption_gaps()
    print(f"слов без субтитра: {len(h)}")
    if h:
        print("  ", h)

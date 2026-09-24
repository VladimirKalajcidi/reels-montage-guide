"""Подгонка границ слов в words_hook*.json ролика 35 под огибающую.

Та же процедура, что в snap35.py: для границ, по которым в блоке хука
проходит рез, берётся локальный минимум энергии и граница двигается туда.
Проверяется, что новая точка лежит внутри обоих соседних слов.

Границы взяты ровно на кадре реза, чтобы рез стоял на кадре, а не между.
Если слова стоят встык, двигаются обе границы; если между ними уже есть зазор
(hook1: «множеств,» 5.519 / «которые» 5.540), двигается только начало второго
слова — конец первого измерен верно и трогать его незачем.
"""
import json
import wave

import numpy as np

BASE = "/Users/vladimirkalajcidi/reels_good"

SNAP = {
    "hook1": {3.439: 102 / 30, 5.540: 167 / 30},
    "hook2": {1.496: 47 / 30, 3.775: 114 / 30},
}


def envelope(path):
    w = wave.open(path)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    sr = w.getframerate()
    n = int(sr * 0.010)
    return np.sqrt(np.convolve(a ** 2, np.ones(n) / n, "same")), sr


def main():
    for name, snaps in SNAP.items():
        env, sr = envelope(f"{BASE}/build/assets/_align35_{name}.wav")
        p = f"{BASE}/videos/35/words_{name}.json"
        words = json.load(open(p))
        for old, new in sorted(snaps.items()):
            j = next((k for k, w in enumerate(words)
                      if abs(w["start"] - old) < 1e-6), None)
            assert j is not None and j > 0, f"{name}: граница {old} не найдена"
            a, b = words[j - 1], words[j]
            assert a["start"] < new < b["end"], f"{name}: {new} вне соседних слов"
            joined = abs(a["end"] - old) < 1e-6
            if joined:
                a["end"] = round(new, 4)
            b["start"] = round(new, 4)
            print(f"{name}  {old:7.3f} -> {new:7.4f}  «{a['word']}» | «{b['word']}»  "
                  f"{'встык' if joined else 'зазор, двигается только начало'}  "
                  f"энергия {env[int(old*sr)]:.4f} -> {env[int(new*sr)]:.4f}")
        json.dump(words, open(p, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

"""Подгонка границ слов в words.json ролика 35 под огибающую сигнала.

align35.py растягивает выдачу whisper линейно на измеренный речевой отрезок,
поэтому границы ВНУТРИ отрезка плывут на десятые доли секунды. Резы ставятся
по этим границам, и промах слышен как «рез на атаке слова» (ролик 33).

Здесь для тех девяти границ, по которым в раскадровке проходит рез, берётся
локальный минимум энергии в окне +-80мс и граница двигается туда. Проверяется,
что новая точка лежит внутри обоих соседних слов, — то есть это уточнение
границы, а не переразметка речи.
"""
import json
import wave

import numpy as np

WORDS = "/Users/vladimirkalajcidi/reels_good/videos/35/words.json"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align35.wav"

# граница whisper -> замеренный минимум огибающей
SNAP = {1.894: 1.954, 5.839: 5.766, 13.411: 13.481, 15.184: 15.105,
        21.284: 21.300, 25.201: 25.245, 29.480: 29.400, 36.516: 36.574,
        38.274: 38.262}


def envelope():
    w = wave.open(WAV)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    sr = w.getframerate()
    n = int(sr * 0.010)
    return np.sqrt(np.convolve(a ** 2, np.ones(n) / n, "same")), sr


def main():
    env, sr = envelope()
    words = json.load(open(WORDS))
    for old, new in sorted(SNAP.items()):
        i = next((k for k, w in enumerate(words)
                  if abs(w["end"] - old) < 1e-6 and k + 1 < len(words)
                  and abs(words[k + 1]["start"] - old) < 1e-6), None)
        assert i is not None, f"граница {old} не найдена"
        a, b = words[i], words[i + 1]
        assert a["start"] < new < b["end"], f"{new} вне соседних слов"
        a["end"] = b["start"] = round(new, 3)
        print(f"{old:7.3f} -> {new:7.3f}  «{a['word']}» | «{b['word']}»  "
              f"энергия {env[int(old*sr)]:.4f} -> {env[int(new*sr)]:.4f}")
    json.dump(words, open(WORDS, "w"), ensure_ascii=False, indent=1)
    print("границ поправлено:", len(SNAP))


if __name__ == "__main__":
    main()

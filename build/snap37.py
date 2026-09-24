"""Подгонка границ слов в words.json ролика 37 под огибающую сигнала.

align37.py растягивает выдачу whisper линейно на измеренный речевой отрезок,
поэтому границы ВНУТРИ отрезка плывут на десятые доли секунды. Резы ставятся
по этим границам, и промах слышен как «рез на атаке слова».

Здесь правятся семь границ, по которым в раскадровке проходит рез: берётся
локальный минимум энергии (или измеренная атака слова) и проверяется, что
новая точка лежит внутри соседних слов, — то есть это уточнение границы,
а не переразметка речи.

SHARED — общая граница двух слов (конец одного = начало другого).
ENDS / STARTS — односторонние правки: между словами была пауза, и промахнулась
только одна её сторона.
"""
import json
import wave

import numpy as np

WORDS = "/Users/vladimirkalajcidi/reels_good/videos/37/words.json"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align37.wav"

SHARED = {1.823: 1.800, 10.120: 10.100, 18.346: 18.400, 42.658: 42.733,
          51.455: 51.533}
ENDS = {21.321: 21.300}     # «-2,» — хвост слова whisper поставил позже тишины
STARTS = {6.641: 6.667,     # «вторая» — атака на 0.026с позже
          21.484: 21.345,   # «ведь» — атака слышна на 0.139с раньше
          28.022: 28.100}   # «которые» — атака на 0.078с позже


def envelope():
    w = wave.open(WAV)
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    sr = w.getframerate()
    n = int(sr * 0.010)
    return np.sqrt(np.convolve(a ** 2, np.ones(n) / n, "same")), sr


def main():
    env, sr = envelope()
    words = json.load(open(WORDS))
    e = lambda t: env[int(t * sr)]

    for old, new in sorted(SHARED.items()):
        i = next((k for k, w in enumerate(words)
                  if abs(w["end"] - old) < 1e-6 and k + 1 < len(words)
                  and abs(words[k + 1]["start"] - old) < 1e-6), None)
        if i is None and any(abs(w["end"] - new) < 1e-6 for w in words):
            continue                                    # правка уже применена
        assert i is not None, f"общая граница {old} не найдена"
        a, b = words[i], words[i + 1]
        assert a["start"] < new < b["end"], f"{new} вне соседних слов"
        a["end"] = b["start"] = round(new, 3)
        print(f"общая  {old:7.3f} -> {new:7.3f}  «{a['word']}» | «{b['word']}»  "
              f"энергия {e(old):.4f} -> {e(new):.4f}")

    for old, new in sorted(ENDS.items()):
        i = next((k for k, w in enumerate(words) if abs(w["end"] - old) < 1e-6), None)
        if i is None:
            continue                                    # правка уже применена
        assert words[i]["start"] < new <= words[i + 1]["start"]
        words[i]["end"] = round(new, 3)
        print(f"конец  {old:7.3f} -> {new:7.3f}  «{words[i]['word']}»  "
              f"энергия {e(old):.4f} -> {e(new):.4f}")

    for old, new in sorted(STARTS.items()):
        i = next((k for k, w in enumerate(words) if abs(w["start"] - old) < 1e-6), None)
        if i is None:
            continue                                    # правка уже применена
        assert words[i - 1]["end"] <= new < words[i]["end"]
        words[i]["start"] = round(new, 3)
        print(f"начало {old:7.3f} -> {new:7.3f}  «{words[i]['word']}»  "
              f"энергия {e(old):.4f} -> {e(new):.4f}")

    json.dump(words, open(WORDS, "w"), ensure_ascii=False, indent=1)
    print("границ поправлено:", len(SHARED) + len(ENDS) + len(STARTS))


if __name__ == "__main__":
    main()

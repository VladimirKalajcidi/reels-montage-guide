"""Подгонка границ слов в words_hook*.json ролика 37 под огибающую.

align37hooks.py растягивает выдачу whisper линейно на измеренный речевой
отрезок, поэтому границы ВНУТРИ отрезка плывут. Резы блока хука ставятся по
этим границам, и `qa37hooks.cuts_inside_word` честно ловит промах, хотя на
самой дорожке в этих точках тишина.

Правятся две границы, по которым проходит рез:
* hook1 «вероятности,» 1.978 -> 1.900 и «но» 1.978 -> 1.998 — silencedetect
  показывает паузу 1.897-1.998, whisper растянул слово на всю паузу;
* hook2 «и» 1.355 -> 1.310 и «знаешь» 1.355 -> 1.360 — энергия на 1.30
  0.0079, на 1.3333 0.0034, на 1.3667 уже 0.0415 (атака «знаешь»).
"""
import json
import wave

import numpy as np

BASE = "/Users/vladimirkalajcidi/reels_good"

# файл -> список правок (индекс не используется, ищем по значению)
ENDS = {"hook1": {1.978: 1.900}, "hook2": {1.355: 1.310}}
STARTS = {"hook1": {1.978: 1.998}, "hook2": {1.355: 1.360}}


def envelope(name):
    w = wave.open(f"{BASE}/build/assets/_align37_{name}.wav")
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    sr = w.getframerate()
    n = int(sr * 0.010)
    return np.sqrt(np.convolve(a ** 2, np.ones(n) / n, "same")), sr


def main():
    for name in ("hook1", "hook2"):
        env, sr = envelope(name)
        e = lambda t: float(env[int(t * sr)])
        path = f"{BASE}/videos/37/words_{name}.json"
        words = json.load(open(path))
        for old, new in ENDS[name].items():
            i = next((k for k, w in enumerate(words) if abs(w["end"] - old) < 1e-6), None)
            if i is None:
                continue                                   # правка уже применена
            assert words[i]["start"] < new, f"{new} вне слова"
            words[i]["end"] = round(new, 3)
            print(f"{name} конец  {old:6.3f} -> {new:6.3f}  «{words[i]['word']}»  "
                  f"энергия {e(old):.4f} -> {e(new):.4f}")
        for old, new in STARTS[name].items():
            i = next((k for k, w in enumerate(words) if abs(w["start"] - old) < 1e-6), None)
            if i is None:
                continue
            assert new < words[i]["end"], f"{new} вне слова"
            words[i]["start"] = round(new, 3)
            print(f"{name} начало {old:6.3f} -> {new:6.3f}  «{words[i]['word']}»  "
                  f"энергия {e(old):.4f} -> {e(new):.4f}")
        json.dump(words, open(path, "w"), ensure_ascii=False, indent=1)
    print("готово")


if __name__ == "__main__":
    main()

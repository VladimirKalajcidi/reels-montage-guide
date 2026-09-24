"""Пословные тайминги ролика 10 (слот 43 — см. videos/10/status.md).

whisper medium, ru, word_timestamps по всему дублю (videos/10/work/source.json).
Сплошной прогон здесь не дрейфует: все 20 пауз silencedetect совпадают с паузами
транскрипта. Результат — videos/10/words.json, дальше границы уточняет snap43.py.
"""
import json

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/10"
d = json.load(open(f"{ROOT}/work/source.json"))
words = [dict(start=round(w["start"], 3), end=round(w["end"], 3), word=w["word"].strip())
         for s in d["segments"] for w in s["words"]]
json.dump(words, open(f"{ROOT}/words.json", "w"), ensure_ascii=False, indent=1)
print("слов:", len(words))

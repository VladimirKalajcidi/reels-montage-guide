"""Пословные тайминги ролика 66 (слот 166 = 100 + номер папки).

OpenAI Whisper API (whisper-1, ru, verbose_json, word) по work/voice16k.wav (.mov 111 МБ > 25 МБ):
videos/66/work/source.json → videos/66/words.json, дальше начала слов уточняет realign166.py.

Текст сверен gpt-4o-transcribe (work/check_4o.txt) и вырезками (cuts166.py/.log):
- «Но бесконечность времени ограничена» (25.8–27.7) — так слышат обе модели целиком, но по смыслу фраза
  противоречит ролику (урок 36). Хвост 26.72–27.80 отдельно все три модели (whisper-1, gpt-4o, gpt-4o-mini) слышат
  «время не ограничено»; начало 25.70–26.76 — 4o «Но о/в бесконечности». Принято «Но в бесконечности время не
  ограничено»: «в» 25.99–26.04 (провал огибающей между «Но» и «бес»), «время/не/ограничено» — по вырезке d2;
- «Возраст Вселенной не хватит» — whisper-1 и 4o одинаково, оставлено как сказано.
"""
import json

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/66"
d = json.load(open(f"{ROOT}/work/source.json"))
src = [dict(start=round(w["start"], 3), end=round(w["end"], 3), word=w["word"].strip().strip("«»\"")) for w in d["words"]]
assert len(src) == 101
assert [w["word"] for w in src[65:69]] == ["Но", "бесконечность", "времени", "ограничена"], src[65:69]
FIX = [dict(start=25.78, end=25.99, word="Но"), dict(start=25.99, end=26.04, word="в"),
       dict(start=26.04, end=26.76, word="бесконечности"), dict(start=26.76, end=27.02, word="время"),
       dict(start=27.02, end=27.12, word="не"), dict(start=27.12, end=27.68, word="ограничено")]
out = src[:65] + FIX + src[69:]
out = [w for w in out if w["word"] not in ("–", "-", "—", "")]
json.dump(out, open(f"{ROOT}/words.json", "w"), ensure_ascii=False, indent=1)
print("слов:", len(out))

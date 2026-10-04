"""Ролик 56 (слот 156): words.json из сырой расшифровки whisper-1 (videos/56/words_raw.json).

whisper-1, ru, verbose_json, word — по videos/56/voice16k.wav (.mov 157 МБ > 25 МБ лимита API).
Текст сверен gpt-4o-transcribe (полный файл + вырезка 14.4–16.8с) и повторной расшифровкой вырезок
(cuts156.py / cuts156.log):
  «друг по другому» → «друг под другом» (gpt-4o на вырезке; whisper: «по другому» / «по другам»)
  «УГЭ» → «ОГЭ» (обе модели ошиблись: «УГЭ», «фуге»; по смыслу — экзамен 9 класса)
  «70» + «ти» → «70-ти» (whisper разрезал суффикс, урок ролика 31)
  «18» + «5» → «18,5» (одно число; в субтитр не идёт — его несёт синее R5b)
Границы слов уточняет snap156.py.
"""
import json

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/56"
FIX = {38: ("по", "под"), 39: ("другому", "другом"), 132: ("УГЭ", "ОГЭ")}   # индексы words_raw
MERGE = [(77, 78, "70", "ти", "70-ти"), (89, 90, "18", "5", "18,5")]          # индексы words_raw


def main():
    raw = json.load(open(f"{ROOT}/words_raw.json"))["words"]
    w = [dict(word=x["word"], start=round(x["start"], 3), end=round(x["end"], 3)) for x in raw]
    for k, (old, new) in FIX.items():
        assert w[k]["word"] == old, (k, w[k]["word"], old)
        w[k]["word"] = new
    for a, b, wa, wb, new in sorted(MERGE, reverse=True):
        assert (w[a]["word"], w[b]["word"]) == (wa, wb) and b == a + 1
        w[a] = dict(word=new, start=w[a]["start"], end=w[b]["end"])
        del w[b]
    assert len(w) == 156
    json.dump(w, open(f"{ROOT}/words.json", "w"), ensure_ascii=False, indent=1)
    print("align156: words.json —", len(w), "слов")


if __name__ == "__main__":
    main()

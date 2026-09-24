"""Пословные тайминги ролика 29 (Санкт-Петербургский парадокс).

Whisper small используется как быстрый акустический выравниватель. Три места,
где модель систематически путает математическую лексику, исправляются по
сценарию без изменения измеренных границ слов: дроби вероятностей,
«посчитать» и «гигантского»/«санкт-петербургским».
"""
import json
import os

SRC_JSON = "/tmp/reels29_whisper_small/source.json"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/29/words.json"


def replacement(start, end, items):
    return [dict(start=a, end=b, word=w) for a, b, w in items]


def main():
    if not os.path.exists(SRC_JSON):
        raise SystemExit(
            "Сначала запусти whisper: whisper videos/29/source.mov --model small "
            "--language ru --word_timestamps True --output_format json "
            "--output_dir /tmp/reels29_whisper_small"
        )
    raw = json.load(open(SRC_JSON))
    words = [dict(start=round(w["start"], 3), end=round(w["end"], 3),
                  word=w["word"].strip())
             for s in raw["segments"] for w in s.get("words", [])]

    fixes = [
        replacement(17.60, 22.24, [
            (17.600, 18.020, "вероятность"),
            (18.020, 18.440, "выигрыша"),
            (18.440, 18.600, "2"),
            (18.600, 18.960, "рубля"),
            (18.960, 19.200, "одна"),
            (19.200, 19.620, "вторая"),
            (19.760, 20.220, "4"),
            (20.220, 20.480, "одна"),
            (20.600, 20.980, "четвёртая"),
            (21.000, 21.420, "8"),
            (21.500, 21.780, "одна"),
            (21.860, 22.240, "восьмая"),
        ]),
        replacement(22.64, 23.42, [
            (22.640, 22.960, "если"),
            (22.960, 23.420, "посчитать"),
        ]),
        replacement(27.96, 28.82, [
            (27.960, 28.500, "бесконечно"),
            (28.500, 28.820, "много"),
        ]),
        replacement(40.04, 41.76, [
            (40.040, 40.460, "гигантского"),
            (40.460, 40.900, "выигрыша"),
            (40.900, 41.440, "невероятно"),
            (41.440, 41.760, "мала"),
        ]),
        replacement(46.20, 47.84, [
            (46.200, 47.180, "санкт-петербургским"),
            (47.180, 47.840, "парадоксом"),
        ]),
    ]
    ranges = [(17.60, 22.24), (22.64, 23.42), (27.96, 28.82),
              (40.04, 41.76), (46.20, 47.84)]
    kept = [w for w in words if not any(a <= w["start"] < b for a, b in ranges)]
    words = sorted(kept + [w for group in fixes for w in group], key=lambda w: w["start"])
    with open(OUT, "w") as f:
        json.dump(words, f, ensure_ascii=False, indent=1)
    print("слов:", len(words), "->", OUT)


if __name__ == "__main__":
    main()

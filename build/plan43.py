"""Монтажный лист ролика 10 («задача Монти Холла») из раскадровки storyboard43.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard43 import SHOTS, CAPS, DUR, NAMES, STOCK, DOOR_STATES, FACE_KINDS

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k in FACE_KINDS)
lens = sorted(t1 - t0 for t0, t1, _, _ in SHOTS)

print("# Монтажный лист — «задача Монти Холла»\n")
print(f"Исходник: `videos/10/source.mov` · 720×1280 → холст 1080×1920 · 30 fps · {DUR:.2f}с")
print("Готовый ролик: `videos/10/monty_hall_edit.mp4` · скрипты: слот 43 "
      "(`storyboard43.py`, `doors43.py`, `render43.py`, `sfx43.py`, `qa43.py`)\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens[len(lens)//2]:.2f}с · мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%** · первый рез {SHOTS[1][0]:.2f}с")
print(f"**Субтитров:** {len(CAPS)} фраз · синих чисел 3 (2/3 ×3) · белых 2 (50/50, 1/3) · "
      "R6 нет (v3: текст только внизу) · R3 нет · R8 нет\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(SHOTS, 1):
    rec, desc = NAMES[kind]
    if kind == "stock":
        desc = STOCK[prm["clip"]]
    elif kind == "doors":
        desc = DOOR_STATES[prm["state"]]
    elif kind == "num":
        desc = f"{'синее' if prm['blue'] else 'белое'} число {prm['digits']}"
    if prm.get("topword"):
        rec += " + R6"
        desc += f"; над головой «{prm['topword']}»"
    txt = " / ".join(" ".join(r[0] for r in runs) for (a, b, runs, _) in CAPS if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt or '—'} |")

print("\n## Источники стоковых материалов")
print("Видео — Pixabay (Pixabay Content License, коммерческое использование без атрибуции), "
      "через API, `https://pixabay.com/videos/id-<id>/`. Все клипы лежат в `videos/10/stock/`, "
      "изолированы под этот ролик, каждый использован ровно один раз. Длительность файлов "
      "сверена с ответом API, кадры просмотрены, склеек внутри используемых отрезков нет.")
for k, v in STOCK.items():
    print(f"- `stock_{k}.mp4` — {v}")
print("\n## Субтитры (v3, 2026-09-24)")
print("SF Pro Expanded Black капсом (ось ширины 150, вес 1000), одна фраза в 1–2 строки, кегль 62, "
      "только внизу карточки; каждое слово всплывает из размытия на своём тайминге (0.23с); "
      "тень вместо свечения (+3/+4px, размытие 2, 75%). Цифры графики тем же шрифтом без свечения. "
      "Цветные слова: ЯРОСТИ, НЕ ТАК, ПОДВОДИТ — красный #FF3B30; ПУСТУЮ, УВЕРЕННО, ВЕРОЯТНОСТЬ — "
      "бирюзовый #2DE1C2. Вставки светлые, без притемнения. Прошлые версии — `monty_hall_edit_v1.mp4`, `monty_hall_edit_v2.mp4`.")
print("\n## Собственная графика")
print("Сцена «три двери» на вертикальной сетке (`doors43.py`): белые контурные двери с номерами, "
      "силуэт козы (альфа эмодзи, только белый), лаймовая метка выбора под дверью 1 — "
      "единственный элемент #C3DB4E, скобка под дверями 2–3, числа 1/3 (белое) и 2/3 (синее).")
print("\n## Звук")
print("Голос без резов; whoosh на каждом резе (0.040 от пика), impact на синих 2/3 (0.070), "
      "soft tick на дверях/метке/белых числах (0.030), мягкий «пуф» появления текста ≈ раз в 3с (−16 дБ от RMS голоса, ~0.9–4 кГц); музыка `audios/song1.mp3` (−19 dB от RMS "
      "голоса, петля с кроссфейдом 0.25с, фейды 0.8/2.0с); loudnorm I=-14 TP=-1.5 LRA=7 + "
      "доводка громкости по замеру готового файла (см. qa-report.md).")

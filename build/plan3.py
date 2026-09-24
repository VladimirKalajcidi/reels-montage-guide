"""Генерирует монтажный лист ролика 2 («актуарий») из раскадровки."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard3 import SHOTS, CAPS, DUR, LABELS, NUMBERS

NAMES = {
    "A1": ("R1", "лицо, говорящая голова"),
    "stock": ("R2", "сток-видео в карточке B"),
    "num": ("R4+R5", "число-ревил на сетке"),
}

STOCK_DESC = {
    "38512_signing_document": "подпись документа (Mixkit 38512)",
    "13231_team_office": "команда в офисе планирует проект (Mixkit 13231)",
    "4401_crowd_street": "толпа людей на перекрёстке (Mixkit 4401)",
    "4547_meeting_table": "рабочая встреча за столом (Mixkit 4547)",
    "46533_signing_paperwork": "подписание документов (Mixkit 46533)",
    "4623_teacher_equations": "преподаватель объясняет уравнения (Mixkit 4623)",
    "4619_blackboard_formulas": "решение формул у доски (Mixkit 4619)",
    "28321_frustrated_student": "расстроенный школьник за конспектами (Mixkit 28321)",
    "42656_woman_office": "сосредоточенная работа в офисе (Mixkit 42656)",
    "314_exit_building_door": "выход через дверь бизнес-центра (Mixkit 314)",
    "34124_calculator_tallying": "бизнесмен считает на калькуляторе (Mixkit 34124)",
    "4533_calculator_accounts": "расчёты на калькуляторе (Mixkit 4533)",
}

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k == "A1")
lens = [t1 - t0 for t0, t1, _, _ in SHOTS]
lens_sorted = sorted(lens)

print("# Монтажный лист — «актуарий» (профессии, требующие математики)\n")
print(f"Исходник: `videos/2/source.mov` · 720×1280 → холст 1080×1920 · 30 fps · {DUR:.2f}с")
print(f"Готовый ролик: `videos/2/aktuary_edit.mp4`\n")
print("Скрипты сборки этого ролика используют суффикс **3** (не 2) — слот `*2.py` в `build/` "
      "занят незаконченным роликом другого проекта (reels_good), см. `videos/2/status.md`.\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens_sorted[len(lens_sorted)//2]:.2f}с · мин {lens_sorted[0]:.2f}с · макс {lens_sorted[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%**")
print(f"**Субтитров:** {len(CAPS)} фраз · **чисел-ревилов:** {len(NUMBERS)} "
      f"(синих: {sum(1 for n in NUMBERS if n['blue'])})\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(SHOTS, 1):
    rec, desc = NAMES[kind]
    if kind == "stock":
        desc = STOCK_DESC.get(prm["clip"], prm["clip"])
        for lt0, lt1, ltxt in LABELS:
            if abs(lt0 - t0) < 0.01:
                desc += f" + подпись «{ltxt}»"
    if kind == "num":
        desc = f'{prm["digits"]} / {prm["label"]}' + (" (синее)" if prm.get("blue") else " (белое)")
    txt = "" if kind == "num" else " / ".join(
        "".join(r[0] for r in runs).strip() for (a, b, runs, _) in CAPS if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt} |")

print("\n## Источники стоковых материалов")
print("Видео — Mixkit (свободная лицензия, коммерческое использование без атрибуции), "
      "прямые ссылки `assets.mixkit.co/videos/<id>/<id>-<res>.mp4`. Все материалы лежат "
      "в `videos/2/stock/`, изолированы под этот ролик, каждый клип использован ровно один раз.")
print("\n## Звук")
print("Голос без резов, whoosh на каждом резе картинки (0.040 от пика), impact на ревиле "
      "синего числа (0.070), музыка `audios/song2.mp3` (−19 dB от RMS голоса, петля с "
      "кроссфейдом 0.25с, фейды 0.8/2.0с), финальный loudnorm I=-14 TP=-1.5 LRA=7.")

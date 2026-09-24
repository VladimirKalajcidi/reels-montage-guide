"""Генерирует монтажный лист ролика 1 из раскадровки."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard1 import SHOTS, CAPS, DUR, LABELS, MONEY

NAMES = {
    "A1": ("R1", "лицо, говорящая голова"),
    "stock": ("R2", "сток-видео в карточке B"),
    "photo": ("R2 (фото)", "сток-фото в карточке B — по правке на этот ролик"),
    "money": ("R5a", "единственная цифра-ревил ролика"),
}

STOCK_DESC = {
    "01_glass_tower": "стеклянная башня, элитный офис (Pexels 34724848)",
    "02_nyse_wallstreet": "фасад Нью-Йоркской биржи (Pexels 4319342)",
    "03_trading_desk_monitors": "трейдинг-деск, несколько мониторов с графиками (Pexels 38055932)",
    "05_chart_growth": "биржевой график на экране (Pexels 38670063)",
    "06_team_data_meeting": "команда обсуждает данные и графики (Pexels 3246669)",
    "06b_team_graph_photo": "фото: команда смотрит на бизнес-график (Pexels 7693700)",
    "08_man_stressed_chart": "человек в напряжении смотрит на график (Pexels 6799669)",
    "09b_formula_glass_photo": "фото: формула на стекле (Pexels 9301830)",
    "10_student_blackboard_math": "школьник решает пример у доски (Pexels 8617059)",
    "11b_boy_math_photo": "фото: мальчик решает пример в тетради (Pexels 6256077)",
    "12_chart_loss": "падающий/красный биржевой график (Pexels 38736274)",
    "13_running_catchup": "мужчина быстро идёт мимо стеклянного фасада (Pexels 7868438)",
}

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k == "A1")
lens = [t1 - t0 for t0, t1, _, _ in SHOTS]
lens_sorted = sorted(lens)

print("# Монтажный лист — «квонт» (профессии, требующие математики)\n")
print(f"Исходник: `videos/1/source.mov` · 720×1280 → холст 1080×1920 · 30 fps · {DUR:.2f}с")
print(f"Готовый ролик: `videos/1/kvant_edit.mp4`\n")
print("**Правка стиля для этого ролика** (по прямой просьбе, в отличие от общего брендкита):")
print("минимум собственной графики (одна цифра-ревил на весь ролик вместо инфографики на сетке), "
      "максимум стоковых вставок — видео **и фото** (фото — осознанное исключение из правила «только видео»).\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens_sorted[len(lens_sorted)//2]:.2f}с · мин {lens_sorted[0]:.2f}с · макс {lens_sorted[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%**")
print(f"**Субтитров:** {len(CAPS)} фраз\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(SHOTS, 1):
    rec, desc = NAMES[kind]
    if prm.get("clip"):
        desc = STOCK_DESC.get(prm["clip"], prm["clip"])
        for lt0, lt1, ltxt in LABELS:
            if abs(lt0 - t0) < 0.01:
                desc += f" + подпись «{ltxt}»"
    if kind == "money":
        desc = f'{MONEY["digits"]} {MONEY["sub"]}'
    txt = "" if kind == "money" else " / ".join(
        "".join(r[0] for r in runs).strip() for (a, b, runs, _) in CAPS if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt} |")

print("\n## Источники стоковых материалов")
print("Видео — Pexels (свободная лицензия, коммерческое использование без атрибуции), "
      "прямые ссылки `pexels.com/download/video/<id>/`. Фото — Pexels CDN "
      "`images.pexels.com/photos/<id>/...`. Все материалы лежат в `videos/1/stock/`, "
      "изолированы под этот ролик.")
print("\n## Звук")
print("Голос без резов, whoosh на каждом резе картинки (0.040 от пика), один impact на ревиле "
      "суммы (0.070), музыка `audios/song1.mp3` (−19 dB от RMS голоса, петля с кроссфейдом 0.25с, "
      "фейды 0.8/2.0с), финальный loudnorm I=-14 TP=-1.5 LRA=7.")

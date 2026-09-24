"""Монтажный лист ролика 7."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard7 import SHOTS, CAPS, DUR, CUTS, STOCK_NAMES

NAMES = {
    "A1": ("R1", "лицо, базовая крупность"),
    "A2": ("R1", "лицо, крупнее"),
    "stock": ("R2", "стоковая видео-вставка"),
    "rule37": ("R4+R5b", "сетка + синее число 37%"),
    "grid_rule": ("R10", "сетка: правило в три шага"),
    "grid_pick": ("R10", "сетка: первый вариант лучше всех предыдущих"),
    "grid_early": ("R10", "сетка: риск остановиться слишком рано"),
    "grid_late": ("R10", "сетка: риск смотреть слишком долго"),
    "balance": ("R10", "сетка: баланс двух рисков"),
    "formula": ("R4+R5b", "сетка: 1/e и e≈2,7"),
    "grid_max": ("R4+R5b", "сетка: максимум 37%, не 100%"),
}

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k in ("A1", "A2"))
lens = sorted(t1 - t0 for t0, t1, _, _ in SHOTS)

print("# Монтажный лист — «правило 37% при выборе лучшего варианта»\n")
print("Исходник: `videos/7/source.mov` · 1080×1920 · 30 fps · 82.36с")
print("Очищенная речь: `videos/7/source_cut.mp4`")
print("Готовый ролик: `videos/7/secretary_edit.mp4`\n")
print("**Вырезанные паузы:** " + ", ".join(f"{a:.2f}–{b:.2f}" for a, b in CUTS))
print(f"**Длительность после резов:** {DUR:.2f}с\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens[len(lens)//2]:.2f}с · мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%**")
print(f"**Субтитров:** {len(CAPS)} фраз\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(SHOTS, 1):
    rec, desc = NAMES[kind]
    if kind == "stock":
        desc = STOCK_NAMES.get(prm["clip"], desc)
    if prm.get("topword"):
        desc += f" · крупное слово «{prm['topword']}»"
    txt = " / ".join("".join(r[0] for r in runs).strip()
                     for a, _, runs, _ in CAPS if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt} |")

"""Генерирует монтажный лист ролика 9 из раскадровки."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard9 import SHOTS, CAPS, DUR, STOCK_NAMES, FACE_KINDS

NAMES = {
    "A1": ("R1", "лицо, базовая крупность"),
    "A2": ("R1", "лицо, крупнее (смена дубля)"),
    "stock": ("R2", "стоковая видео-вставка в карточке B"),
    "five": ("R4+R5a", "сетка: белое «5»"),
    "formula": ("R4+R10+R5b", "сетка: формула e^(iπ)+1 строится по речи, «0» синим"),
    "constants": ("R4+R10", "сетка: пять констант e i π 0 1 каскадом"),
    "e_symbol": ("R4+R10", "сетка: символ e + кривая роста"),
    "growth": ("R4+R10", "сетка: экспоненциальная кривая растёт"),
    "imaginary": ("R4+R10", "сетка: мнимая ось, точка поворачивается на 90°"),
    "pi_circle": ("R4+R10", "сетка: окружность разворачивается в 3,14 диаметра"),
    "zero_one": ("R4+R5a", "сетка: белые 0 и 1"),
    "converge": ("R4+R10", "сетка: три круга сходятся в диаграмму Венна"),
}

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k in FACE_KINDS)
lens = sorted(t1 - t0 for t0, t1, _, _ in SHOTS)

print("# Монтажный лист — «формула Эйлера»\n")
print(f"Исходник: `videos/9/source.mov` · 1080×1920 · 30 fps · {DUR:.2f}с")
print(f"Готовый ролик: `videos/9/euler_edit.mp4`\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens[len(lens)//2]:.2f}с · мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%**")
print(f"**Субтитров:** {len(CAPS)} фраз\n")
print("Зоны на планах с графикой: субтитры y 300…600, графика y 700…1560 "
      "(`brand-kit.md §5`). Где смысл несёт графика — субтитра нет.\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(SHOTS, 1):
    rec, desc = NAMES[kind]
    if prm.get("clip"):
        desc = STOCK_NAMES[prm["clip"]]
    txt = " / ".join("".join(r[0] for r in runs).strip()
                     for (a, b, runs) in CAPS if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} "
          f"| {txt or '— (говорит графика)'} |")

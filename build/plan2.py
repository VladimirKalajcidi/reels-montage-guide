"""Генерирует монтажный лист из раскадровки.

Раскадровка выбирается переменной окружения SB (по умолчанию `storyboard2`):
    SB=storyboard4 python3 plan2.py > ../videos/4/montage-plan.md

Расшифровку планов раскадровка отдаёт сама: TITLE, NAMES (kind -> (рецепт, описание))
и STOCK (id клипа -> описание источника).
"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SB = importlib.import_module(os.environ.get("SB", "storyboard2"))
SHOTS, CAPS, DUR = SB.SHOTS, SB.CAPS, SB.DUR
NAMES, STOCK = SB.NAMES, SB.STOCK

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k in ("A1", "A2"))
lens = sorted(t1 - t0 for t0, t1, _, _ in SHOTS)

src = os.path.relpath(SB.SRC, "/Users/vladimirkalajcidi/reels_good")
out = os.path.relpath(SB.OUT, "/Users/vladimirkalajcidi/reels_good")

print(f"# Монтажный лист — «{SB.TITLE}»\n")
print(f"Исходник: `{src}` · 1080×1920 · 30 fps · {DUR:.2f}с")
print(f"Готовый ролик: `{out}`\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens[len(lens)//2]:.2f}с · мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%**")
print(f"**Субтитров:** {len(CAPS)} фраз · в среднем {DUR/len(CAPS):.2f}с на фразу\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(SHOTS, 1):
    rec, desc = NAMES[kind]
    if prm.get("topword"):
        desc += f" · крупное слово наверху «{prm['topword']}»"
    if prm.get("clip"):
        desc = STOCK[prm["clip"]]
    txt = " / ".join("".join(r[0] for r in runs).strip()
                     for (a, b, runs, _) in CAPS if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt} |")

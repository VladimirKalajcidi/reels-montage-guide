"""Генерирует монтажный лист из раскадровки — базовый шаблон.

Для нового ролика: скопируй в plan<N>.py, поменяй импорт на storyboard<N> и подставь
номер/название ролика в шапке ниже. Запускать после render<N>.py + sfx<N>.py:
    cd build && python3 render<N>.py && python3 sfx<N>.py && python3 plan<N>.py > ../videos/<N>/montage-plan.md
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard import SHOTS, CAPS, DUR, NAMES, STOCK

NO_FLOW_CAPS = {"numw"}  # типы планов, где текст на экране несёт графика, а не субтитр

VIDEO_NUM = 1  # <- номер ролика, тот же, что в render.py/sfx.py
TITLE = "название ролика"  # <- подставь своё

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k in ("A1", "A2"))
lens = [t1 - t0 for t0, t1, _, _ in SHOTS]
lens.sort()

print(f"# Монтажный лист — «{TITLE}»\n")
print(f"Исходник: `videos/{VIDEO_NUM}/source.mov` · 1080×1920 · 30 fps · {DUR:.2f}с")
print(f"Готовый ролик: `videos/{VIDEO_NUM}/edit.mp4`\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens[len(lens)//2]:.2f}с · мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%**")
print(f"**Субтитров:** {len(CAPS)} фраз\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(SHOTS, 1):
    rec, desc = NAMES[kind]
    if prm.get("clip"):
        desc = STOCK[prm["clip"]]
    if kind == "numw":
        desc = f"{prm['digits']} / {prm['sub']}"
    txt = "" if kind in NO_FLOW_CAPS else " / ".join("".join(r[0] for r in runs).strip()
                     for (a, b, runs, _) in CAPS if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt} |")

print("\n## Источники стоковых материалов")
print(f"Все материалы лежат в `videos/{VIDEO_NUM}/stock/`, изолированы под этот ролик, "
      "каждый клип использован ровно один раз — см. assets-manifest.md §3 (Pixabay/Mixkit).")
print("\n## Звук")
print("Голос без резов, whoosh на каждом резе картинки (0.040 от пика), impact на "
      "ревиле каждого числа (0.070), музыка из `audios/` (−19 dB от RMS голоса, петля с "
      "кроссфейдом 0.25с, фейды 0.8/2.0с), финальный loudnorm I=-14 TP=-1.5 LRA=7.")

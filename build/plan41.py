"""Генерирует монтажный лист ролика 7 («инженер компьютерного зрения») из раскадровки."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard41 import SHOTS, CAPS, DUR, NAMES, STOCK

NO_FLOW_CAPS = {"numw"}

face = sum(t1 - t0 for t0, t1, k, _ in SHOTS if k in ("A1", "A2"))
lens = [t1 - t0 for t0, t1, _, _ in SHOTS]
lens.sort()

print("# Монтажный лист — «инженер компьютерного зрения» (профессии, требующие математики)\n")
print("Исходник: `videos/7/source.mov` · 720×1280 → холст 1080×1920 · 30 fps · "
      f"{DUR:.2f}с")
print("Готовый ролик: `videos/7/cv_engineer_edit.mp4`\n")
print(f"**Планов:** {len(SHOTS)} · средняя длина **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens[len(lens)//2]:.2f}с · мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {DUR:.1f}с = **{100*face/DUR:.0f}%**")
print(f"**Субтитров:** {len(CAPS)} фраз · чисел-ревилов: {sum(1 for _,_,k,_ in SHOTS if k=='numw')} (белых)\n")
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
print("Видео — Mixkit (свободная лицензия, коммерческое использование без атрибуции), "
      "прямые ссылки `assets.mixkit.co/videos/<id>/<id>-<res>.mp4`. Все материалы лежат "
      "в `videos/7/stock/`, изолированы под этот ролик, каждый клип использован ровно один раз.")
print("\n## Звук")
print("Голос без резов, whoosh на каждом резе картинки (0.040 от пика), impact на "
      "ревиле каждого белого числа (0.070), музыка `audios/song2.mp3` (−19 dB от RMS "
      "голоса, петля с кроссфейдом 0.25с, фейды 0.8/2.0с), финальный loudnorm "
      "I=-14 TP=-1.5 LRA=7.")

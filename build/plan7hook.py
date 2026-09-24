"""Монтажный лист версии с другим хуком: python3 plan7hook.py h1"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hook7 import HOOKS, CUT
import storyboard7 as SB
from plan7 import NAMES
from style import FPS

key = sys.argv[1]
cfg = HOOKS[key]
nf = cfg["frames"]
HD = nf / FPS
dt = HD - CUT
total = HD + (SB.DUR - CUT)
n = key[-1]

face = sum(b - a for a, b, k, _ in cfg["shots"] if k in SB.FACE_KINDS)
face += sum(b - a for a, b, k, _ in SB.SHOTS if a >= CUT and k in SB.FACE_KINDS)
lens = sorted([b - a for a, b, _, _ in cfg["shots"]] +
              [b - a for a, b, _, _ in SB.SHOTS if a >= CUT])

print(f"# Монтажный лист — «правило 37%», версия с хуком {n}\n")
print(f"Хук: `videos/7/hook{n}.mov` — {cfg['title']}")
print(f"Тело: кадры готового `secretary_edit.mp4` с **{CUT:.2f}с**, кадр в кадр, не пересобиралось.")
print(f"Готовый ролик: `videos/7/{os.path.basename(cfg['out'])}`\n")
print(f"**Длительность:** {total:.2f}с (хук {HD:.2f}с + тело {SB.DUR-CUT:.2f}с)")
print(f"**Планов:** {len(lens)} · средняя **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens[len(lens)//2]:.2f}с · мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с · "
      f"первый рез **{cfg['shots'][0][1]:.2f}с**")
print(f"**Лицо в кадре:** {face:.1f}с = **{100*face/total:.0f}%**")
print(f"**Уникализация (версия {cfg['version']} из 3):** трек "
      f"`{os.path.basename(cfg['music'])}` · деформация "
      f"`{cfg['post_v'] or 'нет'}`{' + `' + cfg['post_a'] + '`' if cfg['post_a'] else ''}")
print(f"**Цвет:** дубль снят в HDR (bt2020nc/HLG), приведён к SDR bt709 через "
      f"tonemap npl=260 — как основной исходник\n")

print("## Блок хука (перерисован)\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(cfg["shots"], 1):
    rec, desc = NAMES[kind]
    if kind == "stock":
        desc = SB.STOCK_NAMES[prm["clip"]]
    txt = " / ".join("".join(r[0] for r in runs).strip()
                     for (a, b, runs) in cfg["caps"] if t0 <= a < t1)
    print(f"| {i} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt or '—'} |")

print(f"\n## Тело (готовые кадры, сдвиг +{dt:.2f}с)\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
j = len(cfg["shots"])
for (t0, t1, kind, prm) in SB.SHOTS:
    if t0 < CUT:
        continue
    j += 1
    rec, desc = NAMES[kind]
    if kind == "stock":
        desc = SB.STOCK_NAMES[prm["clip"]]
    if prm.get("topword"):
        rec += "+R6"
        desc += f" · крупное слово в карточке «{prm['topword']}»"
    txt = " / ".join("".join(r[0] for r in runs).strip()
                     for (a, b, runs) in SB.CAPS if t0 <= a < t1)
    print(f"| {j} | {t0+dt:.2f}–{t1+dt:.2f} | {t1-t0:.2f}с | {rec} | {desc} | {txt or '—'} |")

print(f"\n> Точка реза **{CUT:.2f}с** — граница плана, с которого начинается "
      f"«есть точная математическая стратегия» (первое слово тела на 12.06с). "
      f"Всё до неё — хук: и вопрос «как выбрать кандидата / квартиру / партнёра», "
      f"и условия задачи («смотришь варианты один за другим», «без права вернуться»). "
      f"Планы 1–8 старого начала и их субтитры выброшены целиком.")
print(f"> Пауза между концом речи хука и первым словом тела — **0.14с** "
      f"(норма 0.10–0.20с). Хвост хука держит графический план: файл дубля короче "
      f"нужной длины, а стоп-кадр лица ставить нельзя.")
print("> Звук пересобран целиком: голос хука + голос тела с 4.30с, "
      f"постель со сдвигом {dt:+.2f}с, музыка с нуля, затем loudnorm. "
      f"События постели из выброшенного куска (каскад на 7.90с) отброшены, не сдвинуты.")

"""Монтажный лист версии ролика 9 с другим хуком: python3 plan9hook.py h1"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hook9 import HOOKS, CUT, CUT_F
import storyboard9 as SB
from style import FPS

# plan9.py — скрипт без guard'а: он печатает лист ролика прямо при импорте.
# Переписывать сданный скрипт нельзя, поэтому глушим его вывод и берём только NAMES.
with contextlib.redirect_stdout(io.StringIO()):
    from plan9 import NAMES

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

print(f"# Монтажный лист — «формула Эйлера», версия с хуком {n}\n")
print(f"Хук: `videos/9/hook{n}.mov` — {cfg['title']}")
print(f"Тело: кадры готового `euler_edit.mp4` с **{CUT:.2f}с** (кадр {CUT_F}), "
      f"кадр в кадр, не пересобиралось.")
print(f"Готовый ролик: `videos/9/{os.path.basename(cfg['out'])}`\n")
print(f"**Длительность:** {total:.2f}с (хук {HD:.2f}с + тело {SB.DUR-CUT:.2f}с)")
print(f"**Планов:** {len(lens)} · средняя **{sum(lens)/len(lens):.2f}с** · "
      f"медиана {lens[len(lens)//2]:.2f}с · мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с · "
      f"первый рез **{cfg['shots'][0][1]:.2f}с**")
print(f"**Лицо в кадре:** {face:.1f}с = **{100*face/total:.0f}%**")
print(f"**Уникализация (версия {cfg['version']} из 3):** трек "
      f"`{os.path.basename(cfg['music'])}` · деформация "
      f"`{cfg['post_v'] or 'нет'}`" +
      (f" + `{cfg['post_a']}`" if cfg.get("post_a") else ""))
print(f"**Цвет:** дубль снят в HDR (10 бит, bt2020nc / HLG), основной исходник — SDR bt709. "
      f"Приведён к SDR через tonemap `npl=160`, дальше штатный тракт A-roll без поправок.\n")

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
    if prm.get("clip"):
        desc = SB.STOCK_NAMES[prm["clip"]]
    txt = " / ".join("".join(r[0] for r in runs).strip()
                     for (a, b, runs) in SB.CAPS if t0 <= a < t1)
    print(f"| {j} | {t0+dt:.2f}–{t1+dt:.2f} | {t1-t0:.2f}с | {rec} | {desc} "
          f"| {txt or '— (говорит графика)'} |")

print(f"\n> Точка реза **{CUT:.2f}с** — конец хуковой фразы «…в истории математики» "
      f"(`source.srt`) и ровно граница плана 2. Планы 1–2 старого начала "
      f"(лицо + учёная у доски) и их субтитры выброшены целиком.")
print(f"> Вставка в хуке — та же самая (учёная у доски, Mixkit 4619) и с той же точки "
      f"входа: это визуальная интрига начала, менять её незачем.")
print(f"> Пауза между концом речи хука и первым словом тела — **0.29с** "
      f"(в сданном ролике 0.24с). Кадры {CUT_F}…106 тела — та самая пауза-вдох, "
      f"она перенесена как есть, а не переписана.")
if cfg.get("post_a"):
    print(f"> Тайминги в таблице — монтажная таймлиния до ускорения. В отданном файле "
          f"всё на 2% короче: {total:.2f}с -> {total/1.02:.2f}с, картинка и звук вместе.")
print(f"> Звук пересобран целиком: голос хука + голос тела с {CUT:.2f}с, "
      f"постель (whoosh/impact/tick) со сдвигом {dt:+.2f}с, музыка с нуля, затем loudnorm "
      f"до уровня сданного ролика (−15.0 LUFS). События постели из выброшенного куска "
      f"отброшены, не сдвинуты.")

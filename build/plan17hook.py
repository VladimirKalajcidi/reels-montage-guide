"""Монтажный лист версии ролика 17 с другим хуком.

    python3 plan17hook.py h1 > ../videos/17/montage-plan-hook1.md
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import storyboard17 as SB
import hook17 as H
from plan17 import NAMES

key = sys.argv[1]
cfg = H.HOOKS[key]
FPS = 30
nf_hook = cfg["frames"]
hook_len = nf_hook / FPS
dt = hook_len - H.CUT
body = [s for s in SB.SHOTS if s[0] >= H.CUT]
total = (nf_hook + 1736 - H.CUT_F) / FPS
post = "кроп ×0.98" if cfg.get("post_v", "").startswith("crop") else "ускорение ×1.02"
if cfg.get("post_a"):
    total /= 1.02

# у ускоренной версии тайминги приводим к сданному файлу, иначе лист
# разойдётся с ним на 2% и по нему нельзя будет ничего найти
tsc = 1.02 if cfg.get("post_a") else 1.0
shots = [(a, b, k, p) for a, b, k, p in cfg["shots"]] + \
        [(a + dt, b + dt, k, p) for a, b, k, p in body]
shots = [(a / tsc, b / tsc, k, p) for a, b, k, p in shots]
lens = sorted(b - a for a, b, _, _ in shots)
face = sum(b - a for a, b, k, _ in shots if k in SB.FACE_KINDS)

print(f"# Монтажный лист — «алгоритм Луна», версия {cfg['version']} (hook{key[-1]})\n")
print(f"Хук: `videos/17/hook{key[-1]}.mov` — {cfg['title']}")
print(f"Тело: кадры {H.CUT_F}+ из `assets/_video_luhn.mp4`, не пересобиралось.")
print(f"Готовый ролик: `{os.path.basename(cfg['out'])}` · 1080×1920 · 30 fps · {total:.2f}с\n")
print(f"**Точка реза в старом ролике:** кадр {H.CUT_F} = **{H.CUT:.4f}с**, граница планов "
      "`card16` → `check`. От старого начала (планы 1–5, 0.00–10.40с) не осталось "
      "ни кадра, ни субтитра.\n")
print(f"**Блок хука:** {nf_hook} кадров = {hook_len:.4f}с · {len(cfg['shots'])} плана · "
      f"первый рез {cfg['shots'][0][1]:.2f}с")
print(f"**Всего планов:** {len(shots)} · средняя {sum(lens)/len(lens):.2f}с · "
      f"мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с · лицо {100*face/total:.0f}%")
print(f"**Уникализация:** трек `{os.path.basename(cfg['music'])}` + {post}\n")
print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
for i, (t0, t1, kind, prm) in enumerate(shots, 1):
    rec, desc = NAMES[kind]
    if prm.get("clip"):
        desc = SB.STOCK_NAMES[prm["clip"]]
    src_caps = [(a / tsc, b / tsc, r, s) for a, b, r, s in
                (cfg["caps"] if i <= len(cfg["shots"]) else
                 [(x + dt, y + dt, r, s) for x, y, r, s in SB.CAPS])]
    txt = " / ".join("".join(r[0] for r in runs).strip()
                     for (a, b, runs, _s) in src_caps if t0 - 1e-6 <= a < t1)
    mark = " **(хук)**" if i <= len(cfg["shots"]) else ""
    print(f"| {i}{mark} | {t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} "
          f"| {txt or '— (говорит графика)'} |")

print("\n## Что сделано с хуком")
print(f"- **Приведение формата:** дубль снят 720×1280 в полном диапазоне "
      f"(`yuvj420p` / `color_range=pc`), основной исходник — 1080×1920 `tv`. "
      f"Приведено к 1080×1920 `tv` до тракта A-roll. Ни цвет, ни экспозиция не трогались.")
print(f"- **Тракт A-roll штатный:** те же `FRAMINGS`, тот же безусловный `hflip` "
      f"(надпись на футболке читается зеркально во всех трёх дублях), грейда нет — "
      f"как и в основном ролике.")
print(f"- **Тишина в голове дубля срезана** ({cfg['src_ss']:.2f}с), речь начинается сразу.")
print(f"- **Графика хука — та же**, что стояла в старом начале: `card16` и `scan` "
      f"плюс та же стоковая вставка mixkit 17618. Менять визуальную интригу незачем.")
print(f"- **Скорость анимации подогнана под длину плана** (`lts`), чтобы движение "
      f"доигрывало ровно к резу, а не обрывалось и не замирало.")
print("\n## Расхождение дублей — измерено, не исправлено")
print("Хук снят отдельным дублем, и по числам он не совпадает с основным. "
      "По правилу START-HERE это НЕ правится кропом и лутом — расхождение "
      "называется цифрами, решение за съёмочной стороной.\n")
print("| что | основной дубль | хук |")
print("|---|---|---|")
print("| размер кадра | 1080×1920 `tv` | 720×1280 `pc` -> приведено |")
print(f"| ширина лица (детектор) | 472px | {cfg['face_d']}px |")
print(f"| верх лица по кадру | y=591 | y={cfg['face_y']} |")
print(f"| средний RGB в карточке A | 95/78/71 | {cfg['rgb']} |")
print()
print(f"- **Крупность головы совпадает** ({cfg['face_d']} против 472 — разница "
      f"{abs(cfg['face_d']-472)/472*100:.0f}%), это в пределах разброса между дублями.")
print(f"- **Голова сидит на ~{cfg['face_y']-591}px ниже**: камера смотрела выше, "
      "и в карточке над головой остаётся больше потолка, чем в теле. "
      "Кропом это не подтянуто.")
print("- **Хук светлее тела на 16–20 единиц по каналу** при норме ~10. "
      "Дубль снят при более ярком свете. Экспозиция не правилась.")
print("- Оба расхождения решаются на съёмке: та же точка, та же дистанция, "
      "тот же свет, экспорт в том же размере и диапазоне.")
print("\n## Звук")
print("- Дорожка пересобрана целиком: голос хука + голос тела с 10.40с, "
      "постель (whoosh / impact / tick) со сдвигом, музыка с нуля, потом loudnorm.")
print("- Голос хука приведён по RMS к голосу тела.")
print(f"- Музыка `{os.path.basename(cfg['music'])}` — своя на эту версию, "
      "−19 dB от RMS голоса.")
print(f"- Целевая громкость — как у оригинала ролика ({H.TARGET_LUFS} LUFS), "
      "а не абстрактные −14.")

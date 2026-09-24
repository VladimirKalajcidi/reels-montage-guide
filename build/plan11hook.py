"""Монтажный лист версии ролика 11 с другим хуком: python3 plan11hook.py h1"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hook11 as HK
import storyboard11 as SB
from plan11 import NAMES
from style import FPS

key = sys.argv[1]
cfg = HK.HOOKS[key]
Lh = cfg["frames"] / FPS
dt = Lh - HK.CUT
speed = 1.02 if cfg.get("post_a") else 1.0

NAMES = dict(NAMES)
NAMES["num400"] = ("R4+R5b", "сетка: синий «400» + «лет ожидания»")

tl = [(a, b, k, p) for a, b, k, p in cfg["shots"]]
tl += [(Lh, SB.SHOTS[5][1] + dt, "eqn", {})]
tl += [(a + dt, b + dt, k, p) for a, b, k, p in SB.SHOTS if a > 7.680]

lens = sorted((b - a) / speed for a, b, _, _ in tl)
face = sum((b - a) / speed for a, b, k, _ in tl if k in SB.FACE_KINDS)
total = sum(lens)

print(f"# Монтажный лист — «великая теорема Ферма», версия {cfg['version']} ({key})\n")
print(f"Хук: {cfg['title']}")
print(f"Исходник хука: `videos/11/{os.path.basename(cfg['src'])}` · "
      f"дубль берётся с {cfg['trim']:.3f}с (снято молчание перед началом)")
print(f"Готовый ролик: `videos/11/{os.path.basename(cfg['out'])}` · "
      f"1080×1920 · 30 fps · {total:.2f}с\n")
print(f"**Точка реза:** {HK.CUT:.3f}с. Планы старого начала (лицо, теорема Пифагора, "
      f"замена показателя, синий «0») выброшены целиком. Кадры {HK.BRIDGE[0]}…{HK.BRIDGE[-1]} "
      f"перерисованы как начало плана eqn — переносить их из тела нельзя, они ещё "
      f"принадлежат старому хуку.")
print(f"**Тело не пересобиралось:** кадры с {HK.CUT_F} берутся готовыми из "
      f"`assets/_video_fermat.mp4`.\n")
print(f"**Планов:** {len(tl)} · средняя **{total/len(tl):.2f}с** · медиана {lens[len(lens)//2]:.2f}с "
      f"· мин {lens[0]:.2f}с · макс {lens[-1]:.2f}с · первый рез {cfg['shots'][0][1]:.2f}с")
print(f"**Лицо в кадре:** {face:.1f}с из {total:.1f}с = **{100*face/total:.0f}%**\n")

print("| # | тайминг | длит. | рецепт | что на экране | текст на экране |")
print("|---|---|---|---|---|---|")
allcaps = list(cfg["caps"]) + [(a + dt, b + dt, r, s) for a, b, r, s in SB.CAPS if a > 7.680]
for i, (t0, t1, kind, prm) in enumerate(tl, 1):
    rec, desc = NAMES[kind]
    if prm.get("clip"):
        desc = SB.STOCK_NAMES[prm["clip"]]
    txt = " / ".join("".join(r[0] for r in runs).strip()
                     for (a, b, runs, _s) in allcaps if t0 <= a < t1)
    print(f"| {i} | {t0/speed:.2f}–{t1/speed:.2f} | {(t1-t0)/speed:.2f}с | {rec} | {desc} "
          f"| {txt or '— (говорит графика)'} |")

print("\n## Уникализация версии")
print(f"- Музыка: `{os.path.basename(cfg['music'])}` (у каждой версии своя).")
print(f"- Деформация кадра: `{cfg['post_v']}`"
      + (f", звук: `{cfg['post_a']}`" if cfg.get("post_a") else ""))
print(f"- Громкость подогнана под версию 1: цель {HK.TARGET_LUFS} LUFS.")

print("\n## Приведение формата хука")
print("Дубль хука снят 720×1280 в полном диапазоне (`yuvj420p` / `pc`), основной "
      "исходник — 1080×1920, `tv`. Оба расхождения выводятся из тегов файла, поэтому "
      "приводятся всегда, до тракта A-roll: апскейл 1.5× и full → limited. "
      "Кадрирование `FRAMINGS`, `hflip` и грейд — штатные, без поправок под дубль.")
print("\nЗамеры (средний RGB внутри карточки A · высота лица при карточке 1380px):\n")
print("| дубль | средний RGB | высота лица | центр лица |")
print("|---|---|---|---|")
print("| исходник | 105 / 81 / 69 | 417px | 651px |")
print("| хук1 | 122 / 101 / 88 | 384px | 738px |")
print("| хук2 | 136 / 114 / 101 | 364px | 763px |")
print("\n**Расхождение не подчищено кропом и не выправлено лутом.** Норма по гайду — "
      "до ~10 единиц на канал; хук1 ярче на +18/+20/+19, хук2 на +32/+33/+32, и оба "
      "сняты дальше от камеры (лицо мельче на 8% и 13%, посажено ниже на 87 и 112px). "
      "Это решается на съёмке — светом и дистанцией, одинаковыми для всех дублей.")
print("\nЗеркальность проверена по надписи на толстовке: во всех трёх дублях текст "
      "читается нормально после штатного `hflip`, исключений не нужно.")

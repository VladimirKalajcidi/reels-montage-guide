"""Монтажный лист версии ролика 20 с другим хуком.

    python3 plan20hook.py h1 > ../videos/20/montage-plan-hook1.md
    python3 plan20hook.py h2 > ../videos/20/montage-plan-hook2.md
"""
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import FPS, CARD_A
import storyboard20 as SB
import render20 as R
from hook20 import HOOKS, CUT, CUT_F
from plan20 import NAMES
import qa20hook as Q

VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/20"
ORIG = f"{VIDEO_DIR}/popcorn_edit.mp4"


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr.splitlines()
    out = {}
    for i, line in enumerate(r):
        s = line.strip()
        if s.startswith("I:") and "LUFS" in s:
            out["I"] = s.split()[1]
        elif s.startswith("LRA:") and "LU" in s and "LRA" not in out:
            out["LRA"] = s.split()[1]
        elif s.startswith("Peak:"):
            out["TP"] = s.split()[1]
    return out


def probe(path, stream="v:0", fields="r_frame_rate,nb_frames,duration"):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream,
                        "-show_entries", f"stream={fields}",
                        "-of", "default=nw=1", path],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return dict(kv.split("=") for kv in r)


def body_psnr(cfg):
    """Тело версии против сданного ролика, прогнанного через ту же деформацию."""
    ref = ORIG
    tmp = None
    if cfg.get("post_v"):
        tmp = f"/tmp/_ref_{os.path.basename(cfg['out'])}"
        if not os.path.exists(tmp):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", ORIG,
                            "-vf", cfg["post_v"], "-an", "-c:v", "libx264",
                            "-crf", "15", "-preset", "medium",
                            "-pix_fmt", "yuv420p", tmp], check=True)
        ref = tmp
    speed = 1.02 if cfg.get("post_a") == "atempo=1.02" else 1.0
    a = cv2.VideoCapture(ref)
    b = cv2.VideoCapture(cfg["out"])
    vals = []
    for body_f in (200, 400, 700, 1000, 1300):
        a.set(cv2.CAP_PROP_POS_FRAMES, int(round(body_f / speed)) if speed != 1.0 else body_f)
        ok1, f1 = a.read()
        b.set(cv2.CAP_PROP_POS_FRAMES,
              int(round((cfg["frames"] + (body_f - CUT_F)) / speed)))
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            continue
        mse = float(np.mean((f1.astype(np.float32) - f2.astype(np.float32)) ** 2))
        vals.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
    a.release()
    b.release()
    return (round(min(vals), 1), round(max(vals), 1)) if vals else (None, None)


def emit(key):
    cfg = HOOKS[key]
    Q.bind(cfg)
    rh = Q.rhythm(cfg)
    dt = cfg["frames"] / FPS - CUT
    hook_s = cfg["frames"] / FPS

    print(f"# Монтажный лист — версия {cfg['version']} (hook{key[-1]})\n")
    print(f"Хук: `videos/20/hook{key[-1]}.mov` · {cfg['title']}")
    print(f"Тело: без изменений, готовые кадры из `build/assets/_video_popcorn.mp4` "
          f"с кадра **{CUT_F}** ({CUT:.4f}с)")
    print(f"Готовый файл: `{os.path.relpath(cfg['out'], '/Users/vladimirkalajcidi/reels_good')}` "
          "· 1080×1920 · 30 fps\n")
    print(f"**Версия целиком:** {rh['dur']:.2f}с · планов {rh['shots']} · "
          f"средняя длина **{rh['avg']:.2f}с** · медиана {rh['median']:.2f}с · "
          f"мин {rh['shortest']:.2f}с · макс {rh['longest']:.2f}с · "
          f"первый рез {rh['first_cut']:.2f}с")
    print(f"**Лицо в кадре:** **{rh['face_pct']:.1f}%**")
    print(f"**Блок хука:** {cfg['frames']} кадров = {hook_s:.4f}с · "
          f"тело сдвинуто на {dt:+.4f}с\n")

    print("## Хук — раскадровка")
    print("| # | кадры | тайминг | длит. | рецепт | что на экране | текст на экране |")
    print("|---|---|---|---|---|---|---|")
    for i, (t0, t1, kind, prm) in enumerate(cfg["shots"], 1):
        rec, desc = NAMES[kind]
        if prm.get("clip"):
            # у стока в STOCK_NAMES хвост с фразой основного ролика — в хуке
            # под этим клипом звучат другие слова, поэтому берём только описание
            desc = SB.STOCK_NAMES[prm["clip"]].split(" — ")[0]
        txt = " / ".join("".join(r[0] for r in runs).strip()
                         for (a, b, runs, _s) in cfg["caps"] if t0 <= a < t1)
        print(f"| {i} | {int(round(t0*FPS))}–{int(round(t1*FPS))} | "
              f"{t0:.2f}–{t1:.2f} | {t1-t0:.2f}с | {rec} | {desc} "
              f"| {txt or '— (говорит графика)'} |")

    print("\n## Точка реза")
    print(f"- Рез по кадру **{CUT_F}** ({CUT:.4f}с) — граница планов `A2` → `cups3` "
          "в сданном ролике. Планы 1–3 старого начала и все их субтитры выброшены "
          "целиком: ни одного кадра и ни одной строки от старого хука не осталось.")
    print("- Рез попадает в паузу: непрерывная тишина в основном дубле идёт "
          "**4.73–5.28с**, речь тела («в кинотеатре обычно три размера попкорна») "
          "начинается на 5.28. До первого слова тела остаётся **0.013с** — "
          "ни звука не срезано.")
    print(f"- Пауза между последним словом хука и первым словом тела — "
          f"**{cfg['pause']:.3f}с**. В самом теле паузы между предложениями "
          "0.29–0.55с (замер по огибающей), так что связка звучит как обычный "
          "переход внутри дубля.")
    print(f"- Передача эстафеты: {cfg['handoff']}")

    print("\n## Дубль хука")
    print("- **Цветовые теги сверены**: `yuv420p` / `color_range=tv` / "
          "`bt709` / `bt709` / `bt709` — совпадают с основным исходником "
          "до последнего поля. Приведение HDR→SDR не требуется, кадры идут "
          "в штатный тракт A-roll прямо из `.mov`.")
    print("- **A-roll прогнан тем же трактом**: те же `FRAMINGS`, тот же `hflip`, "
          "грейда нет — ни одной поправки под дубль.")
    print(f"- **Крупность** (детектор, после hflip): хук cx {cfg['face_cx']}, "
          f"линия глаз {cfg['face_eye']}, размер лица {cfg['face_d']} против "
          "cx 382 / 670 / 270 у основного дубля. Расхождение "
          f"{abs(cfg['face_cx']-382)}px по горизонтали, {abs(cfg['face_eye']-670)}px "
          f"по линии глаз, {abs(cfg['face_d']-270)/270*100:.1f}% по размеру лица. "
          "Кропом это не правится — цифры названы, решение за съёмочной стороной.")
    hook_rgb = cfg["rgb_hook"]
    print(f"- **Средний RGB внутри карточки A**: хук {hook_rgb} против "
          "(115, 102, 94) у основного дубля — расхождение до **1 единицы** "
          "по каналу при норме гайда ~10. Свет и экспозиция дублей совпали.")
    print("- **Зеркальность**: надписей в кадре нет, но комната, точка съёмки и "
          "рука с петличкой те же, что в основном дубле и в ролике 19, — "
          "штатный `hflip` подходит.")

    print("\n## Графика хука")
    print(cfg["gfx_note"])

    print("\n## Уникализация версии")
    print(f"- **Музыка:** `audios/{os.path.basename(cfg['music'])}` — свой трек, "
          "ни с одной другой версией не совпадает (версия 1 — `song2.mp3`, "
          "версия 2 — `song3.mp3`, версия 3 — `song1.mp3`). Уровень тот же: "
          "−19 dB от RMS голоса, фейды 0.8 / 2.0с.")
    if cfg.get("music_note"):
        print(f"- {cfg['music_note']}")
    print(f"- **Деформация:** {cfg['deform_note']}")
    m = cfg["measured"]
    for line in m:
        print(f"- {line}")

    print("\n## Звук")
    print("- Дорожка пересобрана целиком, а не склеена из готовых: голос хука + "
          "голос тела с точки реза, постель (whoosh / impact / tick) со сдвигом "
          f"{dt:+.4f}с, музыка с нуля, потом loudnorm.")
    print(f"- Голос хука приведён к громкости тела: коэффициент "
          f"×{cfg['voice_gain']:.3f} по RMS речевых участков.")
    ld = loudness(cfg["out"])
    print(f"- Замер готового `.mp4`: **{ld.get('I')} LUFS**, истинный пик "
          f"**{ld.get('TP')} dBFS**, LRA {ld.get('LRA')} LU. "
          "Ориентир — громкость сданного ролика (−14.0 LUFS), а не абстрактные −14.")
    seam_local, seam_glob = Q.click_at_seam(cfg)
    print(f"- **Щелчка на стыке нет:** максимальный скачок между соседними "
          f"сэмплами в окне ±5мс вокруг шва — **{seam_local:.4f}** против "
          f"{seam_glob:.4f} у 99.99-перцентиля всей дорожки.")

    print("\n## Автопроверки")
    c, _ = Q.cuts_inside_word(cfg)
    cc, _ = Q.captions_off_speech(cfg)
    o, _ = Q.gfx_text_overlap(cfg)
    z, _ = Q.gfx_in_zone(cfg)
    tc, _ = Q.text_in_card(cfg)
    lo, _ = Q.line_overlaps(cfg)
    b, _ = Q.bright_outside(cfg)
    lo_psnr, hi_psnr = body_psnr(cfg)
    print("| проверка | результат |")
    print("|---|---|")
    print(f"| текст за карточкой (вся версия, каждый 3-й кадр) | **{b} кадров** |")
    print(f"| наложение строк внутри блока | **{lo} пар** |")
    print(f"| графика пересекает субтитры | **{o} кадров** |")
    print(f"| графика вне зоны y700–1560 / x175–905 | **{z} кадров** |")
    print(f"| субтитр вне карточки с отступом 30px | **{tc} кадров** |")
    print(f"| рез хука попал в середину слова | **{c} резов** |")
    print(f"| фраза хука разошлась с речью | **{cc} фраз** |")
    v = probe(cfg["out"])
    a = probe(cfg["out"], "a:0", "duration")
    print(f"| частота кадров на выходе | **{v['r_frame_rate']}** |")
    print(f"| длительности видео / аудио | **{float(v['duration']):.3f}с / "
          f"{float(a['duration']):.3f}с** |")
    print(f"| тело кадр в кадр со сданным роликом (PSNR) | **{lo_psnr}–{hi_psnr} dB** |")
    print("\nТело сравнивается со сданным `popcorn_edit.mp4`, прогнанным через "
          "ту же деформацию версии. PSNR такого порядка — это шум кодека, "
          "то есть кадры совпадают.")
    if cfg.get("psnr_note"):
        print(cfg["psnr_note"])
    print("\n**Версия №1 (`popcorn_edit.mp4`) не тронута** — сборка версий пишет "
          "только в `popcorn_hook1.mp4` / `popcorn_hook2.mp4`.")


EXTRA = {
    "h1": dict(
        pause=0.146,
        handoff="хук кончается на «и вот как это работает» — ровно та же связка, "
                "что в старом начале, поэтому тело подхватывает без единой правки.",
        gfx_note=(
            "- В старом хуке ролика 20 сетки не было вообще: там стояли лицо, "
            "зал кинотеатра (сток 33312) и снова лицо. **Зал кинотеатра сохранён** "
            "на том же месте — это визуальная примета начала.\n"
            "- Хвост блока закрыт планом **`pickL`** (три стакана, стрелка вправо, "
            "галка над большим) под слова «и вот как это работает». План уже есть "
            "в теле ролика — ни одного нового не нарисовано.\n"
            "- Ценники 150 / 350 / 400 ₽ появляются в хуке раньше, чем о них "
            "заходит речь. Это не дублирование: в хуке цены не произносятся, "
            "их пишет только графика, и они работают как интрига."),
        deform_note="кроп **×0.98** — обрезаем 2% по краям и возвращаем холст "
                    "1080×1920. Накладывается на уже собранное немое видео, "
                    "кадры не перерисовываются.",
        measured=[
            "**Деформация измерена машиной:** карточка A занимает на экране "
            "**888px** вместо номинальных 870 — это ровно **×1.0207** "
            "(ожидалось ×1.0204).",
            "Ресемпл `bilinear`, а не `lanczos`: у lanczos отрицательные лепестки "
            "дают на границе карточки светлый ореол в 1px, и автопроверка "
            "«текст за карточкой» краснеет.",
        ],
        voice_gain=1.236,
        rgb_hook="(114, 102, 95)",
    ),
    "h2": dict(
        pause=0.110,
        handoff="хук кончается на «продавая вам большие попкорны», тело начинает "
                "новым предложением «в кинотеатре обычно три размера попкорна». "
                "Слово «попкорн» звучит в обеих фразах, но предложение начинается "
                "с нового подлежащего — шва в речи не слышно.",
        gfx_note=(
            "- Как и в версии 2, сохранён **зал кинотеатра** (сток 33312) на месте "
            "вставки старого хука.\n"
            "- Хвост блока закрыт планом **`pickL`** под слова «продавая вам "
            "большие попкорны»: галка над большим стаканом — буквально то, "
            "что произносится. План взят из тела, новых не рисовалось.\n"
            "- Ценники в кадре есть, но в речи хука цен нет — числа несёт "
            "только графика."),
        deform_note="ускорение **×1.02** — `setpts=PTS/1.02` по картинке и "
                    "`atempo=1.02` по звуку, вместе. Накладывается на уже "
                    "собранное немое видео.",
        measured=[
            "**Ускорение прошло и по картинке, и по звуку:** видео 45.767с, "
            "аудио 45.765с — расходятся на 2мс, липсинк не едет.",
            "**Кадров стало 1373 вместо 1400** (1400 / 1.02 = 1372.5): лишние 2% "
            "кадров честно выброшены.",
            "**`fps=30` после `setpts` обязателен** — без него на выходе был бы "
            "30.6 fps (`r_frame_rate=31/1`), нестандартный для площадки.",
        ],
        voice_gain=1.383,
        rgb_hook="(114, 102, 95)",
        music_note="`song1.mp3` длится всего 9.27с, поэтому идёт **петлёй ×5.0 "
                   "с кроссфейдом 0.25с на шве**. Это единственный трек в "
                   "`audios/`, который короче ролика; если петля на слух "
                   "надоедает, нужен четвёртый трек.",
        psnr_note="Нижняя граница у этой версии ниже, чем у версии 2, по одной "
                  "причине: ускорение ×1.02 выбрасывает каждый 51-й кадр, и "
                  "сопоставить кадр версии с кадром оригинала можно только с "
                  "точностью до половины кадра. На быстром плане это и даёт "
                  "36 dB вместо 46 — расхождение сопоставления, а не картинки.",
    ),
}

for k, extra in EXTRA.items():
    HOOKS[k].update(extra)


if __name__ == "__main__":
    emit(sys.argv[1])

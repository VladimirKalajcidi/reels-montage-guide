"""Ролик 35 («парадокс Рассела»), альтернативные хуки.

Тело ролика (всё с кадра 131) не пересобирается — берётся готовыми кадрами из
`assets/_video_russell.mp4` и склеивается за новым хуком. Заново рисуется
только блок хука. render35.py этот скрипт не трогает.

    python3 hook35.py h1
    python3 hook35.py h2

--- точка реза ------------------------------------------------------------
Старое начало — «может ли существовать множество всех множеств, которые не
содержат сами себя» (0.156–4.263). Оно занимает план `A1` 0.000–1.954 и план
`boxes` 1.954–4.336. Оба новых дубля кончаются **той же самой фразой**, так
что вопрос из ролика не пропадает — меняется только подводка к нему.

Рез — по границе плана `boxes` -> `A1`, то есть по 4.336. В кадрах это
ceil(4.336 x 30) = **131** (округление вверх: кадр 130 показывает ещё сетку
с коробками, и в версии от старого хука остался бы целый кадр графики).

--- шов: настоящий рез, а не продолжение плана ------------------------------
Блок хука кончается графическим планом `boxesB`, тело начинается планом `A1`
с лицом. Карточка меняется, значит на шве стоит обычный прямой рез и на нём
ставится whoosh. Кончать блок хука лицом было нельзя: A-roll на шве
переключается с дубля хука на основной, и без смены карточки это читалось бы
как прыжок внутри одного плана.

--- голос на шве -----------------------------------------------------------
Первое слово тела — «Ответ» на 4.806, то есть **на 0.44с позже** кадра реза.
Огибающая основного дубля на 4.3667 даёт 0.00014 (тишина), поэтому J-cut не
нужен: голос тела берётся ровно с CUT_V = 131/30.

--- пауза хук -> тело -------------------------------------------------------
В основном дубле между «сами себя» (4.263) и «ответ» (4.806) стоит пауза
**0.543с** — это граница предложений, и версия обязана её воспроизвести,
иначе ритм в этом месте разойдётся с оригиналом:

    hook1  речь кончается 6.983, блок 212 кадров (7.0667) -> хвост 0.084
           + 0.439 тишины тела = пауза 0.523с
    hook2  речь кончается 6.737, блок 204 кадра (6.8000) -> хвост 0.063
           + 0.439 = пауза 0.502с

Диапазон 0.10–0.20с из delivery-specs §6 здесь неприменим: он описывает шов
внутри абзаца (ролик 34), а тут шов приходится на границу предложений.

--- хвостовая тишина дублей -------------------------------------------------
hook1: речь до 6.983 при длине 7.102 — 0.119с тишины, последнее слово целое.
hook2: речь идёт до конца файла (6.737 при 6.753). Последнее слово успевает
затухнуть до 8% от пика (0.0154 против 0.1885), но записью подрезано, поэтому
на последние 45мс дубля кладётся фейд — он снимает щелчок; вернуть срезанный
хвост слова монтаж не может, это к записи.

--- передача эстафеты -------------------------------------------------------
Тело начинается фразой «ответ да, и оно называется множество Рассела».
Местоимение «оно» опирается на «множество всех множеств, которые не содержат
сами себя» — и **оба дубля произносят эту формулировку целиком**, поэтому
антецедент на месте. Графика на шве тоже сходится: блок хука кончается
пятёркой коробок (три обычных + две с копией себя), а тело на 5.766 берёт
ровно эту картинку и вычёркивает самосодержащие.

--- формат хуков ------------------------------------------------------------
Оба дубля совпадают с основным исходником по всем тегам: 720x1280, yuv420p,
color_range=tv, bt709/bt709/bt709. Приведение цветового пространства не
требуется, HDR ни в одном файле нет. Надпись на футболке после штатного
`hflip` читается нормально в обоих дублях — hflip подходит, отключать не надо.

Крупность дублей (детектор, после hflip, среднее по дублю):
    основной  cx 374  линия глаз 687  ширина лица 271
    hook1     cx 369  линия глаз 684  ширина лица 270
    hook2     cx 370  линия глаз 687  ширина лица 265
Кропом это не правится: оба дубля идут через те же `FRAMINGS`, что и основной
исходник, числа названы в сдаче.
"""
import json
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render35 as R
import storyboard35 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/35"
BODY = f"{BUILD}/assets/_video_{SB.TAG}.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 131                                    # первый кадр тела (ceil(4.336*30))
CUT_V = CUT_F / FPS                            # 4.3667с — картинка тела
CUT_A = CUT_F / FPS                            # голос тела: на шве тишина, J-cut не нужен

TARGET_LUFS = -14.1                            # замер сданного russell_edit.mp4

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        words=f"{VIDEO_DIR}/words_hook1.json",
        src_frames=213,                  # 7.102с, речь 0.113-6.983
        speech_end=6.983,
        tail_fade=0.0,                   # хвостовая тишина есть, фейд не нужен
        cuts=(52, 102, 167),             # A1 | boxesA | A1 | boxesB
        frames=212,                      # 7.0667с — пауза до тела 0.523с
        out=f"{VIDEO_DIR}/russell_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«многим объясняли этот парадокс в школе — итак, вопрос»",
        face_cx=369, face_eye=684, face_d=270,
        caps=[
            (0.113, 0.950, [("многим ", R_, 44), ("объясняли", R_, 52)], "A"),
            (0.950, 52 / FPS, [("этот ", R_, 44), ("парадокс", S_, 54)], "A"),

            (1.742, 2.249, [("в ", R_, 42), ("школе", R_, 52)], "G"),
            (2.249, 2.953, [("итак ", R_, 42), ("вопрос", S_, 54)], "G"),

            (3.400, 4.160, [("может ли ", R_, 42), ("существовать", R_, 50)], "A"),
            (4.160, 167 / FPS, [("множество всех ", R_, 40), ("множеств", R_, 52)], "A"),

            (167 / FPS, 5.964, [("которые ", R_, 42), ("не", R_, 50)], "G"),
            (5.964, 6.983, [("содержат сами ", R_, 40), ("себя", S_, 54)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        words=f"{VIDEO_DIR}/words_hook2.json",
        src_frames=202,                  # 6.737с, речь 0.186-6.737
        speech_end=6.737,
        tail_fade=0.045,                 # хвостовой тишины нет — снимаем щелчок
        cuts=(47, 114, 154),             # A1 | boxesA | A1 | boxesB
        frames=204,                      # 6.8000с — пауза до тела 0.502с
        out=f"{VIDEO_DIR}/russell_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«это самый известный парадокс теории множеств»",
        face_cx=370, face_eye=687, face_d=265,
        caps=[
            (0.186, 0.534, [("это ", R_, 44), ("самый", R_, 52)], "A"),
            (0.534, 47 / FPS, [("известный парадокс ", R_, 40), ("теории", R_, 50)], "A"),

            # 1.57-2.83 «теории множества» идёт без субтитра: слово «множества»
            # несёт сама графика — на сетке в этот момент встают коробки
            (2.826, 3.800, [("может ли ", R_, 42), ("существовать", R_, 50)], "G"),

            (3.800, 154 / FPS, [("множество всех ", R_, 40), ("множеств", R_, 52)], "A"),

            (5.175, 5.590, [("которые ", R_, 42), ("не", R_, 50)], "G"),
            (5.590, 6.737, [("содержат сами ", R_, 40), ("себя", S_, 54)], "G"),
        ],
    ),
}

KINDS = ("A1", "boxesA", "A1", "boxesB")


def shots_of(cfg):
    bounds = (0,) + tuple(cfg["cuts"]) + (cfg["frames"],)
    return [(bounds[i] / FPS, bounds[i + 1] / FPS, KINDS[i], {})
            for i in range(4)]


def card_rgb(path, t, kind="A1"):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    card = np.array(R.source_card(fr, kind))
    return tuple(int(v) for v in card.reshape(-1, 3).mean(axis=0))


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    shots = shots_of(cfg)
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, s=shots: next((x for x in s if x[0] <= t < x[1]), s[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(cfg["src"])

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    sheet, last_img, held = [], None, 0
    for f in range(nf_hook):
        t = f / FPS
        ok, fr = cap.read()
        if ok:
            last_img = fr
        else:
            fr = last_img          # только под графическим планом, лицо не морозим
            held += 1
        t0, t1, kind, prm = R.shot_at(t)
        if not ok and kind in R.FACE_KINDS:
            raise RuntimeError(f"A-roll кончился на кадре {f} под планом с лицом {kind}")

        canvas = R.background(kind, prm, fr, t, t0)
        gl = R.graphics_layer(kind, t - t0)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 4 == 0:
            sheet.append((t, np.array(rgb)))
    cap.release()

    body = cv2.VideoCapture(BODY)
    body.set(cv2.CAP_PROP_POS_FRAMES, CUT_F)
    n_body = 0
    while True:
        ok, img = body.read()
        if not ok:
            break
        ff.stdin.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
        n_body += 1
    body.release()

    ff.stdin.close()
    ff.wait()
    print(f"видео: хук {nf_hook} кадров (планы {[c for c in cfg['cuts']]}, "
          f"из них {held} сверх дубля) + тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook35.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path]
    if start:
        cmd += ["-ss", f"{start:.5f}"]
    cmd += ["-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp]
    subprocess.run(cmd, check=True)
    d = read_wav(tmp)
    os.remove(tmp)
    return d


def speech_rms(x):
    m = np.abs(x).mean(axis=1)
    sel = m > m.max() * 0.12
    return float(np.sqrt((x[sel] ** 2).mean()))


def build_audio(key, cfg, nf_hook, n_body, vid, out):
    shots = shots_of(cfg)
    hook = wav_of(cfg["src"])
    body = wav_of(SRC, CUT_A)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = (hook * g).copy()
    print("хук: коррекция громкости голоса x%.3f" % g)

    tf = cfg.get("tail_fade", 0.0)
    if tf:                                   # дубль обрезан записью — снимаем щелчок
        r = int(tf * SR)
        hook[-r:] *= np.linspace(1, 0, r)[:, None]
        print("хвостовой фейд дубля: %.0f мс" % (tf * 1000))

    n_hook = int(round(nf_hook / FPS * SR))
    n_bod = int(round(n_body / FPS * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    if len(body) < n_bod:
        body = np.vstack([body, np.zeros((n_bod - len(body), 2), np.float32)])
    else:
        body = body[:n_bod].copy()
    r = int(0.008 * SR)
    hook[-r:] *= np.linspace(1, 0, r)[:, None]
    body[:r] *= np.linspace(0, 1, r)[:, None]

    voice = np.vstack([hook, body]).astype(np.float32)
    n = voice.shape[0]
    peak = np.abs(voice).max()
    bed = np.zeros((n, 2), np.float32)

    def add(sig, t, gain):
        i = int(t * SR)
        m = min(len(sig), n - i)
        if i >= 0 and m > 0:
            bed[i:i + m, 0] += sig[:m] * gain
            bed[i:i + m, 1] += sig[:m] * gain

    dt = nf_hook / FPS - CUT_V          # SFX тела считаются от КАРТИНКИ
    wh, im, tk = low_whoosh(), impact(), tick()
    for t0 in [s[0] for s in shots[1:]]:            # три реза внутри блока хука
        add(wh, t0, peak * 0.040)
    add(wh, nf_hook / FPS, peak * 0.040)            # шов: карточка меняется, рез настоящий
    for t0, _t1, _k, _p in SB.SHOTS:
        if t0 > CUT_V + 0.01:
            add(wh, t0 + dt, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= CUT_V:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT_V:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t0 in (cfg["cuts"][0] / FPS, cfg["cuts"][2] / FPS):   # каскады коробок в хуке
        for k in range(3):
            add(tk, t0 + 0.10 + k * 0.085, peak * 0.030)

    music = cfg["music"]
    mw = f"{BUILD}/assets/_music_{os.path.basename(music).split('.')[0]}.wav"
    if not os.path.exists(mw):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", music,
                        "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
    mus = read_wav(mw)
    if mus.shape[1] == 1:
        mus = np.repeat(mus, 2, axis=1)
    if len(mus) >= n:
        track = mus[:n].copy()
        print("музыка: %s, петли нет" % os.path.basename(music))
    else:
        xf = int(0.25 * SR)
        core, tail = mus[:len(mus) - xf], mus[len(mus) - xf:]
        ramp = np.linspace(0, 1, xf)[:, None]
        loop = core.copy()
        loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
        track = np.tile(loop, (int(np.ceil(n / len(loop))) + 1, 1))[:n]
        print("музыка: %s, петля %.2fс x%.1f" %
              (os.path.basename(music), len(loop) / SR, n / len(loop)))
    vr = np.sqrt((voice ** 2).mean()) + 1e-9
    mr = np.sqrt((track ** 2).mean()) + 1e-9
    track = track * (vr / mr) * (10 ** (-19 / 20))
    fi, fo = int(0.8 * SR), int(2.0 * SR)
    track[:fi] *= np.linspace(0, 1, fi)[:, None]
    track[-fo:] *= np.linspace(1, 0, fo)[:, None]
    bed += track.astype(np.float32)

    mix = voice + bed
    m = np.abs(mix).max()
    if m > 0.99:
        mix *= 0.99 / m
    mix_path = f"{BUILD}/assets/_mix_{SB.TAG}{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    src = f"{BUILD}/assets/_video_{SB.TAG}{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_{SB.TAG}{key}_post.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
                    "-pix_fmt", "yuv420p", dst], check=True)
    print("правка кадра:", vf)
    return dst


def measure_i(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr
    m = json.loads(s[s.rindex("{"):s.rindex("}") + 1])
    return float(m["input_i"]), float(m["input_tp"])


def mux(vid, mix_path, out, target=TARGET_LUFS, extra_af=None):
    def run(post_db):
        af = (f"{extra_af}," if extra_af else "") + \
             "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
        if abs(post_db) > 0.02:
            af += f",volume={post_db:+.2f}dB,alimiter=limit=0.80:level=disabled"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", vid, "-i", mix_path, "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-af", af, "-ar", str(SR),
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out,
        ], check=True)
        i, tp = measure_i(out)
        print("  loudnorm: post %+.2f dB -> I=%.2f LUFS, TP=%.2f dBFS" % (post_db, i, tp))
        return i, tp

    post_db, best = 0.0, None
    for _ in range(4):
        i, tp = run(post_db)
        if tp <= -1.0 and (best is None or abs(i - target) < abs(best[1] - target)):
            best = (post_db, i)
        if abs(i - target) < 0.15 and tp <= -1.0:
            best = None
            break
        post_db += target - i
    if best is not None:
        run(best[0])
    print("готово:", out)


def main(key):
    cfg = HOOKS[key]
    tmp = f"{BUILD}/assets/_video_{SB.TAG}{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    print("средний RGB в карточке A: хук", card_rgb(cfg["src"], 1.0),
          "· основной дубль", card_rgb(SRC, 1.0))

    build_audio(key, cfg, nf_hook, n_body, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook35_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_{SB.TAG}{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

"""Ролик 36 («парадокс мальчика и девочки»), альтернативные хуки.

Тело ролика (всё с кадра 124) не пересобирается — берётся готовыми кадрами из
`assets/_video_kids.mp4` и склеивается за новым хуком. Заново рисуется только
блок хука. render36.py этот скрипт не трогает.

    python3 hook36.py h1
    python3 hook36.py h2

--- точка реза ------------------------------------------------------------
Старое начало — «если ты учил в школе теорию вероятности, ты должен решить
эту задачку» (0.000–3.621). Оно занимает три плана: `A1` 0.000–1.413,
`stock` 1.413–2.446 и `A1` 2.446–4.110. Ближайшая граница плана после конца
фразы — **4.110**, по ней и режем: всё раньше выбрасывается целиком.

В кадрах это ceil(4.110 x 30) = **124** (округление вверх, а не к ближайшему:
кадр 123 стоит на 4.100 и показывает ещё лицо старого хука — при round от
него остался бы целый кадр).

Тело начинается планом `family` (4.110–6.230), у которого на 4.16 стартует
каскад: первый кадр тела — пустая сетка, ровно как в сданной версии.

--- шов ---------------------------------------------------------------------
В отличие от ролика 34, шов здесь совпадает с настоящим резом: блок хука
кончается стоковой вставкой (карточка B), тело начинается сеткой (карточка A).
Поэтому whoosh на шве ставится как на любой склейке.

--- голос на шве: J-cut в два кадра -----------------------------------------
Первое слово тела («у мужчины») по огибающей атакует в 4.095:
    4.080 = -80 дБ, 4.090 = -61, 4.100 = -43, 4.110 = -31, 4.120 = -27.
Картинка тела начинается кадром 124 (4.1333) — рез голоса ровно по картинке
срезал бы 38мс уже звучащего слова. Голос тела берётся с кадра 122
(`CUT_A` = 4.0667, там -80 дБ, тишина) и кладётся в новую дорожку на два
кадра раньше картинки тела. Синхрон не едет: смещение кратно кадру, и каждый
кадр тела стоит против своего звука. На двух последних кадрах блока хука
звучат 66мс тишины из тела — это обычный J-cut.

--- пауза хук -> тело -------------------------------------------------------
    hook1  речь кончается 3.601, слово тела звучит в 3.777 -> пауза 0.176с
    hook2  речь кончается 5.026, слово тела звучит в 5.177 -> пауза 0.148с
           (после ускорения x1.02 — 0.145с)
По огибающей атака на 15мс раньше слова из расшифровки — пауза от
этого только короче на 15мс и из диапазона не выходит.
Оба значения в диапазоне 0.10–0.20с из delivery-specs §6. Хвостовая тишина
в обоих дублях есть (hook1 3.601 при длине 3.767, hook2 5.026 при 5.167),
последнее слово записью не обрезано.

--- передача эстафеты -------------------------------------------------------
Тело начинается самостоятельным предложением «у мужчины двое детей»: своё
подлежащее, ни одного местоимения, отсылок к выброшенному тексту нет. Оба
новых дубля кончаются словами «...по теории вероятностей» / «...неправильный
ответ» и ведут ровно туда же, куда вело старое начало.

--- графика блока хука ------------------------------------------------------
Та же, что стояла в старом хуке: стоковая вставка со школьным классом
(mixkit 35954). Другой графики в старом начале не было, менять её незачем.
Окна клипа взяты новые, чтобы вставка не повторяла кадр в кадр сданную
версию: v1 ss=16.6, hook1 ss=8.0, hook2 ss=12.5 и 19.4.
Окна не пересекаются между версиями — иначе один и тот же кадр стока
стоял бы в двух версиях подряд, и площадка честно считала бы их дублями.

--- число «90%» в hook2 -----------------------------------------------------
Дубль 2 говорит «но 90% людей дают на неё неправильный ответ». Это
риторическая оценка, а не измеренная величина, и источника у неё нет.
На экран она не выносится: в ролике синее число = посчитанный результат
(1/3 и 1/2), и оформить ею неподтверждённую цифру значило бы выдать её
за такой же результат. В речи фраза остаётся как есть — речь не режем.

--- формат хуков ------------------------------------------------------------
Оба дубля совпадают с основным исходником по всем тегам: 720x1280, yuv420p,
color_range=tv, bt709/bt709/bt709. Приведение цветового пространства не
требуется, HDR ни в одном файле нет. Зеркальность проверена по кадру:
надпись на футболке в обоих дублях читается задом наперёд, штатный `hflip`
подходит обоим.

Крупность дублей (детектор, после hflip, среднее по дублю):
    основной  cx 372  линия глаз 683  ширина лица 264
    hook1     cx 365  линия глаз 691  ширина лица 270
    hook2     cx 368  линия глаз 689  ширина лица 270
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
import render36 as R
import storyboard36 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/36"
BODY = f"{BUILD}/assets/_video_{SB.TAG}.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 124                                    # первый кадр тела (ceil(4.110*30))
CUT_V = CUT_F / FPS                            # 4.1333с — картинка тела
CUT_A = (CUT_F - 2) / FPS                      # 4.0667с — голос тела, J-cut в 2 кадра

TARGET_LUFS = -14.0                            # замер сданного kids_edit.mp4

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        src_frames=113,                  # 3.767с, речь 0.107-3.601
        speech_end=3.601,
        frames=114,                      # 3.800с — пауза до тела 0.176с
        # рез 57 (1.900) — граница «решить»/«эту», промах 4мс
        shots=[(0, "A1", {}),
               (57, "stock", dict(clip="35954", ss=8.0, cx=0.62, cy=0.5))],
        out=f"{VIDEO_DIR}/kids_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«если ты сдаёшь егэ ты обязан решить эту задачу по теории вероятностей»",
        face_cx=365, face_eye=691, face_d=270,
        caps=[
            (0.107, 1.137, [("если ты ", R_, 42), ("сдаёшь егэ", R_, 52)], "A"),
            (1.137, 1.900, [("ты обязан ", R_, 42), ("решить", R_, 54)], "A"),

            (1.904, 2.510, [("эту ", R_, 44), ("задачу", R_, 52)], "B"),
            (2.510, 3.601, [("по теории ", R_, 42), ("вероятностей", S_, 50)], "B"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        src_frames=155,                  # 5.167с, речь 0.178-5.026
        speech_end=5.026,
        frames=156,                      # 5.200с — пауза до тела 0.136с
        # резы 45 (1.500), 89 (2.9667), 120 (4.000) — все на границах слов.
        # Хвост держим 36 кадров: после ускорения x1.02 план должен остаться
        # не короче 1с (1.200 -> 1.176), 30 кадров дали бы 0.98с.
        shots=[(0, "A1", {}),
               (45, "stock", dict(clip="35954", ss=12.5, cx=0.62, cy=0.5)),
               (89, "A1", {}),
               (120, "stock", dict(clip="35954", ss=19.4, cx=0.62, cy=0.5))],
        out=f"{VIDEO_DIR}/kids_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«это простая задача по теории вероятности но 90% людей дают "
              "на неё неправильный ответ»",
        face_cx=368, face_eye=689, face_d=270,
        caps=[
            (0.178, 1.090, [("это простая ", R_, 42), ("задача", R_, 52)], "A"),
            (1.090, 1.500, [("по ", R_, 44), ("теории", R_, 52)], "A"),

            (1.500, 2.374, [("вероятности ", R_, 42), ("но", R_, 50)], "B"),

            # 2.374-2.975 «90 процентов» — риторическая оценка без источника,
            # на экран не выносится (см. шапку файла)
            (2.975, 3.907, [("людей ", R_, 44), ("дают", R_, 52)], "A"),

            (4.011, 5.026, [("на неё ", R_, 42), ("неправильный ответ", S_, 46)], "B"),
        ],
    ),
}


def shots_of(cfg):
    """Планы блока хука в секундах: (t0, t1, kind, prm)."""
    fr = cfg["shots"] + [(cfg["frames"], None, None)]
    return [(fr[i][0] / FPS, fr[i + 1][0] / FPS, fr[i][1], fr[i][2])
            for i in range(len(cfg["shots"]))]


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
            fr = last_img          # только под стоковым планом, лицо не морозим
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
        if f % 5 == 0:
            sheet.append((t, np.array(rgb)))
    cap.release()
    R._STOCK.release()

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
    face_n = sum(1 for f in range(nf_hook)
                 if R.shot_at(f / FPS)[2] in R.FACE_KINDS)
    print(f"видео: хук {nf_hook} кадров (лицо {face_n}, сток {nf_hook - face_n}, "
          f"из них {held} сверх дубля) + тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook36.wav"
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
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    # голос тела кладётся на кадр N-2 — на два кадра раньше картинки тела
    n_hook = int(round((nf_hook - 2) / FPS * SR))
    n_bod = int(round((n_body + 2) / FPS * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    if len(body) < n_bod:
        body = np.vstack([body, np.zeros((n_bod - len(body), 2), np.float32)])
    else:
        body = body[:n_bod]
    r = int(0.008 * SR)
    hook = hook.copy()
    body = body.copy()
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
    for t0 in [s[0] for s in shots[1:]]:            # резы внутри блока хука
        add(wh, t0, peak * 0.040)
    add(wh, nf_hook / FPS, peak * 0.040)            # шов: настоящий рез B -> A
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
    csp = f"{BUILD}/test/hook36_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_{SB.TAG}{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

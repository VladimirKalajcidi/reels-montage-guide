"""Ролик 31 («парадокс инспекции»), альтернативные хуки.

Тело ролика (всё с кадра 171) не пересобирается — берётся готовыми кадрами из
`assets/_video_inspection.mp4` и склеивается за новым хуком. Заново рисуется
только блок хука. render31.py этот скрипт не трогает: графика хука собрана
из тех же примитивов (`_axis` / `_tick` / `_band` / `_drop` / `draw_number`),
но со своим расписанием стадий, чтобы ревилы падали на слова НОВОГО дубля.

    python3 hook31.py h1
    python3 hook31.py h2

--- точка реза ------------------------------------------------------------
Старое начало — «если автобусы ходят в среднем раз в 10 минут, это ещё
не значит, что вы будете ждать в среднем 5 минут» (0.00–5.157) — занимало три
плана: `A1` 0.000–1.118, `schedTicks` 1.118–2.444, `naiveWait` 2.444–5.669.
Граница плана 5.669 (`naiveWait` -> `A1` «итак допустим...») и есть точка реза:
`CUT_F = ceil(5.669 * 30) = 171`. Планы 1–3 и их пять субтитров выброшены
целиком; ни одного кадра и ни одного субтитра старого хука не остаётся.

--- почему звук тела берётся с 5.669, а картинка с 5.700 -------------------
Границы планов в этом ролике поставлены по НАЧАЛАМ слов, поэтому граница
5.669 совпадает не с тишиной, а с атакой слова «Итак». Огибающая source.mov
(шаг 10мс): 5.62 −82 дБ, 5.63 −57, 5.66 −38, 5.67 −33, 5.70 −25, пик слова
−9 дБ. То есть кадр 171 (5.700) стоит уже на 35% амплитуды слова — обрезав
звук по нему, мы срезали бы атаку «И» и получили бы «...так допустим».

Поэтому картинка тела идёт с кадра 171 (5.700 — иначе в версию попал бы кадр
старого хука), а голос тела берётся с 5.669, с настоящей атаки слова. Разница
ровно один кадр, 31мс, и знак у неё безопасный: КАРТИНКА ИДЁТ ВПЕРЕДИ ЗВУКА
(звук задержан). Порог заметности для задержки звука ~125мс против ~45мс для
опережения — 31мс лежит вчетверо ниже порога. Смещение постоянное на всё тело,
никаких склеек внутри дорожки нет. SFX при этом считаются от КАРТИНКИ
(`dt = seam - CUT_V`), чтобы whoosh падал ровно на кадр реза.

--- передача эстафеты -----------------------------------------------------
Тело начинается самостоятельным предложением: «итак, допустим, интервалы между
автобусами разные». Подлежащее своё, отсылок к выброшенному тексту нет — оба
новых хука говорят про автобусы и ожидание, так что «интервалы между
автобусами» подхватывается без дефицита антецедента.

hook1 кончается на «и этому есть математическое объяснение», hook2 — на
«давайте разберёмся, почему это так работает». Оба ведут ровно туда же, куда
вело старое начало.

--- графика хука ----------------------------------------------------------
Новых объектов не нарисовано: оба блока используют ту же графику, что стояла
в старом начале, — линию расписания с равными промежутками (`schedTicks`)
и один промежуток с меткой прихода и полосой ожидания (`naiveWait`).
Числа показываются только там, где они звучат: у hook2 это «10» и «5»
(те же, что в старом хуке), у hook1 чисел в речи нет — и на экране их нет,
графика идёт без ревилов и без подсветки размеченного промежутка.

Оба блока КОНЧАЮТСЯ ГРАФИКОЙ, а не лицом: первый план тела (`A1` 5.669–8.374)
— говорящая голова, и стык «лицо другого дубля -> лицо» читался бы как
джамп-кат.

--- формат хуков ----------------------------------------------------------
Оба дубля совпадают с основным исходником по всем тегам: 720x1280, yuv420p,
color_range=tv, bt709/bt709/bt709, 30 fps. Приведение цветового пространства
не требуется, HDR ни в одном файле нет.

Крупность дублей (детектор, после hflip, среднее по дублю):
    основной  cx 401  линия глаз 701  ширина лица 273
    hook1     cx 385  линия глаз 699  ширина лица 275   (cx на 16px левее)
    hook2     cx 399  линия глаз 705  ширина лица 279   (совпадает)
Кропом это НЕ правится: оба дубля идут через те же FRAMINGS, что и основной
исходник, числа названы в сдаче.

--- пауза хук -> тело -------------------------------------------------------
Оба дубля кончаются речью у самого конца файла (хвостовой тишины нет:
hook1 речь до 7.813 при длине 7.833, hook2 до 9.010 при длине 9.033).
Пауза набирается кадрами блока хука: hook1 239 кадров (7.967с) -> пауза
0.154с, hook2 275 кадров (9.167с) -> пауза 0.157с. Оба в диапазоне
0.10–0.20с из delivery-specs §6. Кадры сверх A-roll добиваются графическим
планом — последним планом обоих блоков стоит графика, лицо не морозится.
"""
import json
import math
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render31 as R
import storyboard31 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/31"
BODY = f"{BUILD}/assets/_video_{SB.TAG}.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_T = 5.669                                  # граница планов naiveWait -> A1
CUT_F = math.ceil(CUT_T * FPS - 1e-6)          # 171 — граница плана в кадрах
CUT_V = CUT_F / FPS                            # 5.700с — с этого кадра идёт картинка тела
CUT_A = CUT_T                                  # 5.669с — с этой точки идёт голос тела

TARGET_LUFS = -14.0                            # замер сданного inspection_edit.mp4

R_ = "r"
S_ = "s"


# --------------------------------------------------------------- графика хука
# Те же объекты, что в старом начале, но со своим расписанием стадий: ревил
# числа должен падать на слово НОВОГО дубля, а не на тайминг старого.

def _sched(mark, num_at):
    """Линия расписания: равные промежутки по 10 минут (плана `schedTicks`).
    mark=False — без яркой подсветки размеченного промежутка и без числа."""
    def draw(lay, lt):
        d = ImageDraw.Draw(lay)
        R._axis(d, R.L_X0, R.L_X1, R.AX_Y, R.ease_out(R.clamp01(lt / 0.24)))
        prog = R.stagger(lt, SB.UNIFORM_N + 1, t0=0.10, step=0.032, dur=0.22)
        for i, x in enumerate(R.UNI_X):
            R._tick(d, x, R.AX_Y, prog[i])
        for i in range(SB.UNIFORM_N):
            R._band(d, R.UNI_X[i] + 4, R.UNI_X[i + 1] - 4, prog[i + 1], alpha=0.26)
        if mark:
            R._band(d, R.UNI_X[R.UNI_MARK], R.UNI_X[R.UNI_MARK + 1],
                    R.ease_out(R.clamp01((lt - 0.46) / 0.24)))
        if num_at is not None:
            q = R.pop(lt, num_at)
            if q > 0:
                cx = (R.UNI_X[R.UNI_MARK] + R.UNI_X[R.UNI_MARK + 1]) / 2
                R.draw_number(lay, str(SB.SCHED_MEAN), (cx, R.NUM_Y), R.NUM_BIG,
                              color=WHITE, glow=WHITE, glow_a=0.55,
                              scale=0.6 + 0.4 * q)
    return draw


def _wait(arrow_at, band_at, num_at):
    """Один промежуток крупно: метка прихода и полоса ожидания (план `naiveWait`)."""
    def draw(lay, lt):
        d = ImageDraw.Draw(lay)
        R._axis(d, R.G_X0, R.G_X1, R.AX_Y, R.ease_out(R.clamp01(lt / 0.22)))
        p0 = R.stagger(lt, 2, t0=0.06, step=0.10, dur=0.24)
        R._tick(d, R.G_X0, R.AX_Y, p0[0])
        R._tick(d, R.G_X1, R.AX_Y, p0[1])
        R._band(d, R.G_X0, R.G_X1, R.ease_out(R.clamp01((lt - 0.20) / 0.28)), alpha=0.26)
        R._drop(d, R.G_MID, R.ease_out(R.clamp01((lt - arrow_at) / 0.30)))
        R._band(d, R.G_MID, R.G_X1, R.ease_out(R.clamp01((lt - band_at) / 0.32)))
        if num_at is not None:
            q = R.pop(lt, num_at)
            if q > 0:
                R.draw_number(lay, str(SB.NAIVE_WAIT), ((R.G_MID + R.G_X1) / 2, R.NUM_Y),
                              R.NUM_BIG, color=WHITE, glow=WHITE, glow_a=0.55,
                              scale=0.6 + 0.4 * q)
    return draw


HOOK_GFX = {
    # hook1 — чисел в речи нет, поэтому и на экране их нет
    "schedPlain": _sched(mark=False, num_at=None),
    "waitPlain": _wait(arrow_at=0.55, band_at=1.10, num_at=None),
    # hook2 — «раз в 10 минут» и «больше 5 минут»: те же числа, что в старом хуке
    "schedTen": _sched(mark=True, num_at=1.082),
    "waitFive": _wait(arrow_at=0.45, band_at=1.20, num_at=1.810),
}

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        src_frames=235,                  # 7.833с, речь до 7.813
        frames=239,                      # 7.967с — пауза до тела 0.154с
        out=f"{VIDEO_DIR}/inspection_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«на автобусных остановках автобусы иногда приходится ждать "
              "дольше, чем должно быть по расписанию»",
        face_cx=385, face_eye=699, face_d=275,
        num_reveals=[],                  # чисел в блоке хука нет
        shots=[
            (0.000, 1.350, "A1", {}),
            (1.350, 3.150, "schedPlain", {}),
            (3.150, 4.664, "A1", {}),
            (4.664, 6.085, "schedPlain", {}),
            (6.085, 239 / 30, "waitPlain", {}),
        ],
        caps=[
            (0.000, 0.491, [("я думаю ", R_, 44), ("вы", R_, 48)], "A"),
            (0.491, 1.350, [("замечали ", R_, 44), ("что на", R_, 48)], "A"),

            (1.350, 2.414, [("автобусных ", R_, 42), ("остановках", R_, 50)], "G"),
            (2.414, 3.150, [("автобусы ", R_, 44), ("иногда", R_, 48)], "G"),

            (3.150, 4.132, [("приходится ", R_, 42), ("ждать", R_, 50)], "A"),
            (4.132, 4.664, [("дольше ", R_, 46), ("чем", R_, 42)], "A"),

            (4.664, 5.094, [("должно ", R_, 44), ("быть", R_, 48)], "G"),
            (5.094, 5.769, [("по ", R_, 42), ("расписанию", S_, 52)], "G"),

            (6.085, 6.484, [("и этому ", R_, 42), ("есть", R_, 48)], "G"),
            (6.484, 7.813, [("математическое ", R_, 42), ("объяснение", R_, 50)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        src_frames=271,                  # 9.033с, речь до 9.010
        frames=275,                      # 9.167с — пауза до тела 0.157с
        out=f"{VIDEO_DIR}/inspection_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«автобусы ходят раз в 10 минут, а ждёте вы почему-то всегда "
              "больше 5 минут»",
        face_cx=399, face_eye=705, face_d=279,
        num_reveals=[1.373 + 1.082, 3.183 + 1.810],     # 2.455 («10») и 4.993 («5»)
        shots=[
            (0.000, 1.373, "A1", {}),
            (1.373, 3.183, "schedTen", {}),
            (3.183, 5.742, "waitFive", {}),
            (5.742, 7.341, "A1", {}),
            (7.341, 275 / 30, "schedPlain", {}),
        ],
        caps=[
            (0.249, 0.665, [("я ", R_, 42), ("думаю", R_, 48)], "A"),
            (0.665, 1.373, [("у вас было ", R_, 42), ("такое", R_, 50)], "A"),

            (1.373, 2.371, [("автобусы ходят ", R_, 42), ("раз", R_, 50)], "G"),
            # 2.371-3.183 — «в 10 минут»: число несёт линия расписания

            (3.183, 3.765, [("а ждете ", R_, 44), ("вы", R_, 46)], "G"),
            (4.369, 4.993, [("всегда ", R_, 44), ("больше", R_, 50)], "G"),
            # 4.993-5.742 — «5 минут»: число несёт график

            (5.742, 6.366, [("то есть ", R_, 42), ("половина", S_, 52)], "A"),
            (6.366, 7.341, [("от этого ", R_, 42), ("времени", R_, 48)], "A"),

            (7.341, 8.049, [("давайте ", R_, 42), ("разберемся", R_, 50)], "G"),
            (8.049, 9.010, [("почему это ", R_, 42), ("так работает", R_, 48)], "G"),
        ],
    ),
}


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
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()
    R.GFX.update(HOOK_GFX)

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

        canvas = R.background(kind, prm, fr, t)
        gl = R.graphics_layer(kind, t - t0)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 6 == 0:
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
    print(f"видео: хук {nf_hook} кадров (из них {held} добито графикой) "
          f"+ тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook31.wav"
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
    hook = wav_of(cfg["src"])
    body = wav_of(SRC, CUT_A)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = int(round(nf_hook / FPS * SR))
    n_bod = int(round(n_body / FPS * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    if len(body) < n_bod:
        body = np.vstack([body, np.zeros((n_bod - len(body), 2), np.float32)])
    else:
        body = body[:n_bod]
    r = int(0.008 * SR)
    hook[-r:] *= np.linspace(1, 0, r)[:, None]
    body = body.copy()
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

    seam = nf_hook / FPS
    dt = seam - CUT_V              # SFX считаются от КАРТИНКИ, не от голоса
    wh, im, tk = low_whoosh(), impact(), tick()
    for t0 in [s[0] for s in cfg["shots"][1:]]:
        add(wh, t0, peak * 0.040)
    add(wh, seam, peak * 0.040)                      # шов «хук -> тело»
    for t0, _t1, _k, _p in SB.SHOTS:
        if t0 > CUT_V + 0.01:
            add(wh, t0 + dt, peak * 0.040)
    for t in cfg["num_reveals"]:
        add(im, t, peak * 0.070)
    for t in SB.NUM_REVEALS:
        if t >= CUT_V:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT_V:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t0, _t1, kind, _p in cfg["shots"]:
        if kind not in R.FACE_KINDS:
            for k in range(3):
                add(tk, t0 + 0.12 + k * 0.085, peak * 0.030)

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
            af += f",volume={post_db:+.2f}dB,alimiter=limit=0.78:level=disabled"
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
    csp = f"{BUILD}/test/hook31_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_{SB.TAG}{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

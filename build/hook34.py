"""Ролик 34 («парадокс крокодила»), альтернативные хуки.

Тело ролика (всё с кадра 99) не пересобирается — берётся готовыми кадрами из
`assets/_video_croc.mp4` и склеивается за новым хуком. Заново рисуется только
блок хука. render34.py этот скрипт не трогает.

    python3 hook34.py h1
    python3 hook34.py h2

--- точка реза ------------------------------------------------------------
Старое начало — «эта логическая задача существует больше 2000 лет»
(0.000–2.804). Оно занимает план `A1` 0.000–1.891 и хвостом заходит в план
`ages` 1.891–3.283, где на сетке выстраиваются двадцать засечек-веков и
всплывает белое 2000. Эта графика иллюстрирует ровно ту фразу, которую мы
выбрасываем: **ни один из новых дублей возраст задачи не называет**, поэтому
план `ages` уходит из версий целиком вместе с хуком.

Рез — по границе плана `ages` -> `stock`, то есть по 3.283. В кадрах это
ceil(3.283 x 30) = **99** (округление вверх, а не к ближайшему: кадр 98
показывает ещё сетку с числом 2000, и в версии от старого хука остался бы
целый кадр графики).

--- шов внутри плана, а не на резе ------------------------------------------
Тело начинается стоковой вставкой с крокодилом (3.283–5.330). Блок хука
кончается **тем же планом**: последние кадры блока — тот же клип, отмотанный
назад по времени от шва (`rel = 0.017 - (N - f)/30`), поэтому на шве не
меняется ни карточка, ни кадр — плана два, а рез между ними один.
Whoosh на шве не ставится: визуального реза там нет.

Единственный рез блока хука — `A1` -> `stock`:
    hook1  кадр 49 (1.6333) — граница «очень»/«простой», промах 14мс;
    hook2  кадр 47 (1.5667) — граница «задач»/«на», промах 8мс.
Оба промаха меньше 20мс, в середину слова рез не попадает.

--- голос на шве: J-cut в один кадр -----------------------------------------
Слово «крокодил» в основном дубле начинается в 3.283, то есть **раньше**
кадра 99 (3.300): по огибающей 3.260 = -67 дБ, 3.270 = -50, 3.280 = -28,
3.300 = -21. Рез голоса ровно по картинке срезал бы 17мс уже звучащего слова.
Поэтому голос тела берётся с кадра 98 (`CUT_A` = 3.2667, там -55 дБ, тишина)
и кладётся в новую дорожку на кадр N-1 — на один кадр раньше картинки тела.
Синхрон при этом не едет: смещение кратно кадру, и каждый кадр тела стоит
против своего звука. На последнем кадре блока хука звучат 33мс тишины
из тела — это обычный J-cut, а не рассинхрон.

--- пауза хук -> тело -------------------------------------------------------
    hook1  речь кончается 2.568, слово тела начинается 2.716 -> пауза 0.148с
    hook2  речь кончается 2.307, слово тела начинается 2.450 -> пауза 0.143с
Оба значения в диапазоне 0.10–0.20с из delivery-specs §6. Хвостовая тишина
в обоих дублях есть (hook1 2.568 при длине 2.667, hook2 2.307 при 2.567),
последнее слово записью не обрезано.

--- передача эстафеты -------------------------------------------------------
Тело начинается самостоятельным предложением «крокодил похищает ребёнка
и говорит отцу»: своё подлежащее, ни одного местоимения, отсылок к
выброшенному тексту нет. Оба новых дубля кончаются словом «парадокс» /
«логику» и ведут ровно туда же, куда вело старое начало.

--- формат хуков ------------------------------------------------------------
Оба дубля совпадают с основным исходником по всем тегам: 720x1280, yuv420p,
color_range=tv, bt709/bt709/bt709. Приведение цветового пространства не
требуется, HDR ни в одном файле нет.

Крупность дублей (детектор, после hflip, среднее по дублю):
    основной  cx 377  линия глаз 684  ширина лица 269
    hook1     cx 366  линия глаз 689  ширина лица 278
    hook2     cx 371  линия глаз 691  ширина лица 277
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
import render34 as R
import storyboard34 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/34"
BODY = f"{BUILD}/assets/_video_{SB.TAG}.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 99                                     # первый кадр тела (ceil(3.283*30))
CUT_V = CUT_F / FPS                            # 3.3000с — картинка тела
CUT_A = (CUT_F - 1) / FPS                      # 3.2667с — голос тела, J-cut в кадр
STOCK_T0 = 3.283                               # начало плана stock в теле
STOCK_PRM = dict(clip="27364", ss=7.04, cx=0.5, cy=0.5)

TARGET_LUFS = -14.1                            # замер сданного croc_edit.mp4

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        src_frames=80,                   # 2.667с, речь 0.125-2.568
        speech_end=2.568,
        a1_frames=49,                    # рез 1.6333 — граница «очень»/«простой»
        frames=82,                       # 2.7333с — пауза до тела 0.148с
        out=f"{VIDEO_DIR}/croc_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«это очень известный при этом очень простой парадокс»",
        face_cx=366, face_eye=689, face_d=278,
        caps=[
            (0.000, 0.993, [("это очень ", R_, 42), ("известный", R_, 52)], "A"),
            (0.993, 1.633, [("при этом ", R_, 42), ("очень", R_, 52)], "A"),

            (1.633, 2.568, [("простой ", R_, 42), ("парадокс", S_, 56)], "B"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        src_frames=77,                   # 2.567с, речь 0.000-2.307
        speech_end=2.307,
        a1_frames=47,                    # рез 1.5667 — граница «задач»/«на»
        frames=74,                       # 2.4667с — пауза до тела 0.143с
        out=f"{VIDEO_DIR}/croc_hook2.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«это одна из самых запутанных задач на логику»",
        face_cx=371, face_eye=691, face_d=277,
        caps=[
            (0.000, 0.684, [("это одна ", R_, 42), ("из самых", R_, 50)], "A"),
            # 1.566 = кадр 47, ровно рез A1 -> stock: блок субтитра обрывается
            # на последнем кадре плана с лицом, следующий стартует на первом
            # кадре стока (иначе строка живёт один кадр через рез)
            (0.684, 1.566, [("запутанных ", R_, 42), ("задач", S_, 56)], "A"),

            (1.566, 2.307, [("на ", R_, 42), ("логику", S_, 56)], "B"),
        ],
    ),
}


def shots_of(cfg):
    a1 = cfg["a1_frames"] / FPS
    return [(0.000, a1, "A1", {}),
            (a1, cfg["frames"] / FPS, "stock", dict(STOCK_PRM))]


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
    a1n = cfg["a1_frames"]
    # кадры стока в блоке хука — продолжение того же клипа назад от шва:
    # на кадре N клип стоит на 7.04 + (CUT_V - STOCK_T0), значит на кадре f
    # он должен стоять на столько же минус (N - f)/FPS. Это ровно t - t0
    # при постоянном t0 = N/FPS - (CUT_V - STOCK_T0).
    stock_t0 = nf_hook / FPS - (CUT_V - STOCK_T0)
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

        canvas = R.background(kind, prm, fr, t, stock_t0 if kind == "stock" else t0)
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
    print(f"видео: хук {nf_hook} кадров (A-roll {a1n}, сток {nf_hook - a1n}, "
          f"из них {held} сверх дубля) + тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook34.wav"
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

    # голос тела кладётся на кадр N-1 — на один кадр раньше картинки тела
    n_hook = int(round((nf_hook - 1) / FPS * SR))
    n_bod = int(round((n_body + 1) / FPS * SR))
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
    for t0 in [s[0] for s in shots[1:]]:            # единственный рез блока хука
        add(wh, t0, peak * 0.040)
    # на шве whoosh не ставится: визуального реза там нет, план stock продолжается
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
    csp = f"{BUILD}/test/hook34_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_{SB.TAG}{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

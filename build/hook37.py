"""Ролик 37 («парадокс трёх карт»), альтернативные хуки.

Тело ролика (всё с кадра 111) не пересобирается — берётся готовыми кадрами из
`assets/_video_cards.mp4` и склеивается за новым хуком. Заново рисуется только
блок хука. render37.py этот скрипт не трогает.

    python3 hook37.py h1
    python3 hook37.py h2

--- точка реза ------------------------------------------------------------
Старое начало — «если ты учил в школе теорию вероятности, ты должен решить
эту задачку» (0.090–3.354). Оно занимает два плана: `A1` 0.000–1.800 и
`stock` 1.800–3.6667. Ближайшая граница плана после конца фразы — **3.6667**,
по ней и режем: всё раньше выбрасывается целиком.

В кадрах это ceil(3.6667 x 30) = **111** (округление вверх, а не к
ближайшему: кадр 110 стоит на 3.6667 и по `shot_at` принадлежит ещё
стоковому плану старого хука — при round от него остался бы целый кадр).
Проверено прямым перебором: `shot_at(110/30)` = stock, `shot_at(111/30)` =
cards3.

Тело начинается планом `cards3` (3.6667–5.2333), у которого на 3.72 стартует
каскад: первый кадр тела — пустая сетка, ровно как в сданной версии.

--- шов ---------------------------------------------------------------------
Шов совпадает с настоящим резом: блок хука кончается стоковой вставкой
(карточка B) у h1 и планом с лицом (карточка A) у h2, тело начинается сеткой
(карточка A). Whoosh на шве ставится как на любой склейке.

--- голос на шве: J-cut в один кадр -----------------------------------------
Первое слово тела («итак») по огибающей атакует в 3.705:
    3.690 = -84 дБ, 3.700 = -68, 3.710 = -49, 3.720 = -39, 3.730 = -33.
Картинка тела начинается кадром 111 (3.7000). Голос тела берётся с кадра 110
(`CUT_A` = 3.66667, там -86 дБ, чистая тишина) и кладётся в новую дорожку на
один кадр раньше картинки. Синхрон не едет: смещение кратно кадру.

--- пауза хук -> тело -------------------------------------------------------
    hook1  речь кончается 3.150, слово тела звучит в 3.323 -> пауза 0.173с
    hook2  речь кончается 4.365, слово тела звучит в 4.553 -> пауза 0.188с
           (после ускорения x1.02 — 0.184с)
Считается по расшифровке тела («итак» на 3.723), то есть по верхней оценке:
по огибающей атака на 18мс раньше, и настоящая пауза ещё короче.
Оба значения в диапазоне 0.10–0.20с из delivery-specs §6.

--- хвостовая тишина: у hook1 её нет ----------------------------------------
hook2 после последнего слова (4.365) даёт 0.17с настоящей тишины — дубль
записью не обрезан.
**hook1 хвостовой тишины не имеет**: слово кончается 3.150, а с 3.22 до конца
файла (3.325) идёт дыхание на -27...-37 dB. Само слово целое, обрезано не оно,
но если оставить хвост как есть, в паузе перед телом будет слышен вдох.
Поэтому дорожка хука гасится фейдом с 3.170 за 0.080с (`tail_fade`) — пауза
получается тихой, а последнее слово не тронуто.

--- передача эстафеты -------------------------------------------------------
Тело начинается самостоятельным предложением «итак, перед вами три карты»:
своё подлежащее, ни одного местоимения, отсылок к выброшенному тексту нет.
Оба новых дубля кончаются словами «...с небольшим подвохом» / «...обязан
решить эту задачу» и ведут ровно туда же, куда вело старое начало.

--- «подвох» в hook1 --------------------------------------------------------
Дубль 1 говорит «но она с небольшим подвохом», а в теле на 24.93 стоит R6
«ПОДВОХ». Это не дублирование по `brand-kit §5`: правило запрещает повтор
в ОДНОМ кадре, а здесь между ними 21 секунда, и R6 работает как расплата за
обещание из хука. Субтитр хука слово оставляет: без него фраза разваливается.

--- графика блока хука ------------------------------------------------------
Та же, что стояла в старом хуке: стоковая вставка с руками и колодой
(mixkit 100377). Другой графики в старом начале не было, менять её незачем.
Окна клипа взяты новые и не пересекаются между версиями:
    v1     ss=2.920 (2.920–4.787)
    hook1  ss=4.850 (4.850–6.250)
    hook2  ss=1.210 (1.210–2.877)
Иначе один и тот же кадр стока стоял бы в двух версиях подряд, и площадка
честно считала бы их дублями.

--- формат хуков ------------------------------------------------------------
Оба дубля совпадают с основным исходником по всем тегам: 720x1280, yuv420p,
color_range=tv, bt709/bt709/bt709. Приведение цветового пространства не
требуется, HDR ни в одном файле нет. Зеркальность проверена по кадру:
надпись на футболке в обоих дублях читается задом наперёд, штатный `hflip`
подходит обоим.

Крупность дублей (детектор, после hflip, среднее по дублю):
    основной  cx 378  линия глаз 684  ширина лица 264
    hook1     cx 372  линия глаз 696  ширина лица 277
    hook2     cx 371  линия глаз 718  ширина лица 279
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
import render37 as R
import storyboard37 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/37"
BODY = f"{BUILD}/assets/_video_{SB.TAG}.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 111                                    # первый кадр тела (ceil(3.6667*30))
CUT_V = CUT_F / FPS                            # 3.7000с — картинка тела
CUT_A = (CUT_F - 1) / FPS                      # 3.6667с — голос тела, J-cut в 1 кадр

TARGET_LUFS = -14.2                            # замер сданного cards_edit.mp4

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        src_frames=100,                  # 3.333с, речь 0.120-3.150
        speech_end=3.150,
        frames=99,                       # 3.300с — пауза до тела 0.173с
        tail_fade=(3.170, 0.080),        # хвостовой тишины в дубле нет, см. шапку
        # рез 58 (1.9333) — пауза «вероятности»/«но», энергия 0.0002
        shots=[(0, "A1", {}),
               (58, "stock", dict(clip="100377", ss=4.850, cx=0.35))],
        out=f"{VIDEO_DIR}/cards_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«это лёгкая задача по теории вероятности но она с небольшим подвохом»",
        face_cx=372, face_eye=696, face_d=277,
        caps=[
            (0.120, 0.989, [("это ", R_, 46), ("лёгкая задача", R_, 50)], "A"),
            (0.989, 1.9333, [("по теории ", R_, 42), ("вероятности", S_, 50)], "A"),

            (1.978, 2.443, [("но ", R_, 46), ("она", R_, 52)], "B"),
            (2.443, 3.150, [("с небольшим ", R_, 42), ("подвохом", S_, 50)], "B"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        src_frames=136,                  # 4.533с, речь 0.180-4.365
        speech_end=4.365,
        frames=136,                      # 4.533с — пауза до тела 0.173с
        tail_fade=None,                  # 0.17с настоящей тишины в дубле есть
        # резы 40 (1.3333, энергия 0.0034) и 90 (3.0000, энергия 0.0003) —
        # оба в паузах между словами. Хвостовой план держим 46 кадров: после
        # ускорения x1.02 он должен остаться не короче 1с (1.533 -> 1.503).
        shots=[(0, "A1", {}),
               (40, "stock", dict(clip="100377", ss=1.210, cx=0.35)),
               (90, "A1", {})],
        out=f"{VIDEO_DIR}/cards_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«если ты сдаёшь егэ и знаешь что такое условная вероятность "
              "ты обязан решить эту задачу»",
        face_cx=371, face_eye=718, face_d=279,
        caps=[
            (0.180, 0.489, [("если ", R_, 46), ("ты", R_, 52)], "A"),
            (0.489, 1.3333, [("сдаёшь ", R_, 44), ("егэ", R_, 54)], "A"),

            (1.355, 1.870, [("знаешь ", R_, 44), ("что", R_, 52)], "B"),
            (1.870, 2.984, [("такое условная ", R_, 40), ("вероятность", S_, 48)], "B"),

            (3.000, 3.458, [("ты ", R_, 46), ("обязан", R_, 52)], "A"),
            (3.458, 4.365, [("решить эту ", R_, 42), ("задачу", R_, 52)], "A"),
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
    tmp = f"{BUILD}/assets/_tmp_hook37.wav"
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

    tf = cfg.get("tail_fade")
    if tf:
        i0 = int(tf[0] * SR)
        n_f = int(tf[1] * SR)
        if i0 < len(hook):
            m = min(n_f, len(hook) - i0)
            hook[i0:i0 + m] *= np.linspace(1, 0, m)[:, None]
            hook[i0 + m:] = 0.0
            print("хук: хвост погашен фейдом с %.3fс за %.3fс "
                  "(хвостовой тишины в дубле нет)" % tf)

    # голос тела кладётся на кадр N-1 — на кадр раньше картинки тела
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
    for t0 in [s[0] for s in shots[1:]]:            # резы внутри блока хука
        add(wh, t0, peak * 0.040)
    add(wh, nf_hook / FPS, peak * 0.040)            # шов: настоящий рез
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
    csp = f"{BUILD}/test/hook37_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_{SB.TAG}{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

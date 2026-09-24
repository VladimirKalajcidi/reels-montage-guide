"""Ролик 11 («великая теорема Ферма»), альтернативные хуки.

Тело ролика (всё с 7.600с) не пересобирается — берётся готовыми кадрами из
assets/_video_fermat.mp4 и склеивается за новым хуком. Заново рисуется только хук:
свой A-roll, свои субтитры, та же графика (замена показателя 2 -> n, треугольник
Пифагора), что стояла в старом хуке — это визуальная интрига начала.

    python3 hook11.py h1
    python3 hook11.py h2

Точка реза. Хуковая фраза старого начала кончается на 7.200 («…нет решений вообще»),
дальше тело: «Уравнение x в степени n…». Граница плана — 7.680 (zero -> eqn), но
акустический онсет «Уравнение» замерен на 7.620, то есть на 60мс раньше, чем его
показывает Whisper. Рез по кадровой границе 7.700 (кадр 231) срезал бы 80мс «У».
Поэтому звук тела берётся с 7.600, а три кадра 228…230 (они ещё принадлежат плану
zero старого хука) не переносятся, а перерисовываются заново как начало плана eqn —
там всё равно только сетка: формула проявляется с 8.030. Ни одного кадра старого
хука в версии не остаётся, и при этом слово не обрезано.

Формат хуков. Дубли сняты в 720x1280 и в полном диапазоне (yuvj420p / pc), основной
исходник — 1080x1920, tv. И то и другое выводится из тегов файла, значит приводится
всегда, до тракта A-roll: апскейл 1.5x (кадрирование FRAMINGS не трогаем — оно в
относительных долях остаётся тем же) и full -> limited. Экспозицию, баланс белого и
крупность НЕ подгоняем — расхождения замерены и названы в сдаче, см. README ниже.

Замеры (средний RGB внутри карточки A / высота лица при карточке 1380px):
    исходник  105/81/69   лицо 417px   центр 651px
    хук1      122/101/88  лицо 384px   центр 738px   (+18/+20/+19, лицо -8%)
    хук2      136/114/101 лицо 364px   центр 763px   (+32/+33/+32, лицо -13%)
Норма расхождения по гайду — до ~10 единиц на канал. Оба дубля ярче и сняты дальше;
это решается на съёмке (свет, дистанция), а не кропом и не лутом на монтаже.
Зеркальность проверена по надписи на толстовке: во всех трёх дублях текст читается
нормально после штатного hflip — исключений не нужно.
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
import render11 as R
import storyboard11 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/11"
BODY = f"{BUILD}/assets/_video_fermat.mp4"
STOCK_DIR = f"{VIDEO_DIR}/stock"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT = 7.600                  # откуда берём звук тела (до онсета «Уравнение» 7.620)
CUT_F = 231                  # первый переносимый кадр тела (7.700)
BRIDGE = [228, 229, 230]     # кадры, которые перерисовываем как начало плана eqn
TARGET_LUFS = -14.2          # как в уже сданном fermat_edit.mp4 (замер: -14.2)

# приведение формата хука к формату основного исходника (объективное, по тегам)
CONFORM = ("scale=1080:1920:flags=lanczos:in_range=full:out_range=limited,"
           "format=yuv420p")

S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        trim=1.900,                        # снимаем 1.9с молчания перед дублем
        frames=117,                        # 3.900с; речь кончается на 3.760
        out=f"{VIDEO_DIR}/fermat_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=lanczos",
        post_a=None,
        title="«доказательства этой теории мы, математики, ждали около 400 лет»",
        shots=[
            (0.000, 1.420, "A1",     {}),
            (1.420, 2.860, "swap",   {}),
            (2.860, 117 / 30, "num400", {}),
        ],
        caps=[
            (0.140, 0.860, [("доказательства", S, 50)], "A"),
            (0.860, 1.420, [("этой теории", S, 52)], "A"),
            (1.440, 2.160, [("мы математики", S, 48)], "G"),
            (2.160, 2.860, [("ждали около", S, 50)], "G"),
            # 2.860-3.900 — «400 лет» показывает графика, субтитра нет
        ],
        num_reveals=[2.920],
        cascades=[1.480],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        trim=4.100,                        # снимаем 4.1с молчания перед дублем
        frames=203,                        # 6.767с; речь кончается на 6.640
        out=f"{VIDEO_DIR}/fermat_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«это одна из самых великих теорем в истории математики»",
        # название теоремы отдано лицу, а не графике: держать один график 2.25с
        # нечем — его анимация кончается за 0.9с, и хвост встаёт мёртвым кадром
        shots=[
            (0.000, 1.860, "A1",   {}),
            (1.860, 3.240, "pyth", {}),
            (3.240, 4.520, "swap", {}),
            (4.520, 203 / 30, "A2", {}),
        ],
        caps=[
            (0.140, 1.020, [("это одна из", S, 50)], "A"),
            (1.020, 1.860, [("самых великих", S, 50)], "A"),
            (1.860, 2.640, [("теорем в истории", S, 46)], "G"),
            (2.640, 3.440, [("математики", S, 52)], "G"),
            (3.780, 4.520, [("и называется она", S, 48)], "G"),
            (4.520, 5.340, [("соответствующе", S, 48)], "A"),
            # Whisper слышит фамилию как «фирма» — на экране правильное написание
            (5.340, 203 / 30, [("великая теорема ферма", "s", 58)], "A"),
        ],
        num_reveals=[],
        cascades=[1.920, 3.300],
    ),
}


# ------------------------------------------------------- графика только для хука
def g_num400(lay, lt):
    """R5b: «около 400 лет». В теле синих чисел остаётся три (1637/358/1994),
    старый синий «0» уезжает вместе с хуком — бюджет в четыре числа соблюдён."""
    p, o = R.blue_pop(lt, 0.38)
    if p <= 0:
        return
    lay.alpha_composite(R.text_layer((W, H), [
        R.gtext("400", (R.GCX, 1100), int(212 * (0.55 + 0.45 * p) * o), blue=True),
        R.gtext("лет", (R.GCX, 1285), 62, op=R.clamp01((lt - 0.38) / 0.28)),
        R.gtext("ожидания", (R.GCX, 1412), 60, serif=True,
                op=R.clamp01((lt - 0.72) / 0.32)),
    ]))


R.GFX["num400"] = g_num400


# ----------------------------------------------------------------- A-roll хука
def conform(key, cfg):
    """Хук -> формат основного исходника: 1080x1920, limited range, bt709.
    Звук здесь не трогаем, он берётся из оригинала дубля."""
    out = f"{BUILD}/assets/_hook11_{key}_conf.mp4"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{cfg['trim']:.3f}", "-i", cfg["src"],
                        "-vf", CONFORM, "-an",
                        "-color_range", "tv", "-colorspace", "bt709",
                        "-color_trc", "bt709", "-color_primaries", "bt709",
                        "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                        out], check=True)
        print("хук приведён к формату исходника:", os.path.basename(out))
    return out


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_fermat.mp4.

    Раскадровка render11 подменяется на хуковую: SHOTS/CAPS читаются из глобалей
    модуля, поэтому достаточно подменить их и пересобрать раскладку блоков."""
    aroll = conform(key, cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(aroll)
    stock_cap, stock_key = None, None

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    sheet, last = [], None
    for f in range(nf_hook):
        t = f / FPS
        ok, fr = cap.read()
        if ok:
            last = fr
        else:
            fr = last
        t0, t1, kind, prm = R.shot_at(t)
        lt = t - t0

        def read_stock():
            nonlocal stock_cap, stock_key
            k = (t0, prm["clip"])
            if k != stock_key:
                if stock_cap is not None:
                    stock_cap.release()
                stock_cap = cv2.VideoCapture(f"{STOCK_DIR}/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
                stock_key = k
            ok_s, sfr = stock_cap.read()
            if not ok_s:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok_s, sfr = stock_cap.read()
            return sfr

        canvas = R.background(kind, prm, fr, t, read_stock)
        gl = R.graphics_layer(kind, lt)
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
    if stock_cap is not None:
        stock_cap.release()

    # мост: кадры 228…230 тела принадлежат плану zero старого хука, поэтому
    # рисуем их заново как начало плана eqn — там одна сетка, формула с 8.030
    for bf in BRIDGE:
        t = bf / FPS
        canvas = R.background("eqn", {}, None, t)
        ff.stdin.write(canvas.convert("RGB").tobytes())

    # тело: готовые кадры, начиная с CUT_F
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
    print(f"видео: хук {nf_hook} + мост {len(BRIDGE)} + тело {n_body} кадров "
          f"-> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook11.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp]
    subprocess.run(cmd, check=True)
    d = read_wav(tmp)
    os.remove(tmp)
    return d


def speech_rms(x):
    m = np.abs(x).mean(axis=1)
    sel = m > m.max() * 0.12
    return float(np.sqrt((x[sel] ** 2).mean()))


def build_audio(key, cfg, nf_hook, vid, out):
    """Дорожка пересобирается целиком: голос хука + голос тела с точки реза,
    постель со сдвигом (длина_хука − CUT), музыка с нуля, потом loudnorm."""
    hook = wav_of(cfg["src"], cfg["trim"])
    body = wav_of(SRC, CUT)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = nf_hook * SR // FPS
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    # склейка в тишине: гасим последние 8мс хука и первые 8мс тела, чтобы не было щелчка
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

    dt = nf_hook / FPS - CUT                 # сдвиг тела на новой таймлинии
    wh, im, tk = low_whoosh(), impact(), tick()
    # склейки: внутри хука, сам стык хук->тело и все склейки тела после реза
    starts = ([s[0] for s in cfg["shots"][1:]]
              + [nf_hook / FPS]
              + [s[0] + dt for s in SB.SHOTS if s[0] > 7.680])
    for t0 in starts:
        add(wh, t0, peak * 0.040)
    for t in cfg["num_reveals"]:
        add(im, t, peak * 0.070)
    for t in SB.NUM_REVEALS:
        if t > 7.680:
            add(im, t + dt, peak * 0.070)
    for t in cfg["cascades"]:
        for k in range(3):
            add(tk, t + k * 0.085, peak * 0.030)
    for t in SB.CASCADES:
        if t > 7.680:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)

    music = cfg["music"]
    mw = f"{BUILD}/assets/_music_{os.path.basename(music).split('.')[0]}.wav"
    if not os.path.exists(mw):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", music,
                        "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
    print("музыка:", os.path.basename(music))
    mus = read_wav(mw)
    if mus.shape[1] == 1:
        mus = np.repeat(mus, 2, axis=1)
    xf = int(0.25 * SR)
    core, tail = mus[:len(mus) - xf], mus[len(mus) - xf:]
    ramp = np.linspace(0, 1, xf)[:, None]
    loop = core.copy()
    loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
    track = np.tile(loop, (int(np.ceil(n / len(loop))) + 1, 1))[:n]
    vr = np.sqrt((voice ** 2).mean()) + 1e-9
    mr = np.sqrt((track ** 2).mean()) + 1e-9
    track *= (vr / mr) * (10 ** (-19 / 20))
    fi, fo = int(0.8 * SR), int(2.0 * SR)
    track[:fi] *= np.linspace(0, 1, fi)[:, None]
    track[-fo:] *= np.linspace(1, 0, fo)[:, None]
    bed += track.astype(np.float32)

    mix = voice + bed
    m = np.abs(mix).max()
    if m > 0.99:
        mix *= 0.99 / m
    mix_path = f"{BUILD}/assets/_mix_fermat{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Лёгкая деформация кадра/темпа, чтобы площадка не сочла версии дублями.
    Работает по уже собранному немому видео — кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_fermat{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_fermat{key}_post.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
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
    """Догоняем громкость до уровня уже сданного ролика."""
    def run(post_db):
        af = (f"{extra_af}," if extra_af else "") + "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
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
    tmp = f"{BUILD}/assets/_video_fermat{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)
    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook11_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_fermat{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

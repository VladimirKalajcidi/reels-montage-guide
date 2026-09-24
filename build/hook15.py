"""Ролик 15 («доказательство теоремы Пифагора, придуманное Гарфилдом»),
альтернативные хуки.

Тело ролика (всё с 3.6667с) не пересобирается — берётся готовыми кадрами из
assets/_video_garfield.mp4 и склеивается за новым хуком. Заново рисуется только
блок хука: свой A-roll, свои субтитры и та же графика, что стояла в старом начале
(классический чертёж Пифагора) — это визуальная интрига ролика, менять её незачем.

    python3 hook15.py h1
    python3 hook15.py h2

Точка реза 3.6667с. Хуковая фраза оригинала — «Президент США доказал теорему
Пифагора» — кончается на 2.580, но следующая фраза «Её доказывали больше 370
разными способами» начинается уже на 2.880, внутри плана 2 (`pyth`, 1.94–3.64).
Резать по 2.88 нельзя: тело начиналось бы с середины анимации чертежа и с плана
длиной 0.76с. Поэтому рез поставлен на ближайшую границу плана ПОСЛЕ неё — 3.64,
в кадрах 109.2, то есть по первому кадру плана `n370`: кадр 110 = 3.6667с.
Планы 1–2 старого начала и их субтитры выброшены целиком.

Связка «её доказывали» (2.88–3.64) в новых хуках не произносится, поэтому она
берётся из исходной дорожки и играет под последним планом блока хука; в блоке она
и перерисована субтитром. Голос при этом остаётся непрерывным: дорожка хука
кончается, дальше идёт исходник сплошняком с 2.73с.

Формат хуков: оба дубля SDR bt709, как и основной исходник — тонмаппинг не нужен.
Расходятся два формальных свойства, оба приводятся (START-HERE, «приведение
формата делается всегда»):
  * разрешение 720x1280 против 1080x1920 — апскейл lanczos до холста тракта;
  * color_range=pc (yuvj420p) против tv — приведение к limited.
Замер: приведение range меняет картинку на 3-4 единицы по каналу
(144.5/121.0/107.8 -> 141.2/116.5/103.8). Остальное расхождение по свету —
съёмочное, оно НЕ правится, числа названы в монтажном листе.

Зеркальность проверена по кадру: надпись на футболке читается задом наперёд и в
основном дубле, и в обоих хуках — штатный hflip в source_card подходит всем трём.
"""
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render15 as R
import storyboard15 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/15"
BODY = f"{BUILD}/assets/_video_garfield.mp4"
STOCK_DIR = f"{VIDEO_DIR}/stock"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 110                        # первый кадр плана `n370`; 110/30 = 3.6667с
CUT = CUT_F / FPS
# связка «её доказывали» начинается на 2.880; берём на 0.10с раньше. Больше брать
# нельзя: пауза между хуком и первой фразой тела складывается из этого запаса и
# хвоста дубля и должна остаться в 0.10-0.20с, а хвост нужен, чтобы не срезать
# последний слог хуковой фразы (у обоих дублей запас 0.07-0.09с)
BRIDGE = 2.78
BRIDGE_LEN = CUT - BRIDGE
TARGET_LUFS = -14.0                # как в уже сданном garfield_edit.mp4 (замер: -14.0)

S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        aroll=f"{VIDEO_DIR}/prepared/hook1_aroll.mp4",
        head=15,                          # 15 кадров тишины в начале дубля срезаны
        frames=175,                       # блок хука 5.8333с
        out=f"{VIDEO_DIR}/garfield_hook1.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=lanczos",
        post_a=None,
        title="«а вы знали, что один из президентов США вошёл в историю, "
              "доказав теорему Пифагора?»",
        shots=[
            (0.000, 1.820, "A1", {}),
            (1.820, 3.680, "stock", {"clip": "11800", "ss": 2.5}),
            (3.680, 175 / 30, "pyth", {}),
        ],
        caps=[
            (0.08, 1.10, [("а вы знали", S, 48)], "A"),
            (1.22, 1.82, [("что один из", S, 48)], "A"),
            (1.82, 2.68, [("президентов сша", S, 46)], "B"),
            (2.68, 3.48, [("вошел в ", S, 46), ("историю", "s", 62)], "B"),
            (3.68, 4.28, [("доказав теорему", S, 46)], "G"),
            (4.28, 4.86, [("пифагора", "s", 72)], "G"),
            # связка из исходной дорожки, перерисована субтитром в блоке хука
            (5.05, 5.81, [("ее доказывали", S, 50)], "G"),
        ],
        cascades=[3.78],                  # каскад квадратов в чертеже
        reveals=[],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        aroll=f"{VIDEO_DIR}/prepared/hook2_aroll.mp4",
        head=31,                          # 31 кадр тишины в начале дубля срезан
        frames=215,                       # блок хука 7.1667с
        out=f"{VIDEO_DIR}/garfield_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",  # fps=30 обязателен: без него на выходе 30.6 fps
        post_a="atempo=1.02",
        title="«вы точно удивитесь, узнав, что одно из самых красивых доказательств "
              "теоремы Пифагора принадлежит президенту США»",
        shots=[
            (0.000, 1.770, "A1", {}),
            (1.770, 4.130, "pyth", {}),
            (4.130, 5.290, "A2", {}),
            (5.290, 215 / 30, "stock", {"clip": "11800", "ss": 4.0}),
        ],
        caps=[
            (0.11, 1.11, [("вы точно", S, 48)], "A"),
            (1.11, 1.77, [("удивитесь", "s", 66)], "A"),
            (1.77, 2.61, [("узнав что одной из", S, 44)], "G"),
            (2.61, 3.17, [("самых ", S, 46), ("красивых", "s", 58)], "G"),
            (3.17, 3.79, [("доказательств", "s", 56)], "G"),
            # whisper услышал «аремы» — в субтитре стоит слово, которое произнесено
            (3.79, 4.13, [("теоремы", S, 50)], "G"),
            (4.13, 4.67, [("пифагора", "s", 70)], "A"),
            (4.67, 5.29, [("принадлежит", S, 50)], "A"),
            (5.29, 5.85, [("президенту", S, 48)], "B"),
            (5.85, 6.38, [("сша", "s", 66)], "B"),
            # связка из исходной дорожки, перерисована субтитром в блоке хука
            (6.38, 7.14, [("ее доказывали", S, 50)], "B"),
        ],
        cascades=[1.87],
        reveals=[],
    ),
}


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_garfield.mp4.

    SHOTS/CAPS читаются функциями render15 из глобалей модуля — поэтому достаточно
    подменить их и пересобрать раскладку блоков."""
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(cfg["aroll"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, cfg["head"])   # срезаем тишину в начале дубля

    stock = dict(cap=None, key=None)

    def read_stock(prm, t0):
        k = (t0, prm["clip"])
        if k != stock["key"]:
            if stock["cap"] is not None:
                stock["cap"].release()
            stock["cap"] = cv2.VideoCapture(f"{STOCK_DIR}/stock_{prm['clip']}.mp4")
            stock["cap"].set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
            stock["key"] = k
        ok_s, sfr = stock["cap"].read()
        if not ok_s:
            stock["cap"].set(cv2.CAP_PROP_POS_MSEC, 0)
            ok_s, sfr = stock["cap"].read()
        return sfr

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    sheet, last_img = [], None
    for f in range(nf_hook):
        t = f / FPS
        ok, fr = cap.read()
        if ok:
            last_img = fr
        else:
            fr = last_img
        t0, t1, kind, prm = R.shot_at(t)

        canvas = R.background(kind, prm, fr, t, lambda: read_stock(prm, t0))
        gl = R.graphics_layer(kind, t - t0)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)

        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 8 == 0:
            sheet.append((t, np.array(rgb)))
    cap.release()
    if stock["cap"] is not None:
        stock["cap"].release()

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
    print(f"видео: хук {nf_hook} кадров + тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook15.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.4f}"]
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
    """Дорожка пересобирается целиком: голос хука + исходная дорожка с 2.73с
    (связка «её доказывали» и дальше тело сплошняком), постель со сдвигом,
    музыка с нуля, потом loudnorm."""
    hook = wav_of(cfg["src"], cfg["head"] / FPS)
    body = wav_of(SRC, BRIDGE)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = int(round((nf_hook / FPS - BRIDGE_LEN) * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    # склейка в тишине: гасим последние 8мс хука и первые 8мс исходника
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
    # склейка хук/тело — такой же рез, как все прочие, и свой whoosh обязана получить
    starts = [s[0] for s in cfg["shots"][1:]] + [nf_hook / FPS] + \
             [s[0] + dt for s in SB.SHOTS if s[0] >= CUT]
    for t0 in starts:
        add(wh, t0, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= CUT:
            add(im, t + dt, peak * 0.070)
    for t in cfg["reveals"]:
        add(im, t, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t in cfg["cascades"]:
        for k in range(3):
            add(tk, t + k * 0.085, peak * 0.030)

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
    mix_path = f"{BUILD}/assets/_mix_garfield{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Лёгкая деформация кадра/темпа, чтобы площадка не сочла версии дублями.
    Работает по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_garfield{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_garfield{key}_post.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                    "-pix_fmt", "yuv420p", dst], check=True)
    print("правка кадра:", vf)
    return dst


def measure_i(path):
    import json
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr
    m = json.loads(s[s.rindex("{"):s.rindex("}") + 1])
    return float(m["input_i"]), float(m["input_tp"])


def mux(vid, mix_path, out, target=TARGET_LUFS, extra_af=None):
    """Догоняем громкость до уровня уже сданного ролика.

    alimiter=...:level=disabled обязателен: автонормализация у alimiter включена
    по умолчанию и разгоняет громкость мимо цели, выбивая истинный пик в ноль.
    """
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
    tmp = f"{BUILD}/assets/_video_garfield{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)
    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook15_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h1")

"""Ролик 23 («как телефон находит себя»), альтернативные хуки.

Тело ролика (всё с кадра 148) не пересобирается — берётся готовыми кадрами из
assets/_video_gps.mp4 и склеивается за новым хуком. Заново рисуется только хук.

    python3 hook23.py h1
    python3 hook23.py h2

Точка реза: кадр 148 (4.9333с) — граница планов `teaser` -> `A2`, то есть конец
старой хуковой фразы «ваш телефон узнаёт точное место расположения, просто решая
систему уравнений в реальном времени». Речь там смолкает на 4.9238 по
silencedetect, план кончается на 4.92; ceil(4.92*30) = 148. Планы 1-3 старого
начала (лицо, сток с телефоном, teaser) и их субтитры выброшены целиком,
ни одного кадра от них не осталось.

Рез попадает в паузу: до первого слова тела («И да…», 5.34) остаётся 0.407с,
после конца хуковой речи — 0.0095с. Ни звука не срезано.

Передача эстафеты. Тело начинается с «и да, GPS-спутники не фотографируют вас
сверху и не отслеживают телефон напрямую». Проверено на антецеденты:
* hook1 кончается на «…основана на точных математических расчётах» — «и да»
  читается как подтверждение только что заявленного, GPS в хуке назван;
* hook2 кончается на «…отслеживание вас по вашему же телефону» — тело прямо
  отвечает на этот вопрос («не отслеживают телефон напрямую»), связка даже
  плотнее оригинальной.
Ни в одном варианте первая фраза тела не опирается на местоимение без
антецедента: «GPS-спутники» — существительное, оно вводит себя само.

Формат хуков. Оба дубля совпадают с основным исходником по всем тегам:
720x1280, yuv420p, color_range=tv, bt709/bt709/bt709, 30 fps. Приведение
цветового пространства не требуется — кадры читаются прямо из .mov и идут
в штатный тракт A-roll без промежуточного кодека.

Графика хука. В старом начале стояли лицо -> сток -> `teaser`. Структура
сохранена в обоих вариантах, `teaser` (три сферы сходятся в точку) оставлен:
это визуальная интрига ролика, менять её незачем. Сток выбран по словам
самого дубля: hook1 говорит «работа GPS» — идёт спутник (47401), hook2 говорит
«по вашему же телефону» — идёт телефон с картой (22149). Оба клипа уже скачаны
под этот ролик, ни одного нового не добавлено.
"""
import json
import math
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render23 as R
import storyboard23 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/23"
BODY = f"{BUILD}/assets/_video_gps.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = math.ceil(4.92 * FPS)       # 148 — первый кадр тела, граница teaser -> A2
CUT = CUT_F / FPS                   # 4.9333с
TARGET_LUFS = -14.2                 # как в уже сданном gps_edit.mp4 (замер)

# (длительность полной анимации, запас до реза). Масштаб времени применяется
# только чтобы УСКОРИТЬ анимацию под короткий план хука; замедлять её ниже
# темпа тела нельзя — иначе тот же план в хуке и в теле живёт по-разному.
GFX_FULL = {"teaser": (1.62, 0.06)}

S = "r"
I = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        # речь дубля 0.160-4.140 по огибающей RMS, файл 4.155с (124 кадра).
        # 125 кадров: хвост 0.027с + 0.407с в голове тела = пауза 0.434с,
        # ровно как между предложениями в самом теле (0.42с).
        # Последний кадр приходится на `teaser` — лицо не морозится ни на кадр.
        src_ss=0.0,
        frames=125,
        out=f"{VIDEO_DIR}/gps_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«а вы знали, что работа GPS основана на точных математических расчётах?»",
        # резы: 42 -> конец слова «GPS» (1.39), 86 -> конец «точных» (2.87)
        shots=[
            (0 / 30,   42 / 30,  "A1", {}),
            (42 / 30,  86 / 30,  "stock", {"clip": "47401", "ss": 5.0}),
            (86 / 30,  125 / 30, "teaser", {}),
        ],
        caps=[
            (0.16,    0.72,     [("а вы знали что", S, 48)], "A"),
            (0.72,    42 / 30,  [("работа ", S, 50), ("gps", I, 56)], "A"),

            (42 / 30, 2.87,     [("основана на точных", S, 46)], "B"),

            (86 / 30, 4.14,     [("математических ", S, 42), ("расчётах", I, 52)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # речь дубля 0.150-3.390 по огибающей, файл 3.473с (104 кадра).
        # 105 кадров: последний кадр добивается планом `teaser` (сетка исходный
        # кадр не использует), лицо не морозится. Хвост 0.110с + 0.407с
        # в голове тела = пауза 0.517с.
        src_ss=0.0,
        frames=105,
        out=f"{VIDEO_DIR}/gps_hook2.mp4",
        # song1 всего 9.53с — единственный оставшийся трек, идёт петлёй
        # с кроссфейдом 0.25с. Названо в монтажном листе.
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«а вы знаете, как работает отслеживание вас по вашему же телефону?»",
        # резы: 40 -> в паузе перед «отслеживание», 74 -> в паузе перед «вашему»
        shots=[
            (0 / 30,   40 / 30,  "A1", {}),
            (40 / 30,  74 / 30,  "stock", {"clip": "22149", "ss": 6.0}),
            (74 / 30,  105 / 30, "teaser", {}),
        ],
        caps=[
            (0.15,    0.56,     [("а вы знаете", S, 50)], "A"),
            (0.56,    40 / 30,  [("как ", S, 48), ("работает", I, 54)], "A"),

            (40 / 30, 2.4667,   [("отслеживание вас по", S, 44)], "B"),

            (74 / 30, 3.39,     [("вашему же ", S, 46), ("телефону", I, 54)], "G"),
        ],
    ),
}


def lts_for(cfg):
    """Масштаб времени анимации на каждый графический план хука.

    max(1.0, ...) — анимацию можно только ускорить, чтобы она успела доиграть
    до реза. Замедлять нельзя: тот же план в теле идёт своим темпом.
    """
    out = {}
    for t0, t1, kind, _p in cfg["shots"]:
        if kind in GFX_FULL:
            full, hold = GFX_FULL[kind]
            out[kind] = max(1.0, full / max(0.4, (t1 - t0) - hold))
    return out


def face_metrics(path, n=8):
    """Крупность дубля детектором, после hflip: cx, линия глаз, ширина лица."""
    casc = cv2.CascadeClassifier(cv2.data.haarcascades
                                 + "haarcascade_frontalface_default.xml")
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    cap = cv2.VideoCapture(path)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for f in np.linspace(2, max(3, nf - 2), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.flip(fr, 1)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        det = casc.detectMultiScale(g, 1.15, 6, minSize=(120, 120))
        if len(det) == 0:
            continue
        x, y, w, h = max(det, key=lambda r: r[2] * r[3])
        roi = g[y:y + int(h * 0.6), x:x + w]
        ee = eyes.detectMultiScale(roi, 1.12, 8, minSize=(25, 25))
        ey = y + int(np.mean([e[1] + e[3] / 2 for e in ee])) if len(ee) >= 2 \
            else y + int(h * 0.42)
        rows.append((x + w // 2, ey, w))
    cap.release()
    if not rows:
        return None
    a = np.array(rows, float)
    return tuple(int(round(v)) for v in a.mean(axis=0))


def card_rgb(path, t, kind="A1"):
    """Средний RGB внутри карточки A после штатного тракта A-roll."""
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
    """Кадры хука рисуем, кадры тела берём готовыми из _video_gps.mp4.

    A-roll читается из .mov последовательно, без промежуточного кодека:
    форматы дубля и основного исходника совпадают, приводить нечего.
    """
    lts = lts_for(cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(cfg["src"])
    for _ in range(int(round(cfg["src_ss"] * FPS))):      # отбрасываем голову
        cap.read()

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    stock = R.StockReader()
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

        canvas = R.background(kind, prm, fr, t,
                              lambda: stock.frame(prm["clip"], prm.get("ss", 0), t - t0))
        # lts — масштаб времени анимации: графика тела рассчитана на длину плана
        # в теле, на более коротком плане хука она иначе оборвётся на полудвижении
        gl = R.graphics_layer(kind, (t - t0) * lts.get(kind, 1.0))
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
    stock.release()

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
    """-ss после -i: точный отброс, чтобы звук встал кадр в кадр с картинкой."""
    tmp = f"{BUILD}/assets/_tmp_hook23.wav"
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


def build_audio(key, cfg, nf_hook, vid, out):
    """Дорожка пересобирается целиком: голос хука + голос тела с точки реза,
    постель со сдвигом, музыка с нуля, потом loudnorm."""
    hook = wav_of(cfg["src"], cfg["src_ss"])
    body = wav_of(SRC, CUT)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = int(round(nf_hook / FPS * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    r = int(0.008 * SR)                                  # мягкий стык, без щелчка
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
    for t0 in [s[0] for s in cfg["shots"][1:]]:      # резы внутри хука
        add(wh, t0, peak * 0.040)
    add(wh, nf_hook / FPS, peak * 0.040)             # шов «хук -> тело»
    for t0, _t1, _k, _p in SB.SHOTS:
        if t0 > CUT + 0.01:
            add(wh, t0 + dt, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= CUT:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t0, _t1, kind, _p in cfg["shots"]:       # каскады графики самого хука
        if kind in SB.GRID_KINDS:
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
    mix_path = f"{BUILD}/assets/_mix_gps{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_gps{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_gps{key}_post.mp4"
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
    """Догоняем громкость до уровня уже сданного ролика, следя за истинным пиком."""
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
    tmp = f"{BUILD}/assets/_video_gps{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    # расхождение дублей — измеряем и называем, а не чиним кропом
    print("крупность (cx, линия глаз, ширина лица), после hflip:")
    print("   хук          ", face_metrics(cfg["src"]))
    print("   основной дубль", face_metrics(SRC))
    print("средний RGB в карточке A: хук", card_rgb(cfg["src"], 0.8),
          "· основной дубль", card_rgb(SRC, 0.8))

    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook23_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_gps{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

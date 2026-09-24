"""Ролик 20 («эффект приманки — попкорн в кино»), альтернативные хуки.

Тело ролика (всё с кадра 158) не пересобирается — берётся готовыми кадрами из
assets/_video_popcorn.mp4 и склеивается за новым хуком. Заново рисуется только хук.

    python3 hook20.py h1
    python3 hook20.py h2

Точка реза: кадр 158 (5.2667с) — граница планов `A2` -> `cups3`, то есть конец
старой хуковой фразы «и вот как это работает» (слово «работает» кончается на
4.90 по whisper, реально смолкает на 4.73 по огибающей; план — на 5.2667).
Планы 1-3 старого начала и их субтитры выброшены целиком, ни одного кадра
от них не осталось.

Рез попадает в паузу: непрерывная тишина в дубле идёт 4.73–5.28, речь тела
(«в кинотеатре обычно три размера попкорна») начинается на 5.28. Рез по 5.2667
не срезает ни звука — до первого слова тела остаётся 0.013с.

Передача эстафеты. hook1 кончается на «и вот как это работает» — ровно та же
связка, что в старом начале. hook2 кончается на «продавая вам большие попкорны»,
и тело подхватывает «в кинотеатре обычно три размера попкорна»: слово другое,
но предложение начинается с нового подлежащего, шва в речи не слышно.

Формат хуков. Оба дубля совпадают с основным исходником по всем тегам:
720x1280, yuv420p, color_range=tv, bt709/bt709/bt709, 30 fps. Приведение
цветового пространства не требуется — кадры читаются прямо из .mov и идут
в штатный тракт A-roll без промежуточного кодека.

Крупность дублей (детектор, после hflip, среднее по 6 кадрам):
    основной  cx 382  линия глаз 670  размер лица 270
    hook1     cx 389  линия глаз 678  размер лица 273
    hook2     cx 377  линия глаз 680  размер лица 274
Расхождение до 7px по горизонтали, 10px по линии глаз и 1.5% по размеру лица —
самое малое из всех роликов. Кропом это НЕ правится: числа названы в сдаче.

Графика хука. В старом хуке ролика 20 сетки не было вообще — там стояли лицо,
зал кинотеатра (сток 33312) и снова лицо. Зал кинотеатра сохранён в обоих новых
хуках. Второй графический план взят по словам дубля: оба хука говорят про
«большой попкорн» / «большие попкорны», поэтому стоит `pickL` — три стакана
и галка над большим. Это уже существующий план ролика, ни одного нового
не нарисовано.
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
import render20 as R
import storyboard20 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/20"
BODY = f"{BUILD}/assets/_video_popcorn.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 158                         # первый кадр тела = граница плана A2 -> cups3
CUT = CUT_F / FPS                   # 5.2667с
TARGET_LUFS = -14.0                 # как в уже сданном popcorn_edit.mp4 (замер)

# (длительность полной анимации, запас до реза). Масштаб времени применяется
# только чтобы УСКОРИТЬ анимацию под короткий план хука; замедлять её ниже
# темпа тела нельзя — иначе тот же план в хуке и в теле живёт по-разному.
GFX_FULL = {"pickL": (1.12, 0.08)}

S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        # дубль звучит с первого кадра (речь с 0.009), тишины в голове нет.
        # A-roll декодируется 167 кадров (5.567с; FRAME_COUNT врёт про 177),
        # речь в аудио до 5.591с.
        # 172 кадра: хвост 0.133с + 0.013с в голове тела = пауза 0.146с.
        # Последние 30 кадров — план pickL, лицо не морозится ни на кадр.
        src_ss=0.0,
        frames=172,
        out=f"{VIDEO_DIR}/popcorn_hook1.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«я уверен, что многие из вас попадались в эту ловушку и брали "
              "большой попкорн в кинотеатре. и вот как это работает»",
        face_cx=389, face_eye=678, face_d=273,
        # структура старого хука сохранена: лицо -> зал кинотеатра -> лицо.
        # Графики в старом начале не было вообще, не появилась и здесь.
        shots=[
            (0 / 30,   55 / 30,  "A1", {}),
            (55 / 30,  87 / 30,  "stock", {"clip": "33312", "ss": 3.0}),
            (87 / 30,  142 / 30, "A2", {}),
            (142 / 30, 172 / 30, "pickL", {}),
        ],
        caps=[
            (0.000, 0.600, [("я уверен что", S, 48)], "A"),
            (0.600, 1.100, [("многие из вас", S, 46)], "A"),
            (1.100, 55 / 30, [("попадались в эту", S, 44)], "A"),

            (1.840, 87 / 30, [("ловушку", "s", 62)], "B"),

            (2.900, 3.660, [("большой попкорн", S, 48)], "A"),
            (3.720, 142 / 30, [("в кинотеатре", S, 50)], "A"),

            (142 / 30, 5.040, [("и вот как", S, 48)], "G"),
            (5.040, 172 / 30, [("это ", S, 46), ("работает", "s", 60)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # в голове дубля 0.180с тишины — оставлены: обрезать их значило бы
        # сдвинуть все субтитры относительно whisper, а выигрыш 6 кадров.
        # A-roll 160 кадров (5.333с), речь 0.180–5.270.
        # 161 кадр: хвост 0.097с + 0.013с в голове тела = пауза 0.110с.
        # Последний кадр добивается планом pickL, лицо не морозится.
        # Все резы считаются в системе самого файла, без сдвига.
        src_ss=0.0,
        frames=161,
        out=f"{VIDEO_DIR}/popcorn_hook2.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«многие не знают этот трюк, на котором кинотеатры зарабатывают "
              "большие деньги, продавая вам большие попкорны»",
        face_cx=377, face_eye=680, face_d=274,
        # лицо -> зал кинотеатра -> лицо, как в старом начале; хвост добивается
        # планом pickL (галка над большим) — «продавая вам большие попкорны».
        shots=[
            (0 / 30,   49 / 30,  "A1", {}),
            (49 / 30,  90 / 30,  "stock", {"clip": "33312", "ss": 4.5}),
            (90 / 30,  127 / 30, "A2", {}),
            (127 / 30, 161 / 30, "pickL", {}),
        ],
        caps=[
            (0.000, 1.040, [("многие не знают", S, 48)], "A"),
            (1.040, 49 / 30, [("этот ", S, 46), ("трюк", "s", 62)], "A"),

            (1.640, 2.440, [("на котором ", S, 42), ("кинотеатры", "s", 52)], "B"),
            (2.440, 90 / 30, [("зарабатывают", S, 50)], "B"),

            (3.000, 3.660, [("большие деньги", S, 50)], "A"),
            (3.840, 127 / 30, [("продавая вам", S, 48)], "A"),

            (4.400, 4.760, [("большие", S, 52)], "G"),
            (4.760, 161 / 30, [("попкорны", "s", 58)], "G"),
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
    """Кадры хука рисуем, кадры тела берём готовыми из _video_popcorn.mp4.

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
        if f % 6 == 0:
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
    tmp = f"{BUILD}/assets/_tmp_hook20.wav"
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
    mix_path = f"{BUILD}/assets/_mix_popcorn{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_popcorn{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_popcorn{key}_post.mp4"
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
    tmp = f"{BUILD}/assets/_video_popcorn{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    # расхождение дублей — измеряем и называем, а не чиним кропом
    print("средний RGB в карточке A: хук", card_rgb(cfg["src"], 1.0),
          "· основной дубль", card_rgb(SRC, 1.0))

    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook20_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_popcorn{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

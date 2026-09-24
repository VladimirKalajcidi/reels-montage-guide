"""Ролик 8, альтернативные хуки.

Тело ролика (всё после 6.940с) не пересобирается — берётся готовыми кадрами из
assets/_video_v8.mp4 и склеивается за новым хуком. Заново рисуется только хук:
свой A-roll, свои субтитры, та же графика «√−1 перечёркнуто», что стояла в старом хуке.

    python3 hook8.py h1
    python3 hook8.py h2

Точка реза 6.940с — граница плана 4 (SHOTS[3], лицо «ни один телефон»), ровно на
конце старой хуковой фразы (речь кончается на 6.46, план — на 6.940). До этой точки
выброшено всё: планы 1–4 и их субтитры (столетиями … ни один телефон).
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
import render8 as R
from sfx import low_whoosh, impact, tick, read_wav
import storyboard8 as SB8
from sfx8 import NUM_REVEALS as SB8_NUM_REVEALS, CASCADES as SB8_CASCADES

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/8"
BODY = f"{BUILD}/assets/_video_v8.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = "/Users/vladimirkalajcidi/reels_good/videos/8/source.mov"
SR = 48000

CUT = 6.940                        # граница плана 4; старый хук выброшен целиком
CUT_F = 209                        # первый кадр тела на таймлинии 30fps (6.940*30=208.2 -> 209)
TARGET_LUFS = -14.5                # как в уже сданном imaginary_edit.mp4

S, Z = "r", "s"

# Зеркальность и крупность проверены по кадру: оба дубля сняты с той же дистанции и
# точки, что основной исходник (лого «CLASSIC SPORTS» читается задом наперёд в обоих).
# Компенсирующий кроп не нужен — штатный FRAMINGS["A1"] подходит без поправок.
#
# Причина цветового расхождения — НЕ грейд и не свет на площадке, а формат записи:
#   source.mov : 8 бит, bt709 / bt709 — обычный SDR.
#   hook*.mov  : 10 бит, bt2020nc / arib-std-b67 (HLG) — HDR-запись с телефона.
# Без конверсии HLG декодируется как bt709 и картинка уезжает в светлый плоский «туман».
# Три собственные попытки поправить это множителями/кривыми/regression по каналам были
# лечением симптома не там — весь последующий GRADE считает, что на входе уже SDR.
# Как ролик 7 (тот же формат хуков): сначала честный tonemap HDR -> SDR, потом штатный
# тракт (hflip+FRAMINGS+GRADE) без единой самодельной поправки цвета поверх.
TONEMAP = ("zscale=t=linear:npl=260,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def sdr_source(key, cfg):
    """HDR-дубль -> SDR bt709. Звук берём из оригинала, здесь только картинка."""
    out = f"{BUILD}/assets/_hook8_{key}_sdr.mp4"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", cfg["src"], "-vf", TONEMAP, "-an",
                        "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                        "-pix_fmt", "yuv420p", out], check=True)
        print("хук приведён к SDR bt709:", os.path.basename(out))
    return out

HOOKS = {
    "h1": dict(
        src=f"{VIDEO_DIR}/hook1.mov",
        frames=164,                       # речь кончается на 5.30 (159 кадров); +5 кадров паузы
        speech_end=5.300,
        out=f"{VIDEO_DIR}/imaginary_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=lanczos",
        title="«что будет если взять число которого не существует»",
        shots=[
            (0.000, 1.600, "A1", {}),
            (1.600, 3.600, "nonsense", {}),
            (3.600, 164 / 30, "A1", {}),
        ],
        caps=[
            (0.000, 0.780, [("что будет", S, 56)], "A"),
            (0.880, 1.600, [("если взять ", S, 48), ("число", "s", 66)], "A"),
            (2.020, 2.380, [("которого вообще", S, 50)], (540, 700, "mm")),
            (2.380, 3.160, [("не ", S, 50), ("существует", "s", 74)], (540, 1180, "mm")),
            (3.280, 3.780, [("и построить", S, 52)], "A"),
            (3.780, 4.440, [("на нём целый", S, 48)], "A"),
            (4.440, 5.300, [("раздел ", S, 48), ("математики", "s", 68)], "A"),
        ],
        cascades=[],
    ),
    "h2": dict(
        src=f"{VIDEO_DIR}/hook2.mov",
        frames=138,                       # речь кончается на 4.46; хвост дубля обрезан до паузы ~0.14с
        speech_end=4.460,
        out=f"{VIDEO_DIR}/imaginary_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: без него на выходе 30.6 fps
        post_a="atempo=1.02",
        title="«одно из самых полезных чисел математики не существует»",
        shots=[
            (0.000, 1.480, "A1", {}),
            (1.480, 3.660, "nonsense", {}),
            (3.660, 138 / 30, "A1", {}),
        ],
        caps=[
            (0.000, 0.740, [("одно из самых", S, 50)], "A"),
            (0.740, 1.480, [("полезных чисел", S, 52)], "A"),
            (1.480, 2.540, [("математики вообще", S, 48)], (540, 700, "mm")),
            (2.540, 3.260, [("не ", S, 50), ("существует", "s", 74)], (540, 1180, "mm")),
            (3.260, 138 / 30, [("среди обычных ", S, 46), ("чисел", "s", 62)], "A"),
        ],
        cascades=[],
    ),
}

CX, CY, CW, CH = CARD_A


# ----------------------------------------------------------------- A-roll хука
def prep_hook_aroll(key, cfg):
    """Тот же тракт, что у основного A-roll: hflip + FRAMINGS['A1'] + GRADE.
    Перед этим — TONEMAP (HDR->SDR), без него дубль в принципе декодируется неверно."""
    out = f"{BUILD}/assets/aroll_8{key}_A1.mp4"
    if not os.path.exists(out):
        sdr = sdr_source(key, cfg)
        w, h, x, y = R.FRAMINGS["A1"]
        vf = f"hflip,crop={w}:{h}:{x}:{y},scale={CW}:{CH}:flags=lanczos,{R.GRADE}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", sdr, "-vf", vf, "-an", "-c:v", "libx264",
                        "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p",
                        out], check=True)
        print("A-roll хука:", out)
    return out


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_v8.mp4."""
    aroll = prep_hook_aroll(key, cfg)

    # переключаем раскадровку в render8 на хуковую и пересобираем раскладку блоков
    R.SHOTS = cfg["shots"]
    R.CAPS = cfg["caps"]
    R.BLOCKS = R._build_blocks()
    R.BLOCK_LAY = R._block_layout()

    nf_hook = cfg["frames"]
    maskA = rounded_mask(CW, CH, R_A)
    cap = cv2.VideoCapture(aroll)

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    grid_cache = {}

    def grid_for(f):
        k = f // 3
        if k not in grid_cache:
            grid_cache.clear()
            grid_cache[k] = grid_canvas(CW, CH, phase=k * 0.06)
        return grid_cache[k]

    sheet = []
    si = 0
    last_img = None
    for f in range(nf_hook):
        t = f / FPS
        cap.grab()
        while si < len(R.SHOTS) - 1 and t >= R.SHOTS[si][1]:
            si += 1
        t0, t1, kind, prm = R.SHOTS[si]
        lt = t - t0

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        if kind == "A1":
            okr, img = cap.retrieve()
            if okr:
                last_img = img
            elif last_img is not None:
                # дубль кончился раньше nf_hook (хвостовая пауза перед телом) — держим
                # последний кадр вместо чёрного, чтобы не было мёртвого кадра на стыке
                img = last_img
            else:
                img = np.zeros((CH, CW, 3), np.uint8)
            canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                         (CX, CY), maskA)
        else:
            canvas.paste(grid_for(f), (CX, CY), maskA)

        layers, titems = R.shot_layers(kind, prm, lt, t1 - t0, t)
        for lay in layers:
            canvas.alpha_composite(lay)
        items = titems + R.caption_items(t, kind)
        if items:
            canvas.alpha_composite(text_layer((W, H), items))

        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 6 == 0:
            sheet.append((t, np.array(rgb)))
    cap.release()

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

    ff.stdin.close(); ff.wait()
    print(f"видео: хук {nf_hook} кадров + тело {n_body} кадров -> {tmp}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook8.wav"
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
    hook = wav_of(cfg["src"])
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
    starts = [s[0] for s in cfg["shots"][1:]] + \
             [s[0] + dt for s in SB8.SHOTS if s[0] >= CUT]
    for t0 in starts:
        add(wh, t0, peak * 0.040)
    for t in SB8_NUM_REVEALS:
        add(im, t + dt, peak * 0.070)
    for t in SB8_CASCADES:
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
    mix_path = f"{BUILD}/assets/_mix_v8{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Лёгкая деформация кадра/темпа, чтобы площадка не сочла версии дублями.
    Работает по уже собранному немому _video_v8<key>.mp4, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_v8{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_v8{key}_post.mp4"
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
    """Догоняем громкость до уровня уже сданного ролика."""
    def run(post_db):
        af = (f"{extra_af}," if extra_af else "") + "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
        if abs(post_db) > 0.02:
            af += f",volume={post_db:+.2f}dB,alimiter=limit=0.80:level=disabled"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", vid, "-i", mix_path, "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-af", af,
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
    tmp = f"{BUILD}/assets/_video_v8{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)
    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        im = Image.fromarray(arr).resize((180, 320))
        cs.paste(im, ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook8_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_v8{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"], apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

"""Ролик 7, альтернативные хуки.

Тело ролика (всё после 4.300с) не пересобирается — берётся готовыми кадрами из
assets/_video_secretary.mp4 и склеивается за новым хуком. Заново рисуется только хук:
свой A-roll, свои субтитры, та же графика и тот же сток, что в старом хуке.

    python3 hook7.py h1
    python3 hook7.py h2

Точка реза 12.02с — граница плана 9, с которого начинается «есть точная математическая
стратегия». Всё до неё — хук: и вопрос «как выбрать кандидата / квартиру / партнёра»,
и условия задачи («смотришь варианты один за другим», «без права вернуться»).
Выброшены планы 1–8 и их субтитры.
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
import render7 as R
import storyboard7 as SB
from sfx7 import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/7"
BODY = f"{BUILD}/assets/_video_secretary.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SR = 48000

# Уникализация версий под площадку (START-HERE.md «Уникализация версий»):
# версия 1 — исходный secretary_edit.mp4, song1, без деформации, эталон;
# версия 2 — hook1, song2, кроп ×0.98;
# версия 3 — hook2, song3, ускорение ×1.02 (setpts + atempo + обязательный fps=30).
CROP_098 = "crop=1058:1882:11:19,scale=1080:1920:flags=lanczos"
SPEED_102_V = "setpts=PTS/1.02,fps=30"
SPEED_102_A = "atempo=1.02"

CUT = 12.02                      # граница плана 9 «есть точная математическая стратегия»
CUT_F = int(round(CUT * FPS))    # 361
TARGET_LUFS = -14.6              # как в уже сданном secretary_edit.mp4

S, Z = "r", "s"

# Зеркальность проверена по логотипу на толстовке: в обоих дублях надпись
# «CLASSIC SPORTS» читается задом наперёд — как и в основном исходнике.
# Значит штатный hflip из R.source_card подходит обоим, исключений нет.
HOOKS = {
    "h1": dict(
        src=f"{VIDEO_DIR}/hook1.mov",
        # 271 кадра = 9.033с. Речь кончается на 8.92, дальше 0.11с хвоста:
        # пауза до первого слова тела выходит 0.14с. Сам файл дубля 268 кадров,
        # поэтому хвост держит графический план, а не стоп-кадр лица.
        frames=271,
        out=f"{VIDEO_DIR}/secretary_hook1.mp4",
        title="«представьте, что вы сидите на авито»",
        version=2,
        music=f"{AUDIO_DIR}/song2.mp3",
        post_v=CROP_098,
        post_a=None,
        shots=[
            (0.00, 1.46, "A1", {}),
            (1.46, 3.20, "stock", {"clip": "4919", "ss": 0.8}),
            (3.20, 4.88, "A2", {}),
            (4.88, 6.06, "stock", {"clip": "29302", "ss": 3.0}),
            (6.06, 7.38, "A1", {}),
            (7.38, 271 / FPS, "row_scan", {}),
        ],
        caps=[
            (0.00, 0.86, [("представьте", S, 58)]),
            (0.86, 1.46, [("что вы сидите", S, 48)]),
            (1.46, 1.96, [("на ", S, 44), ("авито", Z, 68)]),
            (1.96, 2.98, [("и пытаетесь например", S, 42)]),
            (3.20, 3.52, [("себе найти", S, 50)]),
            (3.52, 4.88, [("репетитора", Z, 62), (" по математике", S, 40)]),
            (4.88, 5.92, [("и не понимаете", S, 48)]),
            (6.06, 6.60, [("кого конкретно", S, 46)]),
            (6.60, 7.06, [("выбрать", Z, 72)]),
            (7.38, 8.04, [("вот как это нужно", S, 44)]),
            (8.04, 8.92, [("делать ", S, 42), ("на самом деле", Z, 54)]),
        ],
        cascades=[7.38],
    ),
    "h2": dict(
        src=f"{VIDEO_DIR}/hook2.mov",
        # 427 кадров = 14.233с, речь кончается на 14.12 — те же 0.14с паузы до тела.
        # Файл дубля 424 кадра, хвост держит графический план.
        frames=427,
        out=f"{VIDEO_DIR}/secretary_hook2.mp4",
        title="«у вас точно хоть раз такое было»",
        version=3,
        music=f"{AUDIO_DIR}/song3.mp3",
        post_v=SPEED_102_V,
        post_a=SPEED_102_A,
        shots=[
            (0.00, 1.74, "A1", {}),
            (1.74, 3.24, "stock", {"clip": "4919", "ss": 0.8}),
            (3.24, 4.24, "A2", {}),
            (4.24, 6.90, "stock", {"clip": "3110", "ss": 2.0}),
            (6.90, 8.62, "stock", {"clip": "5660", "ss": 2.0}),
            (8.62, 10.48, "A2", {}),
            (10.48, 12.46, "stock", {"clip": "29302", "ss": 3.0}),
            (12.46, 427 / FPS, "row_scan", {}),
        ],
        caps=[
            (0.00, 0.72, [("у вас точно", S, 50)]),
            (0.72, 1.74, [("хоть раз такое было", S, 44)]),
            (2.02, 2.54, [("представьте", S, 58)]),
            (2.54, 3.24, [("что вы сидите", S, 46)]),
            (3.36, 4.06, [("пытаетесь например", S, 44)]),
            (4.24, 4.90, [("на сайте выбрать", S, 46)]),
            (4.90, 6.28, [("себе ", S, 42), ("квартиру", Z, 66)]),
            (6.28, 6.90, [("для аренды", S, 46)]),
            (7.12, 7.68, [("может быть", S, 46)]),
            (7.68, 8.62, [("кандидата на работу", S, 46)]),
            (8.62, 9.60, [("или ", S, 44), ("репетитора", Z, 62)]),
            (9.60, 10.48, [("по математике", S, 48)]),
            (10.82, 11.30, [("и всё никак", S, 48)]),
            (11.30, 12.46, [("не можете ", S, 44), ("выбрать", Z, 68)]),
            (12.64, 13.16, [("вот что на этот", S, 44)]),
            (13.16, 14.12, [("счёт говорит ", S, 42), ("математик", Z, 56)]),
        ],
        cascades=[12.46],
    ),
}

CX, CY, CW, CH = CARD_A

# --- приведение хуков к тому же цветовому пространству, что у основного дубля ---
# source.mp4 : 8 бит, bt709 / bt709 — обычный SDR.
# hook*.mov  : 10 бит, bt2020nc / arib-std-b67 (HLG) — HDR-запись с телефона.
# Без конверсии HLG декодируется как bt709 и картинка уезжает в светлый плоский
# «туман». Это не грейд и не фильтр «на вкус» — это недостающее приведение
# формата; грейд GRADE применяется потом одинаково к хуку и к телу.
# npl=260 подобран по средним RGB кадра: хук выходит на (89,76,65) / (85,74,64)
# против (88,74,67) у основного дубля.
TONEMAP = ("zscale=t=linear:npl=260,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def sdr_source(key, cfg):
    """HDR-дубль -> SDR bt709. Звук берём из оригинала, здесь только картинка."""
    out = f"{BUILD}/assets/_hook7_{key}_sdr.mp4"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", cfg["src"], "-vf", TONEMAP, "-an",
                        "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                        "-pix_fmt", "yuv420p", out], check=True)
        print("хук приведён к SDR bt709:", os.path.basename(out))
    return out


def patch_render(cfg):
    """Подменяем раскадровку в render7 на хуковую и пересобираем раскладку блоков."""
    shots = cfg["shots"]

    def shot_at(t):
        for s in shots:
            if s[0] <= t < s[1]:
                return s
        return shots[-1]

    R.shot_at = shot_at
    R.CAPS = cfg["caps"]
    R.BLOCKS = R.build_blocks()
    return shot_at


def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_secretary.mp4."""
    shot_at = patch_render(cfg)
    nf_hook = cfg["frames"]
    maskA = rounded_mask(CW, CH, R_A)
    maskB = rounded_mask(CARD_B[2], CARD_B[3], R_B)

    cap = cv2.VideoCapture(sdr_source(key, cfg))
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    stock_cap, stock_key, last = None, None, None
    sheet = []
    for f in range(nf_hook):
        ok, fr = cap.read()
        if ok:
            last = fr
        else:
            fr = last
        t = f / FPS
        t0, t1, kind, prm = shot_at(t)
        lt = t - t0

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        if kind in SB.FACE_KINDS:
            # тот же тракт, что у основного A-roll: hflip + CROPS + GRADE, без поправок
            canvas.paste(R.source_card(fr, kind), (CX, CY), maskA)
        elif kind == "stock":
            k = (t0, prm["clip"])
            if k != stock_key:
                if stock_cap is not None:
                    stock_cap.release()
                stock_cap = cv2.VideoCapture(f"{SB.STOCK_DIR}/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
                stock_key = k
            oks, sfr = stock_cap.read()
            if not oks:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                oks, sfr = stock_cap.read()
            canvas.paste(R.stock_frame_from_bgr(sfr), (CARD_B[0], CARD_B[1]), maskB)
        else:
            canvas.paste(grid_canvas(CW, CH, phase=t * 0.7), (CX, CY), maskA)

        gl = R.graphics_layer(kind, prm, lt)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)

        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 8 == 0:
            sheet.append(np.array(rgb))
    cap.release()
    if stock_cap is not None:
        stock_cap.release()

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
    print(f"видео: хук {nf_hook} кадров + тело {n_body} кадров -> {tmp}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------------- звук

def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook7.wav"
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
    body = wav_of(SB.SRC, CUT)

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
             [s[0] + dt for s in SB.SHOTS if s[0] >= CUT]
    for t0 in starts:
        add(wh, t0, peak * 0.040)
    # события тела берём только те, что после точки реза: каскад на 7.90с жил
    # в выброшенном куске, со сдвигом он бы уехал внутрь нового хука
    for t in SB.NUM_REVEALS:
        if t >= CUT:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t in cfg["cascades"]:
        for k in range(3):
            add(tk, t + k * 0.085, peak * 0.030)

    track_name = os.path.splitext(os.path.basename(cfg["music"]))[0]
    mw = f"{BUILD}/assets/_music_{track_name}.wav"
    if not os.path.exists(mw):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", cfg["music"],
                        "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
    print(f"музыка версии {cfg['version']}: {os.path.basename(cfg['music'])}")
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
    mix_path = f"{BUILD}/assets/_mix_secretary_{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, post_a=cfg["post_a"])


def measure_i(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr
    m = json.loads(s[s.rindex("{"):s.rindex("}") + 1])
    return float(m["input_i"]), float(m["input_tp"])


def deform(cfg, vid):
    """Деформация версии накладывается на уже собранное немое видео, одним проходом.
    Кадры не перерисовываются; звук трогается отдельно, в mux (post_a)."""
    if not cfg["post_v"]:
        return vid
    out = vid.replace(".mp4", "_def.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", vid,
                    "-vf", cfg["post_v"], "-an", "-c:v", "libx264", "-crf", "16",
                    "-preset", "medium", "-pix_fmt", "yuv420p", out], check=True)
    print(f"деформация версии {cfg['version']}: {cfg['post_v']}")
    return out


def mux(vid, mix_path, out, target=TARGET_LUFS, post_a=None):
    """Догоняем громкость до уровня уже сданного ролика (-14.6 LUFS)."""
    def run(post_db):
        af = "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
        if post_a:                      # ускорение звука вместе с картинкой
            af = f"{post_a},{af}"
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
    tmp = f"{BUILD}/assets/_video_secretary_{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)
    # порядок: сначала правка кадра, потом мукс — иначе звук переживает лишнее пересжатие
    build_audio(key, cfg, nf_hook, deform(cfg, tmp), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, arr in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook7_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(f"{BUILD}/assets/_video_secretary_{k}.mp4",
            f"{BUILD}/assets/_mix_secretary_{k}.wav", HOOKS[k]["out"],
            post_a=HOOKS[k]["post_a"])
    else:
        main(k)

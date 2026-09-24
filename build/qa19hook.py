"""QA версий ролика 19 с другим хуком — чеклист delivery-specs §6.

    python3 qa19hook.py h1
    python3 qa19hook.py h2
"""
import hashlib
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import CARD_A, CARD_B, FPS, W, H
import storyboard19 as SB
import render19 as R
from hook19 import HOOKS, CUT_F, CUT, BUILD, BODY

V1 = SB.OUT
ALPHA = 40


def probe(path, *entries, stream="v:0"):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", "stream=" + ",".join(entries), "-of", "json", path],
        capture_output=True, text=True).stdout
    return json.loads(out)["streams"][0]


def dur_of(path, stream):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", "stream=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip()
    return float(out)


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    tail = r[r.rindex("Integrated loudness"):]
    i = float(tail.split("I:")[1].split("LUFS")[0])
    tp = float(tail.split("Peak:")[1].split("dBFS")[0])
    return i, tp


def panel_width(path, t, band):
    """Ширина белой QR-панели — объект для замера деформации.

    Меряем только внутри горизонтальной полосы по центру панели: если брать
    весь кадр, в замер попадает субтитр (он белый и шире панели) и деформация
    считается неправильно.
    """
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * FPS)))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[band[0]:band[1], :]
    ys, xs = np.nonzero(g > 170)
    if len(xs) == 0:
        return None
    return int(xs.max() - xs.min() + 1)


def card_rgb_at(path, t, scale=1.0, dx=0, dy=0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * FPS)))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    x, y, w, h = CARD_A
    x = int(x * scale + dx)
    y = int(y * scale + dy)
    w = int(w * scale)
    h = int(h * scale)
    crop = img[y + 60:y + h - 60, x + 60:x + w - 60]
    return tuple(int(v) for v in cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3).mean(0))


# --- проверки картинки по блоку хука (до деформации) -----------------------
def patch(cfg):
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()


def hook_layers(cfg):
    """Графика не залезает на субтитры, субтитры внутри карточки, графика в зоне."""
    patch(cfg)
    y0, y1 = SB.GFX_ZONE
    x0, x1 = SB.GFX_X
    ov = zone = card = 0
    nf = cfg["frames"]
    for f in range(0, nf, 2):
        t = f / FPS
        t0, _, kind, _ = R.shot_at(t)
        gl = R.graphics_layer(kind, t - t0)
        cl = R.caption_layer(t)
        if gl is not None:
            a = np.array(gl.split()[3])
            ys, xs = np.nonzero(a > ALPHA)
            if len(ys) and (ys.min() < y0 or ys.max() > y1
                            or xs.min() < x0 or xs.max() > x1):
                zone += 1
            if cl is not None:
                ca = np.array(cl.split()[3])
                if np.count_nonzero((a > ALPHA) & (ca > ALPHA)):
                    ov += 1
        if cl is not None:
            ca = np.array(cl.split()[3])
            ys, xs = np.nonzero(ca > 120)
            if len(ys):
                cx, cy, cw, ch = CARD_B if kind == "stock" else CARD_A
                if (xs.min() < cx + 30 or xs.max() > cx + cw - 30
                        or ys.min() < cy + 30 or ys.max() > cy + ch - 30):
                    card += 1
    return ov, zone, card


def bright_outside(path, cfg, n_hook, scale=1.0, dx=0, dy=0, tscale=1.0):
    """Нет пикселей ярче 200 вне прямоугольника карточки — по всей версии."""
    patch(cfg)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bad, sample = 0, []
    for f in range(0, total, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        t = f / FPS * tscale
        if t < n_hook / FPS:
            kind = R.shot_at(t)[2]
        else:
            kind = SB.shot_at(t - n_hook / FPS + CUT)[2]
        rect = CARD_B if kind == "stock" else CARD_A
        x, y, w, h = rect
        # после кропа x0.98 границы карточки дробные (правый край 983.88):
        # маску считаем floor/ceil от преобразованных углов плюс 1px допуска,
        # иначе ресемплированный краевой пиксель самой карточки читается как
        # «свет за карточкой» и проверка краснеет на ровном месте
        fx0, fy0 = x * scale + dx, y * scale + dy
        fx1, fy1 = (x + w) * scale + dx, (y + h) * scale + dy
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[max(0, int(np.floor(fy0)) - 1):int(np.ceil(fy1)) + 1,
             max(0, int(np.floor(fx0)) - 1):int(np.ceil(fx1)) + 1] = 1
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.any((g > 200) & (mask == 0)):
            bad += 1
            sample.append(round(f / FPS, 2))
    cap.release()
    return bad, sample[:8]


def body_identical(key, cfg):
    """Тело версии до деформации совпадает с уже сданным роликом кадр в кадр."""
    a = cv2.VideoCapture(f"{BUILD}/assets/_video_qr{key}.mp4")
    b = cv2.VideoCapture(BODY)
    worst, checked = 99.0, 0
    for k in range(0, 1370, 137):
        a.set(cv2.CAP_PROP_POS_FRAMES, cfg["frames"] + k)
        b.set(cv2.CAP_PROP_POS_FRAMES, CUT_F + k)
        oka, fa = a.read()
        okb, fb = b.read()
        if not (oka and okb):
            continue
        mse = np.mean((fa.astype(np.float64) - fb.astype(np.float64)) ** 2)
        p = 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
        worst = min(worst, p)
        checked += 1
    a.release()
    b.release()
    return worst, checked


def seam_click(path, n_hook, tscale=1.0):
    """На стыке нет щелчка: скачок между соседними сэмплами того же порядка,
    что внутри дорожки."""
    tmp = "/tmp/_seam19.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                    "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", tmp], check=True)
    import wave
    w = wave.open(tmp)
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    w.close()
    os.remove(tmp)
    d = np.abs(np.diff(x))
    t = (n_hook / FPS) / tscale
    i = int(t * 48000)
    local = d[max(0, i - 240):i + 240].max()
    return float(local), float(np.percentile(d, 99.99)), float(d.max())


def hook_pause(key, cfg, tscale=1.0):
    """Пауза между последним словом хука и первым словом тела."""
    def env(path):
        tmp = "/tmp/_p19.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                        "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", tmp], check=True)
        import wave
        w = wave.open(tmp)
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        w.close()
        os.remove(tmp)
        win = int(0.01 * 48000)
        return np.convolve(np.abs(x), np.ones(win) / win, mode="same"), 48000

    e, sr = env(cfg["src"])
    end_hook = np.nonzero(e > e.max() * 0.05)[0][-1] / sr - cfg["src_ss"]
    e2, sr2 = env(SB.SRC)
    seg = e2[int(7.30 * sr2):int(7.90 * sr2)]
    start_body = 7.30 + np.nonzero(seg > e2.max() * 0.05)[0][0] / sr2
    gap = (cfg["frames"] / FPS - end_hook) + (start_body - CUT)
    return gap / tscale


if __name__ == "__main__":
    key = sys.argv[1]
    cfg = HOOKS[key]
    out = cfg["out"]
    tscale = 1.02 if cfg.get("post_a") else 1.0
    scale = 1 / 0.98 if "crop" in (cfg.get("post_v") or "") else 1.0
    dx = -1080 * 0.01 / 0.98 if scale != 1.0 else 0
    dy = -1920 * 0.01 / 0.98 if scale != 1.0 else 0

    print(f"=== версия {cfg['version']} ({key}) -> {os.path.basename(out)}")
    st = probe(out, "width", "height", "r_frame_rate", "nb_frames")
    print(f"контейнер: {st['width']}x{st['height']} · r_frame_rate={st['r_frame_rate']} "
          f"· кадров {st.get('nb_frames')}")
    dv, da = dur_of(out, "v:0"), dur_of(out, "a:0")
    print(f"длительности: видео {dv:.3f}с · аудио {da:.3f}с · расхождение {abs(dv-da)*1000:.0f}мс")

    lens = [t1 - t0 for t0, t1, _, _ in cfg["shots"]]
    print(f"хук: {len(cfg['shots'])} планов, средняя {sum(lens)/len(lens):.2f}с, "
          f"первый рез {lens[0]:.2f}с, блок {cfg['frames']/FPS:.4f}с")

    ov, zone, card = hook_layers(cfg)
    print(f"gfx_text_overlap_frames={ov} gfx_out_of_zone_frames={zone} "
          f"text_out_of_card_frames={card}")

    b, bs = bright_outside(out, cfg, cfg["frames"], scale, dx, dy, tscale)
    print(f"bright_outside_card_frames={b} sample={bs}")

    p, n = body_identical(key, cfg)
    print(f"тело против сданной версии: худший PSNR {p:.1f} dB на {n} кадрах")

    band1 = (1000, 1120)                       # полоса по центру панели, без текста
    band2 = (int(1000 * scale + dy), int(1120 * scale + dy))
    pw1 = panel_width(V1, 9.90, band1)
    tpw = (9.90 - CUT + cfg["frames"] / FPS) / tscale
    pw2 = panel_width(out, tpw, band2)
    print(f"деформация: панель QR {pw1}px -> {pw2}px = x{pw2/pw1:.4f} "
          f"(ожидается x{scale:.4f})")

    r1 = card_rgb_at(V1, 1.0)
    r2 = card_rgb_at(out, 1.0 / tscale, scale, dx, dy)
    print(f"средний RGB карточки A: хук {r2} · сданная версия (тело) {r1} "
          f"· max Δ {max(abs(np.array(r1)-np.array(r2))):.0f}")

    loc, p9999, mx = seam_click(out, cfg["frames"], tscale)
    print(f"стык: скачок {loc:.5f} · p99.99 по дорожке {p9999:.5f} · max {mx:.5f} "
          f"-> {'ок' if loc <= p9999 else 'ЩЕЛЧОК'}")

    print(f"пауза хук->тело: {hook_pause(key, cfg, tscale)*tscale:.3f}с (норма 0.10-0.20)")

    i, tp = loudness(out)
    i1, tp1 = loudness(V1)
    print(f"громкость: {i:.2f} LUFS / TP {tp:.2f} · сданная версия {i1:.2f} / {tp1:.2f}")

    print(f"музыка: {os.path.basename(cfg['music'])}")
    h = hashlib.sha256(open(V1, 'rb').read()).hexdigest()[:16]
    print(f"версия №1 sha256[:16]={h} size={os.path.getsize(V1)}")

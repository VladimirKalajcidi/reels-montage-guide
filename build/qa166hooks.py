"""Автопроверки версий с другим хуком ролика 66 (слот 166) — START-HERE «альтернативный хук», delivery-specs §6.
Проверки 1–4 qa166 — по немому видео версии ДО деформации (_video_ape166<key>.mp4: кроп/ускорение — равномерное
преобразование всего кадра поверх уже проверенной геометрии), на шкале «блок хука + тело со сдвигом».
Плюс: тело кадр в кадр = сданное тело v1; пауза и щелчок на шве; md5 версии 1 не изменился; итоговый файл — fps, длительности, громкость/пик, ошибки декодирования;
деформация измерена (кроп ×0.98 → ×1.0204 на экране, ускорение ×1.02 → кадров ÷1.02); RGB карточки A хук/тело.
    python3 qa166hooks.py h1"""
import os, sys, json, subprocess
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import FPS
import hook166 as K
import render166 as R
import qa166 as Q

key = sys.argv[1]
cfg = K.HOOKS[key]
hook_shots = K.setup(key)
NF_H = cfg["frames"]
dt = NF_H / FPS - K.CUT_V
# границы тела — в кадрах: сдвиг 12.30 + dt в плавающей точке ложится на волос позже кадра 225, и кадр сетки
# считался лицом («пустая карточка» на шве A1 → tobe, в v1 тот же кадр проверку прошёл)
fr_ = lambda t: round(t * FPS) / FPS
body = [(fr_(max(t0, K.CUT_V) + dt), fr_(t1 + dt), k, p) for t0, t1, k, p in K.BODY_SHOTS if t1 > K.CUT_V + 1e-6]
ALL = hook_shots + body
VID = f"{K.BUILD}/assets/_video_ape166{key}.mp4"
nf_vid = int(cv2.VideoCapture(VID).get(cv2.CAP_PROP_FRAME_COUNT))

# проверки 1 и 4 — по всему немому видео версии; 2 и 3 — по блоку хука (тело — готовые кадры, проверены qa166)
R.SHOTS = ALL
Q.SHOTS = ALL
Q.DUR = nf_vid / FPS
R.OUT = VID
print(f"=== QA хук {key} (версия {cfg['version']}), ролик 66 ===")
n1 = Q.check_text_outside_card()
R.SHOTS = hook_shots
Q.SHOTS = hook_shots
Q.CAPS = cfg["caps"]
Q.DUR = NF_H / FPS
n2 = Q.check_line_overlap()
n3 = Q.check_gfx_vs_text()
R.SHOTS = ALL
Q.SHOTS = ALL
Q.DUR = nf_vid / FPS
n4 = Q.check_card_not_black()
# 9. каждая фраза хука живёт хотя бы кадр (v1 хуков: начало 7.0333 < реза 211/30 — фраза уходила в прошлый план и не показывалась,
#    проверки 1–4 этого не видят: нет текста — нет брака)
invis = [" ".join(r[0] for r in c[2]) for i, c in enumerate(cfg["caps"]) if R.BLOCKS[i][1] - c[0] < 1 / FPS]
print(f"9. невидимые фразы хука: {len(invis)} из {len(cfg['caps'])}" + (f" — {invis}" if invis else ""))

# тело кадр в кадр
a, b = cv2.VideoCapture(VID), cv2.VideoCapture(K.BODY)
n_body = int(b.get(cv2.CAP_PROP_FRAME_COUNT)) - K.CUT_F
diffs = []
for j in list(range(0, n_body, 25)) + [n_body - 1]:
    a.set(cv2.CAP_PROP_POS_FRAMES, NF_H + j); b.set(cv2.CAP_PROP_POS_FRAMES, K.CUT_F + j)
    ok1, x = a.read(); ok2, y = b.read()
    if ok1 and ok2:
        d = np.abs(x.astype(np.int16) - y.astype(np.int16)).max(axis=2)
        diffs.append(float((d > 8).mean() * 100))     # доля пикселей с заметной разницей, % (гайд: сотые доли процента)
# выравнивание: сдвиг 0 должен быть лучше ±1 кадра (иначе тело съехало на кадр)
def _d(i, k):
    a.set(cv2.CAP_PROP_POS_FRAMES, NF_H + i); b.set(cv2.CAP_PROP_POS_FRAMES, K.CUT_F + i + k)
    x, y = a.read()[1], b.read()[1]
    return float((np.abs(x.astype(np.int16) - y.astype(np.int16)).max(axis=2) > 8).mean() * 100)
aligned = all(_d(i, 0) < min(_d(i, -1), _d(i, 1)) for i in (1, 400, 800))
print(f"5. тело = сданное v1: {len(diffs)} кадров, пикселей с разницей >8 — max {max(diffs):.3f}% "
      f"(шум повторного кодирования, больше всего на подвижном стоке), выравнивание {'сдвиг 0 — верно' if aligned else 'СЪЕХАЛО'}, "
      f"кадров тела {nf_vid - NF_H} из {n_body}")

# RGB карточки A: планы лица хука против планов лица тела (норма — до ~10 по каналу)
def card_mean(fr):
    x, y, w, h = R.CARD_A if hasattr(R, "CARD_A") else (105, 238, 870, 1380)
    return fr[y + 60:y + h - 60, x + 60:x + w - 60].reshape(-1, 3).mean(0)[::-1]
hk = [f for f in range(NF_H) if hook_shots[R._shot_idx(f / FPS)][2] in R.FACE_KINDS][::6]
bd = [NF_H + f for f in range(0, nf_vid - NF_H, 45) if ALL[R._shot_idx((NF_H + f) / FPS)][2] in R.FACE_KINDS]
def mean_of(frames):
    out = []
    for f in frames:
        a.set(cv2.CAP_PROP_POS_FRAMES, f); ok, x = a.read()
        if ok:
            out.append(card_mean(x))
    return np.mean(out, axis=0)
mh, mb = mean_of(hk), mean_of(bd)
print(f"6. RGB карточки A: хук {tuple(int(v) for v in mh)} · тело {tuple(int(v) for v in mb)} · "
      f"расхождение max {np.abs(mh - mb).max():.1f}")

# итоговый файл
OUT = cfg["out"]
pr = lambda s: subprocess.run(["ffprobe", "-v", "error", "-select_streams", s, "-show_entries",
                               "stream=r_frame_rate,duration,nb_frames,width,height", "-of", "json", OUT],
                              capture_output=True, text=True).stdout
v, au = json.loads(pr("v:0"))["streams"][0], json.loads(pr("a:0"))["streams"][0]
err = subprocess.run(["ffmpeg", "-v", "error", "-i", OUT, "-f", "null", "-"], capture_output=True, text=True).stderr
i, tp = K.measure(OUT)
print(f"7. файл: {v['width']}×{v['height']}, r_frame_rate {v['r_frame_rate']}, кадров {v['nb_frames']}, "
      f"видео {float(v['duration']):.3f}с / звук {float(au['duration']):.3f}с, {i:.1f} LUFS / {tp:.1f} dBTP, "
      f"ошибок декодирования {len(err.strip().splitlines()) if err.strip() else 0}")

# деформация — машиной: ширина карточки A на кадре с лицом до и после
def card_width(path, t):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = c.read(); c.release()
    row = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[925]
    xs = np.nonzero(row > 12)[0]
    return int(xs[-1] - xs[0] + 1)
tf = (hk[len(hk) // 2]) / FPS
if "crop" in cfg["post_v"]:
    w0, w1 = card_width(VID, tf), card_width(OUT, tf)
    print(f"8. деформация кроп ×0.98: карточка A {w0}px → {w1}px = ×{w1 / w0:.4f} (норма ×1.0204)")
else:
    exp = round(nf_vid / 1.02)
    print(f"8. деформация ×1.02: кадров {nf_vid} → {v['nb_frames']} (ожидание ≈{exp}), fps {v['r_frame_rate']}")

# 10. шов: пауза «конец речи хука → первое слово тела» (норма 0.10–0.20с) и щелчок (скачок сэмплов на шве ~ как внутри)
wav = f"{K.BUILD}/assets/_qa166_{key}.wav"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", OUT, "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", wav], check=True)
import wave as _w
ww = _w.open(wav); xs = np.frombuffer(ww.readframes(ww.getnframes()), np.int16).astype(np.float32) / 32768; ww.close()
os.remove(wav)
sr = 48000
seam = (NF_H - 1) / FPS if "atempo" not in (cfg.get("post_a") or "") else (NF_H - 1) / FPS / 1.02
# пауза — по дорожкам голоса отдельно (в миксе под паузой музыка, порог по миксу её не видит):
# хвост речи хука до шва голоса (NF_H − 1)/FPS и начало речи тела после CUT_A, RMS 10 мс ≥ −36 дБ
def _env(path):
    w_ = _w.open(path); x_ = np.frombuffer(w_.readframes(w_.getnframes()), np.int16).astype(np.float32) / 32768
    return 20 * np.log10(np.sqrt(np.convolve(x_ ** 2, np.ones(160) / 160, "same"))[::160] + 1e-6)
dh = _env(f"{K.VIDEO_DIR}/work/hook{key[1]}_16k.wav")
dbb = _env(f"{K.VIDEO_DIR}/work/voice16k.wav")
kh = int((NF_H - 1) / FPS * 100)
last = max(k for k in range(0, min(kh, len(dh))) if dh[k] > -36) / 100
kb = int(K.CUT_A * 100)
first = min(k for k in range(kb, kb + 200) if dbb[k] > -36) / 100
pause = ((NF_H - 1) / FPS - last) + (first - K.CUT_A)
if "atempo" in (cfg.get("post_a") or ""):
    pause /= 1.02
i0 = int(seam * sr)
jump = np.abs(np.diff(xs[i0 - 480:i0 + 480])).max()
inner = np.percentile(np.abs(np.diff(xs[int(sr * 1):int(sr * 4)])), 99.9)
print(f"10. шов на {seam:.3f}с: пауза {pause:.3f}с (хук до {last:.2f}, тело с {first:.2f}; норма 0.10–0.20), "
      f"скачок сэмплов {jump:.4f} при p99.9 внутри дорожки {inner:.4f}")
md5_now = subprocess.run(["md5", "-q", f"{K.VIDEO_DIR}/monkey_edit.mp4"], capture_output=True, text=True).stdout.strip()
md5_v1 = open(f"{K.VIDEO_DIR}/work/v1.md5").read().strip()
print(f"11. версия 1 (monkey_edit.mp4) не тронута: md5 {'совпадает' if md5_now == md5_v1 else 'ИЗМЕНИЛСЯ'}")

ok = n1 == 0 and n2 == 0 and n3 == 0 and n4 == 0 and not invis and max(diffs) < 0.5 and aligned and nf_vid - NF_H == n_body and 0.10 <= pause <= 0.20 \
     and jump <= 2 * inner and md5_now == md5_v1
print("ИТОГО:", "ПРОШЁЛ" if ok else f"брак: outside={n1} overlap={n2} gfx={n3} black={n4}")

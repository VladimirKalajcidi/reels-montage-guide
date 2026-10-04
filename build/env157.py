"""Огибающая громкости ролика 57 (RMS 10 мс) — начала слов, которые whisper сжал до ~0с.
Для каждого окна печатает локальные минимумы RMS (дБ) — кандидаты на границу слов."""
import numpy as np, wave
w = wave.open("/Users/vladimirkalajcidi/reels_challenge/videos/57/work/voice16k.wav")
x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
hop = 157
rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same"))[::hop]
db = 20 * np.log10(rms + 1e-6)
def dips(a, b):
    i0, i1 = int(a * 100), int(b * 100)
    seg = db[i0:i1]
    return [(round((i0 + k) / 100, 2), round(float(seg[k]), 1)) for k in range(2, len(seg) - 2)
            if seg[k] == seg[k - 2:k + 3].min()]
WIN = [(0.6, 1.3, "парадокс | в котором математическое"), (12.6, 13.8, "рублей | Кажется что"),
       (13.6, 14.5, "во втором с"), (8.0, 8.9, "чем в другом"), (28.8, 30.1, "сумме | Значит | менять"),
       (41.9, 42.5, "знать | как изначально"), (45.1, 45.6, "хотите | чтобы ваш"), (50.1, 51.2, "вероятность | и я проведу с ним"),
       (24.9, 25.7, "в том что"), (35.2, 35.9, "потому что мы")]
if __name__ == "__main__":
    for a, b, what in WIN:
        print(what, dips(a, b))
        print("   rms:", " ".join(f"{(int(a*100)+k)/100:.2f}:{db[int(a*100)+k]:.0f}" for k in range(0, int((b-a)*100), 3)))

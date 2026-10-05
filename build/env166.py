"""Огибающая громкости ролика 66 (RMS 10 мс): провалы между слогами в спорных окнах (→ env166.log)."""
import numpy as np, wave, sys
w = wave.open("/Users/vladimirkalajcidi/reels_challenge/videos/66/work/voice16k.wav")
x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
hop = 160
rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same"))[::hop]
db = 20 * np.log10(rms + 1e-6)
def dips(a, b):
    i0, i1 = int(a * 100), int(b * 100)
    seg = db[i0:i1]
    return [(round((i0 + k) / 100, 2), round(float(seg[k]), 1)) for k in range(2, len(seg) - 2)
            if seg[k] == seg[k - 2:k + 3].min()]
WIN = [(25.6, 27.9, "Но бесконечность времени ограничена")]
if __name__ == "__main__":
    for a, b, what in (WIN if len(sys.argv) < 3 else [(float(sys.argv[1]), float(sys.argv[2]), "окно")]):
        print(what, dips(a, b))
        print("   rms:", " ".join(f"{(int(a*100)+k)/100:.2f}:{db[int(a*100)+k]:.0f}" for k in range(0, int((b-a)*100), 2)))

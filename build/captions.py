"""Субтитры v3 — общий движок (brand-kit.md, «Субтитры v3»). Подключается из render.py.

Правила (прямые правки автора, 2026-09-24):
- шрифт — SF Pro Expanded Black (системный SFNS.ttf, оси Width=150, Weight=1000), КАПС;
  тем же шрифтом — весь текст ролика, включая цифры графики (num_item);
- на экране одна текущая фраза, 1–2 строки, перенос по словам; кегль 62 (кап-высота ≈2.4%
  экрана); текст ТОЛЬКО внизу карточки — сверху ничего;
- без свечения: тёмная тень вправо-вниз;
- каждое слово всплывает снизу из размытия в момент, когда его произносят (words.json);
- изредка слово-акцент цветом: runs с начертанием "red" / "teal" вместо "r".

Использование в render.py:
    CAP = Captions(CAPS, SHOTS, DUR, f"{VIDEO_DIR}/words.json")
    layer = CAP.layer(t)          # RGBA на весь холст или None — накладывать последним
    CAP.items(t)                  # слова в конечных позициях — для проверки наложения строк
    CAP.phrase_starts()           # моменты первых слов фраз — для звука появления текста (sfx.py)
"""
import json
import math
import os

from PIL import Image, ImageFilter, ImageFont

from style import W, H, WHITE, CARD_A, CARD_B, SANS, text_layer, ease_out

SIZE = 62                                        # кегль — все планы
BASE = {"A": 1470, "B": 1160, "G": 1500}         # базовая линия ПОСЛЕДНЕЙ строки фразы
PAD = 30                                         # отступ текста от краёв карточки
COLORS = {"r": WHITE, "s": WHITE, "i": WHITE,
          "red": (255, 59, 48), "teal": (45, 225, 194)}
NUM_BLUE = (91, 124, 255)                        # синий числа без свечения (#5B7CFF)
SH_DX, SH_DY, SH_BLUR, SH_A = 3, 4, 2, 0.75      # тень: сдвиг, размытие, непрозрачность
POP_DUR, POP_DY, POP_BLUR = 0.23, 28, 10         # всплытие слова: 0.23с, подъём 28px, блюр 10→0

# какой зоне принадлежит план: B — сток в карточке B, G — графика/число на карточке A, A — лицо
STOCK_KINDS = {"stock"}
FACE_KINDS = {"A1", "A2"}

_wf = {}


def wide(size):
    """SF Pro Expanded Black. На Linux/Windows подставь широкий жирный гротеск (см. README)."""
    if size not in _wf:
        f = ImageFont.truetype(SANS, size)
        try:
            f.set_variation_by_axes([150, 28, 400, 1000])   # Width, opsz, GRAD, Weight
        except Exception:
            pass
        _wf[size] = f
    return _wf[size]


def num_item(t, t0, text, xy, size, blue=False, max_w=None):
    """Число-ревил тем же шрифтом, без свечения. R5b (синее): 0.38с, 0.55→1.0, оверщут ~3%,
    разряды слева направо. R5a (белое): 80мс, 0.6→1.0, без оверщута."""
    lt = t - t0
    if lt < 0:
        return []
    if max_w:
        size = min(size, int(size * max_w / wide(size).getlength(text)))
    if blue:
        p = min(1.0, lt / 0.38)
        sc = 0.55 + 0.45 * ease_out(p)
        if 0.55 < p < 1.0:
            sc += 0.03 * math.sin((p - 0.55) / 0.45 * math.pi)
    else:
        p = min(1.0, lt / 0.08)
        sc = 0.6 + 0.4 * ease_out(p)
    f = wide(max(8, int(size * sc)))
    it = dict(text=text, font=f, xy=xy, anchor="mm", fill=NUM_BLUE if blue else WHITE,
              glow_r=0, opacity=min(1.0, lt / (0.12 if blue else 0.06)))
    if blue and p < 1.0:
        wdt = f.getlength(text)
        it["reveal"] = ("wipe", xy[0] - wdt / 2 + wdt * min(1.0, p / 0.8) + 4)
    return [it]


def label_item(text, xy, size=50, opacity=1.0, anchor="mm"):
    """Служебная подпись (R7, подпись под числом) — тем же шрифтом, без свечения."""
    return dict(text=text.upper(), font=wide(size), xy=xy, anchor=anchor,
                fill=WHITE, glow_r=0, opacity=opacity)


def shadowed_layer(items, size=(W, H)):
    """Текст с тенью вместо свечения."""
    sh = [dict(it, fill=(0, 0, 0), xy=(it["xy"][0] + SH_DX, it["xy"][1] + SH_DY)) for it in items]
    lay = text_layer(size, sh)
    lay.putalpha(lay.split()[3].filter(ImageFilter.GaussianBlur(SH_BLUR))
                 .point(lambda v: int(v * SH_A)))
    lay.alpha_composite(text_layer(size, items))
    return lay


class Captions:
    def __init__(self, caps, shots, dur, words_json=None):
        self.caps, self.shots, self.dur = caps, shots, dur
        self.words = json.load(open(words_json)) if words_json and os.path.exists(words_json) else []
        self.ends = self._phrase_ends()
        self.word_t = [self._word_times(i) for i in range(len(caps))]
        self.lay = [self._layout(i) for i in range(len(caps))]
        self._cache = {}

    # ---------------------------------------------------------------- тайминг
    def _shot_idx(self, t):
        for i, s in enumerate(self.shots):
            if s[0] <= t < s[1]:
                return i
        return len(self.shots) - 1

    def _zone(self, i):
        kind = self.shots[self._shot_idx(self.caps[i][0])][2]
        return "B" if kind in STOCK_KINDS else ("A" if kind in FACE_KINDS else "G")

    def _phrase_ends(self):
        """Фраза живёт до следующей фразы (+≤0.30с после своего конца), но не дальше своего плана."""
        ends = []
        for i, (t0, t1, runs, slot) in enumerate(self.caps):
            nxt = self.caps[i + 1][0] if i + 1 < len(self.caps) else self.dur
            shot_end = self.shots[self._shot_idx(t0)][1]
            ends.append(min(max(t1, min(nxt, t1 + 0.30)), shot_end))
        return ends

    def _word_times(self, i):
        t0, t1, runs, slot = self.caps[i]
        n = sum(len(txt.split()) for (txt, k, sz) in runs)
        ws = [w["start"] for w in self.words if t0 - 0.06 <= w["start"] < t1]
        if len(ws) >= n:
            return [max(t0, x) for x in ws[:n]]
        step = (t1 - t0) / max(1, n)              # нет words.json — равномерно по фразе
        return [t0 + k * step for k in range(n)]

    # ---------------------------------------------------------------- раскладка
    @staticmethod
    def _wrap(words, f, avail):
        lines, cur, sp = [], [], f.getlength(" ")
        for w in words:
            cand = cur + [w]
            if cur and sum(f.getlength(x[0]) for x in cand) + sp * (len(cand) - 1) > avail:
                lines.append(cur); cur = [w]
            else:
                cur = cand
        return lines + [cur]

    def _fit(self, words, avail):
        """Не больше 2 строк и самое длинное слово влезает в карточку; иначе кегль вниз по 2px."""
        sz = SIZE
        while True:
            f = wide(sz)
            lines = self._wrap(words, f, avail)
            sp = f.getlength(" ")
            widest = max(sum(f.getlength(x[0]) for x in ln) + sp * (len(ln) - 1) for ln in lines)
            if (len(lines) <= 2 and widest <= avail) or sz <= 30:
                return f, sz, lines
            sz -= 2

    def _layout(self, i):
        t0, t1, runs, slot = self.caps[i]
        zone = self._zone(i)
        card = CARD_B if zone == "B" else CARD_A
        words = [(w.upper(), COLORS.get(k, WHITE)) for (txt, k, sz) in runs for w in txt.split()]
        f, sz, lines = self._fit(words, card[2] - 2 * PAD)
        strs = [" ".join(w for w, _ in ln) for ln in lines]
        bb = [f.getbbox(x, anchor="ls") for x in strs]
        pitch = max([bb[k][3] - bb[k + 1][1] + sz * 0.18 for k in range(len(lines) - 1)] or [0])
        sp = f.getlength(" ")
        out, k_w = [], 0
        for k, ln in enumerate(lines):
            x = card[0] + card[2] / 2 - (sum(f.getlength(w) for w, _ in ln) + sp * (len(ln) - 1)) / 2
            y = BASE[zone] - (len(lines) - 1 - k) * pitch
            for w, col in ln:
                out.append(dict(text=w, font=f, xy=(x, y), anchor="ls", fill=col, glow_r=0,
                                t_word=self.word_t[i][k_w]))
                k_w += 1
                x += f.getlength(w) + sp
        return out

    # ---------------------------------------------------------------- вывод
    def _active(self, t):
        act = [i for i, c in enumerate(self.caps) if c[0] <= t < self.ends[i]]
        return act[-1] if act else None

    def items(self, t):
        i = self._active(t)
        if i is None:
            return []
        return [{k: v for k, v in it.items() if k != "t_word"}
                for it in self.lay[i] if t >= it["t_word"]]

    def layer(self, t):
        i = self._active(t)
        if i is None:
            return None
        out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        for it in self.lay[i]:
            lt = t - it["t_word"]
            if lt < 0:
                continue
            key = (i, it["text"], it["xy"])
            if key not in self._cache:
                if len(self._cache) > 64:
                    self._cache.clear()
                self._cache[key] = shadowed_layer([{k: v for k, v in it.items() if k != "t_word"}])
            lay = self._cache[key]
            p = min(1.0, lt / POP_DUR)
            if p < 1.0:
                e = ease_out(p)
                lay = lay.transform(lay.size, Image.AFFINE, (1, 0, 0, 0, 1, -int(round(POP_DY * (1 - e)))))
                r = POP_BLUR * (1 - e)
                if r > 0.3:
                    lay = lay.filter(ImageFilter.GaussianBlur(r))
                lay = lay.copy()
                lay.putalpha(lay.split()[3].point(lambda v: int(v * e)))
            out.alpha_composite(lay)
        return out

    def phrase_starts(self):
        return [wt[0] for wt in self.word_t if wt]

"""Id стока, уже стоявшего в роликах (урок ролика 39): имена videos/*/stock/**/(stock_)?<id>*.mp4 и поля "clip"
во всех build/storyboard*.py (включая свежие раскадровки соседних сессий). Путь rejected/ помечается отдельно
(урок 41/51: чужой брак — смотреть причину). -> videos/66/used_ids.txt (id \t где)"""
import glob, os, re
ROOT = "/Users/vladimirkalajcidi/reels_challenge"
used = {}
for p in glob.glob(f"{ROOT}/videos/*/stock/**/*.mp4", recursive=True):
    if "/videos/66/" in p:
        continue
    m = re.match(r"^(stock_)?(?:mk_|pb_|px_)?(\d+)", os.path.basename(p))
    if m and "prepared" not in p:
        used.setdefault(m.group(2), set()).add(os.path.relpath(p, ROOT))
for p in glob.glob(f"{ROOT}/build/storyboard*.py"):
    if p.endswith("storyboard166.py"):
        continue
    for cid in re.findall(r"[\"']clip[\"']\s*:\s*[\"'](?:mk:)?(\d+)", open(p, encoding="utf-8", errors="ignore").read()):
        used.setdefault(cid, set()).add(os.path.basename(p))
with open(f"{ROOT}/videos/66/used_ids.txt", "w") as f:
    for k in sorted(used, key=int):
        f.write(f"{k}\t{'; '.join(sorted(used[k]))[:300]}\n")
print("id:", len(used))

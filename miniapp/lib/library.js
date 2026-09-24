const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');
const VIDEOS_DIR = path.join(ROOT, 'videos');
const DATA_DIR = path.join(__dirname, '..', 'data');
const META_FILE = path.join(DATA_DIR, 'video-meta.json');

fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(META_FILE)) fs.writeFileSync(META_FILE, '{}\n');

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return fallback;
  }
}

function writeJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2) + '\n');
}

function safeReadDir(dir) {
  try {
    return fs.readdirSync(dir);
  } catch {
    return [];
  }
}

function readText(file) {
  try {
    const text = fs.readFileSync(file, 'utf8').trim();
    return text || null;
  } catch {
    return null;
  }
}

function relVideo(num, file) {
  return `/videos/${num}/${encodeURIComponent(file)}`;
}

function loadMeta() {
  return readJson(META_FILE, {});
}

function saveMeta(meta) {
  writeJson(META_FILE, meta);
}

function detectTitle(dir, num, metaTitle) {
  if (metaTitle) return metaTitle;
  const plan = path.join(dir, 'montage-plan.md');
  const head = readText(plan);
  if (head) {
    const m = head.slice(0, 400).match(/[«"]([^»"]+)[»"]/);
    if (m) return m[1].trim();
  }

  const main = detectMainVideo(dir);
  if (main) return main.replace(/_edit\.(mp4|mov)$/i, '').replace(/[_-]+/g, ' ');
  return `Ролик ${num}`;
}

function detectMainVideo(dir) {
  const files = safeReadDir(dir);
  return files.find((file) => /_edit\.(mp4|mov)$/i.test(file)) || null;
}

function detectHookVideos(dir) {
  return safeReadDir(dir)
    .filter((file) => /_hook\d+(?:_edited)?\.(mp4|mov)$/i.test(file))
    .sort((a, b) => a.localeCompare(b, 'ru'));
}

function createdAt(dir) {
  try {
    const files = safeReadDir(dir)
      .filter((file) => /\.(mp4|mov)$/i.test(file))
      .map((file) => fs.statSync(path.join(dir, file)).mtimeMs);
    if (files.length) return new Date(Math.max(...files)).toISOString();
    return new Date(fs.statSync(dir).mtimeMs).toISOString();
  } catch {
    return new Date().toISOString();
  }
}

function listVideos() {
  const meta = loadMeta();
  return safeReadDir(VIDEOS_DIR)
    .filter((name) => /^\d+$/.test(name))
    .map((name) => {
      const num = Number(name);
      const dir = path.join(VIDEOS_DIR, name);
      if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return null;

      const saved = meta[name] || {};
      const mainVideo = detectMainVideo(dir);
      const hookVideos = detectHookVideos(dir);
      const teleprompterText = typeof saved.teleprompterText === 'string'
        ? saved.teleprompterText
        : (readText(path.join(dir, 'source.txt')) || '');
      const note = typeof saved.note === 'string' ? saved.note : '';
      const posted = Boolean(saved.posted);

      return {
        num,
        title: detectTitle(dir, num, saved.title),
        createdAt: createdAt(dir),
        statuses: {
          edited: Boolean(mainVideo),
          posted,
          hasText: Boolean(teleprompterText),
        },
        emoji: {
          edited: Boolean(mainVideo) ? '📹' : '',
          posted: posted ? '✅' : '',
        },
        mainVideo: mainVideo ? {
          file: mainVideo,
          url: relVideo(num, mainVideo),
        } : null,
        hooks: hookVideos.map((file) => ({
          file,
          url: relVideo(num, file),
        })),
        teleprompterText,
        note,
        updatedAt: saved.updatedAt || null,
      };
    })
    .filter(Boolean)
    .sort((a, b) => b.num - a.num);
}

function getVideo(num) {
  return listVideos().find((item) => item.num === Number(num)) || null;
}

function updateVideo(num, patch) {
  const key = String(num);
  const meta = loadMeta();
  const prev = meta[key] || {};
  const next = { ...prev };

  if (Object.prototype.hasOwnProperty.call(patch, 'posted')) next.posted = Boolean(patch.posted);
  if (Object.prototype.hasOwnProperty.call(patch, 'title')) next.title = patch.title || '';
  if (Object.prototype.hasOwnProperty.call(patch, 'teleprompterText')) next.teleprompterText = patch.teleprompterText || '';
  if (Object.prototype.hasOwnProperty.call(patch, 'note')) next.note = patch.note || '';
  next.updatedAt = new Date().toISOString();

  meta[key] = next;
  saveMeta(meta);
  return getVideo(num);
}

module.exports = {
  ROOT,
  VIDEOS_DIR,
  META_FILE,
  listVideos,
  getVideo,
  updateVideo,
};

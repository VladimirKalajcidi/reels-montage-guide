const fs = require('fs');
const path = require('path');

const DEFAULT_PROJECT_ROOT = path.resolve(__dirname, '..');
const PROJECT_ROOT = process.env.REELS_PROJECT_ROOT
  ? path.resolve(process.env.REELS_PROJECT_ROOT)
  : DEFAULT_PROJECT_ROOT;
const VIDEOS_DIR = process.env.REELS_VIDEOS_DIR
  ? path.resolve(process.env.REELS_VIDEOS_DIR)
  : path.join(PROJECT_ROOT, 'videos');
const DATA_DIR = path.join(__dirname, 'data');
const META_FILE = path.join(DATA_DIR, 'video-meta.json');

fs.mkdirSync(DATA_DIR, { recursive: true });

function ensureMetaFile() {
  if (!fs.existsSync(META_FILE)) {
    fs.writeFileSync(META_FILE, '{}\n');
  }
}

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return fallback;
  }
}

function readText(file) {
  try {
    return fs.readFileSync(file, 'utf8').trim() || null;
  } catch {
    return null;
  }
}

function loadMetaMap() {
  ensureMetaFile();
  return readJson(META_FILE, {});
}

function saveMetaMap(metaMap) {
  ensureMetaFile();
  fs.writeFileSync(META_FILE, JSON.stringify(metaMap, null, 2) + '\n');
}

function relVideoPath(num, file) {
  return `/videos/${num}/${file}`;
}

function folderTitle(num) {
  const dir = path.join(VIDEOS_DIR, String(num));
  for (const name of ['montage-plan.md', 'montage-plan-hook1.md']) {
    const p = path.join(dir, name);
    if (!fs.existsSync(p)) continue;
    const head = readText(p);
    if (!head) continue;
    const m = head.slice(0, 400).match(/[«"]([^»"]+)[»"]/);
    if (m) return m[1].trim();
  }
  const main = detectMainVideo(dir);
  if (main) {
    return main.replace(/_edit\.(mp4|mov)$/i, '').replace(/[_-]+/g, ' ');
  }
  return `Ролик ${num}`;
}

function folderCreatedAt(dir) {
  try {
    const deliverables = fs.readdirSync(dir)
      .filter((f) => /\.(mp4|mov)$/i.test(f) && !/^source\b/i.test(f) && !/^hook\d+\./i.test(f))
      .map((f) => fs.statSync(path.join(dir, f)).mtimeMs);
    if (deliverables.length) return new Date(Math.max(...deliverables)).toISOString();
    return new Date(fs.statSync(dir).mtimeMs).toISOString();
  } catch {
    return new Date().toISOString();
  }
}

function detectMainVideo(dir) {
  const preferred = ['output_v1.mp4'];
  for (const name of preferred) {
    if (fs.existsSync(path.join(dir, name))) return name;
  }
  const files = safeReadDir(dir);
  return files.find((f) => /_edit\.(mp4|mov)$/i.test(f)) || null;
}

function safeReadDir(dir) {
  try {
    return fs.readdirSync(dir);
  } catch {
    return [];
  }
}

function listHookVideos(dir) {
  return safeReadDir(dir)
    .filter((f) => /_hook\d+\.(mp4|mov)$/i.test(f))
    .sort((a, b) => a.localeCompare(b, 'ru'));
}

function listRawHooks(dir) {
  return safeReadDir(dir)
    .filter((f) => /^hook\d+\.(mp4|mov)$/i.test(f))
    .sort((a, b) => a.localeCompare(b, 'ru'));
}

function buildDownloadables(num, dir) {
  const items = [];
  const main = detectMainVideo(dir);
  if (main) {
    items.push({ key: 'main', label: 'Основной монтаж', file: main, path: relVideoPath(num, main) });
  }

  for (const hook of listHookVideos(dir)) {
    const match = hook.match(/_hook(\d+)\.(mp4|mov)$/i);
    items.push({
      key: `hook${match ? match[1] : hook}`,
      label: `Хук ${match ? match[1] : hook}`,
      file: hook,
      path: relVideoPath(num, hook),
    });
  }

  return items;
}

function getVideoLibraryItem(num, metaMap = loadMetaMap()) {
  const dir = path.join(VIDEOS_DIR, String(num));
  if (!fs.existsSync(dir)) return null;

  const meta = metaMap[String(num)] || {};
  const mainVideo = detectMainVideo(dir);
  const hookVideos = listHookVideos(dir);
  const teleprompterText = typeof meta.teleprompterText === 'string'
    ? meta.teleprompterText
    : readText(path.join(dir, 'source.txt'));
  const transcriptFile = fs.existsSync(path.join(dir, 'source.txt')) ? 'source.txt' : null;
  const note = typeof meta.note === 'string' ? meta.note : '';
  const posted = Boolean(meta.posted);
  const edited = Boolean(mainVideo);
  const hasSource = safeReadDir(dir).some((f) => /^source\.(mp4|mov)$/i.test(f));
  const rawHookCount = listRawHooks(dir).length;

  return {
    num,
    title: meta.title || folderTitle(num),
    createdAt: folderCreatedAt(dir),
    path: dir,
    statuses: {
      edited,
      posted,
      hasTeleprompterText: Boolean(teleprompterText),
      hasSource,
    },
    tags: [
      edited ? 'edited' : 'not_edited',
      posted ? 'posted' : 'not_posted',
      teleprompterText ? 'has_text' : 'no_text',
    ],
    mainVideo: mainVideo ? { file: mainVideo, path: relVideoPath(num, mainVideo) } : null,
    hookVideos: hookVideos.map((file) => ({ file, path: relVideoPath(num, file) })),
    downloadables: buildDownloadables(num, dir),
    rawHookCount,
    transcriptFile,
    teleprompterText,
    note,
    meta: {
      postedAt: meta.postedAt || null,
      updatedAt: meta.updatedAt || null,
    },
  };
}

function listVideoLibrary() {
  const metaMap = loadMetaMap();
  return safeReadDir(VIDEOS_DIR)
    .filter((name) => /^\d+$/.test(name) && fs.statSync(path.join(VIDEOS_DIR, name)).isDirectory())
    .map((name) => getVideoLibraryItem(parseInt(name, 10), metaMap))
    .filter(Boolean)
    .sort((a, b) => b.num - a.num);
}

function updateVideoMeta(num, patch) {
  const metaMap = loadMetaMap();
  const key = String(num);
  const prev = metaMap[key] || {};
  const next = { ...prev };

  if (Object.prototype.hasOwnProperty.call(patch, 'title')) next.title = patch.title || null;
  if (Object.prototype.hasOwnProperty.call(patch, 'note')) next.note = patch.note || '';
  if (Object.prototype.hasOwnProperty.call(patch, 'teleprompterText')) {
    next.teleprompterText = patch.teleprompterText || '';
  }
  if (Object.prototype.hasOwnProperty.call(patch, 'posted')) {
    next.posted = Boolean(patch.posted);
    next.postedAt = next.posted ? new Date().toISOString() : null;
  }

  next.updatedAt = new Date().toISOString();
  metaMap[key] = next;
  saveMetaMap(metaMap);
  return getVideoLibraryItem(num, metaMap);
}

module.exports = {
  DATA_DIR,
  META_FILE,
  PROJECT_ROOT,
  VIDEOS_DIR,
  getVideoLibraryItem,
  listVideoLibrary,
  loadMetaMap,
  updateVideoMeta,
};

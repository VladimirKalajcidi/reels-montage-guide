const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const readline = require('readline');
const { spawn } = require('child_process');
const { getVideoLibraryItem, listVideoLibrary, updateVideoMeta } = require('./video-library');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const VIDEOS_DIR = path.join(PROJECT_ROOT, 'videos');
const JOBS_DIR = path.join(__dirname, 'jobs');
fs.mkdirSync(JOBS_DIR, { recursive: true });

const app = express();
app.use(express.json());
app.use('/videos', express.static(VIDEOS_DIR));
app.use(express.static(path.join(__dirname, 'public')));

// ---------------------------------------------------------------- storage
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 4 * 1024 * 1024 * 1024 } });

/** @type {Map<string, Job>} */
const jobs = new Map();

function nextVideoNum() {
  let max = 0;
  for (const name of fs.readdirSync(VIDEOS_DIR)) {
    if (/^\d+$/.test(name) && fs.statSync(path.join(VIDEOS_DIR, name)).isDirectory()) {
      max = Math.max(max, parseInt(name, 10));
    }
  }
  return max + 1;
}

function jobFile(id) { return path.join(JOBS_DIR, `${id}.json`); }
function logFile(id) { return path.join(JOBS_DIR, `${id}.log`); }

function saveJob(job) {
  const { emitter, ...plain } = job;
  fs.writeFileSync(jobFile(job.id), JSON.stringify(plain, null, 2));
}

function appendLog(job, text) {
  fs.appendFileSync(logFile(job.id), text);
  job.emitter && job.emitter.emit('log', text);
}

function loadAllJobs() {
  for (const f of fs.readdirSync(JOBS_DIR)) {
    if (f.endsWith('.json')) {
      const plain = JSON.parse(fs.readFileSync(path.join(JOBS_DIR, f), 'utf8'));
      const EventEmitter = require('events');
      plain.quickEdits = plain.quickEdits || {};
      plain.editedVideo = plain.editedVideo || {};
      jobs.set(plain.id, { ...plain, emitter: new EventEmitter() });
    }
  }
}
loadAllJobs();

// ---------------------------------------------------------------- claude runner
function summarizeToolInput(name, input) {
  try {
    if (name === 'Bash') return (input.command || '').slice(0, 160);
    if (name === 'Read' || name === 'Write' || name === 'Edit') return input.file_path || '';
    if (name === 'Skill') return input.skill || '';
    return JSON.stringify(input).slice(0, 160);
  } catch { return ''; }
}

function runClaude(job, prompt, { resume }) {
  const args = ['-p', prompt, '--permission-mode', 'bypassPermissions',
    '--output-format', 'stream-json', '--verbose'];
  if (resume) args.push('--resume', job.sessionId);
  else args.push('--session-id', job.sessionId);

  appendLog(job, `\n\n===== ${resume ? 'ПРАВКИ / ВЕРСИИ V2-V3' : 'ОСНОВНОЙ МОНТАЖ'} — старт =====\n\n`);

  const child = spawn('claude', args, { cwd: PROJECT_ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
  const rl = readline.createInterface({ input: child.stdout });

  rl.on('line', (line) => {
    if (!line.trim()) return;
    let obj;
    try { obj = JSON.parse(line); } catch { return; }

    if (obj.type === 'assistant' && obj.message && obj.message.content) {
      for (const block of obj.message.content) {
        if (block.type === 'text' && block.text) {
          appendLog(job, block.text + '\n');
        } else if (block.type === 'tool_use') {
          appendLog(job, `\n▶ ${block.name} ${summarizeToolInput(block.name, block.input)}\n`);
        }
      }
    } else if (obj.type === 'result') {
      if (obj.is_error) {
        appendLog(job, `\n\n[ошибка] ${obj.result || obj.subtype}\n`);
      }
    }
  });

  child.stderr.on('data', (d) => appendLog(job, `[stderr] ${d}`));

  child.on('close', (code) => {
    appendLog(job, `\n\n===== процесс завершён, код ${code} =====\n`);
    finalizeStage(job, resume, code);
  });

  child.on('error', (err) => {
    job.status = 'error';
    job.error = String(err);
    appendLog(job, `\n[не удалось запустить claude] ${err}\n`);
    saveJob(job);
    job.emitter.emit('done');
  });
}

function relVideo(num, filename) {
  return `/videos/${num}/${filename}`;
}

// ---------------------------------------------------------------- library (videos/ folder is the source of truth)
const THUMBS_DIR = path.join(__dirname, 'thumbs');
fs.mkdirSync(THUMBS_DIR, { recursive: true });

/** Reads the human title straight out of the montage plan: `# Монтажный лист — «...»`. */
function folderTitle(num) {
  const dir = path.join(VIDEOS_DIR, String(num));
  for (const name of ['montage-plan.md', 'montage-plan-hook1.md']) {
    const p = path.join(dir, name);
    if (!fs.existsSync(p)) continue;
    try {
      const head = fs.readFileSync(p, 'utf8').slice(0, 400);
      const m = head.match(/[«"]([^»"]+)[»"]/);
      if (m) return m[1].trim();
    } catch {}
  }
  // fallback: derive from the deliverable's filename (ramsey_edit.mp4 → ramsey)
  try {
    const f = fs.readdirSync(dir).find((x) => /_edit\.(mp4|mov)$/i.test(x));
    if (f) return f.replace(/_edit\.(mp4|mov)$/i, '').replace(/[_-]+/g, ' ');
  } catch {}
  return null;
}

/** Date of the finished edit — not of the raw camera source, whose mtime can be far older. */
function folderCreatedAt(num) {
  const dir = path.join(VIDEOS_DIR, String(num));
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

/**
 * Poster frame for a clip, cached on disk and keyed by mtime so it refreshes on re-render.
 * Seeks a fifth of the way in and lets ffmpeg's `thumbnail` filter pick a representative
 * frame from there — a fixed early timestamp lands on the black/fade-in of most reels.
 */
async function thumbFor(num, file, mtime) {
  const key = `${num}_${file.replace(/[^\w.-]/g, '_')}_${Math.round(mtime)}.jpg`;
  const out = path.join(THUMBS_DIR, key);
  if (fs.existsSync(out)) return out;

  const src = path.join(VIDEOS_DIR, String(num), file);
  let seek = 3;
  try { seek = Math.max(1, (await ffprobeDuration(src)) * 0.2); } catch {}

  await new Promise((resolve, reject) => {
    const p = spawn('ffmpeg', ['-y', '-ss', String(seek.toFixed(2)), '-i', src,
      '-vf', 'thumbnail=90,scale=320:-2', '-frames:v', '1', '-q:v', '3', out]);
    p.on('close', (code) => (code === 0 && fs.existsSync(out) ? resolve() : reject(new Error('thumb failed'))));
    p.on('error', reject);
  });
  return out;
}

/** Scans videos/<num>/ and classifies every deliverable clip found there. */
function scanFolderVideos(num) {
  const dir = path.join(VIDEOS_DIR, String(num));
  if (!fs.existsSync(dir)) return [];
  let files;
  try { files = fs.readdirSync(dir); } catch { return []; }

  const entries = [];
  const order = { v1: 0, main: 0, v2: 1, v3: 2 };
  for (const f of files) {
    if (!/\.(mp4|mov)$/i.test(f)) continue;
    if (/^source\b/i.test(f)) continue;
    if (/_edited\.(mp4|mov)$/i.test(f)) continue;
    if (/^hook\d+\.(mp4|mov)$/i.test(f)) continue; // raw hook dubs are input, not a deliverable

    let key, label, deform = null, m;
    if (/^output_v1\.mp4$/i.test(f)) { key = 'v1'; label = 'Версия 1 · эталон'; }
    else if (/^output_v2\.mp4$/i.test(f)) { key = 'v2'; label = 'Версия 2 · инстаграм-тест'; deform = 'другой трек · кроп ×0.98'; }
    else if (/^output_v3\.mp4$/i.test(f)) { key = 'v3'; label = 'Версия 3 · инстаграм-тест'; deform = 'другой трек · ускорение ×1.02'; }
    else if ((m = f.match(/^(.+)_hook(\d+)\.(mp4|mov)$/i))) { key = `hook${m[2]}`; label = `Хук ${m[2]}`; }
    else if (/_edit\.(mp4|mov)$/i.test(f)) { key = 'main'; label = 'Основной монтаж'; }
    else { key = f.replace(/\.(mp4|mov)$/i, ''); label = f; }

    const full = path.join(dir, f);
    const editedName = f.replace(/\.(mp4|mov)$/i, '_edited.mp4');
    const editedFull = path.join(dir, editedName);

    const mtime = fs.statSync(full).mtimeMs;
    entries.push({
      key, label, deform,
      file: f,
      src: relVideo(num, f),
      editedSrc: fs.existsSync(editedFull) ? relVideo(num, editedName) : null,
      thumb: `/api/thumb/${num}/${encodeURIComponent(f)}`,
      strip: `/api/filmstrip/${num}/${encodeURIComponent(f)}`,
      mtime,
    });
  }

  entries.sort((a, b) => (order[a.key] ?? 5) - (order[b.key] ?? 5) || a.mtime - b.mtime);
  return entries;
}

function ensureJobForFolder(num) {
  for (const j of jobs.values()) if (j.num === num) return j;
  const id = crypto.randomUUID();
  const EventEmitter = require('events');
  const job = {
    id, num, ext: '.mp4', sessionId: null,
    title: folderTitle(num),
    status: 'imported',
    resultVideo: null,
    variants: { v2: null, v3: null },
    quickEdits: {},
    editedVideo: {},
    error: null,
    imported: true,
    createdAt: folderCreatedAt(num),
    emitter: new EventEmitter(),
  };
  jobs.set(id, job);
  fs.writeFileSync(logFile(id), '');
  saveJob(job);
  return job;
}

function syncLibrary() {
  if (!fs.existsSync(VIDEOS_DIR)) return;
  for (const name of fs.readdirSync(VIDEOS_DIR)) {
    const full = path.join(VIDEOS_DIR, name);
    if (/^\d+$/.test(name) && fs.statSync(full).isDirectory()) {
      ensureJobForFolder(parseInt(name, 10));
    }
  }
}
syncLibrary();

function finalizeStage(job, wasResume, exitCode) {
  const dir = path.join(VIDEOS_DIR, String(job.num));
  if (!wasResume) {
    const p = path.join(dir, 'output_v1.mp4');
    if (exitCode === 0 && fs.existsSync(p)) {
      job.status = 'ready';
      job.resultVideo = relVideo(job.num, 'output_v1.mp4');
    } else {
      job.status = 'error';
      job.error = `Основной файл output_v1.mp4 не найден (код завершения ${exitCode}). Смотри лог.`;
    }
  } else {
    const p2 = path.join(dir, 'output_v2.mp4');
    const p3 = path.join(dir, 'output_v3.mp4');
    const ok2 = fs.existsSync(p2), ok3 = fs.existsSync(p3);
    if (exitCode === 0 && ok2 && ok3) {
      job.status = 'variants-ready';
    } else {
      job.status = 'variants-error';
      job.error = `Не найдены output_v2.mp4/output_v3.mp4 (код завершения ${exitCode}). Смотри лог.`;
    }
    job.variants = {
      v2: ok2 ? relVideo(job.num, 'output_v2.mp4') : null,
      v3: ok3 ? relVideo(job.num, 'output_v3.mp4') : null,
    };
    // v1 may have been re-cut if feedback changed the body
    const p1 = path.join(dir, 'output_v1.mp4');
    if (fs.existsSync(p1)) job.resultVideo = relVideo(job.num, 'output_v1.mp4');
  }
  saveJob(job);
  job.emitter.emit('done');
}

function buildMainPrompt(job, meta) {
  const ext = job.ext;
  return `Я загрузил гайд по монтажу моих вертикальных роликов о математике — начни с START-HERE.md (он лежит в корне проекта). Исходник уже лежит в videos/${job.num}/source${ext}.

Тема ролика: ${meta.topic ? meta.topic : 'не указана — определи по содержанию речи'}.
Обязательные числа/факты, которые должны прозвучать: ${meta.numbers ? meta.numbers : 'не указаны'}.

Сделай по порядку, СТРОГО не задавая уточняющих вопросов и не останавливаясь на промежуточное согласование монтажного плана — работай от начала до конца автономно, это прямое указание владельца гайда:
1. Прочитай весь гайд по порядку, указанному в START-HERE.md (brand-kit.md → shot-recipes.md → editing-taste.md → delivery-specs.md, затем остальные файлы по ссылкам).
2. Разбери исходник по протоколу «При получении сырого исходника — основной сценарий» из START-HERE.md: техпаспорт (ffprobe), транскрипция с пословными таймингами (whisper), разметка нарратива, план вставок, подбор материала, субтитры.
3. Подбор вставок — строго видео, буквально соответствующее произнесённому слову/объекту/ситуации (не «по настроению» и не абстрактно). Если подходящего материала нет — своя графика вместо стока. Все вставки этого ролика клади в videos/${job.num}/stock/, не бери сток из других роликов.
4. Собери ролик кодом из build/: заведи storyboard${job.num}.py (и при необходимости render${job.num}.py/sfx${job.num}.py по конвенции репозитория, переиспользуя style.py/graph.py), N = ${job.num}.
5. Прогони обе обязательные автопроверки из START-HERE.md (текст за карточкой, наложение строк) — обе обязаны дать ноль; если не ноль — почини и перепроверь. Проверь машиной, что субтитр и графика на экране никогда не дублируют один и тот же текст (это не должно повторяться никогда).
6. Собери контакт-лист ключевых кадров и проверь по ручному чек-листу из START-HERE.md: графика буквально соответствует объекту из речи, геометрия/математика верны, крупные числа читаются, подписи не перекрывают важный объект.
7. Финальный рендер сохрани в videos/${job.num}/ обычным для репозитория именем, и ОБЯЗАТЕЛЬНО сделай его копию (cp) под именем videos/${job.num}/output_v1.mp4 — по этому пути веб-инструмент найдёт готовое видео.
8. В последнем сообщении кратко напиши: путь к финальному файлу, монтажный план (кратко, таблицей) и результаты обеих автопроверок.

Это версия №1 (эталон) — без деформаций и без альтернативной музыки. Версии для инстаграм-теста (v2/v3) закажу отдельно после того как посмотрю эту, сейчас их не делай.`;
}

function buildFeedbackPrompt(job, comment, mainFile) {
  return `Ролик videos/${job.num}/${mainFile} в целом устраивает и утверждается как версия 1 (эталон). Мои комментарии, которые нужно учесть при подготовке тестовых версий для инстаграма:

${comment || '(без правок — просто сделай уникализацию версий 2 и 3 как есть)'}

Сделай по разделу «Уникализация версий под площадку» из START-HERE.md:
- Если в комментариях выше есть просьбы про правки монтажа (другие вставки, текст на экране, тайминги, музыка, что-то не нравится и т.п.) — сначала внеси эти правки в тело ролика. Если правки не касаются исключительно хука, обнови и ${mainFile}, и videos/${job.num}/output_v1.mp4 (заведи output_v1.mp4, если его ещё не было) обновлённой копией — по этому пути веб-инструмент ищет эталон.
- Версия 2: другой трек из audios/ + кроп ×0.98 (обрезать 2% по краям, вернуть холст 1080×1920).
- Версия 3: третий трек из audios/ + ускорение ×1.02 (setpts И atempo вместе, обязательно fps=30 после setpts, звук мукшируется после правки кадра).
- Деформации проверяй машиной (измерение объекта в обеих версиях), а не на глаз.

Сохрани версии как videos/${job.num}/output_v2.mp4 и videos/${job.num}/output_v3.mp4 — по этим путям веб-инструмент найдёт готовые файлы.

Не задавай уточняющих вопросов, не останавливайся на согласование — сделай всё от начала до конца сама и заверши работу готовыми файлами. В конце одним сообщением перечисли: что именно изменил по моим комментариям (если что-то менял), какие треки и деформации применены к v2 и v3.`;
}

// ---------------------------------------------------------------- quick edit (ffmpeg, non-destructive)
function ffprobeDuration(file) {
  return new Promise((resolve, reject) => {
    const p = spawn('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file]);
    let out = '';
    p.stdout.on('data', (d) => (out += d));
    p.on('close', (code) => {
      if (code !== 0) return reject(new Error('ffprobe failed'));
      resolve(parseFloat(out.trim()));
    });
  });
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, Number.isFinite(v) ? v : lo)); }

const NEUTRAL = { brightness: 0, contrast: 100, saturation: 100, warmth: 0, volume: 100 };

function eqFilter(a, enable) {
  const b = clamp((a.brightness ?? 0) / 100, -1, 1) * 0.4;
  const c = clamp((a.contrast ?? 100) / 100, 0.5, 2);
  const s = clamp((a.saturation ?? 100) / 100, 0, 2);
  const parts = [`eq=brightness=${b.toFixed(3)}:contrast=${c.toFixed(3)}:saturation=${s.toFixed(3)}`];
  const w = clamp((a.warmth ?? 0) / 100, -1, 1) * 0.3;
  if (Math.abs(w) > 0.001) {
    parts.push(`colorbalance=rm=${w.toFixed(3)}:gm=${(w * 0.6).toFixed(3)}:bm=${(-w).toFixed(3)}`);
  }
  return parts.map((p) => (enable ? `${p}:enable='${enable}'` : p)).join(',');
}

function isNeutralColor(a) {
  return (a.brightness ?? 0) === NEUTRAL.brightness && (a.contrast ?? 100) === NEUTRAL.contrast
    && (a.saturation ?? 100) === NEUTRAL.saturation && (a.warmth ?? 0) === NEUTRAL.warmth;
}

/**
 * Colour/volume work is applied on the ORIGINAL timeline, and the cuts land last —
 * so every `enable=between(t,…)` window matches the seconds the user sees while
 * scrubbing the untouched clip, no mental arithmetic for removed pieces.
 */
function buildFilterComplex(edits, duration) {
  const trimStart = clamp(edits.trimStart || 0, 0, duration);
  const trimEnd = clamp(edits.trimEnd || duration, trimStart, duration);
  const cuts = Array.isArray(edits.cuts) ? edits.cuts
    .map((c) => ({ start: clamp(c.start, 0, duration), end: clamp(c.end, 0, duration) }))
    .filter((c) => c.end > c.start) : [];

  const ranges = [...cuts];
  if (trimStart > 0.001) ranges.push({ start: 0, end: trimStart });
  if (trimEnd < duration - 0.001) ranges.push({ start: trimEnd, end: duration });

  const selExpr = ranges.length
    ? ranges.map((r) => `not(between(t,${r.start.toFixed(3)},${r.end.toFixed(3)}))`).join('*')
    : null;

  const segments = (Array.isArray(edits.segments) ? edits.segments : [])
    .map((s) => ({
      ...s,
      start: clamp(s.start, 0, duration),
      end: clamp(s.end, 0, duration),
    }))
    .filter((s) => s.end > s.start);

  const vParts = [eqFilter(edits, null)];
  const aParts = [`volume=${clamp((edits.volume ?? 100) / 100, 0, 3).toFixed(3)}`];

  for (const s of segments) {
    const enable = `between(t,${s.start.toFixed(3)},${s.end.toFixed(3)})`;
    if (!isNeutralColor(s)) vParts.push(eqFilter(s, enable));
    const sv = clamp((s.volume ?? 100) / 100, 0, 3);
    if (Math.abs(sv - 1) > 0.001) aParts.push(`volume=${sv.toFixed(3)}:enable='${enable}'`);
  }

  if (selExpr) {
    vParts.push(`select='${selExpr}'`, 'setpts=N/FRAME_RATE/TB');
    aParts.push(`aselect='${selExpr}'`, 'asetpts=N/SR/TB');
  }

  return `[0:v]${vParts.join(',')}[vout];[0:a]${aParts.join(',')}[aout]`;
}

function applyQuickEdit(job, key, edits) {
  const dir = path.join(VIDEOS_DIR, String(job.num));
  const entry = scanFolderVideos(job.num).find((v) => v.key === key);
  if (!entry) throw new Error(`видео с ключом ${key} не найдено в папке ролика`);
  const base = path.join(dir, entry.file);
  const outName = entry.file.replace(/\.(mp4|mov)$/i, '_edited.mp4');
  const outPath = path.join(dir, outName);

  return ffprobeDuration(base).then((duration) => {
    const filterComplex = buildFilterComplex(edits, duration);
    const args = ['-y', '-i', base, '-filter_complex', filterComplex,
      '-map', '[vout]', '-map', '[aout]',
      '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
      '-c:a', 'aac', '-b:a', '192k', outPath];
    return new Promise((resolve, reject) => {
      const p = spawn('ffmpeg', args);
      let stderr = '';
      p.stderr.on('data', (d) => (stderr += d));
      p.on('close', (code) => {
        if (code !== 0) return reject(new Error(stderr.slice(-2000)));
        resolve(outName);
      });
    });
  });
}

// ---------------------------------------------------------------- routes
app.post('/api/jobs', upload.single('video'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'нет файла видео' });
  const ext = path.extname(req.file.originalname || '.mp4').toLowerCase() || '.mp4';
  const num = nextVideoNum();
  const dir = path.join(VIDEOS_DIR, String(num));
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, `source${ext}`), req.file.buffer);

  const id = crypto.randomUUID();
  const sessionId = crypto.randomUUID();
  const EventEmitter = require('events');
  const job = {
    id, num, ext, sessionId,
    title: (req.body.topic || '').trim() || null,
    status: 'running',
    resultVideo: null,
    variants: { v2: null, v3: null },
    quickEdits: {},
    editedVideo: {},
    error: null,
    createdAt: new Date().toISOString(),
    emitter: new EventEmitter(),
  };
  jobs.set(id, job);
  fs.writeFileSync(logFile(id), '');
  saveJob(job);

  const meta = { topic: req.body.topic || '', numbers: req.body.numbers || '' };
  const prompt = buildMainPrompt(job, meta);
  runClaude(job, prompt, { resume: false });

  res.json({ id, num });
});

/** Title may only appear once the montage plan is written, so it is resolved on read. */
function decorate(job) {
  const { emitter, ...plain } = job;
  return {
    ...plain,
    title: folderTitle(job.num) || job.title || null,
    createdAt: folderCreatedAt(job.num),
    videos: scanFolderVideos(job.num),
    library: getVideoLibraryItem(job.num),
  };
}

app.get('/api/jobs/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'not found' });
  res.json(decorate(job));
});

app.get('/api/jobs', (req, res) => {
  syncLibrary();
  res.json([...jobs.values()].map(decorate).sort((a, b) => b.num - a.num));
});

app.get('/api/library', (req, res) => {
  res.json(listVideoLibrary());
});

app.get('/api/library/:num', (req, res) => {
  const item = getVideoLibraryItem(parseInt(req.params.num, 10));
  if (!item) return res.status(404).json({ error: 'not found' });
  res.json(item);
});

app.patch('/api/library/:num', (req, res) => {
  const num = parseInt(req.params.num, 10);
  const current = getVideoLibraryItem(num);
  if (!current) return res.status(404).json({ error: 'not found' });
  const item = updateVideoMeta(num, {
    posted: req.body.posted,
    note: req.body.note,
    teleprompterText: req.body.teleprompterText,
    title: req.body.title,
  });
  res.json(item);
});

app.get('/api/thumb/:num/:file', async (req, res) => {
  const num = parseInt(req.params.num, 10);
  if (!Number.isInteger(num)) return res.status(400).end();
  const entry = scanFolderVideos(num).find((v) => v.file === req.params.file);
  if (!entry) return res.status(404).end();
  try {
    const p = await thumbFor(num, entry.file, entry.mtime);
    res.sendFile(p);
  } catch {
    res.status(500).end();
  }
});

/** Horizontal filmstrip for the timeline: FILMSTRIP_N evenly spaced frames tiled into one image. */
const FILMSTRIP_N = 40;
app.get('/api/filmstrip/:num/:file', async (req, res) => {
  const num = parseInt(req.params.num, 10);
  if (!Number.isInteger(num)) return res.status(400).end();
  const entry = scanFolderVideos(num).find((v) => v.file === req.params.file);
  if (!entry) return res.status(404).end();

  const key = `strip_${num}_${entry.file.replace(/[^\w.-]/g, '_')}_${Math.round(entry.mtime)}.jpg`;
  const out = path.join(THUMBS_DIR, key);
  if (fs.existsSync(out)) return res.sendFile(out);

  const src = path.join(VIDEOS_DIR, String(num), entry.file);
  try {
    const dur = await ffprobeDuration(src);
    // sample a few extra frames so `tile` always fills its row and emits an image
    const fps = (FILMSTRIP_N + 3) / dur;
    await new Promise((resolve, reject) => {
      const p = spawn('ffmpeg', ['-y', '-i', src,
        '-vf', `fps=${fps.toFixed(5)},scale=48:-2,tile=${FILMSTRIP_N}x1`,
        '-frames:v', '1', '-q:v', '5', out]);
      p.on('close', (c) => (c === 0 && fs.existsSync(out) ? resolve() : reject(new Error('strip failed'))));
      p.on('error', reject);
    });
    res.sendFile(out);
  } catch {
    res.status(500).end();
  }
});

app.get('/api/jobs/:id/plan', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'not found' });
  const key = req.query.key || 'main';
  const m = String(key).match(/^hook(\d+)$/);
  const name = m ? `montage-plan-hook${m[1]}.md` : 'montage-plan.md';
  const p = path.join(VIDEOS_DIR, String(job.num), name);
  if (!fs.existsSync(p)) return res.json({ text: null });
  res.json({ text: fs.readFileSync(p, 'utf8'), file: name });
});

app.get('/api/jobs/:id/stream', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).end();

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });

  const existing = fs.existsSync(logFile(job.id)) ? fs.readFileSync(logFile(job.id), 'utf8') : '';
  res.write(`event: log\ndata: ${JSON.stringify(existing)}\n\n`);
  res.write(`event: status\ndata: ${JSON.stringify({ status: job.status, resultVideo: job.resultVideo, variants: job.variants, error: job.error })}\n\n`);

  const onLog = (text) => res.write(`event: log\ndata: ${JSON.stringify(text)}\n\n`);
  const onDone = () => {
    res.write(`event: status\ndata: ${JSON.stringify({ status: job.status, resultVideo: job.resultVideo, variants: job.variants, error: job.error })}\n\n`);
  };
  job.emitter.on('log', onLog);
  job.emitter.on('done', onDone);

  req.on('close', () => {
    job.emitter.off('log', onLog);
    job.emitter.off('done', onDone);
  });
});

app.post('/api/jobs/:id/feedback', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'not found' });
  if (!['ready', 'variants-error', 'imported'].includes(job.status)) {
    return res.status(400).json({ error: 'ролик ещё не готов' });
  }
  const videos = scanFolderVideos(job.num);
  const main = videos.find((v) => v.key === 'v1') || videos.find((v) => v.key === 'main') || videos[0];
  if (!main) return res.status(400).json({ error: 'в папке ролика нет готового монтажа, с которого делать версии' });

  const hadSession = !!job.sessionId;
  if (!hadSession) job.sessionId = crypto.randomUUID();
  job.status = 'variants-running';
  job.error = null;
  saveJob(job);
  const prompt = buildFeedbackPrompt(job, req.body.comment || '', main.file);
  runClaude(job, prompt, { resume: hadSession });
  res.json({ ok: true });
});

app.get('/api/jobs/:id/meta', async (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'not found' });
  const key = req.query.key || req.query.version || 'v1';
  const entry = scanFolderVideos(job.num).find((v) => v.key === key);
  if (!entry) return res.status(404).json({ error: `${key} ещё не готова` });
  try {
    const duration = await ffprobeDuration(path.join(VIDEOS_DIR, String(job.num), entry.file));
    res.json({ duration, edits: job.quickEdits[key] || null });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.post('/api/jobs/:id/quick-edit', async (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'not found' });
  const { key, version, edits } = req.body;
  const k = key || version;
  if (!k || !edits) return res.status(400).json({ error: 'нужны key и edits' });
  try {
    const outName = await applyQuickEdit(job, k, edits);
    job.quickEdits[k] = edits;
    job.editedVideo[k] = relVideo(job.num, outName);
    saveJob(job);
    res.json({ ok: true, path: job.editedVideo[k] });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post('/api/jobs/:id/quick-edit/reset', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return res.status(404).json({ error: 'not found' });
  const k = req.body.key || req.body.version;
  if (job.editedVideo[k]) {
    try { fs.unlinkSync(path.join(VIDEOS_DIR, String(job.num), path.basename(job.editedVideo[k]))); } catch {}
  }
  delete job.quickEdits[k];
  delete job.editedVideo[k];
  saveJob(job);
  res.json({ ok: true });
});

const PORT = process.env.PORT || 4173;
app.listen(PORT, () => {
  console.log(`Reels Good Studio: http://localhost:${PORT}`);
});

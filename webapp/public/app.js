const $ = (id) => document.getElementById(id);

let currentJob = null;
let currentKey = null;
let selectedFile = null;
let es = null;

// ---------------------------------------------------------------- helpers
function statusLabel(s) {
  return {
    running: 'монтирую',
    ready: 'готов',
    imported: 'в библиотеке',
    'variants-running': 'делаю версии',
    'variants-ready': 'версии готовы',
    error: 'ошибка',
    'variants-error': 'ошибка версий',
  }[s] || s;
}

function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if ([2, 3, 4].includes(m10) && ![12, 13, 14].includes(m100)) return few;
  return many;
}

function fmtDate(iso) {
  return new Date(iso).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' });
}

// ---------------------------------------------------------------- library
async function loadLibrary() {
  const list = await (await fetch('/api/jobs')).json();
  const grid = $('grid');
  $('lib-count').textContent = `${list.length} ${plural(list.length, 'ролик', 'ролика', 'роликов')}`;

  if (!list.length) {
    grid.innerHTML = '<p class="empty-note">Пока пусто — загрузите первый дубль.</p>';
    return;
  }

  grid.innerHTML = '';
  for (const j of list) {
    const n = (j.videos || []).length;
    const busy = j.status === 'running' || j.status === 'variants-running';
    const poster = n ? `<img src="${j.videos[0].thumb}" alt="" loading="lazy">` : '<div class="noimg">◍</div>';

    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.innerHTML = `
      <div class="tile-poster">
        ${poster}
        <span class="tile-num">#${j.num}</span>
        ${busy
          ? `<span class="tile-badge live">${statusLabel(j.status)}</span>`
          : n > 1 ? `<span class="tile-badge">${n} ${plural(n, 'версия', 'версии', 'версий')}</span>` : ''}
      </div>
      <div class="tile-name">${j.title ? escapeHtml(j.title) : `Ролик #${j.num}`}</div>
      <div class="tile-meta">${fmtDate(j.createdAt)}</div>`;
    tile.addEventListener('click', () => openJob(j.id));
    grid.appendChild(tile);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------------------------------------------------------------- job screen
function showLibrary() {
  if (es) { es.close(); es = null; }
  currentJob = null;
  $('job-screen').classList.add('hidden');
  $('library-screen').classList.remove('hidden');
  loadLibrary();
}
$('back-btn').addEventListener('click', showLibrary);
$('brand-home').addEventListener('click', showLibrary);

async function openJob(id) {
  $('library-screen').classList.add('hidden');
  $('job-screen').classList.remove('hidden');
  $('log-inner').textContent = '';
  currentKey = null;

  const job = await (await fetch(`/api/jobs/${id}`)).json();
  render(job);

  if (es) es.close();
  es = new EventSource(`/api/jobs/${id}/stream`);
  es.addEventListener('log', (e) => {
    const box = $('log-inner');
    box.textContent += JSON.parse(e.data);
    box.parentElement.scrollTop = box.parentElement.scrollHeight;
  });
  es.addEventListener('status', async () => {
    render(await (await fetch(`/api/jobs/${id}`)).json());
  });
}

function render(job) {
  currentJob = job;
  $('job-title').textContent = job.title ? `${job.title}` : `Ролик #${job.num}`;
  const pill = $('job-status');
  pill.textContent = statusLabel(job.status);
  pill.className = 'status-pill ' + job.status;

  const videos = job.videos || [];

  // version tabs
  const tabs = $('version-tabs');
  tabs.innerHTML = '';
  if (!videos.length) {
    tabs.innerHTML = '<span class="seg" style="cursor:default">ещё нет готовых версий</span>';
    $('player').removeAttribute('src');
    $('player-meta').textContent = '';
  } else {
    if (!videos.some((v) => v.key === currentKey)) currentKey = videos[0].key;
    for (const v of videos) {
      const b = document.createElement('button');
      b.className = 'seg' + (v.key === currentKey ? ' active' : '');
      b.textContent = v.label.replace(' · эталон', '').replace(' · инстаграм-тест', '');
      b.addEventListener('click', () => { currentKey = v.key; render(currentJob); });
      tabs.appendChild(b);
    }
    selectVideo(videos.find((v) => v.key === currentKey));
  }

  // versions tab availability
  const hasMain = videos.some((v) => v.key === 'v1' || v.key === 'main');
  $('feedback-btn').disabled = !hasMain || job.status === 'running' || job.status === 'variants-running';

  const vs = $('variants-status');
  if (job.status === 'variants-running') {
    vs.classList.remove('hidden');
    vs.className = 'inline-status running';
    vs.textContent = 'Собираю версии 2 и 3 — следите на вкладке «Процесс».';
  } else if (job.error) {
    vs.classList.remove('hidden');
    vs.className = 'inline-status err';
    vs.textContent = job.error;
  } else {
    vs.classList.add('hidden');
  }
}

function selectVideo(v) {
  if (!v) return;
  currentVideo = v;
  const player = $('player');
  const src = v.editedSrc || v.src;
  if (player.getAttribute('data-src') !== src) {
    player.src = src + '?t=' + Date.now();
    player.setAttribute('data-src', src);
  }
  $('player-meta').textContent = v.editedSrc ? `${v.file} · с вашими правками` : v.file;
  buildEditor(v.key);
  loadPlan(v.key);
}

// ---------------------------------------------------------------- tabs
document.querySelectorAll('#panel-tabs .tab').forEach((t) => {
  t.addEventListener('click', () => {
    document.querySelectorAll('#panel-tabs .tab').forEach((x) => x.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));
    t.classList.add('active');
    $('pane-' + t.dataset.tab).classList.add('active');
  });
});

async function loadPlan(key) {
  const el = $('plan-inner');
  el.textContent = 'загружаю…';
  try {
    const d = await (await fetch(`/api/jobs/${currentJob.id}/plan?key=${encodeURIComponent(key)}`)).json();
    el.textContent = d.text || 'Для этой версии монтажный лист не найден.';
  } catch {
    el.textContent = 'Не удалось прочитать монтажный лист.';
  }
}
// ---------------------------------------------------------------- timeline editor
const NEUTRAL = { brightness: 0, contrast: 100, saturation: 100, warmth: 0, volume: 100 };

let ed = null;          // current edit state
let currentVideo = null;

function blankState(key, duration) {
  return {
    key, duration,
    brightness: 0, contrast: 100, saturation: 100, warmth: 0, volume: 100,
    trimStart: 0, trimEnd: duration,
    cuts: [], segments: [],
    sel: null, activeSeg: -1,
    view: 'original',
  };
}

function fmtT(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

async function buildEditor(key) {
  const el = $('pane-editor');
  el.innerHTML = '<p class="empty-note">Читаю дорожку…</p>';

  let meta;
  try {
    meta = await (await fetch(`/api/jobs/${currentJob.id}/meta?key=${encodeURIComponent(key)}`)).json();
    if (meta.error) throw new Error(meta.error);
  } catch {
    el.innerHTML = '<p class="empty-note">Не удалось прочитать дорожку этой версии.</p>';
    return;
  }

  ed = blankState(key, meta.duration);
  if (meta.edits) Object.assign(ed, {
    ...meta.edits,
    cuts: meta.edits.cuts || [],
    segments: (meta.edits.segments || []).map((s) => ({ ...NEUTRAL, ...s })),
    trimEnd: meta.edits.trimEnd ?? meta.duration,
    sel: null, activeSeg: -1, view: currentVideo && currentVideo.editedSrc ? 'edited' : 'original',
    key, duration: meta.duration,
  });

  el.innerHTML = `
    <div class="tl-wrap">
      <p class="tl-hint">Протяните мышкой по дорожке, чтобы выделить кусок — потом вырежьте его или задайте ему свои настройки. Клик — перемотать.</p>
      <div class="tl" id="tl">
        <div class="tl-strip" id="tl-strip"></div>
        <div id="tl-marks"></div>
        <div class="tl-play" id="tl-play" style="left:0"></div>
      </div>
      <div class="tl-ruler"><span>0:00.0</span><span id="tl-mid"></span><span>${fmtT(ed.duration)}</span></div>

      <div class="tl-bar idle" id="tl-bar">
        <span class="tl-sel-label" id="tl-sel-label">Ничего не выделено</span>
        <button class="mini-btn" id="btn-play-sel" disabled>▶ Проиграть</button>
        <button class="mini-btn" id="btn-seg" disabled>Настроить кусок</button>
        <button class="mini-btn" id="btn-keep" disabled>Оставить только это</button>
        <button class="mini-btn danger" id="btn-cut" disabled>Вырезать</button>
        <button class="mini-btn" id="btn-clear" disabled>✕</button>
      </div>
    </div>

    <div class="qe-section" id="cuts-section"></div>
    <div class="qe-section" id="segs-section"></div>

    <div class="qe-section">
      <div class="qe-label">Весь ролик целиком</div>
      <div class="qe-row"><label>Громкость</label><input type="range" min="0" max="200" data-g="volume"><span class="qe-val"></span></div>
      <div class="qe-row"><label>Яркость</label><input type="range" min="-100" max="100" data-g="brightness"><span class="qe-val"></span></div>
      <div class="qe-row"><label>Контраст</label><input type="range" min="50" max="200" data-g="contrast"><span class="qe-val"></span></div>
      <div class="qe-row"><label>Насыщенность</label><input type="range" min="0" max="200" data-g="saturation"><span class="qe-val"></span></div>
      <div class="qe-row"><label>Теплота</label><input type="range" min="-100" max="100" data-g="warmth"><span class="qe-val"></span></div>
    </div>

    <div class="qe-actions">
      <button class="primary-btn" id="btn-apply">Применить</button>
      <button class="qe-reset" id="btn-reset">Сбросить</button>
      <button class="mini-btn hidden" id="btn-view">Смотреть оригинал</button>
      <span class="qe-status" id="qe-status"></span>
    </div>`;

  if (currentVideo) $('tl-strip').style.backgroundImage = `url(${currentVideo.strip})`;
  $('tl-mid').textContent = fmtT(ed.duration / 2);

  el.querySelectorAll('[data-g]').forEach((r) => {
    const k = r.dataset.g;
    r.value = ed[k];
    r.parentElement.querySelector('.qe-val').textContent = ed[k];
    r.addEventListener('input', () => {
      ed[k] = parseFloat(r.value);
      r.parentElement.querySelector('.qe-val').textContent = r.value;
    });
  });

  wireTimeline();
  $('btn-apply').addEventListener('click', applyEdit);
  $('btn-reset').addEventListener('click', resetEdit);
  $('btn-view').addEventListener('click', toggleView);
  redraw();
}

// ---------------------------------------------------------------- timeline interaction
function wireTimeline() {
  const tl = $('tl');
  const xToTime = (clientX) => {
    const r = tl.getBoundingClientRect();
    return Math.max(0, Math.min(ed.duration, ((clientX - r.left) / r.width) * ed.duration));
  };

  let dragging = false, anchor = 0;

  tl.addEventListener('mousedown', (e) => {
    dragging = true;
    anchor = xToTime(e.clientX);
    ed.sel = { start: anchor, end: anchor };
    redraw();
  });

  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const t = xToTime(e.clientX);
    ed.sel = { start: Math.min(anchor, t), end: Math.max(anchor, t) };
    redraw();
  });

  window.addEventListener('mouseup', (e) => {
    if (!dragging) return;
    dragging = false;
    // resolve from the release point too — a fast drag can outrun mousemove
    const t = xToTime(e.clientX);
    ed.sel = { start: Math.min(anchor, t), end: Math.max(anchor, t) };
    // a click rather than a drag: seek instead of selecting
    if (ed.sel.end - ed.sel.start < 0.25) {
      const p = $('player');
      if (ed.view === 'original') p.currentTime = ed.sel.start;
      ed.sel = null;
    }
    redraw();
  });

  $('player').addEventListener('timeupdate', () => {
    if (!ed || ed.view !== 'original') return;
    $('tl-play').style.left = `${($('player').currentTime / ed.duration) * 100}%`;
  });

  $('btn-clear').addEventListener('click', () => { ed.sel = null; redraw(); });
  $('btn-cut').addEventListener('click', () => {
    if (!ed.sel) return;
    ed.cuts.push({ ...ed.sel });
    ed.sel = null;
    redraw();
  });
  $('btn-keep').addEventListener('click', () => {
    if (!ed.sel) return;
    ed.trimStart = ed.sel.start;
    ed.trimEnd = ed.sel.end;
    ed.sel = null;
    redraw();
  });
  $('btn-seg').addEventListener('click', () => {
    if (!ed.sel) return;
    ed.segments.push({ ...NEUTRAL, ...ed.sel });
    ed.activeSeg = ed.segments.length - 1;
    ed.sel = null;
    redraw();
    document.querySelector('.seg-card.active')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  });
  $('btn-play-sel').addEventListener('click', playSelection);
}

function playSelection() {
  if (!ed.sel || ed.view !== 'original') return;
  const p = $('player');
  const { start, end } = ed.sel;
  p.currentTime = start;
  p.play();
  const stop = () => {
    if (p.currentTime >= end) { p.pause(); p.removeEventListener('timeupdate', stop); }
  };
  p.addEventListener('timeupdate', stop);
}

// ---------------------------------------------------------------- render
const pct = (t) => `${(t / ed.duration) * 100}%`;

function redraw() {
  const marks = $('tl-marks');
  marks.innerHTML = '';

  if (ed.trimStart > 0.01) {
    marks.insertAdjacentHTML('beforeend', `<div class="tl-outside" style="left:0;width:${pct(ed.trimStart)}"></div>`);
  }
  if (ed.trimEnd < ed.duration - 0.01) {
    marks.insertAdjacentHTML('beforeend', `<div class="tl-outside" style="left:${pct(ed.trimEnd)};right:0"></div>`);
  }
  ed.cuts.forEach((c) => {
    marks.insertAdjacentHTML('beforeend', `<div class="tl-cut" style="left:${pct(c.start)};width:${pct(c.end - c.start)}"></div>`);
  });
  ed.segments.forEach((s, i) => {
    marks.insertAdjacentHTML('beforeend',
      `<div class="tl-seg${i === ed.activeSeg ? ' on' : ''}" style="left:${pct(s.start)};width:${pct(s.end - s.start)}">
         <span class="tl-seg-tag">${segTag(s)}</span></div>`);
  });
  if (ed.sel) {
    marks.insertAdjacentHTML('beforeend',
      `<div class="tl-sel" style="left:${pct(ed.sel.start)};width:${pct(ed.sel.end - ed.sel.start)}"></div>`);
  }

  const has = !!ed.sel && ed.sel.end - ed.sel.start > 0.05;
  ['btn-cut', 'btn-seg', 'btn-keep', 'btn-clear', 'btn-play-sel'].forEach((id) => { $(id).disabled = !has; });
  $('tl-bar').classList.toggle('idle', !has);
  $('tl-sel-label').innerHTML = has
    ? `Выделено <b>${fmtT(ed.sel.start)} — ${fmtT(ed.sel.end)}</b> · ${(ed.sel.end - ed.sel.start).toFixed(1)}с`
    : 'Ничего не выделено';

  renderCuts();
  renderSegments();

  const btnView = $('btn-view');
  if (currentVideo && currentVideo.editedSrc) {
    btnView.classList.remove('hidden');
    btnView.textContent = ed.view === 'original' ? 'Смотреть результат' : 'Смотреть оригинал';
  } else {
    btnView.classList.add('hidden');
  }
}

function segTag(s) {
  const bits = [];
  if (s.volume !== 100) bits.push(`гр ${s.volume}%`);
  if (s.brightness !== 0) bits.push(`ярк ${s.brightness > 0 ? '+' : ''}${s.brightness}`);
  if (s.contrast !== 100) bits.push(`конт ${s.contrast}`);
  if (s.saturation !== 100) bits.push(`нас ${s.saturation}`);
  if (s.warmth !== 0) bits.push(`тепл ${s.warmth > 0 ? '+' : ''}${s.warmth}`);
  return bits.join(' · ');
}

function renderCuts() {
  const box = $('cuts-section');
  if (!ed.cuts.length && ed.trimStart <= 0.01 && ed.trimEnd >= ed.duration - 0.01) {
    box.innerHTML = '';
    return;
  }
  let html = '<div class="qe-label">Вырезано</div>';
  if (ed.trimStart > 0.01 || ed.trimEnd < ed.duration - 0.01) {
    html += `<span class="cut-chip">оставлено ${fmtT(ed.trimStart)} — ${fmtT(ed.trimEnd)}
      <button data-trim="1" title="вернуть полную длину">✕</button></span>`;
  }
  ed.cuts.forEach((c, i) => {
    html += `<span class="cut-chip">${fmtT(c.start)} — ${fmtT(c.end)}
      <button data-cut="${i}" title="вернуть кусок">✕</button></span>`;
  });
  box.innerHTML = html;

  box.querySelectorAll('[data-cut]').forEach((b) => b.addEventListener('click', () => {
    ed.cuts.splice(parseInt(b.dataset.cut, 10), 1);
    redraw();
  }));
  const t = box.querySelector('[data-trim]');
  if (t) t.addEventListener('click', () => { ed.trimStart = 0; ed.trimEnd = ed.duration; redraw(); });
}

function renderSegments() {
  const box = $('segs-section');
  if (!ed.segments.length) { box.innerHTML = ''; return; }

  box.innerHTML = '<div class="qe-label">Настройки отдельных кусков</div>' + ed.segments.map((s, i) => `
    <div class="seg-card${i === ed.activeSeg ? ' active' : ''}" data-i="${i}">
      <div class="seg-card-head">
        <span>${fmtT(s.start)} — ${fmtT(s.end)}</span>
        <span class="grow">${(s.end - s.start).toFixed(1)}с</span>
        <button data-del="${i}" title="убрать">✕</button>
      </div>
      <div class="qe-row"><label>Громкость</label><input type="range" min="0" max="200" value="${s.volume}" data-s="${i}" data-k="volume"><span class="qe-val">${s.volume}</span></div>
      <div class="qe-row"><label>Яркость</label><input type="range" min="-100" max="100" value="${s.brightness}" data-s="${i}" data-k="brightness"><span class="qe-val">${s.brightness}</span></div>
      <div class="qe-row"><label>Контраст</label><input type="range" min="50" max="200" value="${s.contrast}" data-s="${i}" data-k="contrast"><span class="qe-val">${s.contrast}</span></div>
      <div class="qe-row"><label>Насыщенность</label><input type="range" min="0" max="200" value="${s.saturation}" data-s="${i}" data-k="saturation"><span class="qe-val">${s.saturation}</span></div>
      <div class="qe-row"><label>Теплота</label><input type="range" min="-100" max="100" value="${s.warmth}" data-s="${i}" data-k="warmth"><span class="qe-val">${s.warmth}</span></div>
    </div>`).join('');

  box.querySelectorAll('[data-del]').forEach((b) => b.addEventListener('click', () => {
    ed.segments.splice(parseInt(b.dataset.del, 10), 1);
    ed.activeSeg = -1;
    redraw();
  }));
  box.querySelectorAll('input[data-s]').forEach((r) => {
    r.addEventListener('input', () => {
      const i = parseInt(r.dataset.s, 10);
      ed.segments[i][r.dataset.k] = parseFloat(r.value);
      r.parentElement.querySelector('.qe-val').textContent = r.value;
      ed.activeSeg = i;
      // repaint only the timeline tag, so the slider keeps focus while dragging
      const seg = document.querySelectorAll('.tl-seg')[i];
      if (seg) seg.querySelector('.tl-seg-tag').textContent = segTag(ed.segments[i]);
    });
  });
  box.querySelectorAll('.seg-card').forEach((c) => c.addEventListener('click', () => {
    ed.activeSeg = parseInt(c.dataset.i, 10);
    redraw();
  }));
}

// ---------------------------------------------------------------- apply / reset / compare
function toggleView() {
  ed.view = ed.view === 'original' ? 'edited' : 'original';
  const p = $('player');
  const src = ed.view === 'edited' ? currentVideo.editedSrc : currentVideo.src;
  p.src = src + '?t=' + Date.now();
  p.setAttribute('data-src', src);
  $('player-meta').textContent = ed.view === 'edited'
    ? `${currentVideo.file} · с вашими правками`
    : `${currentVideo.file} · оригинал`;
  $('tl-play').style.left = '0';
  redraw();
}

function payload() {
  return {
    brightness: ed.brightness, contrast: ed.contrast, saturation: ed.saturation,
    warmth: ed.warmth, volume: ed.volume,
    trimStart: ed.trimStart, trimEnd: ed.trimEnd,
    cuts: ed.cuts, segments: ed.segments,
  };
}

async function applyEdit() {
  const btn = $('btn-apply'), st = $('qe-status');
  btn.disabled = true;
  st.className = 'qe-status';
  st.textContent = 'считаю…';
  try {
    const res = await fetch(`/api/jobs/${currentJob.id}/quick-edit`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ key: ed.key, edits: payload() }),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'ошибка');
    st.className = 'qe-status ok';
    st.textContent = 'готово';
    currentVideo.editedSrc = d.path;
    ed.view = 'edited';
    const p = $('player');
    p.src = d.path + '?t=' + Date.now();
    p.setAttribute('data-src', d.path);
    $('player-meta').textContent = `${currentVideo.file} · с вашими правками`;
    redraw();
  } catch (err) {
    st.className = 'qe-status err';
    st.textContent = String(err.message).slice(0, 140);
  } finally {
    btn.disabled = false;
  }
}

async function resetEdit() {
  await fetch(`/api/jobs/${currentJob.id}/quick-edit/reset`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key: ed.key }),
  });
  currentVideo.editedSrc = null;
  const p = $('player');
  p.src = currentVideo.src + '?t=' + Date.now();
  p.setAttribute('data-src', currentVideo.src);
  $('player-meta').textContent = currentVideo.file;
  buildEditor(ed.key);
}

// ---------------------------------------------------------------- feedback
$('feedback-btn').addEventListener('click', async () => {
  const res = await fetch(`/api/jobs/${currentJob.id}/feedback`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comment: $('feedback-text').value }),
  });
  if (!res.ok) {
    const d = await res.json().catch(() => ({}));
    alert('Не удалось запустить: ' + (d.error || res.status));
    return;
  }
  const vs = $('variants-status');
  vs.classList.remove('hidden');
  vs.className = 'inline-status running';
  vs.textContent = 'Собираю версии 2 и 3 — следите на вкладке «Процесс».';
});

// ---------------------------------------------------------------- upload modal
const modal = $('modal');
$('new-btn').addEventListener('click', () => modal.classList.remove('hidden'));
$('modal-close').addEventListener('click', () => modal.classList.add('hidden'));
modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });

const dz = $('dropzone');
dz.addEventListener('click', () => $('file-input').click());
dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', (e) => {
  e.preventDefault(); dz.classList.remove('drag');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
$('file-input').addEventListener('change', (e) => { if (e.target.files[0]) setFile(e.target.files[0]); });

function setFile(f) {
  selectedFile = f;
  $('dz-file').textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} МБ`;
  $('start-btn').disabled = false;
}

$('start-btn').addEventListener('click', async () => {
  if (!selectedFile) return;
  const btn = $('start-btn');
  btn.disabled = true;
  btn.textContent = 'Загружаю…';

  const fd = new FormData();
  fd.append('video', selectedFile);
  fd.append('topic', $('topic').value);
  fd.append('numbers', $('numbers').value);

  try {
    const res = await fetch('/api/jobs', { method: 'POST', body: fd });
    const d = await res.json();
    if (!res.ok) throw new Error(d.error || 'ошибка');
    modal.classList.add('hidden');
    btn.textContent = 'Смонтировать';
    selectedFile = null;
    $('dz-file').textContent = '';
    openJob(d.id);
  } catch (err) {
    alert('Не удалось запустить монтаж: ' + err.message);
    btn.disabled = false;
    btn.textContent = 'Смонтировать';
  }
});

loadLibrary();

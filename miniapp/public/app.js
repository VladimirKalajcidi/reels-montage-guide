const state = {
  items: [],
  filter: 'all',
  query: '',
  selectedId: null,
  selected: null,
};

const tg = window.Telegram?.WebApp;
if (tg) {
  tg.ready();
  tg.expand();
}

const $ = (selector) => document.querySelector(selector);

function formatDate(iso) {
  return new Date(iso).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' });
}

function titleWithEmoji(item) {
  return [item.emoji.edited, item.emoji.posted, `${item.num}. ${item.title}`].filter(Boolean).join(' ');
}

function statusText(item) {
  return [
    item.statuses.edited ? 'смонтирован' : 'не смонтирован',
    item.statuses.posted ? 'выложен' : 'не выложен',
    item.statuses.hasText ? 'есть текст' : 'нет текста',
  ].join(' • ');
}

async function api(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function loadList() {
  const params = new URLSearchParams();
  if (state.filter !== 'all') params.set('filter', state.filter);
  if (state.query) params.set('q', state.query);
  const data = await api(`/api/videos?${params.toString()}`);
  state.items = data.items;
  if (!state.selectedId && state.items[0]) state.selectedId = state.items[0].num;
  if (state.selectedId && !state.items.some((item) => item.num === state.selectedId)) {
    state.selectedId = state.items[0]?.num || null;
  }
  renderList();
  renderStats();
  if (state.selectedId) await loadDetail(state.selectedId);
  else renderEmpty();
}

async function loadDetail(num) {
  state.selected = await api(`/api/videos/${num}`);
  state.selectedId = state.selected.num;
  renderDetail();
  renderList();
}

function renderStats() {
  const total = state.items.length;
  const edited = state.items.filter((item) => item.statuses.edited).length;
  const posted = state.items.filter((item) => item.statuses.posted).length;
  $('#stats').textContent = `Найдено: ${total}\n📹 ${edited}  ✅ ${posted}`;
  $('#result-count').textContent = `${total} шт.`;
}

function badge(label, cls) {
  return `<span class="badge ${cls}">${label}</span>`;
}

function renderList() {
  const root = $('#video-list');
  root.innerHTML = '';
  const tpl = $('#video-card-template');

  for (const item of state.items) {
    const node = tpl.content.firstElementChild.cloneNode(true);
    if (item.num === state.selectedId) node.classList.add('active');
    node.querySelector('.video-card-title').textContent = titleWithEmoji(item);
    node.querySelector('.video-card-meta').textContent = formatDate(item.createdAt);
    node.querySelector('.video-card-status').textContent = statusText(item);

    const badges = node.querySelector('.badges');
    if (item.statuses.edited) badges.insertAdjacentHTML('beforeend', badge('📹 монтаж', 'edit'));
    if (item.statuses.posted) badges.insertAdjacentHTML('beforeend', badge('✅ выложен', 'ok'));

    node.addEventListener('click', () => loadDetail(item.num));
    root.appendChild(node);
  }
}

function renderEmpty() {
  $('#detail-empty').classList.remove('hidden');
  $('#detail').classList.add('hidden');
}

function renderDetail() {
  const item = state.selected;
  if (!item) return renderEmpty();

  $('#detail-empty').classList.add('hidden');
  const root = $('#detail');
  root.classList.remove('hidden');

  root.innerHTML = `
    <div class="detail-head">
      <div>
        <h2 class="detail-title">${titleWithEmoji(item)}</h2>
        <p class="detail-sub">Папка videos/${item.num} • обновлено ${item.updatedAt ? formatDate(item.updatedAt) : 'авто из файлов'}</p>
      </div>
      <div class="status-row">
        ${item.statuses.edited ? badge('📹 смонтирован', 'edit') : ''}
        ${item.statuses.posted ? badge('✅ выложен', 'ok') : badge('не выложен', '')}
      </div>
    </div>

    <div class="detail-grid">
      <div class="stack">
        <section class="card">
          <h3>Текст для суфлёра</h3>
          <textarea id="teleprompter-input" placeholder="Текст ролика">${escapeHtml(item.teleprompterText || '')}</textarea>
          <div class="actions">
            <button class="btn" id="save-text-btn">Сохранить текст</button>
          </div>
        </section>

        <section class="card">
          <h3>Заметка</h3>
          <textarea id="note-input" placeholder="Любая заметка по ролику">${escapeHtml(item.note || '')}</textarea>
          <div class="actions">
            <button class="btn" id="save-note-btn">Сохранить заметку</button>
          </div>
        </section>
      </div>

      <div class="stack">
        <section class="card">
          <h3>Статусы</h3>
          <p class="detail-sub">${statusText(item)}</p>
          <div class="actions">
            <button class="btn ${item.statuses.posted ? 'warn' : ''}" id="toggle-posted-btn">
              ${item.statuses.posted ? 'Снять отметку "выложен"' : 'Пометить как выложенный'}
            </button>
          </div>
          <div class="save-note">📹 появляется автоматически, если в папке есть файл `*_edit.mp4`.</div>
        </section>

        <section class="card">
          <h3>Файлы</h3>
          <div class="files">
            ${renderFiles(item)}
          </div>
        </section>
      </div>
    </div>
  `;

  $('#save-text-btn').addEventListener('click', () => saveFields({ teleprompterText: $('#teleprompter-input').value }));
  $('#save-note-btn').addEventListener('click', () => saveFields({ note: $('#note-input').value }));
  $('#toggle-posted-btn').addEventListener('click', () => saveFields({ posted: !item.statuses.posted }));
}

function renderFiles(item) {
  const links = [];
  if (item.mainVideo) {
    links.push(fileLink('Основной монтаж', item.mainVideo.file, item.mainVideo.url));
  }
  for (const hook of item.hooks) {
    links.push(fileLink('Хук', hook.file, hook.url));
  }
  if (!links.length) return '<div class="muted">Готовых файлов пока нет.</div>';
  return links.join('');
}

function fileLink(label, file, url) {
  return `
    <a class="file-link" href="${url}" target="_blank" rel="noopener noreferrer">
      <span>${label}<br><small>${file}</small></span>
      <strong>Открыть</strong>
    </a>
  `;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

async function saveFields(patch) {
  if (!state.selectedId) return;
  const item = await api(`/api/videos/${state.selectedId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  state.selected = item;
  await loadList();
  if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
}

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', async () => {
    document.querySelectorAll('.chip').forEach((node) => node.classList.remove('active'));
    chip.classList.add('active');
    state.filter = chip.dataset.filter;
    await loadList();
  });
});

$('#search').addEventListener('input', async (event) => {
  state.query = event.target.value.trim();
  await loadList();
});

loadList().catch((error) => {
  $('#video-list').innerHTML = `<div class="muted">Ошибка загрузки: ${error.message}</div>`;
});

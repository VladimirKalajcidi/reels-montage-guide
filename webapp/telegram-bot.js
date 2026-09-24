const fs = require('fs');
const path = require('path');
const https = require('https');
const { getVideoLibraryItem, listVideoLibrary, VIDEOS_DIR, updateVideoMeta } = require('./video-library');

const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const ALLOWED_CHAT_ID = process.env.TELEGRAM_ALLOWED_CHAT_ID || null;
const PAGE_SIZE = 8;

if (!BOT_TOKEN) {
  console.error('Set TELEGRAM_BOT_TOKEN before starting the bot.');
  process.exit(1);
}

const apiBase = `https://api.telegram.org/bot${BOT_TOKEN}`;
const pendingInput = new Map();
const recentStarts = new Map();
const START_DEDUP_MS = 2500;

function requestJson(method, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const req = https.request(`${apiBase}/${method}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (!parsed.ok) {
            const error = new Error(parsed.description || `${method} failed`);
            error.code = parsed.error_code || null;
            error.description = parsed.description || '';
            return reject(error);
          }
          resolve(parsed.result);
        } catch (error) {
          reject(error);
        }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

function requestMultipart(method, fields, fileField, filePath, filename) {
  return new Promise((resolve, reject) => {
    const boundary = `----reelsgood${Date.now().toString(16)}`;
    const req = https.request(`${apiBase}/${method}`, {
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
      },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (!parsed.ok) return reject(new Error(parsed.description || `${method} failed`));
          resolve(parsed.result);
        } catch (error) {
          reject(error);
        }
      });
    });

    req.on('error', reject);

    for (const [key, value] of Object.entries(fields)) {
      req.write(`--${boundary}\r\n`);
      req.write(`Content-Disposition: form-data; name="${key}"\r\n\r\n`);
      req.write(String(value));
      req.write('\r\n');
    }

    req.write(`--${boundary}\r\n`);
    req.write(`Content-Disposition: form-data; name="${fileField}"; filename="${filename}"\r\n`);
    req.write('Content-Type: application/octet-stream\r\n\r\n');

    const stream = fs.createReadStream(filePath);
    stream.on('error', reject);
    stream.on('end', () => {
      req.end(`\r\n--${boundary}--\r\n`);
    });
    stream.pipe(req, { end: false });
  });
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function filterItems(filter) {
  const items = listVideoLibrary();
  switch (filter) {
    case 'edited':
      return items.filter((item) => item.statuses.edited);
    case 'not_edited':
      return items.filter((item) => !item.statuses.edited);
    case 'posted':
      return items.filter((item) => item.statuses.posted);
    case 'not_posted':
      return items.filter((item) => !item.statuses.posted);
    default:
      return items;
  }
}

function filterLabel(filter) {
  switch (filter) {
    case 'edited': return 'Смонтированные';
    case 'not_edited': return 'Не смонтированные';
    case 'posted': return 'Выложенные';
    case 'not_posted': return 'Не выложенные';
    default: return 'Все ролики';
  }
}

function statusIcons(item) {
  const icons = [];
  if (item.statuses.edited) icons.push('📹');
  if (item.statuses.posted) icons.push('✅');
  return icons.join(' ');
}

function titleWithIcons(item) {
  const icons = statusIcons(item);
  return icons ? `${icons} ${item.num}. ${item.title}` : `${item.num}. ${item.title}`;
}

function mainMenuKeyboard() {
  return {
    inline_keyboard: [
      [{ text: 'Все ролики', callback_data: 'list|all|0' }],
      [{ text: 'Смонтированные', callback_data: 'list|edited|0' }, { text: 'Не смонтированные', callback_data: 'list|not_edited|0' }],
      [{ text: 'Выложенные', callback_data: 'list|posted|0' }, { text: 'Не выложенные', callback_data: 'list|not_posted|0' }],
    ],
  };
}

function buildListKeyboard(filter, page) {
  const items = filterItems(filter);
  const start = page * PAGE_SIZE;
  const slice = items.slice(start, start + PAGE_SIZE);
  const rows = slice.map((item) => ([
    {
      text: titleWithIcons(item).slice(0, 48),
      callback_data: `video|${item.num}|${filter}|${page}`,
    },
  ]));

  const nav = [];
  if (page > 0) nav.push({ text: 'Назад', callback_data: `list|${filter}|${page - 1}` });
  if (start + PAGE_SIZE < items.length) nav.push({ text: 'Дальше', callback_data: `list|${filter}|${page + 1}` });
  if (nav.length) rows.push(nav);
  rows.push([{ text: 'Главное меню', callback_data: 'menu' }]);
  return { inline_keyboard: rows };
}

function summarizeLibrary() {
  const items = listVideoLibrary();
  const edited = items.filter((item) => item.statuses.edited).length;
  const posted = items.filter((item) => item.statuses.posted).length;
  const withText = items.filter((item) => item.statuses.hasTeleprompterText).length;
  return `Всего: ${items.length}\nСмонтированы: ${edited}\nВыложены: ${posted}\nС текстом: ${withText}`;
}

function listText(filter, page) {
  const items = filterItems(filter);
  if (!items.length) return 'Ролики не найдены.';
  return filterLabel(filter);
}

function videoCaption(item, detailMode = 'summary') {
  if (detailMode === 'text') {
    const text = item.teleprompterText?.trim()
      ? item.teleprompterText
      : 'Текст суфлёра пока не задан.';
    return `<b>${escapeHtml(titleWithIcons(item))}</b>\n\n${escapeHtml(text)}`;
  }
  return `<b>${escapeHtml(titleWithIcons(item))}</b>`;
}

function videoKeyboard(item, filter, page, detailMode = 'summary') {
  const rows = [];
  const downloadRow = [];
  if (item.mainVideo) downloadRow.push({ text: '⬇️ 1', callback_data: `send|${item.num}|main|0` });
  if (item.hookVideos[0]) downloadRow.push({ text: '⬇️ 2', callback_data: `send|${item.num}|hook|0` });
  if (item.hookVideos[1]) downloadRow.push({ text: '⬇️ 3', callback_data: `send|${item.num}|hook|1` });
  if (downloadRow.length) rows.push(downloadRow);
  rows.push([
    { text: detailMode === 'text' ? 'Скрыть текст' : 'Текст', callback_data: `showtext|${item.num}|${filter}|${page}|${detailMode}` },
    { text: item.statuses.posted ? '❌' : '✅', callback_data: `toggle|${item.num}|${filter}|${page}` },
  ]);
  rows.push([{ text: 'К списку', callback_data: `list|${filter}|${page}` }]);
  return { inline_keyboard: rows };
}

async function sendMessage(chatId, text, replyMarkup) {
  return requestJson('sendMessage', {
    chat_id: chatId,
    text,
    parse_mode: 'HTML',
    reply_markup: replyMarkup,
  });
}

async function sendMenuMessage(chatId, text) {
  return requestJson('sendMessage', {
    chat_id: chatId,
    text,
    parse_mode: 'HTML',
    reply_markup: mainMenuKeyboard(),
  });
}

async function editMessage(chatId, messageId, text, replyMarkup) {
  try {
    return await requestJson('editMessageText', {
      chat_id: chatId,
      message_id: messageId,
      text,
      parse_mode: 'HTML',
      reply_markup: replyMarkup,
    });
  } catch (error) {
    if ((error.description || '').includes('message is not modified')) {
      return null;
    }
    throw error;
  }
}

async function answerCallbackQuery(callbackQueryId, text) {
  return requestJson('answerCallbackQuery', {
    callback_query_id: callbackQueryId,
    text,
  });
}

async function sendVideoFile(chatId, item, kind, index) {
  let target = null;
  if (kind === 'main') target = item.mainVideo;
  if (kind === 'hook') target = item.hookVideos[index];
  if (!target) {
    await sendMessage(chatId, 'Файл не найден.', mainMenuKeyboard());
    return;
  }

  const filePath = path.join(VIDEOS_DIR, String(item.num), target.file);
  const fileSize = fs.statSync(filePath).size;
  if (fileSize > 49 * 1024 * 1024) {
    await sendMessage(chatId, `Файл ${target.file} больше лимита Bot API (примерно 50 МБ). Путь: ${filePath}`, mainMenuKeyboard());
    return;
  }

  await requestMultipart('sendDocument', {
    chat_id: chatId,
    caption: `${item.num}. ${item.title}`,
    reply_markup: JSON.stringify({
      inline_keyboard: [[{ text: 'Скрыть', callback_data: 'hidefile' }]],
    }),
  }, 'document', filePath, target.file);
}

async function handleQuickMenuText(chatId, text) {
  const normalized = text.trim().toLowerCase();
  const map = {
    'все ролики': 'all',
    'смонтированные': 'edited',
    'не смонтированные': 'not_edited',
    'выложенные': 'posted',
    'не выложенные': 'not_posted',
  };
  const filter = map[normalized];
  if (!filter) return false;
  await requestJson('sendMessage', {
    chat_id: chatId,
    text: listText(filter, 0),
    parse_mode: 'HTML',
    reply_markup: buildListKeyboard(filter, 0),
  });
  return true;
}

function ensureAllowedChat(chatId) {
  return !ALLOWED_CHAT_ID || String(chatId) === String(ALLOWED_CHAT_ID);
}

async function handleStart(chatId) {
  const text = `<b>Библиотека роликов</b>\n\n${escapeHtml(summarizeLibrary())}`;
  await sendMenuMessage(chatId, text);
}

async function handleTextMessage(message) {
  const chatId = message.chat.id;
  if (!ensureAllowedChat(chatId)) return;

  const text = (message.text || '').trim();
  if (text === '/start' || text === '/menu') {
    const now = Date.now();
    const last = recentStarts.get(chatId) || 0;
    if (now - last < START_DEDUP_MS) return;
    recentStarts.set(chatId, now);
    pendingInput.delete(chatId);
    await handleStart(chatId);
    return;
  }

  if (await handleQuickMenuText(chatId, text)) {
    return;
  }

  const pending = pendingInput.get(chatId);
  if (!pending) {
    await sendMessage(chatId, 'Используй /start или кнопки меню.', mainMenuKeyboard());
    return;
  }

  const item = getVideoLibraryItem(pending.num);
  if (!item) {
    pendingInput.delete(chatId);
    await sendMessage(chatId, 'Ролик не найден.', mainMenuKeyboard());
    return;
  }

  if (pending.kind === 'teleprompterText') {
    updateVideoMeta(item.num, { teleprompterText: text });
    pendingInput.delete(chatId);
    await sendMessage(chatId, 'Текст для суфлёра сохранён.', mainMenuKeyboard());
    await sendMenuMessage(chatId, `<b>Меню</b>\nИзменения сохранены.`);
    return;
  }

}

async function handleCallback(callbackQuery) {
  const chatId = callbackQuery.message.chat.id;
  if (!ensureAllowedChat(chatId)) {
    await answerCallbackQuery(callbackQuery.id, 'Этот бот ограничен для одного чата.');
    return;
  }

  const messageId = callbackQuery.message.message_id;
  const parts = String(callbackQuery.data || '').split('|');
  const action = parts[0];

  try {
    if (action === 'menu') {
      pendingInput.delete(chatId);
      await editMessage(chatId, messageId, `<b>Библиотека роликов</b>\n\n${escapeHtml(summarizeLibrary())}`, mainMenuKeyboard());
      await answerCallbackQuery(callbackQuery.id, 'Открыто меню');
      return;
    }

    if (action === 'list') {
      const filter = parts[1] || 'all';
      const page = Number(parts[2] || '0');
      await editMessage(chatId, messageId, listText(filter, page), buildListKeyboard(filter, page));
      await answerCallbackQuery(callbackQuery.id, filterLabel(filter));
      return;
    }

    if (action === 'video') {
      const num = Number(parts[1]);
      const filter = parts[2] || 'all';
      const page = Number(parts[3] || '0');
      const item = getVideoLibraryItem(num);
      if (!item) {
        await answerCallbackQuery(callbackQuery.id, 'Ролик не найден');
        return;
      }
      await editMessage(chatId, messageId, videoCaption(item), videoKeyboard(item, filter, page));
      await answerCallbackQuery(callbackQuery.id, `Ролик ${num}`);
      return;
    }

    if (action === 'toggle') {
      const num = Number(parts[1]);
      const filter = parts[2] || 'all';
      const page = Number(parts[3] || '0');
      const item = getVideoLibraryItem(num);
      if (!item) {
        await answerCallbackQuery(callbackQuery.id, 'Ролик не найден');
        return;
      }
      const updated = updateVideoMeta(num, { posted: !item.statuses.posted });
      await editMessage(chatId, messageId, videoCaption(updated), videoKeyboard(updated, filter, page));
      await answerCallbackQuery(callbackQuery.id, updated.statuses.posted ? 'Помечено как выложенное' : 'Снято с выложенных');
      return;
    }

    if (action === 'showtext') {
      const num = Number(parts[1]);
      const filter = parts[2] || 'all';
      const page = Number(parts[3] || '0');
      const currentMode = parts[4] || 'summary';
      const item = getVideoLibraryItem(num);
      if (!item) {
        await answerCallbackQuery(callbackQuery.id, 'Ролик не найден');
        return;
      }
      const nextMode = currentMode === 'text' ? 'summary' : 'text';
      await answerCallbackQuery(callbackQuery.id, nextMode === 'text' ? 'Открываю текст' : 'Скрываю текст');
      await editMessage(chatId, messageId, videoCaption(item, nextMode), videoKeyboard(item, filter, page, nextMode));
      return;
    }

    if (action === 'hidefile') {
      await answerCallbackQuery(callbackQuery.id, 'Скрываю');
      await requestJson('deleteMessage', {
        chat_id: chatId,
        message_id: messageId,
      });
      return;
    }

    if (action === 'send') {
      const num = Number(parts[1]);
      const kind = parts[2];
      const index = Number(parts[3] || '0');
      const item = getVideoLibraryItem(num);
      if (!item) {
        await answerCallbackQuery(callbackQuery.id, 'Ролик не найден');
        return;
      }
      await answerCallbackQuery(callbackQuery.id, 'Отправляю файл');
      await sendVideoFile(chatId, item, kind, index);
      return;
    }
  } catch (error) {
    console.error(error);
    try {
      await answerCallbackQuery(callbackQuery.id, 'Ошибка, смотри лог сервера');
    } catch {}
  }
}

async function processUpdate(update) {
  if (update.message && update.message.text) {
    await handleTextMessage(update.message);
  }
  if (update.callback_query) {
    await handleCallback(update.callback_query);
  }
}

async function poll() {
  let offset = 0;
  console.log('Telegram bot polling started');
  while (true) {
    try {
      const updates = await requestJson('getUpdates', { offset, timeout: 25, allowed_updates: ['message', 'callback_query'] });
      for (const update of updates) {
        offset = update.update_id + 1;
        await processUpdate(update);
      }
    } catch (error) {
      console.error('Polling error:', error.message || error);
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }
}

async function bootstrapBot() {
  try {
    await requestJson('setMyCommands', {
      commands: [
        {
          command: 'start',
          description: 'Открыть библиотеку роликов',
        },
      ],
    });
  } catch (error) {
    console.error('setMyCommands failed:', error.message || error);
  }
}

bootstrapBot().then(poll);

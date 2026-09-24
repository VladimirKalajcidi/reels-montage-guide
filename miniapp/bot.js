const https = require('https');

const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const MINIAPP_URL = process.env.TELEGRAM_MINIAPP_URL;
const ALLOWED_CHAT_ID = process.env.TELEGRAM_ALLOWED_CHAT_ID || null;

if (!BOT_TOKEN) {
  console.error('Set TELEGRAM_BOT_TOKEN');
  process.exit(1);
}

if (!MINIAPP_URL) {
  console.error('Set TELEGRAM_MINIAPP_URL');
  process.exit(1);
}

const API = `https://api.telegram.org/bot${BOT_TOKEN}`;

function call(method, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const req = https.request(`${API}/${method}`, {
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
          if (!parsed.ok) return reject(new Error(parsed.description || `${method} failed`));
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

function isAllowed(chatId) {
  return !ALLOWED_CHAT_ID || String(chatId) === String(ALLOWED_CHAT_ID);
}

function appKeyboard() {
  return {
    inline_keyboard: [
      [{ text: 'Открыть библиотеку', web_app: { url: MINIAPP_URL } }],
    ],
  };
}

function replyKeyboard() {
  return {
    keyboard: [
      [{ text: 'Открыть библиотеку' }],
      [{ text: '/start' }],
    ],
    resize_keyboard: true,
    persistent: true,
  };
}

async function sendStart(chatId) {
  await call('sendMessage', {
    chat_id: chatId,
    text: 'Открывай библиотеку роликов кнопкой ниже.',
    reply_markup: replyKeyboard(),
  });

  await call('sendMessage', {
    chat_id: chatId,
    text: 'Мини-апп внутри Telegram:',
    reply_markup: appKeyboard(),
  });
}

async function handleMessage(message) {
  const chatId = message.chat.id;
  if (!isAllowed(chatId)) return;

  const text = (message.text || '').trim().toLowerCase();
  if (!text) return;

  if (text === '/start' || text === 'открыть библиотеку') {
    await sendStart(chatId);
    return;
  }

  await call('sendMessage', {
    chat_id: chatId,
    text: 'Нажми "Открыть библиотеку", чтобы открыть mini app.',
    reply_markup: appKeyboard(),
  });
}

async function handleUpdate(update) {
  if (update.message) {
    await handleMessage(update.message);
  }
}

async function main() {
  let offset = 0;
  console.log('Mini app bot polling started');

  while (true) {
    try {
      const updates = await call('getUpdates', {
        offset,
        timeout: 25,
        allowed_updates: ['message'],
      });

      for (const update of updates) {
        offset = update.update_id + 1;
        await handleUpdate(update);
      }
    } catch (error) {
      console.error(error.message || error);
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }
}

main();

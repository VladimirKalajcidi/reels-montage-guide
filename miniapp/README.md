# Reels Mini App

Новый интерфейс с нуля, отдельно от старого `webapp`.

Что умеет:

- список роликов из `videos/`;
- фильтры `все / смонтированы / без монтажа / выложены / не выложены`;
- метки `📹` и `✅`;
- карточка ролика с текстом суфлёра и заметкой;
- переключение статуса `выложен`;
- открытие основного файла и hook-версий;
- работа как обычная мобильная вебка и как Telegram Mini App.

## Запуск

```bash
cd /Users/vladimirkalajcidi/reels_good/miniapp
npm start
```

По умолчанию сервер стартует на:

```text
http://127.0.0.1:4310
```

При необходимости:

```bash
HOST=127.0.0.1 PORT=4310 npm start
```

## Telegram Mini App

Для Telegram нужен публичный `https` URL этого приложения. Локальный `127.0.0.1` внутри Telegram не откроется.

Нужные переменные:

```bash
export TELEGRAM_BOT_TOKEN=123456:token
export TELEGRAM_MINIAPP_URL=https://your-domain.example
export TELEGRAM_ALLOWED_CHAT_ID=123456789
```

Запуск бота:

```bash
cd /Users/vladimirkalajcidi/reels_good/miniapp
npm run bot
```

Бот пришлёт кнопку `Открыть библиотеку`, которая открывает mini app внутри Telegram.

Если хочешь открыть mini app официально через BotFather:

1. Открой `@BotFather`
2. Выбери бота
3. Настрой кнопку меню / Mini App на тот же `TELEGRAM_MINIAPP_URL`

Без публичного HTTPS-адреса Telegram Mini App работать не будет.

## Файлы

- [server.js](/Users/vladimirkalajcidi/reels_good/miniapp/server.js) — HTTP API и статика
- [lib/library.js](/Users/vladimirkalajcidi/reels_good/miniapp/lib/library.js) — сбор библиотеки роликов
- [public/index.html](/Users/vladimirkalajcidi/reels_good/miniapp/public/index.html) — разметка
- [public/app.js](/Users/vladimirkalajcidi/reels_good/miniapp/public/app.js) — клиентская логика
- [public/style.css](/Users/vladimirkalajcidi/reels_good/miniapp/public/style.css) — интерфейс
- [data/video-meta.json](/Users/vladimirkalajcidi/reels_good/miniapp/data/video-meta.json) — ручные статусы и тексты

# Telegram Bot

Бот читает папку с роликами и показывает по каждому ролику:

- смонтирован или нет;
- выложен или нет;
- основной файл и готовые хуки;
- текст для суфлёра;
- заметку.

Ручные данные хранятся в [webapp/data/video-meta.json](/Users/vladimirkalajcidi/reels_good/webapp/data/video-meta.json).

## Запуск

1. Создай бота через BotFather и получи токен.
2. Запусти:

```bash
cd /Users/vladimirkalajcidi/reels_good/webapp
export TELEGRAM_BOT_TOKEN=123456:your_token
export TELEGRAM_ALLOWED_CHAT_ID=123456789
npm run bot
```

`TELEGRAM_ALLOWED_CHAT_ID` не обязателен, но лучше его задать, чтобы бот отвечал только тебе.

## На сервере

Если бот запускается не на ноуте, а на сервере, задай путь к проекту или прямо к папке роликов:

```bash
export REELS_PROJECT_ROOT=/srv/reels_good
# или точнее:
export REELS_VIDEOS_DIR=/srv/reels_good/videos
```

Тогда бот будет читать файлы оттуда.

## Что умеет

- открыть список роликов;
- фильтровать `смонтированные / не смонтированные / выложенные / не выложенные`;
- открыть карточку ролика;
- скачать основной монтаж и готовые hook-версии;
- пометить ролик как выложенный или не выложенный;
- обновить текст для суфлёра;
- сохранить заметку по ролику.

## Замечания

- Бот отправляет файлы через Telegram Bot API. Если файл больше примерно 50 МБ, бот сообщит локальный путь вместо отправки.
- Текст по умолчанию берётся из `videos/<num>/source.txt`, но его можно переопределить через бота.
- Веб-сервер тоже может читать эти метаданные через `/api/library`.

## Как развернуть схему "ноут -> сервер -> бот"

На сервере:

1. Положи проект, например, в `/srv/reels_good`.
2. Запускай бота из `/srv/reels_good/webapp`.
3. Для автозапуска можно взять шаблон [webapp/deploy/reels-bot.service.example](/Users/vladimirkalajcidi/reels_good/webapp/deploy/reels-bot.service.example).

На ноуте:

1. Задай адрес сервера и путь:

```bash
export REELS_SERVER_HOST=user@server
export REELS_SERVER_DIR=/srv/reels_good
```

2. Если SSH на нестандартном порту:

```bash
export REELS_SERVER_PORT=22
```

3. Запускай синк:

```bash
bash /Users/vladimirkalajcidi/reels_good/webapp/scripts/sync-videos-to-server.sh
```

Скрипт отправляет на сервер:

- `*_edit.mp4`
- `*_hook*.mp4`
- `source.txt`
- `montage-plan*.md`
- `webapp/data/video-meta.json`

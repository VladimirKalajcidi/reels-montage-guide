const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');
const { listVideos, getVideo, updateVideo, VIDEOS_DIR } = require('./lib/library');

const PORT = process.env.PORT || 4310;
const HOST = process.env.HOST || '127.0.0.1';
const PUBLIC_DIR = path.join(__dirname, 'public');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.txt': 'text/plain; charset=utf-8',
};

function sendJson(res, code, data) {
  res.writeHead(code, { 'Content-Type': MIME['.json'] });
  res.end(JSON.stringify(data));
}

function sendFile(res, file) {
  const ext = path.extname(file).toLowerCase();
  const type = MIME[ext] || 'application/octet-stream';
  const stream = fs.createReadStream(file);
  stream.on('error', () => sendJson(res, 404, { error: 'not found' }));
  res.writeHead(200, { 'Content-Type': type });
  stream.pipe(res);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 2 * 1024 * 1024) {
        reject(new Error('payload too large'));
        req.destroy();
      }
    });
    req.on('end', () => resolve(body ? JSON.parse(body) : {}));
    req.on('error', reject);
  });
}

function serveStatic(reqPath, res) {
  const rel = reqPath === '/' ? '/index.html' : reqPath;
  const file = path.join(PUBLIC_DIR, rel);
  if (!file.startsWith(PUBLIC_DIR)) return sendJson(res, 403, { error: 'forbidden' });
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) return sendJson(res, 404, { error: 'not found' });
  return sendFile(res, file);
}

function serveVideo(reqPath, res) {
  const parts = reqPath.split('/').filter(Boolean);
  if (parts.length < 3) return sendJson(res, 404, { error: 'not found' });
  const num = parts[1];
  const file = decodeURIComponent(parts.slice(2).join('/'));
  const full = path.join(VIDEOS_DIR, num, file);
  if (!full.startsWith(path.join(VIDEOS_DIR, num))) return sendJson(res, 403, { error: 'forbidden' });
  if (!fs.existsSync(full)) return sendJson(res, 404, { error: 'not found' });
  return sendFile(res, full);
}

function applyFilters(items, searchParams) {
  const filter = searchParams.get('filter') || 'all';
  const q = (searchParams.get('q') || '').trim().toLowerCase();
  let result = items;

  if (filter === 'edited') result = result.filter((item) => item.statuses.edited);
  if (filter === 'not_edited') result = result.filter((item) => !item.statuses.edited);
  if (filter === 'posted') result = result.filter((item) => item.statuses.posted);
  if (filter === 'not_posted') result = result.filter((item) => !item.statuses.posted);

  if (q) {
    result = result.filter((item) =>
      String(item.num).includes(q) ||
      item.title.toLowerCase().includes(q) ||
      item.note.toLowerCase().includes(q) ||
      item.teleprompterText.toLowerCase().includes(q)
    );
  }

  return result;
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);

    if (req.method === 'GET' && url.pathname === '/api/videos') {
      const items = applyFilters(listVideos(), url.searchParams);
      return sendJson(res, 200, { items });
    }

    if (req.method === 'GET' && /^\/api\/videos\/\d+$/.test(url.pathname)) {
      const num = Number(url.pathname.split('/').pop());
      const item = getVideo(num);
      return item ? sendJson(res, 200, item) : sendJson(res, 404, { error: 'not found' });
    }

    if (req.method === 'PATCH' && /^\/api\/videos\/\d+$/.test(url.pathname)) {
      const num = Number(url.pathname.split('/').pop());
      const body = await readBody(req);
      const item = updateVideo(num, {
        posted: body.posted,
        title: body.title,
        note: body.note,
        teleprompterText: body.teleprompterText,
      });
      return item ? sendJson(res, 200, item) : sendJson(res, 404, { error: 'not found' });
    }

    if (req.method === 'GET' && url.pathname.startsWith('/videos/')) {
      return serveVideo(url.pathname, res);
    }

    if (req.method === 'GET') {
      return serveStatic(url.pathname, res);
    }

    return sendJson(res, 404, { error: 'not found' });
  } catch (error) {
    return sendJson(res, 500, { error: error.message || 'server error' });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`Mini app: http://${HOST}:${PORT}`);
});

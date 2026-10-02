"""Loopback-only rack assets and telemetry HTTP handler."""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit
ASSETS=Path(__file__).resolve().parents[1]/"dashboard"

def make_handler(telemetry, rack=False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            # Reject foreign Host headers, including DNS rebinding to loopback.
            host = self.headers.get('Host', '').split(':')[0]
            if host not in ('127.0.0.1', 'localhost'):
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            if path == '/api/status':
                try:
                    payload = json.dumps(telemetry.snapshot(), allow_nan=False).encode()
                except (OSError, ValueError, TypeError):
                    self.send_error(503, 'Telemetry temporarily unavailable')
                    return
                content_type = 'application/json'
            else:
                files = {'/': ('index.html', 'text/html'), '/style.css': ('style.css', 'text/css'),
                         '/app.js': ('app.js', 'text/javascript'),
                         '/fabric-motion.mjs': ('fabric-motion.mjs', 'text/javascript')}
                if rack:
                    files = {'/': ('rack.html', 'text/html'), '/rack': ('rack.html', 'text/html'),
                             '/rack.css': ('rack.css', 'text/css'),
                             '/rack.mjs': ('rack.mjs', 'text/javascript'),
                             '/rack-model.mjs': ('rack-model.mjs', 'text/javascript')}
                if path not in files:
                    self.send_error(404)
                    return
                filename, content_type = files[path]
                try:
                    payload = (ASSETS / filename).read_bytes()
                except OSError:
                    self.send_error(404)
                    return
            self.send_response(200)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

    return Handler

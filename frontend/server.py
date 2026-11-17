"""Dependency-free local AresAtlas server. Run: python3 server.py"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os

os.chdir(Path(__file__).parent)
PORT = int(os.environ.get('PORT', 4173))
print(f'AresAtlas ready at http://localhost:{PORT}')
ThreadingHTTPServer(('0.0.0.0', PORT), SimpleHTTPRequestHandler).serve_forever()

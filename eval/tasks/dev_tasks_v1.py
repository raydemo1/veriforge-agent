"""Deterministic development tasks for the P2.2 agent evaluation.

Each task carries its broken fixture, a hidden test copied in only after the
agent finishes, and a golden change used by ``--validate`` to prove the test
fails before the fix and passes after it.

Fixture code is plain text so the task pack is inspectable without running.
"""

from __future__ import annotations

TASKS: list[dict] = [
    {
        "id": "locate_retry_policy",
        "category": "locate_explain",
        "prompt": (
            "Inspect the project. Write answer.json at the workspace root with exactly "
            'three keys: "function" (the function that computes retry delay), '
            '"default_retries" (the default retry limit), and "env_var" (the environment '
            "variable that overrides the limit). Values must match the code."
        ),
        "files": {
            "README.md": "# Retry helpers\n\nThe retry policy lives in app/retry.py.\n",
            "app/__init__.py": "",
            "app/retry.py": (
                "import os\n\n"
                "DEFAULT_RETRIES = 4\n"
                'ENV_VAR = "APP_RETRY_LIMIT"\n\n\n'
                "def retry_limit() -> int:\n"
                "    return int(os.environ.get(ENV_VAR, DEFAULT_RETRIES))\n\n\n"
                "def backoff_delay(attempt: int) -> float:\n"
                "    return min(2 ** attempt, 30)\n"
            ),
            "app/client.py": (
                "RETRIES = 2\n\n\n"
                "def send(payload):\n"
                "    return {'payload': payload, 'attempts': RETRIES}\n"
            ),
        },
        "hidden_test": (
            "import json\n"
            "from pathlib import Path\n\n\n"
            "def test_answer():\n"
            "    data = json.loads(Path('answer.json').read_text(encoding='utf-8'))\n"
            "    assert data['function'] == 'backoff_delay'\n"
            "    assert data['default_retries'] == 4\n"
            "    assert data['env_var'] == 'APP_RETRY_LIMIT'\n"
        ),
        "golden": {
            "answer.json": (
                '{"function": "backoff_delay", "default_retries": 4, '
                '"env_var": "APP_RETRY_LIMIT"}\n'
            ),
        },
    },
    {
        "id": "locate_config_precedence",
        "category": "locate_explain",
        "prompt": (
            "Inspect settings.py and determine how configuration is resolved. Write "
            "answer.json with exactly three keys: \"precedence\" (a list ordered highest "
            "priority first, using exactly the labels env, file, and defaults), "
            '"default_timeout" (the default value), and "env_prefix".'
        ),
        "files": {
            "settings.py": (
                'DEFAULTS = {"timeout": 10, "retries": 2}\n'
                'ENV_PREFIX = "APP_"\n\n\n'
                "def resolve(env, file_values):\n"
                "    merged = dict(DEFAULTS)\n"
                "    merged.update(file_values)\n"
                "    merged.update(\n"
                "        {\n"
                "            key[len(ENV_PREFIX):].lower(): value\n"
                "            for key, value in env.items()\n"
                "            if key.startswith(ENV_PREFIX)\n"
                "        }\n"
                "    )\n"
                "    return merged\n"
            ),
            "README.md": "# Settings\n\nRuntime settings are resolved by settings.resolve().\n",
        },
        "hidden_test": (
            "import json\n"
            "from pathlib import Path\n\n\n"
            "def test_answer():\n"
            "    data = json.loads(Path('answer.json').read_text(encoding='utf-8'))\n"
            "    assert data['precedence'] == ['env', 'file', 'defaults']\n"
            "    assert data['default_timeout'] == 10\n"
            "    assert data['env_prefix'] == 'APP_'\n"
        ),
        "golden": {
            "answer.json": (
                '{"precedence": ["env", "file", "defaults"], "default_timeout": 10, '
                '"env_prefix": "APP_"}\n'
            ),
        },
    },
    {
        "id": "fix_pagination",
        "category": "single_file_fix",
        "prompt": (
            "paging.page uses one-based page numbers. The current implementation is "
            "wrong. Fix paging.py so page(items, number, size) matches that contract. "
            "Do not change the function signature. Run the project tests to verify."
        ),
        "files": {
            "paging.py": (
                "def page(items, number, size):\n"
                "    start = number * size\n"
                "    return list(items[start:start + size])\n"
            ),
            "tests/test_paging.py": (
                "from paging import page\n\n\n"
                "def test_first_page():\n"
                "    assert page(range(10), 1, 3) == [0, 1, 2]\n\n\n"
                "def test_last_page():\n"
                "    assert page(range(10), 4, 3) == [9]\n"
            ),
        },
        "hidden_test": (
            "from paging import page\n\n\n"
            "def test_pages():\n"
            "    assert page(list(range(10)), 1, 3) == [0, 1, 2]\n"
            "    assert page(list(range(10)), 2, 3) == [3, 4, 5]\n"
            "    assert page(list(range(10)), 4, 3) == [9]\n"
            "    assert page(list(range(10)), 5, 3) == []\n"
        ),
        "golden": {
            "paging.py": (
                "def page(items, number, size):\n"
                "    if number < 1:\n"
                "        return []\n"
                "    start = (number - 1) * size\n"
                "    return list(items[start:start + size])\n"
            ),
        },
    },
    {
        "id": "fix_slugify",
        "category": "single_file_fix",
        "prompt": (
            "text_utils.slugify must lower-case text, replace runs of non-alphanumeric "
            "characters with a single hyphen, and strip leading/trailing hyphens. "
            "Fix text_utils.py and run the project tests."
        ),
        "files": {
            "text_utils.py": (
                "def slugify(value):\n"
                "    return value.lower().replace(' ', '-')\n"
            ),
            "tests/test_text_utils.py": (
                "from text_utils import slugify\n\n\n"
                "def test_punctuation():\n"
                "    assert slugify('Hello, World!') == 'hello-world'\n\n\n"
                "def test_spaces():\n"
                "    assert slugify('  A  B  ') == 'a-b'\n"
            ),
        },
        "hidden_test": (
            "from text_utils import slugify\n\n\n"
            "def test_slug():\n"
            "    assert slugify('Hello, World!') == 'hello-world'\n"
            "    assert slugify('  A  B  ') == 'a-b'\n"
            "    assert slugify('C++ & Python') == 'c-python'\n"
            "    assert slugify('already-slug') == 'already-slug'\n"
            "    assert slugify('---') == ''\n"
        ),
        "golden": {
            "text_utils.py": (
                "import re\n\n\n"
                "def slugify(value):\n"
                "    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')\n"
            ),
        },
    },
    {
        "id": "feature_cli_json",
        "category": "cross_file_feature",
        "prompt": (
            "Add a --json flag to app/cli.py. With the flag, stdout must be a JSON list "
            "of objects with keys name and value. Without it, output must stay as "
            "'name=value' lines. Put the JSON formatting in app/render.py as "
            "render_json(records). Run the CLI in both modes to verify."
        ),
        "files": {
            "app/__init__.py": "",
            "app/data.py": (
                'RECORDS = [{"name": "alpha", "value": 1}, {"name": "beta", "value": 2}]\n'
            ),
            "app/render.py": (
                "def render(records):\n"
                "    return '\\n'.join(f\"{record['name']}={record['value']}\" for record in records)\n"
            ),
            "app/cli.py": (
                "import sys\n\n"
                "from .data import RECORDS\n"
                "from .render import render\n\n\n"
                "def main(argv=None):\n"
                "    print(render(RECORDS))\n"
                "    return 0\n\n\n"
                "if __name__ == '__main__':\n"
                "    raise SystemExit(main(sys.argv[1:]))\n"
            ),
        },
        "hidden_test": (
            "import json\n"
            "import subprocess\n"
            "import sys\n\n\n"
            "def run(*args):\n"
            "    completed = subprocess.run(\n"
            "        [sys.executable, '-m', 'app.cli', *args],\n"
            "        capture_output=True, text=True, check=True,\n"
            "    )\n"
            "    return completed.stdout.strip()\n\n\n"
            "def test_plain_output():\n"
            "    assert run() == 'alpha=1\\nbeta=2'\n\n\n"
            "def test_json_output():\n"
            "    assert json.loads(run('--json')) == [\n"
            "        {'name': 'alpha', 'value': 1},\n"
            "        {'name': 'beta', 'value': 2},\n"
            "    ]\n"
        ),
        "golden": {
            "app/render.py": (
                "import json\n\n\n"
                "def render(records):\n"
                "    return '\\n'.join(f\"{record['name']}={record['value']}\" for record in records)\n\n\n"
                "def render_json(records):\n"
                "    return json.dumps(records)\n"
            ),
            "app/cli.py": (
                "import sys\n\n"
                "from .data import RECORDS\n"
                "from .render import render, render_json\n\n\n"
                "def main(argv=None):\n"
                "    argv = list(sys.argv[1:] if argv is None else argv)\n"
                "    print(render_json(RECORDS) if '--json' in argv else render(RECORDS))\n"
                "    return 0\n\n\n"
                "if __name__ == '__main__':\n"
                "    raise SystemExit(main(sys.argv[1:]))\n"
            ),
        },
    },
    {
        "id": "feature_retry_integration",
        "category": "cross_file_feature",
        "prompt": (
            "Implement retries across app/http.py and app/service.py. fetch(url, attempts) "
            "must call _request_once up to attempts times, retry only on status 500 or 503, "
            "and return the last response. Responses are dicts with a 'status' key. "
            "service.load(urls, attempts=3) must pass attempts through. Keep the existing "
            "signatures and run a quick check."
        ),
        "files": {
            "app/__init__.py": "",
            "app/http.py": (
                "def _request_once(url):\n"
                "    \"\"\"Return a response dict shaped {'status': int, ...} for url.\"\"\"\n"
                "    raise NotImplementedError\n\n\n"
                "def fetch(url, attempts=1):\n"
                "    return _request_once(url)\n"
            ),
            "app/service.py": (
                "from .http import fetch\n\n\n"
                "def load(urls, attempts=1):\n"
                "    return [fetch(url) for url in urls]\n"
            ),
        },
        "hidden_test": (
            "from app import http\n"
            "from app.service import load\n\n\n"
            "def test_fetch_retries_transient_status(monkeypatch):\n"
            "    responses = [\n"
            "        {'status': 503},\n"
            "        {'status': 500},\n"
            "        {'status': 200, 'body': 'ok'},\n"
            "    ]\n"
            "    calls = []\n\n"
            "    def fake(url):\n"
            "        calls.append(url)\n"
            "        return responses.pop(0)\n\n"
            "    monkeypatch.setattr(http, '_request_once', fake)\n"
            "    assert http.fetch('u', attempts=3) == {'status': 200, 'body': 'ok'}\n"
            "    assert len(calls) == 3\n\n\n"
            "def test_fetch_does_not_retry_client_error(monkeypatch):\n"
            "    calls = []\n\n"
            "    def fake(url):\n"
            "        calls.append(url)\n"
            "        return {'status': 404}\n\n"
            "    monkeypatch.setattr(http, '_request_once', fake)\n"
            "    assert http.fetch('u', attempts=3) == {'status': 404}\n"
            "    assert len(calls) == 1\n\n\n"
            "def test_service_passes_attempts(monkeypatch):\n"
            "    seen = []\n\n"
            "    def fake(url, attempts=1):\n"
            "        seen.append((url, attempts))\n"
            "        return {'status': 200, 'body': url}\n\n"
            "    monkeypatch.setattr('app.service.fetch', fake)\n"
            "    assert load(['a', 'b'], attempts=3) == [\n"
            "        {'status': 200, 'body': 'a'},\n"
            "        {'status': 200, 'body': 'b'},\n"
            "    ]\n"
            "    assert seen == [('a', 3), ('b', 3)]\n"
        ),
        "golden": {
            "app/http.py": (
                "def _request_once(url):\n"
                "    \"\"\"Return a response dict shaped {'status': int, ...} for url.\"\"\"\n"
                "    raise NotImplementedError\n\n\n"
                "def fetch(url, attempts=1):\n"
                "    response = None\n"
                "    for _ in range(max(1, attempts)):\n"
                "        response = _request_once(url)\n"
                "        if response['status'] not in {500, 503}:\n"
                "            return response\n"
                "    return response\n"
            ),
            "app/service.py": (
                "from .http import fetch\n\n\n"
                "def load(urls, attempts=1):\n"
                "    return [fetch(url, attempts=attempts) for url in urls]\n"
            ),
        },
    },
    {
        "id": "fix_failing_tests",
        "category": "test_failure_recovery",
        "prompt": (
            "The project test suite fails. Fix the implementation so every test passes. "
            "Do not modify anything under tests/. Run the suite to confirm."
        ),
        "files": {
            "cart.py": (
                "def total(prices, tax_rate=0.1, discount=0.0):\n"
                "    subtotal = sum(prices)\n"
                "    return round(subtotal * (1 + tax_rate) - discount, 2)\n"
            ),
            "tests/test_cart.py": (
                "from cart import total\n\n\n"
                "def test_percentage_discount():\n"
                "    assert total([100], tax_rate=0.1, discount=0.5) == 55.0\n\n\n"
                "def test_plain_total():\n"
                "    assert total([10, 20]) == 33.0\n"
            ),
        },
        "hidden_test": (
            "from cart import total\n\n\n"
            "def test_percentage_discount():\n"
            "    assert total([100], tax_rate=0.1, discount=0.5) == 55.0\n\n\n"
            "def test_plain_total():\n"
            "    assert total([10, 20]) == 33.0\n"
        ),
        "protected": ["tests/test_cart.py"],
        "golden": {
            "cart.py": (
                "def total(prices, tax_rate=0.1, discount=0.0):\n"
                "    subtotal = sum(prices)\n"
                "    discounted = subtotal * (1 - discount)\n"
                "    return round(discounted * (1 + tax_rate), 2)\n"
            ),
        },
    },
    {
        "id": "fix_background_server",
        "category": "background_service",
        "prompt": (
            "server.py exposes /health and /sum. /health must return JSON "
            '{"status": "ok"} and /sum?a=2&b=3 must return {"sum": 5}. Fix the bugs, '
            "then verify by starting the server in the background, requesting both "
            "endpoints, and stopping it."
        ),
        "files": {
            "server.py": (
                "import argparse\n"
                "import json\n"
                "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
                "from urllib.parse import parse_qs, urlparse\n\n\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        parsed = urlparse(self.path)\n"
                "        if parsed.path == '/health':\n"
                "            self._send({'status': health_status})\n"
                "        elif parsed.path == '/sum':\n"
                "            values = parse_qs(parsed.query)\n"
                "            a = int(values.get('a', ['0'])[0])\n"
                "            b = int(values.get('b', ['0'])[0])\n"
                "            self._send({'sum': a - b})\n"
                "        else:\n"
                "            self.send_error(404)\n\n"
                "    def _send(self, payload):\n"
                "        body = json.dumps(payload).encode()\n"
                "        self.send_response(200)\n"
                "        self.send_header('Content-Type', 'application/json')\n"
                "        self.send_header('Content-Length', str(len(body)))\n"
                "        self.end_headers()\n"
                "        self.wfile.write(body)\n\n"
                "    def log_message(self, *args):\n"
                "        pass\n\n\n"
                "def main():\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument('--port', type=int, default=8765)\n"
                "    args = parser.parse_args()\n"
                "    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()\n\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
            "README.md": "# Demo server\n\nRun `python server.py --port 8765`.\n",
        },
        "hidden_test": (
            "import json\n"
            "import socket\n"
            "import subprocess\n"
            "import sys\n"
            "import time\n"
            "import urllib.request\n\n\n"
            "def free_port():\n"
            "    with socket.socket() as sock:\n"
            "        sock.bind(('127.0.0.1', 0))\n"
            "        return sock.getsockname()[1]\n\n\n"
            "def test_endpoints():\n"
            "    port = free_port()\n"
            "    process = subprocess.Popen(\n"
            "        [sys.executable, 'server.py', '--port', str(port)],\n"
            "        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
            "    )\n"
            "    try:\n"
            "        deadline = time.time() + 15\n"
            "        health = None\n"
            "        while time.time() < deadline:\n"
            "            try:\n"
            "                with urllib.request.urlopen(\n"
            "                    f'http://127.0.0.1:{port}/health', timeout=1\n"
            "                ) as response:\n"
            "                    health = json.loads(response.read())\n"
            "                break\n"
            "            except Exception:\n"
            "                time.sleep(0.1)\n"
            "        assert health == {'status': 'ok'}\n"
            "        with urllib.request.urlopen(\n"
            "            f'http://127.0.0.1:{port}/sum?a=2&b=3', timeout=2\n"
            "        ) as response:\n"
            "            assert json.loads(response.read()) == {'sum': 5}\n"
            "    finally:\n"
            "        process.terminate()\n"
            "        process.wait(10)\n"
        ),
        "golden": {
            "server.py": (
                "import argparse\n"
                "import json\n"
                "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
                "from urllib.parse import parse_qs, urlparse\n\n\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        parsed = urlparse(self.path)\n"
                "        if parsed.path == '/health':\n"
                "            self._send({'status': 'ok'})\n"
                "        elif parsed.path == '/sum':\n"
                "            values = parse_qs(parsed.query)\n"
                "            a = int(values.get('a', ['0'])[0])\n"
                "            b = int(values.get('b', ['0'])[0])\n"
                "            self._send({'sum': a + b})\n"
                "        else:\n"
                "            self.send_error(404)\n\n"
                "    def _send(self, payload):\n"
                "        body = json.dumps(payload).encode()\n"
                "        self.send_response(200)\n"
                "        self.send_header('Content-Type', 'application/json')\n"
                "        self.send_header('Content-Length', str(len(body)))\n"
                "        self.end_headers()\n"
                "        self.wfile.write(body)\n\n"
                "    def log_message(self, *args):\n"
                "        pass\n\n\n"
                "def main():\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument('--port', type=int, default=8765)\n"
                "    args = parser.parse_args()\n"
                "    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()\n\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
        },
    },
]

BY_ID = {task["id"]: task for task in TASKS}

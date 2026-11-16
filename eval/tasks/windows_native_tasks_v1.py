"""Windows-native hard regression tasks for the second evaluation stage.

These tasks target the failures the Windows traces actually exposed:

* PowerShell object output and large native-process output reaching the model;
* merging with edits another session already made;
* background service start, verification, and complete cleanup;
* refusing to change code that is already correct;
* locating a regression across two files and fixing it without breaking the
  behaviour that still works.

They run through the same isolated runner as ``dev_tasks_v1``.
"""

from __future__ import annotations

TASKS: list[dict] = [
    {
        "id": "capture_powershell_objects",
        "category": "runtime_output",
        "prompt": (
            "Run this exact command with bash: "
            "Get-ChildItem -Force | Select-Object Mode,Name . "
            "Then use write_file to save the command's returned output verbatim as "
            "ps_output.txt. Do not use cmd /c dir, and do not redirect the command "
            "to a file inside PowerShell: the tool result itself must be recorded."
        ),
        "files": {
            "README.md": "# Inventory fixture\n",
            "data.txt": "payload\n",
            "app/__init__.py": "",
        },
        "hidden_test": (
            "from pathlib import Path\n\n\n"
            "def test_object_output_was_captured():\n"
            "    text = Path('ps_output.txt').read_text(encoding='utf-8')\n"
            "    lines = [line for line in text.splitlines() if line.strip()]\n"
            "    assert len(lines) >= 5, text\n"
            "    header = lines[0]\n"
            "    assert 'Mode' in header and 'Name' in header, header\n"
            "    assert set(lines[1].replace(' ', '')) <= {'-'}, lines[1]\n"
            "    rows = lines[2:5]\n"
            "    assert all(len(row.split()) >= 2 for row in rows), rows\n"
            "    names = {row.split()[-1] for row in rows}\n"
            "    assert {'app', 'data.txt', 'README.md'} <= names, names\n"
        ),
        "golden": {
            "ps_output.txt": (
                "Mode   Name\n"
                "----   ----\n"
                "d----- app\n"
                "-a---- data.txt\n"
                "-a---- README.md\n"
            ),
        },
    },
    {
        "id": "native_pipe_large_output",
        "category": "runtime_output",
        "prompt": (
            "Run python emit.py. Its output is large and contains non-ASCII text. "
            "Then write report.json with exactly three keys: \"token\" (the value "
            "after TOKEN=), \"special_chars\" (how many 甲 characters the output "
            "contains), and \"tail_ok\" (true only if TAIL-OK appears in the output). "
            "Base the values on the real command output."
        ),
        "files": {
            "emit.py": (
                "import hashlib\n\n"
                "TOKEN = hashlib.sha256(b'veriforge').hexdigest()[:16]\n"
                "print('[BEGIN]' + '甲' * 12000 + '[END]')\n"
                "print('TOKEN=' + TOKEN)\n"
                "print('TAIL-OK')\n"
            ),
        },
        "hidden_test": (
            "import json\n"
            "from pathlib import Path\n\n\n"
            "TOKEN = 'ab97eb5ed7407f58'\n\n\n"
            "def test_report_matches_the_real_output():\n"
            "    data = json.loads(Path('report.json').read_text(encoding='utf-8'))\n"
            "    assert set(data) == {'token', 'special_chars', 'tail_ok'}\n"
            "    assert data['token'] == TOKEN\n"
            "    assert data['special_chars'] == 12000\n"
            "    assert data['tail_ok'] is True\n"
        ),
        "golden": {
            "report.json": (
                '{"token": "ab97eb5ed7407f58", "special_chars": 12000, '
                '"tail_ok": true}\n'
            ),
        },
    },
    {
        "id": "merge_concurrent_edits",
        "category": "merge",
        "prompt": (
            "Read app/service.py before editing. Another session is editing the "
            "same file concurrently. Add retry support to fetch(): "
            "fetch(url, attempts=1) must call _request_once and retry only 500 or "
            "503, up to attempts calls in total, returning the last response. "
            "Preserve the other session's audit() change and do not edit tests. "
            "If a write reports that the file version changed, re-read the file "
            "and recompute the change."
        ),
        "files": {
            "app/__init__.py": "",
            "app/audit_notes.txt": "audit() is owned by another session\n",
            "app/service.py": (
                "RETRYABLE = {500, 503}\n\n\n"
                "def _request_once(url):\n"
                "    return {'status': 200, 'url': url}\n\n\n"
                "def fetch(url, attempts=1):\n"
                "    return _request_once(url)\n\n\n"
                "# Added by another session; keep it unchanged.\n"
                "def audit(event):\n"
                "    return {'event': event, 'logged': True}\n"
            ),
            "tests/test_service.py": (
                "from app.service import audit\n\n\n"
                "def test_audit():\n"
                "    result = audit('x')\n"
                "    assert result['event'] == 'x'\n"
                "    assert result['logged'] is True\n"
            ),
        },
        "hidden_test": (
            "import app.service as service\n\n\n"
            "def test_retries_retryable_status():\n"
            "    calls = []\n"
            "    answers = [503, 200]\n\n"
            "    def fake(url):\n"
            "        calls.append(url)\n"
            "        return {'status': answers.pop(0), 'url': url}\n\n"
            "    service._request_once = fake\n"
            "    assert service.fetch('u', 3) == {'status': 200, 'url': 'u'}\n"
            "    assert len(calls) == 2\n\n\n"
            "def test_stops_on_non_retryable_status():\n"
            "    calls = []\n"
            "    answers = [404, 200]\n\n"
            "    def fake(url):\n"
            "        calls.append(url)\n"
            "        return {'status': answers.pop(0), 'url': url}\n\n"
            "    service._request_once = fake\n"
            "    assert service.fetch('u', 5)['status'] == 404\n"
            "    assert len(calls) == 1\n\n\n"
            "def test_exhausts_attempts():\n"
            "    calls = []\n"
            "    answers = [500, 500, 500, 999]\n\n"
            "    def fake(url):\n"
            "        calls.append(url)\n"
            "        return {'status': answers.pop(0), 'url': url}\n\n"
            "    service._request_once = fake\n"
            "    assert service.fetch('u', 3)['status'] == 500\n"
            "    assert len(calls) == 3\n\n\n"
            "def test_audit_is_unchanged():\n"
            "    result = service.audit('x')\n"
            "    assert result['event'] == 'x'\n"
            "    assert result['logged'] is True\n\n\n"
            "def test_controller_change_is_preserved():\n"
            "    assert service.audit('x').get('controller') is True\n"
        ),
        "concurrent_edit": {
            "after_read": "app/service.py",
            "path": "app/service.py",
            "search": (
                "def audit(event):\n"
                "    return {'event': event, 'logged': True}\n"
            ),
            "replace": (
                "def audit(event):\n"
                "    return {'event': event, 'logged': True, 'controller': True}\n"
            ),
        },
        "protected": ["app/audit_notes.txt", "tests/test_service.py"],
        "golden": {
            "app/service.py": (
                "RETRYABLE = {500, 503}\n\n\n"
                "def _request_once(url):\n"
                "    return {'status': 200, 'url': url}\n\n\n"
                "def fetch(url, attempts=1):\n"
                "    response = _request_once(url)\n"
                "    for _ in range(attempts - 1):\n"
                "        if response['status'] not in RETRYABLE:\n"
                "            break\n"
                "        response = _request_once(url)\n"
                "    return response\n\n\n"
                "# Added by another session; keep it unchanged.\n"
                "def audit(event):\n"
                "    return {'event': event, 'logged': True, 'controller': True}\n"
            ),
        },
    },
    {
        "id": "background_service_cleanup",
        "category": "process_lifecycle",
        "prompt": (
            "Start the service with python server.py 28765 as a background Shell "
            "job. Verify it with python check_health.py 28765, which prints the HTTP "
            "status. Then stop the background job and confirm the port is free. "
            "Write report.json with exactly three keys: \"health_status\" "
            "(the integer status), \"stopped\" (true after the job stopped), and "
            "\"port_free\" (true only if nothing is listening on port 28765 anymore)."
        ),
        "files": {
            "server.py": (
                "import json\n"
                "import os\n"
                "import sys\n"
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n\n"
                "PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 28765\n\n\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        if self.path == '/health':\n"
                "            body = json.dumps({'status': 'ok'}).encode()\n"
                "            self.send_response(200)\n"
                "            self.send_header('Content-Type', 'application/json')\n"
                "            self.send_header('Content-Length', str(len(body)))\n"
                "            self.end_headers()\n"
                "            self.wfile.write(body)\n"
                "        else:\n"
                "            self.send_response(404)\n"
                "            self.end_headers()\n\n"
                "    def log_message(self, *args):\n"
                "        pass\n\n\n"
                "if __name__ == '__main__':\n"
                "    open('server.pid', 'w').write(str(os.getpid()))\n"
                "    HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()\n"
            ),
            "check_health.py": (
                "import sys\n"
                "import urllib.request\n\n"
                "port = int(sys.argv[1]) if len(sys.argv) > 1 else 28765\n"
                "with urllib.request.urlopen(\n"
                "    f'http://127.0.0.1:{port}/health', timeout=5\n"
                ") as response:\n"
                "    print(response.status)\n"
                "    print(response.read().decode())\n"
            ),
        },
        "hidden_test": (
            "import json\n"
            "import socket\n"
            "from pathlib import Path\n\n\n"
            "PORT = 28765\n\n\n"
            "def test_report():\n"
            "    data = json.loads(Path('report.json').read_text(encoding='utf-8'))\n"
            "    assert set(data) == {'health_status', 'stopped', 'port_free'}\n"
            "    assert data['health_status'] == 200\n"
            "    assert data['stopped'] is True\n"
            "    assert data['port_free'] is True\n\n\n"
            "def test_port_is_free():\n"
            "    sock = socket.socket()\n"
            "    exclusive = getattr(socket, 'SO_EXCLUSIVEADDRUSE', None)\n"
            "    if exclusive is not None:\n"
            "        sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)\n"
            "    try:\n"
            "        sock.bind(('127.0.0.1', PORT))\n"
            "    finally:\n"
            "        sock.close()\n\n\n"
            "def test_runtime_state_was_cleaned_up():\n"
            "    assert not Path('server.pid').exists()\n"
        ),
        "golden": {
            "report.json": (
                '{"health_status": 200, "stopped": true, "port_free": true}\n'
            ),
        },
    },
    {
        "id": "keep_correct_code_unchanged",
        "category": "negative",
        "prompt": (
            "CI claims clamp(value, low, high) is broken when low > high. Read "
            "CONTRACT.md and the implementation, then decide whether the source is "
            "actually broken under the documented contract. Do not modify any Python "
            "file or test. Write diagnosis.json with exactly two keys: \"changed\" "
            "(false when no source fix is needed) and \"reason\" (why the CI claim "
            "does not describe a real defect)."
        ),
        "files": {
            "CONTRACT.md": (
                "# clamp contract\n\n"
                "clamp(value, low, high) returns low when value < low and high when "
                "value > high, otherwise value. Callers must pass low <= high; the "
                "function does not define behaviour for low > high.\n"
            ),
            "CI_REPORT.md": (
                "# CI report\n\n"
                "clamp(5, 10, 1) returns 10 instead of raising, so clamp is broken.\n"
            ),
            "app/__init__.py": "",
            "app/mathx.py": (
                "def clamp(value, low, high):\n"
                "    if value < low:\n"
                "        return low\n"
                "    if value > high:\n"
                "        return high\n"
                "    return value\n"
            ),
            "tests/test_mathx.py": (
                "from app.mathx import clamp\n\n\n"
                "def test_bounds():\n"
                "    assert clamp(5, 1, 10) == 5\n"
                "    assert clamp(0, 1, 10) == 1\n"
                "    assert clamp(11, 1, 10) == 10\n"
            ),
        },
        "hidden_test": (
            "import json\n"
            "from pathlib import Path\n\n\n"
            "SOURCE = (\n"
            "    'def clamp(value, low, high):\\n'\n"
            "    '    if value < low:\\n'\n"
            "    '        return low\\n'\n"
            "    '    if value > high:\\n'\n"
            "    '        return high\\n'\n"
            "    '    return value\\n'\n"
            ")\n\n\n"
            "def test_no_source_change_was_needed_or_made():\n"
            "    data = json.loads(Path('diagnosis.json').read_text(encoding='utf-8'))\n"
            "    assert set(data) == {'changed', 'reason'}\n"
            "    assert data['changed'] is False\n"
            "    assert data['reason'].strip()\n"
            "    assert Path('app/mathx.py').read_text(encoding='utf-8') == SOURCE\n\n\n"
            "def test_contract_still_holds():\n"
            "    from app.mathx import clamp\n"
            "    assert clamp(5, 1, 10) == 5\n"
            "    assert clamp(0, 1, 10) == 1\n"
            "    assert clamp(11, 1, 10) == 10\n"
        ),
        "protected": ["app/mathx.py", "tests/test_mathx.py"],
        "golden": {
            "diagnosis.json": (
                '{"changed": false, "reason": "clamp is correct for the documented '
                'contract; low > high is outside it and the CI call does not '
                'describe a defect."}\n'
            ),
        },
    },
    {
        "id": "multi_file_regression",
        "category": "regression",
        "prompt": (
            "Reports show EUR amounts with a dollar sign, and amounts have no "
            "thousands separators. Fix app/report.py and app/format.py: EUR must "
            "render with the euro sign, USD must stay '$', and format_amount must "
            "group thousands and keep two decimals (1234.5 -> 1,234.50). Keep the "
            "function signatures and do not edit tests."
        ),
        "files": {
            "app/__init__.py": "",
            "app/format.py": (
                "def format_amount(value):\n"
                "    return f'{value:.2f}'\n"
            ),
            "app/report.py": (
                "from app.format import format_amount\n\n"
                "CURRENCY = {'USD': '$', 'EUR': '$'}\n\n\n"
                "def render(amount, currency):\n"
                "    return f'{CURRENCY[currency]}{format_amount(amount)}'\n"
            ),
            "tests/test_report.py": (
                "from app.report import render\n\n\n"
                "def test_usd_amount():\n"
                "    assert render(9, 'USD') == '$9.00'\n"
            ),
        },
        "hidden_test": (
            "from app.format import format_amount\n"
            "from app.report import render\n\n\n"
            "def test_thousands_grouping():\n"
            "    assert format_amount(1234.5) == '1,234.50'\n"
            "    assert format_amount(999) == '999.00'\n\n\n"
            "def test_currency_symbols():\n"
            "    assert render(1234.5, 'EUR') == '€1,234.50'\n"
            "    assert render(9, 'USD') == '$9.00'\n"
        ),
        "protected": ["tests/test_report.py"],
        "golden": {
            "app/format.py": (
                "def format_amount(value):\n"
                "    return f'{value:,.2f}'\n"
            ),
            "app/report.py": (
                "from app.format import format_amount\n\n"
                "CURRENCY = {'USD': '$', 'EUR': '€'}\n\n\n"
                "def render(amount, currency):\n"
                "    return f'{CURRENCY[currency]}{format_amount(amount)}'\n"
            ),
        },
    },
]

from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness_code_agent.tui.protocol import UI_PROTOCOL_VERSION, validate_ui_event
from harness_code_agent.tui_bridge import BridgeServer, SessionBridge


class TuiProtocolTests(unittest.TestCase):
    def _server(self) -> SessionBridge:
        server = object.__new__(SessionBridge)
        server.cwd = Path("workspace")
        server._session = SimpleNamespace()
        server._session_error = None
        server.project_resources = None
        server._session_thread = SimpleNamespace(is_alive=lambda: False)
        server.responses = []
        server._response = lambda request_id, result=None, error=None: server.responses.append(
            {"id": request_id, "result": result, "error": error}
        )
        return server

    def test_stop_preserves_queued_messages_and_resets_idle_flag(self):
        import queue
        from harness_code_agent.agent.cancellation import CancellationToken
        for active in (False, True):
            server = self._server()
            server._active_lock = threading.Lock()
            server._active_token = CancellationToken() if active else None
            server._stopping = threading.Event()
            server._tasks = queue.Queue()
            server._tasks.put("next message")
            server._session = SimpleNamespace()
            server._send_event = lambda event: None
            server._handle_request({"id": "stop", "method": "cancel", "params": {}})
            self.assertEqual(server._tasks.get_nowait(), "next message")
            self.assertEqual(server.responses[0]["result"]["discardedQueued"], 0)
            self.assertEqual(server._stopping.is_set(), active)
            if active:
                self.assertTrue(server._active_token.is_cancelled)

    def test_initialize_requires_the_shared_protocol_version(self):
        server = object.__new__(BridgeServer)
        server.registry = SimpleNamespace(announce=lambda: None, _records={}, active_id="A")
        responses = []
        server._response = lambda request_id, result=None, error=None: responses.append((result, error))
        server._handle_request({"id": "init", "method": "initialize", "params": {"protocolVersion": UI_PROTOCOL_VERSION}})
        self.assertEqual(responses[-1][0]["protocolVersion"], UI_PROTOCOL_VERSION)
        server._handle_request({"id": "wrong", "method": "initialize", "params": {"protocolVersion": UI_PROTOCOL_VERSION + 1}})
        self.assertIn("不匹配", str(responses[-1][1]))

    def test_failed_interaction_resolution_still_closes_the_modal(self):
        server = self._server()
        server._interactions = SimpleNamespace(resolve=lambda _id, _result: False)
        events = []
        server._send_event = events.append

        SessionBridge._handle_request(server, {
            "id": "resolve",
            "method": "resolve_interaction",
            "params": {"id": "expired", "result": {"decision": "allow"}},
        })

        self.assertIn("交互已失效", server.responses[0]["error"])
        self.assertEqual(events, [{"type": "interaction_closed", "id": "expired"}])

    def test_broken_output_pipe_is_latched_without_raising(self):
        server = object.__new__(BridgeServer)
        server._output_broken = threading.Event()
        server._write_lock = threading.Lock()
        output = SimpleNamespace(write=lambda _text: (_ for _ in ()).throw(BrokenPipeError()), flush=lambda: None)

        with patch("harness_code_agent.tui_bridge.sys.stdout", output):
            server._write({"type": "event"})
            server._write({"type": "event"})

        self.assertTrue(server._output_broken.is_set())

    def test_event_validation_rejects_unknown_and_incomplete_events(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_ui_event({"type": "surprise"})
        with self.assertRaisesRegex(ValueError, "detail"):
            validate_ui_event({"type": "progress", "status": "starting"})

    def test_stale_session_construction_closes_without_publishing(self):
        server = object.__new__(SessionBridge)
        server.cwd = Path("workspace")
        server.profile_name = "general"
        server.profile_explicit = False
        server.runtime_id = "test"
        server.state_root = Path("workspace/.harness")
        server.environment = "local"
        server._interactions = SimpleNamespace()
        server._session_lock = threading.Lock()
        server._session_generation = 2
        server._closing = threading.Event()
        server._session = None
        server.project_resources = None
        server._session_error = None
        closed: list[bool] = []
        stale_session = SimpleNamespace(close=lambda: closed.append(True))

        with patch(
            "harness_code_agent.tui_bridge.InteractiveSession",
            return_value=stale_session,
        ):
            server._construct_session(generation=1)

        self.assertEqual(closed, [True])
        self.assertIsNone(server._session)


if __name__ == "__main__":
    unittest.main()

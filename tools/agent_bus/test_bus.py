#!/usr/bin/env python3
"""Tests for the file-backed agent bus (stdlib only)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bus  # noqa: E402


class BusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_post_inbox_and_broadcast(self) -> None:
        m = bus.post(sender="codex", recipient="cursor", subject="Review figure",
                     body="Check panel B", kind="request", refs=["figs/q2.pdf"], root=self.root)
        self.assertTrue(m["id"].startswith("MSG-"))
        self.assertEqual(bus.inbox("cursor", root=self.root)[0]["id"], m["id"])
        self.assertEqual(bus.inbox("zcode", root=self.root), [])
        bus.post(sender="codex", recipient="all", subject="Standup", root=self.root)
        self.assertEqual(len(bus.inbox("zcode", root=self.root)), 1)

    def test_read_claim_done(self) -> None:
        m = bus.post(sender="zcode", recipient="codex", subject="Check data",
                     kind="request", root=self.root)
        self.assertEqual(bus.read(m["id"], "codex", root=self.root)["status"], "read")
        self.assertEqual(bus.claim(m["id"], "codex", root=self.root)["status"], "claimed")
        done = bus.done(m["id"], "codex", note="checked", root=self.root)
        self.assertEqual(done["status"], "done")
        self.assertEqual(done["result"], "checked")
        self.assertEqual(bus.inbox("codex", unread_only=True, root=self.root), [])
        self.assertEqual(len(bus.inbox("codex", include_done=True, root=self.root)), 1)

    def test_claim_conflict(self) -> None:
        m = bus.post(sender="zcode", recipient="all", subject="Shared task", root=self.root)
        bus.claim(m["id"], "codex", root=self.root)
        with self.assertRaises(PermissionError):
            bus.claim(m["id"], "cursor", root=self.root)

    def test_wrong_recipient_cannot_claim(self) -> None:
        m = bus.post(sender="codex", recipient="cursor", subject="Cursor only", root=self.root)
        with self.assertRaises(PermissionError):
            bus.claim(m["id"], "zcode", root=self.root)

    def test_summary_counts(self) -> None:
        bus.post(sender="codex", recipient="cursor", subject="One", kind="request", root=self.root)
        m2 = bus.post(sender="codex", recipient="cursor", subject="Two", kind="request", root=self.root)
        bus.claim(m2["id"], "cursor", root=self.root)
        s = bus.summary(self.root)
        self.assertEqual(s["total"], 2)
        self.assertEqual(s["active"], 2)
        self.assertEqual(s["by_agent"]["cursor"]["unread"], 1)
        self.assertEqual(s["by_agent"]["cursor"]["claimed"], 1)

    def test_corrupt_line_is_skipped(self) -> None:
        path = bus.bus_path(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json}\n", encoding="utf-8")
        bus.post(sender="codex", recipient="cursor", subject="Still works", root=self.root)
        self.assertEqual(len(bus.inbox("cursor", root=self.root)), 1)


class McpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.proc = subprocess.Popen(
            [sys.executable, str(HERE / "mcp_server.py"), "--agent", "codex", "--root", str(self.root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def tearDown(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        for pipe in (self.proc.stdout, self.proc.stderr):
            if pipe:
                pipe.close()
        self.tmp.cleanup()

    def rpc(self, obj: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        self.assertTrue(line, "server produced no response")
        return json.loads(line)

    def test_initialize_list_and_post(self) -> None:
        init = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(init["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(init["result"]["serverInfo"]["name"], "shuwei-agent-bus")

        listed = self.rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in listed["result"]["tools"]}
        self.assertIn("bus_post", names)
        self.assertIn("bus_done", names)

        called = self.rpc({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "bus_post", "arguments": {
                "to": "cursor", "subject": "MCP hello", "body": "from test",
                "type": "request",
            }},
        })
        self.assertFalse(called["result"]["isError"], called)
        payload = json.loads(called["result"]["content"][0]["text"])
        self.assertEqual(payload["message"]["from"], "codex")
        self.assertEqual(len(bus.inbox("cursor", root=self.root)), 1)

    def test_unknown_tool_returns_error_content(self) -> None:
        resp = self.rpc({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        })
        self.assertTrue(resp["result"]["isError"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

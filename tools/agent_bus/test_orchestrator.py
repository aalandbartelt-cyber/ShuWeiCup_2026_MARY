#!/usr/bin/env python3
"""Tests for orchestration helpers (no model calls)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bus  # noqa: E402
import orchestrator  # noqa: E402


class OrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pick_request_skips_note(self) -> None:
        bus.post(sender="cursor", recipient="codex", subject="Note only", kind="note", root=self.root)
        req = bus.post(sender="cursor", recipient="codex", subject="Do work", kind="request", root=self.root)
        self.assertEqual(orchestrator._pick_message("codex", None, self.root)["id"], req["id"])

    def test_prompt_contains_actor_and_rules(self) -> None:
        msg = bus.post(sender="zcode", recipient="cursor", subject="Review",
                       body="Check panel B", kind="request", root=self.root)
        prompt = orchestrator._render_prompt(orchestrator.BUILD_PROMPT, msg, self.root, "cursor")
        self.assertIn("You are Cursor", prompt)
        self.assertIn("backup primary", prompt)
        self.assertIn("05_论文/待审", prompt)
        self.assertIn("Check panel B", prompt)

    def test_extract_cursor_json_note(self) -> None:
        out = '{"type":"result","result":"VERDICT: PASS"}\n'
        self.assertEqual(orchestrator._extract_note("cursor", out, None), "VERDICT: PASS")

    def test_extract_codex_last_file(self) -> None:
        p = self.root / "last.md"
        p.write_text("RESULT: done", encoding="utf-8")
        self.assertEqual(orchestrator._extract_note("codex", "ignored", p), "RESULT: done")

    def test_next_agent_and_followup(self) -> None:
        msg = bus.post(sender="zcode", recipient="codex", subject="Build",
                       kind="request", root=self.root)
        note = "RESULT: done\nNEXT: review the diff\nNEXT_AGENT: cursor"
        self.assertEqual(orchestrator._next_agent(note), "cursor")
        self.assertEqual(orchestrator._next_action(note), "review the diff")
        p = self.root / "t.md"
        p.write_text("x", encoding="utf-8")
        follow = orchestrator._post_followup("codex", msg, note, p, self.root)
        self.assertIsNotNone(follow)
        self.assertEqual(follow["to"], "cursor")
        self.assertEqual(follow["type"], "request")
        self.assertIn("review the diff", follow["body"])

    def test_no_self_followup(self) -> None:
        msg = bus.post(sender="zcode", recipient="codex", subject="Build",
                       kind="request", root=self.root)
        p = self.root / "t.md"
        p.write_text("x", encoding="utf-8")
        self.assertIsNone(orchestrator._post_followup("codex", msg, "NEXT_AGENT: codex", p, self.root))


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Turn a bus message into one non-interactive agent run.

This is the "auto" part of the collaboration loop:

    bus message -> claim -> agent CLI -> transcript -> reply -> done

It is intentionally one-shot and gated. Default is dry-run; pass `--run` to
spend model quota. It never loops forever and never edits formal deliverables
unless the message explicitly authorizes that.

Auto-driven agents:
- codex  -> `codex exec`
- cursor -> `cursor-agent -p` (requires one-time `agent login`)

dsh/ZCode still participate through the file bus on their next turn; they have
no verified headless CLI here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bus  # noqa: E402

ACTOR_ROLE = {
    "codex": "the primary builder",
    "cursor": "the backup primary (Cursor Agent CLI)",
    "zcode": "the Windows-side data/review assistant",
    "dsh": "the paper reviewer",
}
ACTOR_NAME = {"codex": "Codex", "cursor": "Cursor", "zcode": "ZCode", "dsh": "dsh"}
ACTOR_RULE = {
    "codex": "Formal paper/figure/delivery files stay yours; rewrite them only when the request explicitly authorizes it.",
    "cursor": "Formal paper/figure/delivery files stay Codex-owned. You may write `cursor/` branches and `05_论文/待审/`; writing `05_论文/正式/` or `06_交付/` requires explicit user authorization recorded in `docs/collab/decisions.md`.",
    "zcode": "Do not edit formal paper/figure/delivery files; write analysis or candidates only.",
    "dsh": "Do not edit formal files; write reviews under `docs/collab/reviews/`.",
}

BUILD_PROMPT = """\
You are {actor}, {actor_role} in the ShuWei 2026 workspace.

Repository: {root}
Bus message: {msg_id}
From: {sender}
Subject: {subject}

Before acting:
1. Read AGENTS.md, docs/collab/CURSOR_HANDOFF.md, and the ShuWei skill SKILL.md.
2. Run `git status --short --branch`; do not overwrite uncommitted work.
3. Read the full request below and restate the acceptance criteria in one short paragraph.

Request:
---
{body}
---

References:
{refs}

Rules:
- Work only inside the requested scope. {actor_rule}
- Do not send bus messages or mark this message done; the orchestrator records
  your result. If you are blocked, say exactly what decision or credential is
  missing.
- Run the relevant tests/checks before claiming success. Prefer small, reviewable
  changes; keep a short evidence list (commands + outcomes).
- End with: RESULT (what changed), EVIDENCE (commands/results), NEXT (one action).
- If the next step genuinely needs another agent, add one final line
  `NEXT_AGENT: cursor` (or codex/zcode/dsh). The orchestrator will open a
  follow-up request for that agent. Do not call bus tools yourself.
"""

REVIEW_PROMPT = """\
You are an independent reviewer in the ShuWei 2026 workspace.

Repository: {root}
Bus message: {msg_id}
Subject: {subject}

Review the work described below against the repository state. Read AGENTS.md,
docs/collab/CURSOR_HANDOFF.md, and the relevant skill references. Run only
read-only checks; do not edit files.

Request context:
---
{body}
---

Return a terse verdict:
- VERDICT: PASS / PASS_WITH_ISSUES / FAIL
- P0: correctness or submission-breaking issues (with file:line evidence)
- P1: important but non-blocking issues
- P2: polish
- EVIDENCE: commands you ran

If P0/P1 issues need a builder, add `NEXT_AGENT: codex` and `NEXT: <one action>`.
"""


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _agent_exe(agent: str) -> str | None:
    names = {"codex": ("codex",), "cursor": ("cursor-agent", "agent")}
    for name in names.get(agent, ()):
        exe = shutil.which(name)
        if exe:
            return exe
    candidates = {
        "codex": (Path.home() / ".local" / "bin" / "codex",),
        "cursor": (Path.home() / ".local" / "bin" / "cursor-agent", Path.home() / ".local" / "bin" / "agent"),
    }
    for candidate in candidates.get(agent, ()):
        if candidate.is_file():
            return str(candidate)
    return None


def _pick_message(agent: str, msg_id: str | None, root: Path) -> dict:
    if msg_id:
        for item in bus.load(root):
            if item.get("id") == msg_id:
                if item.get("to") not in (agent, bus.BROADCAST):
                    raise SystemExit(f"{msg_id} is addressed to {item.get('to')}, not {agent}")
                return item
        raise SystemExit(f"message not found: {msg_id}")
    items = bus.inbox(agent, unread_only=True, root=root)
    requests = [m for m in items if m.get("type") in ("request", "handoff")]
    if not requests:
        raise SystemExit(f"no unread request/handoff for {agent}")
    return requests[0]


def _render_prompt(template: str, msg: dict, root: Path, actor: str) -> str:
    refs = "\n".join(f"- {x}" for x in (msg.get("refs") or [])) or "- (none)"
    return template.format(
        root=root,
        actor=ACTOR_NAME.get(actor, actor),
        actor_role=ACTOR_ROLE.get(actor, "an agent"),
        actor_rule=ACTOR_RULE.get(actor, "Do not edit formal files."),
        msg_id=msg["id"],
        sender=msg.get("from", "?"),
        subject=msg.get("subject", ""),
        body=(msg.get("body") or "").strip() or "(no body)",
        refs=refs,
    )


def _write_transcript(out_dir: Path, msg: dict, role: str, prompt: str, stdout: str, stderr: str, code: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{msg['id']}-{role}-{_now_tag()}.md"
    path.write_text(
        f"# {msg['id']} / {role}\n\n"
        f"- time: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n"
        f"- from: {msg.get('from')}\n- subject: {msg.get('subject')}\n"
        f"- exit: {code}\n\n"
        f"## Prompt\n\n```text\n{prompt}\n```\n\n"
        f"## stdout\n\n```text\n{stdout[-20000:]}\n```\n\n"
        f"## stderr\n\n```text\n{stderr[-8000:]}\n```\n",
        encoding="utf-8",
    )
    return path


def _post_result(msg: dict, actor: str, role: str, ok: bool, note: str, transcript: Path, root: Path) -> None:
    recipient = msg.get("from", "codex")
    body = textwrap.dedent(
        f"""\
        Orchestrator {role} run by {actor} for {msg['id']}.

        Outcome: {"OK" if ok else "FAILED"}
        Transcript: {transcript}
        Note: {note or "(none)"}
        """
    ).strip()
    bus.post(
        sender=actor,
        recipient=recipient,
        subject=f"Re: {msg.get('subject', msg['id'])}",
        body=body,
        kind="reply",
        reply_to=msg["id"],
        refs=[str(transcript)],
        root=root,
    )


def _extract_note(agent: str, stdout: str, last_path: Path | None) -> str:
    if last_path is not None and last_path.is_file():
        return last_path.read_text(encoding="utf-8", errors="replace")[:4000]
    text = (stdout or "").strip()
    if agent == "cursor":
        for line in reversed(text.splitlines()):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                for key in ("result", "text", "message", "output"):
                    value = obj.get(key)
                    if isinstance(value, str) and value.strip():
                        return value[:4000]
    return text[-4000:]


def _cursor_auth_ok(exe: str) -> bool:
    try:
        proc = subprocess.run([exe, "status"], text=True, capture_output=True, timeout=20)
    except Exception:
        return False
    return "not logged in" not in ((proc.stdout or "") + (proc.stderr or "")).lower()


def _next_agent(note: str) -> str | None:
    matches = re.findall(r"(?im)^\s*NEXT_AGENT:\s*([a-zA-Z-]+)\s*$", note or "")
    if not matches:
        return None
    try:
        return bus.validate_agent(matches[-1])
    except ValueError:
        return None


def _next_action(note: str) -> str:
    match = re.search(r"(?im)^\s*NEXT:\s*(.+?)\s*$", note or "")
    return match.group(1).strip() if match else ""


def _post_followup(actor: str, msg: dict, note: str, transcript: Path, root: Path) -> dict | None:
    target = _next_agent(note)
    if not target or target == actor:
        return None
    action = _next_action(note) or f"Continue from {msg['id']}"
    body = textwrap.dedent(
        f"""\
        Follow-up opened by {actor} after {msg['id']}.

        Original subject: {msg.get('subject', '')}
        Next action: {action}

        Previous result:
        ---
        {note[:6000]}
        ---

        Transcript: {transcript}
        """
    ).strip()
    return bus.post(
        sender=actor,
        recipient=target,
        subject=f"Follow-up: {msg.get('subject', msg['id'])}",
        body=body,
        kind="request",
        refs=[str(transcript)],
        root=root,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one bus message through Codex or Cursor (gated)")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--agent", default="codex", choices=bus.AGENTS)
    ap.add_argument("--message", help="message id; default: first unread request/handoff")
    ap.add_argument("--role", default="build", choices=("build", "review"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--run", action="store_true", help="actually invoke the model (default: dry-run)")
    ap.add_argument("--keep-claimed-on-failure", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    root = (args.root or bus.default_root()).resolve()
    out_dir = args.out_dir or (root / "docs" / "collab" / "rounds")
    msg = _pick_message(args.agent, args.message, root)
    template = BUILD_PROMPT if args.role == "build" else REVIEW_PROMPT
    prompt = _render_prompt(template, msg, root, args.agent)
    exe = _agent_exe(args.agent)

    if args.agent not in ("codex", "cursor"):
        print(
            f"{args.agent}: no verified headless CLI. The message stays in the bus; "
            "that agent will see it on its next turn. Auto-run supports codex and cursor.",
            file=sys.stderr,
        )
        return 3
    if not exe:
        print(f"{args.agent} CLI not found in PATH or ~/.local/bin", file=sys.stderr)
        return 3

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.agent == "codex":
        last_path: Path | None = out_dir / f"{msg['id']}-{args.role}-{_now_tag()}.last.md"
        cmd = [exe, "exec", "-C", str(root), "-s", "workspace-write", "--color", "never"]
        if args.model:
            cmd += ["-m", args.model]
        cmd += ["-o", str(last_path), "-"]
        input_text: str | None = prompt
    else:
        last_path = None
        cmd = [exe, "-p", "--output-format", "json", "--workspace", str(root), "--trust"]
        if args.role == "review":
            cmd += ["--mode", "plan"]
        else:
            cmd += ["--force"]
        if args.model:
            cmd += ["--model", args.model]
        cmd.append(prompt)
        input_text = None

    print(f"message : {msg['id']}  {msg.get('from')} -> {msg.get('to')}  {msg.get('subject')}")
    print(f"repo    : {root}")
    print(f"command : {' '.join(cmd[:8])}{' ...' if len(cmd) > 8 else ''}")
    print(f"prompt  : {len(prompt)} chars")
    if not args.run:
        print("\n--- DRY RUN (pass --run to invoke) ---\n")
        print(prompt)
        return 0

    if args.agent == "cursor" and not _cursor_auth_ok(exe):
        print(
            "cursor-agent is installed but not logged in. Run once in WSL:\n"
            "  export PATH=\"$HOME/.local/bin:$PATH\"\n"
            "  agent login\n"
            "Then re-run this command.",
            file=sys.stderr,
        )
        return 3

    claimed = bus.claim(msg["id"], args.agent, root=root)
    print(f"claimed : {claimed['id']} by {args.agent}")
    started = dt.datetime.now()
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            cwd=str(root),
            env={**os.environ, "NO_COLOR": "1"},
        )
        code = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTIMEOUT after {args.timeout}s\n"
    elapsed = int((dt.datetime.now() - started).total_seconds())
    transcript = _write_transcript(out_dir, msg, args.role, prompt, stdout, stderr, code)
    note = _extract_note(args.agent, stdout, last_path)
    ok = code == 0
    _post_result(msg, args.agent, args.role, ok, note, transcript, root)
    if ok:
        bus.done(msg["id"], args.agent, note=note, root=root)
        follow = _post_followup(args.agent, msg, note, transcript, root)
        print(f"done    : {msg['id']} in {elapsed}s")
        if follow:
            print(f"follow  : {follow['id']} -> {follow['to']}")
        print(f"result  : {transcript}")
        return 0

    if not args.keep_claimed_on_failure:
        bus.done(msg["id"], args.agent, note=f"FAILED (exit {code}); see {transcript}", root=root)
    print(f"failed  : exit {code} after {elapsed}s; transcript {transcript}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

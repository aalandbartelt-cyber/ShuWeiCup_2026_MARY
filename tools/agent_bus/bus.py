#!/usr/bin/env python3
"""File-backed message bus for the ShuWei multi-agent workspace.

The bus is deliberately boring: one append-mostly JSONL file, a process lock,
and a small CLI. It works without a server, survives WSL restarts, and leaves a
readable audit trail in git.

It does NOT wake an idle agent. Delivery happens when the recipient next reads
its inbox (at the start/end of a turn, or via the orchestrator).

Usage:
    python3 tools/agent_bus/bus.py post --from codex --to cursor \
        --type request --subject "Review Q2 figure" --body "..."
    python3 tools/agent_bus/bus.py inbox --agent cursor --unread
    python3 tools/agent_bus/bus.py claim MSG-... --agent cursor
    python3 tools/agent_bus/bus.py done MSG-... --agent cursor --note "fixed"
    python3 tools/agent_bus/bus.py summary
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

AGENTS = ("codex", "cursor", "zcode", "dsh")
BROADCAST = "all"
TYPES = ("request", "reply", "note", "handoff", "decision", "review")
STATUSES = ("unread", "read", "claimed", "done")
DEFAULT_REL = Path("docs") / "collab" / "AGENT_MESSAGES.jsonl"


def default_root() -> Path:
    env = os.environ.get("SHUWEI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # tools/agent_bus/bus.py -> repo root
    return Path(__file__).resolve().parents[2]


def bus_path(root: Path | None = None) -> Path:
    return (root or default_root()) / DEFAULT_REL


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def make_id() -> str:
    return "MSG-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + f"{random.randrange(16**4):04x}"


def _norm_agent(value: str) -> str:
    v = str(value or "").strip().lower()
    aliases = {"deepseek": "dsh", "deepseek-harness": "dsh", "harness": "dsh", "zc": "zcode"}
    return aliases.get(v, v)


def validate_agent(value: str, allow_all: bool = False) -> str:
    v = _norm_agent(value)
    allowed = set(AGENTS) | ({BROADCAST} if allow_all else set())
    if v not in allowed:
        raise ValueError(f"unknown agent {value!r}; expected one of {sorted(allowed)}")
    return v


@contextmanager
def locked(path: Path):
    """Exclusive lock next to the message file (Linux/WSL and Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    fh = open(lock_path, "a+b")
    try:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            try:
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            except (ImportError, OSError):
                pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        fh.close()


def _read_all(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("id"):
            out.append(item)
    return out


def _write_all(path: Path, items: Iterable[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def post(
    *,
    sender: str,
    recipient: str,
    subject: str,
    body: str = "",
    kind: str = "note",
    priority: str = "normal",
    refs: list[str] | None = None,
    reply_to: str | None = None,
    root: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sender = validate_agent(sender)
    recipient = validate_agent(recipient, allow_all=True)
    if kind not in TYPES:
        raise ValueError(f"unknown type {kind!r}; expected one of {TYPES}")
    subject = re.sub(r"\s+", " ", str(subject or "")).strip()
    if not subject:
        raise ValueError("subject must not be empty")
    if len(subject) > 160:
        subject = subject[:157] + "..."
    path = bus_path(root)
    msg = {
        "id": make_id(),
        "ts": now(),
        "from": sender,
        "to": recipient,
        "type": kind,
        "priority": priority if priority in ("low", "normal", "high", "urgent") else "normal",
        "subject": subject,
        "body": str(body or "").strip(),
        "refs": [str(x) for x in (refs or []) if str(x).strip()],
        "status": "unread",
        "claimed_by": None,
        "claimed_at": None,
        "result": None,
    }
    if reply_to:
        msg["reply_to"] = reply_to
    if meta:
        msg["meta"] = meta
    with locked(path):
        existing = _read_all(path)
        existing.append(msg)
        _write_all(path, existing)
    return msg


def load(root: Path | None = None) -> list[dict[str, Any]]:
    path = bus_path(root)
    with locked(path):
        return _read_all(path)


def _visible(msg: dict[str, Any], agent: str) -> bool:
    return msg.get("to") in (agent, BROADCAST)


def inbox(
    agent: str,
    *,
    unread_only: bool = False,
    include_done: bool = False,
    limit: int | None = None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    agent = validate_agent(agent)
    items = [m for m in load(root) if _visible(m, agent)]
    if unread_only:
        items = [m for m in items if m.get("status") == "unread"]
    elif not include_done:
        items = [m for m in items if m.get("status") != "done"]
    items.sort(key=lambda m: (0 if m.get("priority") in ("urgent", "high") else 1, m.get("ts", ""), m.get("id", "")))
    if limit:
        items = items[:limit]
    return items


def read(msg_id: str, agent: str, root: Path | None = None) -> dict[str, Any]:
    # Keep the read-only path simple: allow marking read only if currently unread.
    path = bus_path(root)
    with locked(path):
        items = _read_all(path)
        for item in items:
            if item.get("id") == msg_id:
                if not _visible(item, agent):
                    raise PermissionError(f"{msg_id} is not addressed to {agent}")
                if item.get("status") == "unread":
                    item["status"] = "read"
                    item["updated_at"] = now()
                    _write_all(path, items)
                return item
    raise KeyError(f"message not found: {msg_id}")


def claim(msg_id: str, agent: str, root: Path | None = None) -> dict[str, Any]:
    path = bus_path(root)
    with locked(path):
        items = _read_all(path)
        for item in items:
            if item.get("id") == msg_id:
                if not _visible(item, agent):
                    raise PermissionError(f"{msg_id} is not addressed to {agent}")
                if item.get("status") == "done":
                    raise PermissionError(f"{msg_id} is already done")
                if item.get("status") == "claimed" and item.get("claimed_by") not in (None, agent):
                    raise PermissionError(f"{msg_id} is claimed by {item.get('claimed_by')}")
                item["status"] = "claimed"
                item["claimed_by"] = agent
                item["claimed_at"] = now()
                item["updated_at"] = now()
                _write_all(path, items)
                return item
    raise KeyError(f"message not found: {msg_id}")


def done(msg_id: str, agent: str, note: str = "", root: Path | None = None) -> dict[str, Any]:
    path = bus_path(root)
    with locked(path):
        items = _read_all(path)
        for item in items:
            if item.get("id") == msg_id:
                if not _visible(item, agent):
                    raise PermissionError(f"{msg_id} is not addressed to {agent}")
                if item.get("status") == "claimed" and item.get("claimed_by") != agent:
                    raise PermissionError(f"{msg_id} is claimed by {item.get('claimed_by')}")
                item["status"] = "done"
                item["claimed_by"] = agent
                item["done_at"] = now()
                item["updated_at"] = now()
                if note:
                    item["result"] = str(note).strip()
                _write_all(path, items)
                return item
    raise KeyError(f"message not found: {msg_id}")


def summary(root: Path | None = None) -> dict[str, Any]:
    items = load(root)
    by_agent: dict[str, dict[str, int]] = {}
    for agent in AGENTS:
        visible = [m for m in items if _visible(m, agent)]
        by_agent[agent] = {
            "unread": sum(1 for m in visible if m.get("status") == "unread"),
            "claimed": sum(1 for m in visible if m.get("status") == "claimed" and m.get("claimed_by") == agent),
            "done": sum(1 for m in visible if m.get("status") == "done"),
        }
    recent = sorted(items, key=lambda m: (m.get("ts", ""), m.get("id", "")), reverse=True)[:8]
    return {
        "path": str(bus_path(root)),
        "total": len(items),
        "active": sum(1 for m in items if m.get("status") != "done" and m.get("type") in ("request", "handoff")),
        "by_agent": by_agent,
        "recent": recent,
    }


def _print_table(items: list[dict[str, Any]]) -> None:
    if not items:
        print("(empty)")
        return
    for m in items:
        print(f"{m.get('id')}  {m.get('ts', '')[:19]}  {m.get('from'):>6} -> {m.get('to'):<6}  "
              f"{m.get('type'):<8} {m.get('status'):<8}  {m.get('subject')}")
        body = (m.get("body") or "").strip()
        if body:
            print("    " + body.replace("\n", "\n    ")[:1200])
        refs = m.get("refs") or []
        if refs:
            print("    refs: " + ", ".join(refs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ShuWei file-backed agent message bus")
    ap.add_argument("--root", type=Path, default=None, help="repo root (default: auto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("post", help="append a message")
    p.add_argument("--from", dest="sender", required=True)
    p.add_argument("--to", dest="recipient", required=True)
    p.add_argument("--type", dest="kind", default="note", choices=TYPES)
    p.add_argument("--priority", default="normal", choices=("low", "normal", "high", "urgent"))
    p.add_argument("--subject", required=True)
    p.add_argument("--body", default="")
    p.add_argument("--body-file", type=Path)
    p.add_argument("--ref", action="append", default=[])
    p.add_argument("--reply-to")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("inbox", help="list messages for an agent")
    p.add_argument("--agent", required=True)
    p.add_argument("--unread", action="store_true")
    p.add_argument("--include-done", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--json", action="store_true")

    for name in ("read", "claim", "done"):
        p = sub.add_parser(name, help=f"mark a message {name}")
        p.add_argument("id")
        p.add_argument("--agent", required=True)
        if name == "done":
            p.add_argument("--note", default="")
        p.add_argument("--json", action="store_true")

    p = sub.add_parser("summary", help="counts per agent")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("watch", help="poll for new messages")
    p.add_argument("--agent", required=True)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--once", action="store_true")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "post":
            body = args.body
            if args.body_file:
                body = args.body_file.read_text(encoding="utf-8")
            msg = post(sender=args.sender, recipient=args.recipient, subject=args.subject,
                       body=body, kind=args.kind, priority=args.priority, refs=args.ref,
                       reply_to=args.reply_to, root=args.root)
            print(json.dumps(msg, ensure_ascii=False, indent=2) if args.json else f"posted {msg['id']}")
            return 0
        if args.cmd == "inbox":
            items = inbox(args.agent, unread_only=args.unread, include_done=args.include_done,
                          limit=args.limit, root=args.root)
            if args.json:
                print(json.dumps(items, ensure_ascii=False, indent=2))
            else:
                _print_table(items)
            return 0
        if args.cmd == "read":
            item = read(args.id, args.agent, root=args.root)
        elif args.cmd == "claim":
            item = claim(args.id, args.agent, root=args.root)
        elif args.cmd == "done":
            item = done(args.id, args.agent, note=args.note, root=args.root)
        elif args.cmd == "summary":
            data = summary(args.root)
            if args.json:
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print(f"bus: {data['path']}  total={data['total']}")
                for agent, counts in data["by_agent"].items():
                    print(f"  {agent:<7} unread={counts['unread']:<3} claimed={counts['claimed']:<3} done={counts['done']}")
                print("recent:")
                _print_table(data["recent"])
            return 0
        elif args.cmd == "watch":
            seen: set[str] = set()
            while True:
                for item in inbox(args.agent, unread_only=True, root=args.root):
                    if item["id"] in seen:
                        continue
                    seen.add(item["id"])
                    print(f"\n[{item['ts']}] {item['from']} -> {item['to']}  {item['type']}  {item['subject']}")
                    if item.get("body"):
                        print(item["body"])
                if args.once:
                    return 0
                time.sleep(max(0.5, args.interval))
        else:
            ap.error("unknown command")
        if getattr(args, "json", False):
            print(json.dumps(item, ensure_ascii=False, indent=2))
        else:
            print(f"{item['id']} {item['status']} {item['subject']}")
        return 0
    except (ValueError, KeyError, PermissionError) as exc:
        print(f"bus error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

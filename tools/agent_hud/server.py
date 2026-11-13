#!/usr/bin/env python3
"""Local AI Ops dashboard for the ShuWei competition workspace.

Read-only. Serves a small HUD at 127.0.0.1 and exposes /api/state.
No network calls, no credentials, no writes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]
SKILL_NAME = "mary-modeling-contest-pipeline-shuwei"
DEFAULT_SKILL = Path.home() / ".codex" / "skills" / SKILL_NAME


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sections(markdown: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current = ""
    buf: list[str] = []
    for line in markdown.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current:
                out[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        elif current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf).strip()
    return out


def find_section(sec: dict[str, str], needle: str) -> str:
    for title, body in sec.items():
        if needle in title:
            return body
    return ""


def bullets(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.+?)\s*$", line)
        if m:
            items.append(re.sub(r"\s+", " ", m.group(1)).strip())
    return items


def git(root: Path, *args: str) -> str:
    try:
        p = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            capture_output=True,
            timeout=5,
        )
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:
        return ""


def skill_info() -> dict:
    path = Path(os.environ.get("SHUWEI_SKILL_DIR", str(DEFAULT_SKILL)))
    if not path.is_dir():
        return {"path": str(path), "exists": False, "files": 0, "updated": None}
    files = [p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    skill_md = path / "SKILL.md"
    updated = None
    if skill_md.is_file():
        updated = dt.datetime.fromtimestamp(skill_md.stat().st_mtime).isoformat(timespec="seconds")
    return {"path": str(path), "exists": True, "files": len(files), "updated": updated}


def codex_sessions() -> dict:
    root = Path.home() / ".codex" / "sessions"
    if not root.is_dir():
        return {"exists": False, "today": 0, "last": None}
    now = dt.datetime.now()
    today = 0
    last = None
    for p in root.rglob("*.jsonl"):
        try:
            m = dt.datetime.fromtimestamp(p.stat().st_mtime)
        except OSError:
            continue
        if m.date() == now.date():
            today += 1
        if last is None or m > last:
            last = m
    return {
        "exists": True,
        "today": today,
        "last": last.isoformat(timespec="seconds") if last else None,
    }


def build_state(root: Path) -> dict:
    cfg_path = HERE / "ai_flow.json"
    cfg = read_json(cfg_path, {})
    handoff_path = root / "docs" / "collab" / "CURSOR_HANDOFF.md"
    handoff_text = ""
    if handoff_path.is_file():
        handoff_text = handoff_path.read_text(encoding="utf-8", errors="replace")
    sec = sections(handoff_text)

    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = git(root, "status", "--short")
    last_commit = git(root, "log", "-1", "--format=%h %s")
    commits = [x for x in git(root, "log", "-5", "--format=%h %s").splitlines() if x]

    tasks = bullets(find_section(sec, "下一步"))
    blockers = bullets(find_section(sec, "阻塞"))
    ownership = bullets(find_section(sec, "正式件所有权"))

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repo": str(root),
        "phase": cfg.get("phase", "unknown"),
        "owner": cfg.get("owner", "codex"),
        "agents": cfg.get("agents", []),
        "flows": cfg.get("flows", []),
        "notes": cfg.get("notes", []),
        "handoff": {
            "path": str(handoff_path),
            "exists": handoff_path.is_file(),
            "age_minutes": int((dt.datetime.now() - dt.datetime.fromtimestamp(handoff_path.stat().st_mtime)).total_seconds() // 60) if handoff_path.is_file() else None,
            "tasks": tasks,
            "blockers": blockers,
            "ownership": ownership,
        },
        "git": {
            "branch": branch,
            "dirty": bool(status),
            "status": status.splitlines()[:20],
            "last_commit": last_commit,
            "recent": commits,
        },
        "skill": skill_info(),
        "codex_sessions": codex_sessions(),
    }


class Handler(BaseHTTPRequestHandler):
    root: Path
    server_version = "ShuWeiAgentHUD/1.0"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            body = json.dumps(build_state(self.root), ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path == "/healthz":
            self._send(200, b"ok", "text/plain; charset=utf-8")
            return
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
        }
        if path in files:
            name, ctype = files[path]
            p = HERE / name
            if p.is_file():
                self._send(200, p.read_bytes(), ctype)
                return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[hud] {fmt % args}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    root = args.root.resolve()
    Handler.root = root
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"AI Ops HUD: http://127.0.0.1:{args.port}/")
    print(f"repo: {root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstop")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

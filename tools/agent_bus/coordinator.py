#!/usr/bin/env python3
"""Watch the bus and let Codex/Cursor take turns automatically.

    python3 tools/agent_bus/coordinator.py --run --max-turns 6

Each turn:
1. find an unread `request`/`handoff` for codex or cursor;
2. call orchestrator.py for that agent (claim -> CLI -> reply -> done);
3. repeat until no work, max turns, max minutes, or a stop file appears.

This is still bounded: no infinite loop, no hidden retries, and formal-file
ownership rules remain in the orchestrator prompt. Default is dry-run.

Stop:
    touch docs/collab/AGENT_BUS_STOP
    # or Ctrl-C
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bus  # noqa: E402
import orchestrator  # noqa: E402

ORCH = HERE / "orchestrator.py"


def _pending(agent: str, root: Path) -> list[dict]:
    return [
        m for m in bus.inbox(agent, unread_only=True, root=root)
        if m.get("type") in ("request", "handoff")
    ]


def _cursor_ready() -> bool:
    exe = orchestrator._agent_exe("cursor")
    return bool(exe and orchestrator._cursor_auth_ok(exe))


def _run_turn(agent: str, root: Path, run: bool, timeout: int, role: str) -> int:
    cmd = [sys.executable, str(ORCH), "--root", str(root), "--agent", agent,
           "--role", role, "--timeout", str(timeout)]
    if run:
        cmd.append("--run")
    print(f"\n[{dt.datetime.now().strftime('%H:%M:%S')}] turn: {agent} ({role})", flush=True)
    proc = subprocess.run(cmd, text=True, timeout=timeout)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bounded Codex/Cursor bus coordinator")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--agents", default="codex,cursor")
    ap.add_argument("--role", default="build", choices=("build", "review"))
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-minutes", type=int, default=120)
    ap.add_argument("--max-failures", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--run", action="store_true", help="actually invoke models (default: dry-run)")
    ap.add_argument("--once", action="store_true", help="one turn, then exit")
    args = ap.parse_args(argv)

    root = (args.root or bus.default_root()).resolve()
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    for agent in agents:
        bus.validate_agent(agent)
    stop_file = root / "docs" / "collab" / "AGENT_BUS_STOP"
    started = dt.datetime.now()
    turns = 0
    failures = 0

    print(f"coordinator: root={root} agents={agents} run={args.run}")
    print(f"stop file : {stop_file}")

    if not args.run:
        for agent in agents:
            pending = _pending(agent, root)
            if pending:
                print(f"\n--- {agent}: {len(pending)} pending; dry-run first ---")
                _run_turn(agent, root, run=False, timeout=60, role=args.role)
        print("\nDry-run only. Pass --run to actually take turns.")
        return 0

    while turns < args.max_turns:
        if stop_file.exists():
            print(f"stop file found: {stop_file}")
            break
        if (dt.datetime.now() - started).total_seconds() > args.max_minutes * 60:
            print("max minutes reached")
            break

        acted = False
        for agent in agents:
            pending = _pending(agent, root)
            if not pending:
                continue
            if agent == "cursor" and not _cursor_ready():
                print("cursor: not logged in; skipping (run `agent login` once)")
                continue
            turns += 1
            acted = True
            code = _run_turn(agent, root, run=True, timeout=args.timeout, role=args.role)
            if code == 0:
                failures = 0
            else:
                failures += 1
                print(f"turn failed: exit {code} (failure {failures}/{args.max_failures})")
            if args.once or failures >= args.max_failures:
                print("coordinator stopping")
                return 0 if failures == 0 else 1
            break
        if args.once:
            break
        if not acted:
            time.sleep(max(1.0, args.interval))
    print(f"coordinator done: turns={turns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Minimal stdio MCP server for the ShuWei agent bus.

Configure it in Codex/Cursor so agents can send and receive messages with tool
calls instead of shell commands:

    [mcp_servers.agent_bus]
    command = "python3"
    args = ["/home/mary/shuwei2026/tools/agent_bus/mcp_server.py", "--agent", "codex"]

Cursor project config (`.cursor/mcp.json`) uses the same command/args shape.

MCP stdio is newline-delimited JSON-RPC 2.0. Logs go to stderr only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bus  # noqa: E402

PROTOCOL = "2024-11-05"
SERVER = {"name": "shuwei-agent-bus", "version": "1.0.0"}


def _schema(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS = [
    {
        "name": "bus_post",
        "description": "Send a message to another agent. Use type=request when you need work, reply when answering, handoff when ending a turn.",
        "inputSchema": _schema(
            {
                "to": {"type": "string", "enum": list(bus.AGENTS) + [bus.BROADCAST]},
                "subject": {"type": "string", "description": "One-line subject"},
                "body": {"type": "string", "description": "Full message; include paths and acceptance criteria"},
                "type": {"type": "string", "enum": list(bus.TYPES), "default": "note"},
                "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"], "default": "normal"},
                "refs": {"type": "array", "items": {"type": "string"}, "description": "Files/commits/URLs"},
                "reply_to": {"type": "string", "description": "Message id being answered"},
            },
            ["to", "subject"],
        ),
    },
    {
        "name": "bus_inbox",
        "description": "List messages for an agent (default: this agent). Unread only by default.",
        "inputSchema": _schema(
            {
                "agent": {"type": "string", "enum": list(bus.AGENTS)},
                "unread": {"type": "boolean", "default": True},
                "include_done": {"type": "boolean", "default": False},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            }
        ),
    },
    {
        "name": "bus_read",
        "description": "Mark a message as read and return it.",
        "inputSchema": _schema({"id": {"type": "string"}, "agent": {"type": "string", "enum": list(bus.AGENTS)}}, ["id"]),
    },
    {
        "name": "bus_claim",
        "description": "Claim a message before working on it, so two agents do not duplicate the task.",
        "inputSchema": _schema({"id": {"type": "string"}, "agent": {"type": "string", "enum": list(bus.AGENTS)}}, ["id"]),
    },
    {
        "name": "bus_done",
        "description": "Mark a claimed request done, with a short result note.",
        "inputSchema": _schema(
            {"id": {"type": "string"}, "agent": {"type": "string", "enum": list(bus.AGENTS)}, "note": {"type": "string"}},
            ["id"],
        ),
    },
    {
        "name": "bus_summary",
        "description": "Show per-agent unread/claimed counts and the most recent messages.",
        "inputSchema": _schema({}),
    },
]


class Server:
    def __init__(self, identity: str, root: Path | None) -> None:
        self.identity = bus.validate_agent(identity)
        self.root = root

    def _agent(self, args: dict[str, Any]) -> str:
        agent = bus.validate_agent(args.get("agent") or self.identity)
        return agent

    def _post(self, args: dict[str, Any]) -> Any:
        # The MCP server identity is the sender; callers cannot impersonate.
        msg = bus.post(
            sender=self.identity,
            recipient=args["to"],
            subject=args["subject"],
            body=args.get("body", ""),
            kind=args.get("type", "note"),
            priority=args.get("priority", "normal"),
            refs=args.get("refs") or [],
            reply_to=args.get("reply_to"),
            root=self.root,
        )
        return {"message": msg}

    def _inbox(self, args: dict[str, Any]) -> Any:
        items = bus.inbox(
            self._agent(args),
            unread_only=args.get("unread", True),
            include_done=args.get("include_done", False),
            limit=args.get("limit"),
            root=self.root,
        )
        return {"count": len(items), "messages": items}

    def _read(self, args: dict[str, Any]) -> Any:
        return {"message": bus.read(args["id"], self._agent(args), root=self.root)}

    def _claim(self, args: dict[str, Any]) -> Any:
        return {"message": bus.claim(args["id"], self._agent(args), root=self.root)}

    def _done(self, args: dict[str, Any]) -> Any:
        return {"message": bus.done(args["id"], self._agent(args), note=args.get("note", ""), root=self.root)}

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        handlers = {
            "bus_post": self._post,
            "bus_inbox": self._inbox,
            "bus_read": self._read,
            "bus_claim": self._claim,
            "bus_done": self._done,
            "bus_summary": lambda _a: bus.summary(self.root),
        }
        if name not in handlers:
            raise KeyError(f"unknown tool: {name}")
        return handlers[name](args)

    def handle(self, req: dict[str, Any]) -> dict[str, Any] | None:
        method = req.get("method")
        req_id = req.get("id")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER,
                },
            }
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
        if method == "tools/call":
            params = req.get("params") or {}
            name = params.get("name", "")
            args = params.get("arguments") or {}
            try:
                payload = self.call_tool(name, args)
                text = json.dumps(payload, ensure_ascii=False, indent=2)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": text}], "isError": False},
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True},
                }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="codex", choices=bus.AGENTS)
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args()
    server = Server(args.agent, args.root)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}), flush=True)
            continue
        if not isinstance(req, dict):
            continue
        resp = server.handle(req)
        if resp is not None:
            print(json.dumps(resp, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# agent_bus

File-backed message bus + bounded Codex/Cursor turn runner for `~/shuwei2026`.

- Protocol and safety boundary: `docs/collab/AGENT_BUS.md`
- Message file: `docs/collab/AGENT_MESSAGES.jsonl` (public repo: no secrets)
- HUD reads `bus.summary()` and shows `BUS n` / per-agent unread.

## Quick start

```bash
cd ~/shuwei2026

# inspect
python3 tools/agent_bus/bus.py summary
python3 tools/agent_bus/bus.py inbox --agent codex --unread

# send a request
python3 tools/agent_bus/bus.py post --from cursor --to codex \
  --type request --subject "Review table 3 sources" \
  --body "Read-only; cite file:line; do not edit the paper."

# one dry-run (no model call)
python3 tools/agent_bus/orchestrator.py --agent codex --role build

# one real turn
python3 tools/agent_bus/orchestrator.py --agent codex --role build --run

# bounded relay (Codex + logged-in Cursor)
python3 tools/agent_bus/coordinator.py --run --max-turns 6
touch docs/collab/AGENT_BUS_STOP   # stop
```

## Cursor

```bash
export PATH="$HOME/.local/bin:$PATH"
agent login          # one time
agent status
```

## Tests

```bash
python3 tools/agent_bus/test_bus.py -q
python3 tools/agent_bus/test_orchestrator.py -q
```

## Files

| File | Purpose |
|---|---|
| `bus.py` | JSONL store, CLI, locking, status transitions |
| `mcp_server.py` | stdio MCP tools `bus_*` for Codex/Cursor |
| `orchestrator.py` | one message → one `codex exec` / `cursor-agent -p` |
| `coordinator.py` | bounded loop over unread requests |
| `test_bus.py` | bus + MCP tests |
| `test_orchestrator.py` | prompt/follow-up tests |

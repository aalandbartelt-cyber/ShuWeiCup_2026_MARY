# Agent Bus — 多 Agent 文件消息总线

> 目标：Codex / Cursor / ZCode / dsh 之间不靠人复制粘贴，也不靠“切窗口传话”。
> 边界：这是**消息队列 + 单次编排器**，不是常驻聊天室，也不会凭空唤醒一个关着的 Agent。

## 1. 它解决什么

| 现在 | 加 bus 之后 |
|---|---|
| 用户把 Codex 的输出复制给 Cursor | Codex `post` 一条 request，Cursor 下一轮 `inbox` 直接读到 |
| 两个 Agent 抢同一件事 | `claim` 先到先得，另一方看到 `claimed_by` |
| 交接靠 `CURSOR_HANDOFF.md` 一段话 | handoff 文件仍保留，bus 负责可追踪的 request/reply/decision |
| HUD 只能看状态 | HUD 显示未读/已认领数量和最新主题 |

它**不解决**：Cursor 没开时自动打开 Cursor；Codex 没额度时自动换号；多个 Agent 同时改同一文件不冲突。这些分别需要 GUI 自动化、额度/账号策略、worktree/锁。

## 2. 文件与状态

消息文件（入库，便于审计）：

```text
docs/collab/AGENT_MESSAGES.jsonl
```

一行一条 JSON。关键字段：

```json
{
  "id": "MSG-20261113-181500-a1b2",
  "ts": "2026-11-13T18:15:00+08:00",
  "from": "codex",
  "to": "cursor",
  "type": "request",
  "priority": "normal",
  "subject": "Review Q2 figure 3",
  "body": "只做审阅，不改正式件；验收标准……",
  "refs": ["05_论文/待审/Q2/fig3.png"],
  "status": "unread",
  "claimed_by": null,
  "claimed_at": null,
  "result": null
}
```

`type`：`request` / `reply` / `note` / `handoff` / `decision` / `review`
`status`：`unread` → `read` → `claimed` → `done`
`to: "all"` 是广播；广播也会出现在每个人的 inbox 里。

消息文件是**公开仓库**的一部分：禁止写密钥、token、个人隐私、未公开的赛题原文。

## 3. 每个 Agent 的一轮协议

**开始一轮**

```bash
python3 tools/agent_bus/bus.py inbox --agent codex --unread
```

有 request/handoff 时先 `claim`：

```bash
python3 tools/agent_bus/bus.py claim MSG-... --agent codex
```

**结束一轮**

```bash
python3 tools/agent_bus/bus.py post --from codex --to cursor \
  --type reply --reply-to MSG-... \
  --subject "Re: Review Q2 figure 3" \
  --body "已改 panel B；证据：..." \
  --ref docs/collab/rounds/MSG-...-codex.md

python3 tools/agent_bus/bus.py done MSG-... --agent codex --note "已交付，等复核"
```

规则：

1. 只 `claim` 自己 `to` 的消息；广播谁先 claim 谁做。
2. 做不完也要回一条 `reply` 说明卡点，再 `done` 或保持 `claimed`。
3. `done` 不是“删掉”，是“这件事有结论”；`result` 写一句可核验的结论。
4. 正式论文/正式图/交付包仍按 `AGENTS.md` 的所有权规则，不因为一条 bus 消息就自动越界。

## 4. MCP 接入（推荐，Agent 用工具调用而不是 shell）

`tools/agent_bus/mcp_server.py` 是一个最小 stdio MCP server，暴露：

- `bus_post`
- `bus_inbox`
- `bus_read`
- `bus_claim`
- `bus_done`
- `bus_summary`

### Codex

在 `~/.codex/config.toml` 增加（路径按本机）：

```toml
[mcp_servers.agent_bus]
command = "python3"
args = ["/home/mary/shuwei2026/tools/agent_bus/mcp_server.py", "--agent", "codex"]
```

重启 Codex 会话后，工具列表里应出现 `bus_*`。若没有，先手工跑一次确认 server 不报错：

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | python3 tools/agent_bus/mcp_server.py --agent codex
```

### Cursor

项目级 `.cursor/mcp.json`（Cursor 在 WSL 远程窗口里读的是 WSL 路径）：

```json
{
  "mcpServers": {
    "agent_bus": {
      "command": "python3",
      "args": ["/home/mary/shuwei2026/tools/agent_bus/mcp_server.py", "--agent", "cursor"]
    }
  }
}
```

Cursor 的 MCP 工具在**一轮对话结束时**才会处理新消息；它不会在空闲时被 bus 唤醒。要在不切窗口的情况下让 Cursor 接手，用第 5 节的 `cursor-agent -p` 编排器（需一次 `agent login`），而不是只配 MCP。

## 5. 编排器（自动跑一轮）

`tools/agent_bus/orchestrator.py` 把一条 bus 消息变成一次非交互 Agent 运行：

```bash
# Codex
python3 tools/agent_bus/orchestrator.py --agent codex --role build            # dry-run
python3 tools/agent_bus/orchestrator.py --agent codex --role build --run

# Cursor（需先一次性登录，见下）
python3 tools/agent_bus/orchestrator.py --agent cursor --role build            # dry-run
python3 tools/agent_bus/orchestrator.py --agent cursor --role build --run
```

指定消息：

```bash
python3 tools/agent_bus/orchestrator.py --agent codex --message MSG-... --run
```

审阅角色（Cursor 会用 `--mode plan`，只读）：

```bash
python3 tools/agent_bus/orchestrator.py --agent codex --role review --message MSG-... --run
```

### Cursor 一次性登录

```bash
export PATH="$HOME/.local/bin:$PATH"
agent login
agent status        # 应显示已登录
```

`cursor-agent` 已安装在 `~/.local/bin/cursor-agent`（`agent` 是同一个二进制）。不登录时编排器会拒绝执行，不会把消息标成失败。

### 连续接力（coordinator）

```bash
# 先 dry-run 看它会接哪些消息
python3 tools/agent_bus/coordinator.py

# 真正连续跑，最多 6 轮 / 120 分钟
python3 tools/agent_bus/coordinator.py --run --max-turns 6 --max-minutes 120

# 停止
touch docs/collab/AGENT_BUS_STOP
```

coordinator 只处理 `request` / `handoff`，每轮只叫一个 Agent；一旦失败达到 `--max-failures` 或超过轮数/时间就退出。它不会自己发明任务。

### Agent 之间怎么“接上话”

一次 Agent 运行的结尾可以写：

```text
RESULT: ...
EVIDENCE: ...
NEXT: 复核 panel B 的误差棒
NEXT_AGENT: cursor
```

编排器会把 `NEXT_AGENT` 变成一条新的 `request`（`Follow-up: ...`），coordinator 下一轮就会叫 Cursor；Cursor 也可以同样把下一步交回 Codex。这样是**有向接力**，不是四个 Agent 同时聊天；轮数由 `--max-turns` 封顶，避免互相踢皮球。

不要在自己的回合里直接 `bus_post` 给另一个 Agent——让编排器记录 `NEXT_AGENT`，才能保证 transcript、reply、done 一致。

安全设计：

- 默认 **dry-run**，必须显式 `--run`。
- 只处理 `request` / `handoff`，不碰 `reply` / `note`。
- 一次只跑一条，不写无限循环。
- 失败会写 transcript 并把消息 `done` 为 `FAILED`；加 `--keep-claimed-on-failure` 可保留 `claimed` 方便重试。
- 正式件是否可改，仍由消息正文明确授权 + `AGENTS.md` 决定。
- `--agent zcode|dsh` 会直接拒绝：本机没有可验证的 headless CLI，不能假装能自动驱动。

## 6. HUD

`tools/agent_hud/server.py` 会读 bus 摘要，HUD 顶部出现 `BUS n`，点开任一 Agent 可看到该 Agent 的未读/已认领数和最新主题。HUD 只读，不 post。

## 7. 最小启动示例

```bash
# Cursor 给 Codex 派活
python3 tools/agent_bus/bus.py post --from cursor --to codex \
  --type request --subject "核对 Q1 表 3 的数字来源" \
  --body "只读核对；给出文件:行号；不要改正文。" \
  --ref 05_论文/待审/Q1/表3.md

# Codex 侧（或编排器）看到后 claim
python3 tools/agent_bus/bus.py inbox --agent codex --unread
python3 tools/agent_bus/bus.py claim MSG-... --agent codex

# 做完
python3 tools/agent_bus/bus.py done MSG-... --agent codex --note "3 处来源已核对，2 处需补"
```

## 8. 什么时候不要用

- 需要即时对话/辩论：用同一轮里的多角色 prompt，不要指望四个 GUI Agent 实时聊天。
- 需要正式论文改写：仍走 Codex，bus 只传任务和结论。
- 需要保密内容：不要进 `AGENT_MESSAGES.jsonl`（public 仓库）；改用本地忽略文件并只传指针。

# docs/collab 文件信箱

> 参照妈妈杯：用文件交接，不依赖对话上下文。静默模式：只在交接/阻塞/决策时发消息。

## 角色

| 角色 | 位置 | 职责 |
|---|---|---|
| Codex | WSL / VS Code Remote-WSL | 默认主环境：建模、代码、结果、正式论文/包、`main` |
| Cursor | WSL | **Codex 无额度/不可用时的备份主力**：接手建模/代码/结果/图表/论文候选；写 `cursor/` 分支与 `05_论文/待审/`；不能恢复 Codex thread，用 `CURSOR_HANDOFF.md` |
| ZCode | Windows | 副环境：数据、文档、复核、图稿候选、提交检查 |
| dsh | Windows | DeepSeek Harness（`C:\Users\mary3\AppData\Local\Programs\DeepSeek Harness\DeepSeek Harness.exe`）；只做论文评审/文字优化，写 `reviews/` |

## 文件

| 文件 | 写入者 | 规则 |
|---|---|---|
| `board.md` | 双方维护区 | 任务/状态/阻塞/活动日志 |
| `CURSOR_HANDOFF.md` | Codex/Cursor 更新 | Cursor 接手/交回的唯一上下文；不依赖 Codex thread |
| `to_codex.md` | 只有 ZCode | Codex 只改状态行 |
| `to_zcode.md` | 只有 Codex | ZCode 只改状态行 |
| `decisions.md` | 双方追加 | D-xxx，不删除 |
| `reviews/` | dsh | `REV-YYYYMMDD-NN.md` |
| `AGENT_MESSAGES.jsonl` | 各 Agent 追加 | 机器可读消息总线；协议见 `AGENT_BUS.md` |
| `rounds/` | 编排器 | 每次自动运行的 prompt/stdout/stderr 记录 |

## 消息格式

```markdown
## MSG-YYYYMMDD-HHMM-xx | 类型 | 状态
- 发起 / 接收 / 时间 / 关联
**标题**
**内容**（背景、要求、约束、路径）
**验收**（命令、产物、指标）
**回复**（接收方填）
```

状态：`OPEN → ACK → DONE / BLOCKED / PAUSED`。

## 机器可读总线（推荐）

跨 Agent 的任务/回复优先走 `AGENT_MESSAGES.jsonl`，不要只写 markdown 信箱：

```bash
python3 tools/agent_bus/bus.py inbox --agent codex --unread
python3 tools/agent_bus/bus.py claim MSG-... --agent codex
python3 tools/agent_bus/bus.py done MSG-... --agent codex --note "..."
```

完整协议、MCP 配置、自动编排与安全边界见 `AGENT_BUS.md`。

## 边界

- 正式论文/图/包默认只由 Codex 写；ZCode 只写 `05_论文/待审/`、`docs/audit/`。
- Cursor 默认只写 `05_论文/待审/` 与 `cursor/` 分支；Codex 无额度/不可用时作为备份主力接手核心工作，正式件需用户明确授权并记录 `decisions.md`。
- Cursor 是 AI 工具，生成内容按 skill `references/18` 记录/声明/报告。
- dsh 不碰代码/结果/正式稿。
- 候选转正式必须过 skill `references/common/13` 的五条件。
- 大文件/参考 PDF 放 D 盘，不入库；消息只写路径。

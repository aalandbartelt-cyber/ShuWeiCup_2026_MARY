# docs/collab 文件信箱

> 参照妈妈杯：用文件交接，不依赖对话上下文。静默模式：只在交接/阻塞/决策时发消息。

## 角色

| 角色 | 位置 | 职责 |
|---|---|---|
| Codex | WSL / VS Code Remote-WSL | 主环境：建模、代码、结果、正式论文/包、`main` |
| ZCode | Windows | 副环境：数据、文档、复核、图稿候选、提交检查 |
| dsh | Windows | 只做论文评审/文字优化，写 `reviews/` |

## 文件

| 文件 | 写入者 | 规则 |
|---|---|---|
| `board.md` | 双方维护区 | 任务/状态/阻塞/活动日志 |
| `to_codex.md` | 只有 ZCode | Codex 只改状态行 |
| `to_zcode.md` | 只有 Codex | ZCode 只改状态行 |
| `decisions.md` | 双方追加 | D-xxx，不删除 |
| `reviews/` | dsh | `REV-YYYYMMDD-NN.md` |

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

## 边界

- 正式论文/图/包只由 Codex 写；ZCode 只写 `05_论文/待审/`、`docs/audit/`。
- dsh 不碰代码/结果/正式稿。
- 候选转正式必须过 skill `references/common/13` 的五条件。
- 大文件/参考 PDF 放 D 盘，不入库；消息只写路径。

# Agent Instructions — ShuWei Cup 2026

本仓库是 2026 数维杯比赛工作区。所有 AI Agent 先读本文件，再读指定 skill，再动文件。

## 角色

| 角色 | 位置 | 职责 |
|---|---|---|
| **Codex** | WSL | 默认主环境：建模、代码、结果、正式论文/包、`main` 集成 |
| **Cursor** | WSL | **Codex 无额度/不可用时的备份主力**：接手建模/代码/结果/图表/论文候选；写 `cursor/` 分支与 `05_论文/待审/` |
| **ZCode** | Windows | 辅助：数据、文档、复核、图稿候选、提交检查 |
| **dsh** | Windows | DeepSeek Harness，路径 `C:\Users\mary3\AppData\Local\Programs\DeepSeek Harness\DeepSeek Harness.exe`；只做论文评审/文字优化，写 `docs/collab/reviews/` |

## 必读 skill

WSL 路径：

```text
/home/mary/.codex/skills/mary-modeling-contest-pipeline-shuwei/SKILL.md
```

再按任务读 `references/01`、`02`、`05`、`06`、`11`、`12`、`13`、`15`、`18`、`19`、`20`、`21`。

Cursor 不会自动加载 Codex skill；必须显式读上述路径。项目规则见 `.cursor/rules/mary-shuwei-skill.mdc`。

## 论文与图表任务必读

写作、绘图、整合或审稿前，必须读取项目内的最新用户反馈规范：

- [论文写作与审稿规范](02_口径与约定/论文写作与审稿规范.md)：用户意见、已实施修改、后续验收，以及全文暂缓项。
- [图表规范](02_口径与约定/图表规范.md)：字体、配色、图例、放大、数值标注、尺寸和图文混排。

其中“当前必须保持的状态”优先于历史 handoff 中的旧安排。不得自动补回暂空的摘要、关键词、全文第 7/8 节，或提前整理附录 A；后续用户新指令到达时同步更新规范。

## Cursor 接手规则

1. 确认终端在 WSL：`uname -a` 为 Linux，`pwd` 为 `/home/mary/shuwei2026`。
2. 先读 `docs/collab/CURSOR_HANDOFF.md` + `git status --short --branch`。
3. 从 handoff 的“下一步”继续，不重做已完成部分；不要尝试恢复 Codex thread。
4. 在 `cursor/<scope>-<topic>` 分支工作；不直接改 `main`。
5. 正式论文/正式图/交付包默认仍由 Codex 写；Cursor 要写正式件必须由用户明确授权并记录在 `docs/collab/decisions.md`。
6. Cursor 是 AI 工具：生成内容按 skill `references/18_AI工具合规与使用报告.md` 标注、声明、写入 AI 报告。

## 数据与安全

- 原始数据/参考 PDF/大文件在 D 盘或忽略目录，不入库；不提交密钥、token、个人信息。
- 不改 `05_论文/正式`、`06_交付` 的正式件，除非用户明确授权。
- 提交前跑 `python scripts/audit_git_hygiene.py --root .` 和相关测试。

## 交接

- 跨 Agent 交接只写文件，不依赖对话上下文。
- 每次接手/交付更新 `docs/collab/CURSOR_HANDOFF.md`（或对应 handoff）。
- 跨 Agent 消息走 `docs/collab/AGENT_MESSAGES.jsonl`（见 `docs/collab/AGENT_BUS.md`）：
  - 开轮先 `python3 tools/agent_bus/bus.py inbox --agent <self> --unread`；
  - 接任务先 `claim`，做完 `done`，做不完回 `reply` 说明卡点；
  - 可用 `mcp_server.py` 把 `bus_*` 工具接进 Codex/Cursor；
  - bus 只传任务/结论/指针，不传密钥、隐私、未公开赛题原文；正式件所有权不变。
- 可选本地只读面板：`tools/agent_hud/server.py` → `http://127.0.0.1:8765/`；它显示 bus 未读数与 handoff，但不替代 handoff，也不会自动唤醒 Agent。
- 需要“自动跑一轮”时用 `tools/agent_bus/orchestrator.py`（默认 dry-run，`--run` 才调模型；支持 Codex 和已登录的 Cursor Agent CLI）；连续接力用 `tools/agent_bus/coordinator.py`，有轮数/时间/失败上限。
- 在编排器里要交给下一个 Agent 时，结尾写 `NEXT: ...` + `NEXT_AGENT: cursor`（或 codex/zcode/dsh），不要自己 `bus_post`；编排器会建 follow-up request。
- 冲突时保留双方内容，重新编号/重跑测试，不覆盖对方未合并工作。

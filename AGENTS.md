# Agent Instructions — ShuWei Cup 2026

本仓库是 2026 数维杯比赛工作区。所有 AI Agent 先读本文件，再读指定 skill，再动文件。

## 角色

| 角色 | 位置 | 职责 |
|---|---|---|
| **Codex** | WSL | 默认主环境：建模、代码、结果、正式论文/包、`main` 集成 |
| **Cursor** | WSL | **Codex 无额度/不可用时的备份主力**：接手建模/代码/结果/图表/论文候选；写 `cursor/` 分支与 `05_论文/待审/` |
| **ZCode** | Windows | 辅助：数据、文档、复核、图稿候选、提交检查 |
| **dsh** | Windows | 只做论文评审/文字优化，写 `docs/collab/reviews/` |

## 必读 skill

WSL 路径：

```text
/home/mary/.codex/skills/mary-modeling-contest-pipeline-shuwei/SKILL.md
```

再按任务读 `references/01`、`02`、`05`、`06`、`11`、`12`、`13`、`15`、`18`、`19`、`20`、`21`。

Cursor 不会自动加载 Codex skill；必须显式读上述路径。项目规则见 `.cursor/rules/mary-shuwei-skill.mdc`。

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
- 冲突时保留双方内容，重新编号/重跑测试，不覆盖对方未合并工作。

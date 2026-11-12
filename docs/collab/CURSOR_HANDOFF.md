# CURSOR_HANDOFF

> 交接文件，不依赖对话上下文。Cursor 每次接手先读本文件；Codex 恢复后也先读本文件。

## 1. 当前状态（2026-11-13）

- 仓库：`~/shuwei2026`
- 当前本地分支：`main`（以 `git status` 实际为准）
- 远端：`origin` = `ShuWeiCup_2026_MARY`（public）
- 远端 `main` 最新（GitHub API）：`e2a6b07 docs(delivery): add AI tool usage report template`
- 注意：本地 `main` 可能有一个内容相同但 hash 不同的本地提交（API 更新导致本地/远端分叉）；**不要盲目 force push**。先 `git status --short --branch` 和 `git log --oneline -5`。
- 本地 `zcode/fig-r2-visual` 分支保留 R2 视觉分析，未合并；不要覆盖。
- 正式论文/正式图/交付包：尚未开始；`05_论文/正式/`、`06_交付/` 仍由 Codex 默认所有。

## 2. Skill 路径

```text
/home/mary/.codex/skills/mary-modeling-contest-pipeline-shuwei/SKILL.md
```

Cursor 不自动加载；必须显式读。任务相关再读 `references/01,02,05,06,11,12,13,15,18,19,20,21`。项目规则：`.cursor/rules/mary-shuwei-skill.mdc`。

## 3. 当前目标与验收

- 当前不是比赛日；没有进行中的正式论文任务。
- 已完成的准备：官方格式/模板、AI 报告模板、专家写作/流程规则、skill 55 文件、双端同步。
- 比赛开始后，从用户粘贴的赛题与 `00_赛题与规则/赛事参数.md` 起，按 skill 的 4 天流程执行。

## 4. 已完成（证据）

- 官方文件：`00_赛题与规则/2026官方更新_20261111/`
- AI 报告模板：`06_交付/04_支撑材料/AI_Tool_Usage_Report_Template.docx`
- Skill 双端：WSL `~/.codex/skills/...`、Windows `~/.agents/skills/...`；各 55 文件。
- 最近提交（skill 私有仓库）：`8f5ae05 feat(skill): add expert review and writing guidance`
- 最近提交（比赛 public）：`e2a6b07 docs(delivery): add AI tool usage report template`

## 5. 下一步 3 件

1. 比赛开始后，确认题号/队号/截止时间写入 `00_赛题与规则/赛事参数.md`。
2. 建 `cursor/<scope>-<topic>` 分支；先跑 `uname -a`、`git status`、skill 自检。
3. 按 skill 的问题链→数据→模型→结果→论文流程推进；每完成一个阶段更新本文件。

## 6. 数据/模型路径

- 比赛数据/参考：D 盘只读资料，按 `01_参考学习/README.md`；不要把原始数据/PDF 提交进 git。
- 当前无训练/结果产物；比赛时由 Codex/Cursor 在 WSL 生成，路径写入本文件。

## 7. 已定决策 / 禁止事项

- Codex 默认主力；Cursor 在 Codex 无额度/不可用时备份主力。
- Cursor 不能恢复 Codex thread；用本 handoff 接手。
- Cursor 默认只写 `05_论文/待审/` 与 `cursor/` 分支；正式件需用户明确授权并记录 `decisions.md`。
- Cursor 是 AI 工具；AI 使用按 skill `references/18` 记录、声明、报告。
- 不改 `05_论文/正式/`、`06_交付/` 正式件；不覆盖对方未提交工作。
- 原始数据/参考 PDF/密钥/个人信息不入库。

## 8. 阻塞与需要用户决定

- 实际报名/提交系统未登录核验（文件槽位、重复提交、附件限制）。
- 若 Codex 在截止前不可用，正式件是否由 Cursor 写，需用户明确授权。
- 若 GitHub 直连 TLS 失败，使用 `gh api` 更新或等网络恢复；不要 force push。

## 9. 正式件所有权状态

- `05_论文/正式/`：Codex（默认）
- `05_论文/待审/`：Cursor/ZCode 可写
- `06_交付/`：Codex（默认）；Cursor 需授权
- `docs/collab/reviews/`：dsh

## 10. 使用提示

Cursor 固定开场：

```text
先读 AGENTS.md、docs/collab/CURSOR_HANDOFF.md、skill 的 SKILL.md；
确认 uname -a 为 Linux、当前分支和未提交改动。
只从“下一步 3 件”继续，不重做已完成部分。
改动前说明计划；完成后跑测试并更新本 handoff。
正式件默认不覆盖；使用 AI 按 references/18 记录。
```

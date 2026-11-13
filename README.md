# ShuWei Cup 2026 · MARY 比赛仓库

数维杯 2026 秋季赛（英文论文，A—D 任选一题，25 页含 Summary/Contents）的比赛工作区。
Skill：`C:\Users\mary3\.agents\skills\mary-modeling-contest-pipeline-shuwei`（Windows） / `~/.codex/skills/mary-modeling-contest-pipeline-shuwei`（WSL）。

## 环境与职责

| 角色 | 位置 | 职责 | 不做什么 |
|---|---|---|---|
| **Codex** | WSL，VS Code Remote-WSL | 默认主环境：建模/代码/结果/正式论文/正式包；`main` 集成 | 不把正式路径交给辅助角色 |
| **Cursor** | WSL | **Codex 无额度/不可用时的备份主力**：接手建模/代码/结果/图表/论文候选；写 `cursor/` 分支与 `05_论文/待审/`；不能恢复 Codex thread，用 `CURSOR_HANDOFF.md` | 默认不改 `05_论文/正式`、`06_交付`；正式件需用户明确授权 |
| **ZCode** | Windows | 副环境：数据、文档、复核、图稿候选、提交检查 | 不直接改 `05_论文/正式`、`06_交付` |
| **dsh** | Windows | 论文评审/文字优化，只写 `docs/collab/reviews/` | 不改代码/结果/正式稿 |

- WSL 主路径：`~/shuwei2026`
- Windows 访问：`\\wsl.localhost\Ubuntu-22.04\home\mary\shuwei2026`
- 赛题/参考大文件放在 `D:\上海大学__\2026第十二届数维杯`，不入库；见 `01_参考学习/README.md`。

## 先读

1. `AGENTS.md`（Codex/Cursor 分工与边界）
2. `00_赛题与规则/赛事参数.md`
3. `02_口径与约定/Git规范.md`
4. `docs/collab/README.md` 与 `docs/collab/CURSOR_HANDOFF.md`
5. `00_赛题与规则/2026官方更新_20261111/`（新格式/提交规范、Summary 模板、LaTeX 模板）
6. `05_论文/正式/` 默认只由 Codex 写
7. Skill 的 `SKILL.md` 与 `references/09_十轮精读计划.md`

## 快速命令

```bash
cd ~/shuwei2026
git pull --ff-only
python scripts/env_doctor.py
python scripts/audit_git_hygiene.py --root .
```

## AI Ops HUD（可选）

本地只读面板，显示 Codex/Cursor/ZCode/dsh 的 owner、handoff 下一步、阻塞、分支与 skill 版本：

```bash
cd ~/shuwei2026
python3 tools/agent_hud/server.py
# http://127.0.0.1:8765/
```

它参考 `agent-hud-open` 的状态卡/额度色思路，但不读密钥、不调用外部 API、不替代 `CURSOR_HANDOFF.md`；额度色在 `tools/agent_hud/ai_flow.json` 手动维护。

## 提交

```bash
git switch -c codex/q1-<topic>
# 改动 + 证据
git add ...
git commit -m "feat(q1): ..."
git push -u origin codex/q1-<topic>
# Codex 合并到 main，打 gate 标签
```

**不要**把原始数据、权重、参考 PDF、`_archive/`、环境目录提交进 git。

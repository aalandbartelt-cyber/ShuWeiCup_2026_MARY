# ShuWei Cup 2026 · MARY 比赛仓库

数维杯 2026 秋季赛（英文论文，A—D 任选一题，25 页含 Summary/Contents）的比赛工作区。
Skill：`C:\Users\mary3\.agents\skills\mary-modeling-contest-pipeline-shuwei`（Windows） / `~/.codex/skills/mary-modeling-contest-pipeline-shuwei`（WSL）。

## 环境与职责

| 角色 | 位置 | 职责 | 不做什么 |
|---|---|---|---|
| **Codex** | WSL，VS Code Remote-WSL | 主环境：建模/代码/结果/正式论文/正式包；`main` 集成 | 不把正式路径交给辅助角色 |
| **ZCode** | Windows | 副环境：数据、文档、复核、图稿候选、提交检查 | 不直接改 `05_论文/正式`、`06_交付` |
| **dsh** | Windows | 论文评审/文字优化，只写 `docs/collab/reviews/` | 不改代码/结果/正式稿 |

- WSL 主路径：`~/shuwei2026`
- Windows 访问：`\\wsl.localhost\Ubuntu-22.04\home\mary\shuwei2026`
- 赛题/参考大文件放在 `D:\上海大学__\2026第十二届数维杯`，不入库；见 `01_参考学习/README.md`。

## 先读

1. `00_赛题与规则/赛事参数.md`
2. `02_口径与约定/Git规范.md`
3. `docs/collab/README.md`
4. `05_论文/正式/` 只由 Codex 写
5. Skill 的 `SKILL.md` 与 `references/09_十轮精读计划.md`

## 快速命令

```bash
cd ~/shuwei2026
git pull --ff-only
python scripts/env_doctor.py
python scripts/audit_git_hygiene.py --root .
```

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

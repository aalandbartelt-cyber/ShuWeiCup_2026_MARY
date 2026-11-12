# 已定事项（Decisions）

## D-001 比赛仓库独立
- 时间：2026-11-09
- 决定：数维杯单独建 GitHub 仓库；不与妈妈杯/亚太杯 skill 仓库混用。
- 理由：赛题、论文、数据、比赛提交物与 skill 本体生命周期不同；避免大文件和正式稿污染 skill。

## D-002 环境与分工
- 时间：2026-11-09
- 决定：Codex（WSL，VS Code Remote-WSL）为主环境；ZCode（Windows）为副环境；dsh（Windows）只做论文评审/文字优化。
- 边界：正式论文/正式图/正式包只由 Codex 写；ZCode 只写 `05_论文/待审/`、`docs/audit/`；dsh 只写 `docs/collab/reviews/`。

## D-003 Git 规范
- 时间：2026-11-09
- 决定：`main` 由 Codex 集成；功能分支 `codex/<scope>-<topic>`、`zcode/<scope>-<topic>`、`dsh/review-<date>`；commit 用 `<type>(<scope>): summary`；阶段门打 annotated tag。
- 影响：`02_口径与约定/Git规范.md`、`docs/collab/README.md`。

## D-004 AnyMath 不使用
- 时间：2026-11-09
- 决定：数维杯不使用 AnyMath；主栈 Python+R；不为厂商专项奖改主流程、图体系或附录可运行性。
- 影响：skill `SKILL.md` 第 10 节；已删除 AnyMath 参考文件。

## D-005 官方格式/提交规范更新
- 时间：2026-11-11
- 决定：以 2026-11-11 官方电子版为准：截止 11-24 10:00（周二），可重复提交、以最后一次为准；正文每页必须有队伍编号+页顶页码；书籍引用标页码；论文 PDF 不夹带程序/工具文件。
- 影响：`00_赛题与规则/赛事参数.md`、`2026官方更新_20261111/`、skill `references/01` 与提交清单。

## D-006 Cursor 作为备份主力
- 时间：2026-11-13
- 决定：Codex 无额度/不可用时，Cursor（WSL）作为备份主力接手同一工作区；不能恢复 Codex thread，用 `docs/collab/CURSOR_HANDOFF.md`；写 `cursor/` 分支与 `05_论文/待审/`。
- 正式件：默认仍由 Codex 写；Cursor 写正式件须用户明确授权并记录本文件。
- AI 合规：Cursor 是 AI 工具，按 skill `references/18` 记录、声明、报告。
- 影响：`AGENTS.md`、`.cursor/rules/mary-shuwei-skill.mdc`、`docs/collab/CURSOR_HANDOFF.md`、skill `references/21`。

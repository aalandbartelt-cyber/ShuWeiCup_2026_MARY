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

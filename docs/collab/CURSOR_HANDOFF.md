# CURSOR_HANDOFF

> 交接文件，不依赖对话上下文。Cursor 每次接手先读本文件；Codex 恢复后也先读本文件。

## 1. 当前状态（2026-11-20）

- 仓库：`~/shuwei2026`
- 当前本地分支：`cursor/c-selection-handoff`（从 `main` 的 `0066667` 开出；以 `git status` 实际为准）
- 远端：`origin` = `ShuWeiCup_2026_MARY`（public）
- 选题：用户已选定 **C 题**。详细交接：`docs/collab/C题选题交接_20261120.md`；决定：`docs/collab/decisions.md` D-008。
- 参考代码：`01_参考学习/RoboCupHomeSim`，GitHub `main` `9a5539e`（RockyV6.1）。该目录被 gitignore，不要提交。不要用 D 盘 `maryv1` 旧检出。
- 本地 `zcode/fig-r2-visual` 分支保留 R2 视觉分析，未合并；不要覆盖。
- 正式论文/正式图/交付包：尚未开始；`05_论文/正式/`、`06_交付/` 仍由 Codex 默认所有。
- 本轮交接文件尚未提交。

## 2. Skill 路径

```text
/home/mary/.codex/skills/mary-modeling-contest-pipeline-shuwei/SKILL.md
```

Cursor 不自动加载；必须显式读。任务相关再读 `references/01,02,05,06,11,12,13,15,18,19,20,21`。项目规则：`.cursor/rules/mary-shuwei-skill.mdc`。

## 3. 当前目标与验收

- 比赛期：2026-11-20 09:00—11-24 09:00（北京时间）；论文提交截止 11-24 10:00。
- 当前目标：按 `docs/collab/C题选题交接_20261120.md` 做 C 题。先读通三套数据里各一条训练记录的数值轨迹，再做第一问基线。
- 验收：状态定义在三问之间保持不变；划分按环境；第二问有遮挡回放和基线；第三问用公开 30 个种子报告全部公共指标。测试种子公布前不评测试集。
- 不把 RockyV6.1 或 EC63 的结果写进论文。三个原始数据包已由 ZCode 解压并校验，数据状态见 `CODEX_HANDOFF_20261120_C_DATA.md`。

## 4. 已完成（证据）

- 官方文件：`00_赛题与规则/2026官方更新_20261111/`
- AI 报告模板：`06_交付/04_支撑材料/AI_Tool_Usage_Report_Template.docx`
- Skill 双端：WSL `~/.codex/skills/...`、Windows `~/.agents/skills/...`；各 56 文件。
- 最近提交（skill 私有仓库）：`2c3d50c docs(submission): enforce 2026-11-17 three-slot rules`
- 最近提交（比赛 public）：`0066667 chore(git): ignore local official release data`（本轮交接尚未提交）
- 2026-11-20：用户选定 C 题；参考克隆与选题交接已落盘，见第 5 节。

## 5. 下一步 3 件

1. 复核 `01_参考学习/RoboCupHomeSim` 的 HEAD 是否仍为 `9a5539e`，并只读 RockyV6.1 的 `decision.hpp` 与 `inspection.hpp`。
2. 用《C 题数据集说明》中的密码解出规则文件和各一套一条训练记录的数值轨迹。读不通就停下报告容器格式，不改题、不改做视频模型。
3. 落第一问数据审计和常数/线性基线；通过后再做第二问遮挡回放和第三问 450 次验证。

## 5.1 多 Agent 消息总线（2026-11-13 新增）

- 协议：`docs/collab/AGENT_BUS.md`
- 开轮自检：`python3 tools/agent_bus/bus.py inbox --agent cursor --unread`
- 接任务先 `claim`，做完 `done`；阻塞就 `reply` 说明。
- MCP：`tools/agent_bus/mcp_server.py`，Codex/Cursor 配置见协议文档。
- 自动跑一轮：`tools/agent_bus/orchestrator.py`（默认 dry-run，`--run` 才调模型；支持 Codex 与已登录的 Cursor Agent CLI）。
- 连续接力：`tools/agent_bus/coordinator.py --run --max-turns 6`；停止文件 `docs/collab/AGENT_BUS_STOP`。
- Cursor CLI：`~/.local/bin/cursor-agent`；需一次性 `agent login`，当前状态为未登录。
- HUD：`tools/agent_hud/server.py` 显示 `BUS n` 未读数；HUD 只读，不会唤醒 Agent。

## 6. 数据/模型路径

- C 题官方英文 PDF 与数据包在 `00_赛题与规则/正式发布/`（gitignore）。外层 zip 约 5.5 GB；三个原始包已由 ZCode 解压并校验，路径与统计见 `CODEX_HANDOFF_20261120_C_DATA.md`。
- 决策结构参考：`01_参考学习/RoboCupHomeSim`，`main` @ `9a5539e`。不入库。
- 选题交接：`docs/collab/C题选题交接_20261120.md`。
- 当前无 C 题训练/结果产物。Codex 生成后把路径写回本文件。

## 7. 已定决策 / 禁止事项

- Codex 默认主力；Cursor 在 Codex 无额度/不可用时备份主力。
- Cursor 不能恢复 Codex thread；用本 handoff 接手。
- Cursor 默认只写 `05_论文/待审/` 与 `cursor/` 分支；正式件需用户明确授权并记录 `decisions.md`。
- Cursor 是 AI 工具；AI 使用按 skill `references/18` 记录、声明、报告。
- 不改 `05_论文/正式/`、`06_交付/` 正式件；不覆盖对方未提交工作。
- 原始数据/参考 PDF/密钥/个人信息不入库。

## 8. 阻塞与需要用户决定

- 三个原始数据包已由 ZCode 解压并校验；Codex 的第一个技术门禁改成“读通各一套一条训练记录的数值轨迹”，不是改题理由。
- 组织者测试种子不在当前包内。验证先用 `2026111700`–`2026111729`。
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

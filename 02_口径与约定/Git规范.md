# Git 规范（数维杯 2026）

> 目标：主环境 Codex、副环境 ZCode/dsh 同时工作时，提交可读、可回滚、不可互相覆盖；正式论文/包有唯一写入者。

---

## 1. 分支模型

| 分支 | 谁写 | 用途 | 合并 |
|---|---|---|---|
| `main` | Codex | 可运行集成线；正式论文/包 | 只接受 Codex 合并 |
| `codex/<scope>-<topic>` | Codex | 实验、Q1—Q4、图、论文、包 | 测试通过后 `--no-ff` 合 main |
| `zcode/<scope>-<topic>` | ZCode | 数据/文档/复核/图稿候选 | 只改 `05_论文/待审`、`docs/audit`；由 Codex 审后合并 |
| `dsh/review-<date>` | dsh | 评审报告 | 只改 `docs/collab/reviews/`；不合并代码 |

- `<scope>`：`q1` / `q2` / `q3` / `q4` / `data` / `fig` / `paper` / `pkg` / `repo`。
- 分支名必须小写、短横线；不要用 `final`、`new`、`test2`。
- `main` 不做 force-push；不重写共享历史。
- 合并前 `git pull --ff-only`；冲突时保留双方内容，重新编号消息，不删对方。
- 比赛期间至少每天结束前 push；阶段门必须打 tag。

## 2. Commit 格式

```text
<type>(<scope>): <summary>

<body: 为什么/证据/命令>
```

`type`：

- `feat` 新功能/模型/图/章节
- `fix` 修错误（数字、路径、格式、代码）
- `data` 数据口径/清洗/特征
- `fig` 图件
- `paper` 论文正文/结构/版式
- `audit` 审计/验收/复核
- `docs` 文档/协作
- `chore` 仓库/环境/脚本

`scope`：`q1`—`q4`、`data`、`fig`、`paper`、`pkg`、`repo`、`collab`。

示例：

```text
feat(q1): add group-shrunk NNLS baseline
fix(q2): separate bootstrap candidate and final fusion metrics
fig(q3): rebuild style profile with 2x2 panels
paper: add 25-page structure and contents check
audit(q1): record repeat stability and interval coverage
```

正文必须写：改了什么、证据文件/页码、运行过的命令；不要只写“update”。

## 3. 路径与所有权

| 路径 | 写入者 | 规则 |
|---|---|---|
| `05_论文/正式/` | Codex | 正式 DOCX/PDF；每版另存 |
| `05_论文/待审/` | ZCode | 候选稿；文件名带 `_待审` |
| `06_交付/` | Codex | 最终论文/附件/结果 |
| `docs/collab/` | 双方按信箱规则 | `to_codex.md` 只 ZCode 写，`to_zcode.md` 只 Codex 写 |
| `docs/collab/reviews/` | dsh | `REV-*.md` |
| `_archive/` | 任何人 | 旧版/快照；不入库 |

**候选转正式五条件**：差异清单 → 数值复核 → 版式检查 → 语言合规 → SHA256。见 skill `references/common/13`。

## 4. 阶段门与 tag

```text
gate/r1-structure
gate/r2-visual
gate/r3-evidence
...
gate/r10-release
paper/v1
submission/v1
```

- 每个 gate 必须：`audit_git_hygiene.py` 通过；链接/审计脚本通过；R<N> skill diff 落盘。
- tag 只打 annotated tag：`git tag -a gate/r2-visual -m "R2 visual gate"`。

## 5. 不提交什么

`.gitignore` 已排除：

- `data/`、`*.zip`、`*.rar`、`*.7z`、`*.pdf`（官方小模板除外）
- `05_论文/**/*.docx` 以外的临时文件？正式 docx 可入库；大附件 zip 不入库
- `_archive/`、`.worktrees/`、`.venv/`、`__pycache__/`、`*.pyc`
- `06_交付/*.zip`、`06_交付/04_支撑材料/` 大文件
- 权重、图片缓存、IDE 配置

判据：单文件 <50 MB；单次提交 <100 MB；未跟踪 <100 MB。

## 6. 冲突与回滚

- 同一文件两人改：先停手，发消息确认唯一写入者。
- 正式路径被非 owner 改：保存 diff、回退、记录，不静默覆盖。
- 回滚用 `git revert <sha>`；不要 `reset --hard` 已推送的 main。
- 大文件误入：`git rm --cached` + `.gitignore`，不重写公共历史；必要时用 Git LFS/Release。

## 7. 每日 checklist

```bash
git status --short
git diff --stat
python scripts/audit_git_hygiene.py --root .
git pull --ff-only
git push
```

- [ ] 没有未跟踪的大文件
- [ ] 没有密钥/个人路径
- [ ] 正式路径只有 Codex 改
- [ ] 阶段产物有证据文件
- [ ] 旧版已归档

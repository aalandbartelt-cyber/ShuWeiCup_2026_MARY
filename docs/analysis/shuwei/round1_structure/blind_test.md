# R1 盲测记录

> 目的：用未参与 R1 结构矩阵的 2 篇论文，验证新 skill 的“三选一 + 25 页预算 + 每问闭环”是否可定位、可发现问题。
> 规则：不查旧笔记，只按更新后的 `SKILL.md` / `assets/structure_archetypes.md` 走。

---

## 1. `ASHUWEICUP2541264`（A/2025，40 页）

### 按新 skill 定位

- 骨架：**问题驱动型**（1 Intro → 2 Symbols/Assumptions → 3 Problem Analysis → 4 Model → 5 Sensitivity → References/Appendix）。
- 每问模型：3.1/3.2/3.3 分析；4.1—4.6 建模/求解交替（p4—24）。
- 检验：5 Sensitivity Analysis（p24—?）。
- 缺口：**没有独立 Conclusion / Strengths & Weaknesses 章**；3.1.1 写成 Problem 2 Analysis、3.1.2 写成 Problem 3 Analysis，编号层级错位（p2 目录）。
- 页数：40 页总，正文约 22 页；按新 skill 应把 Appendix/References 与正文分开，不拿 40 页当正文。

### skill 是否有用

- 有用：三选一表让它一眼归为“问题驱动型”；页数规则能指出“40 页总/22 页正文”要按 25 页预算重排。
- 不足：旧 skill 未把“每问必须有 Conclusion/优缺点”作为硬门；R1 已把“每问 Analysis→Result→Validation→mini-Conclusion”写进 `references/02`，并在 `structure_archetypes.md` 标“无 Conclusion 是缺口”。

### 结论

**通过（可定位）**；但该论文本身是“结构可学、缺口不可学”的样本。

---

## 2. `ASHUWEICUP2551760`（C/2025，92 页）

### 按新 skill 定位

- 骨架：**问题驱动型**（1 Restatement → 2 Problem Analysis → 3 Assumptions → 4 Symbol → 5—9 Q1—Q5 → 10 Model Analysis/Testing → 11 Evaluation/Promotion → Appendix A—E）。
- 每问模型：5—9 每问 Construction+Solution；10.1 Sensitivity / 10.2 Error Analysis；11 Advantages/Limitations/Promotion。
- 检验：10 Model Analysis and Testing（p?），11 Evaluation。
- 缺口：**Summary 跨 p1—p2**（p1 没有 Keywords，p2 继续写），按“Summary 必须一页”是格式风险；658 词也超过 450 目标。
- 页数：92 页总，正文约 44 页；按 2026 规则必须压到 25 页（含 summary/contents），不能沿用旧样本页数。

### skill 是否有用

- 有用：三选一表把它归为“问题驱动型”；`summary_template` 的 300—450 词能直接指出 658 词超限；`audit_shuwei_submission.py` 的“Keywords 在第 1 页”应能标出跨页风险。
- 不足：audit 原先只检查 p1 是否有 summary/keywords，未显式检查“Summary 是否跨页”。R1 已补：**p1 必须有 Keywords，否则 WARN “Summary 可能跨页”**；C2551760 的第 2 页不是 Contents，已能被审计脚本标出。

### 结论

**通过（可定位）**；但暴露 audit 需要补“Summary 单页”检查。

---

## 3. R1 盲测结论

| 检查 | A2541264 | C2551760 | 结果 |
|---|---|---|---|
| 5 分钟内归入三种骨架 | 问题驱动 | 问题驱动 | 通过 |
| 定位每问模型 | 4.1—4.6 | 5—9 | 通过 |
| 定位检验/敏感性 | 5 Sensitivity | 10 Testing / 11 Evaluation | 通过 |
| 发现结构缺口 | 缺 Conclusion/优缺点；编号错位 | Summary 跨页；正文 44 页 | 通过 |
| 是否需要退回 R1 | 否 | 否（但 audit 补 Summary 单页） | 通过 |

**R1 结论：三选一骨架可定位；需要补两条硬门**：

1. 每问必须有 Analysis→Result→Validation→mini-Conclusion；没有 Conclusion/优缺点章是缺口。
2. Summary 必须一页；p1 无 Keywords 时标“可能跨页”。audit 脚本补此检查。

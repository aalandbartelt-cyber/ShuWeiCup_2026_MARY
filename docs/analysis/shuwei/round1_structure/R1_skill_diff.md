# R1 skill diff（结构解剖）

## 证据

- A25 `ASHUWEICUP2537139`：阶段驱动；Model Development→Algorithm→Test→Results→Sensitivity（p9—29）。
- A24 `A2024110230228`：问题驱动；2.1 Data analysis 约 5 页，4.1—4.4 逐问 Model（p3—24）。
- B24 `B2024093029335`：问题驱动；逐问 Modeling p5—24，Sensitivity 独立 p25。
- B24 `B2024101029505`：问题驱动+测试章；Test the Models p27，Sensitivity p28。
- C25 `CSHUWEICUP2561635`：问题驱动；每问 Construction+Solution p6—29，但无独立 Conclusion/优缺点（缺口）。
- C25 `CSHUWEICUP2556821`：问题驱动（Objective/Core/Key）；每问末 validation+sensitivity p6—35。
- D24 `D2024091329034`：阶段驱动/数据先行；Research ideas p2，Data preprocessing p4—10，Test p21。
- D24 `D2024111231024`：问题驱动+Model Overview；Model Overview p4，Proposals p25，Sensitivity p26。

完整矩阵：`structure_matrix.csv`；逐篇：`reading_sheets/`。

## 独立抽查修正（2 个 Explore agent）

- A25：body_pages 27→26（正文 p4—29）。
- A24 30228：3 Symbol/Assumptions p4→p9。
- B24 29335：该文页脚=PDF−1；矩阵统一改为**印刷页码**（1 p2、2 p2、3 p4、4 p4、5 p10、6 p15、7 p23、8 p25、9 p26、10 p27；Refs p27、App p28）。
- B24 29505：1 Introduction p3→p2；body 27→29（该文没有 Conclusion，矩阵不列）。
- C25 2561635：body 27→26（无独立 Conclusion/优缺点，标为反面）。
- D24 1329034：Refs/App 25/26→23/24；6/7/8 改为 Task 1/2/3（p11/15/17）；骨架从“阶段驱动”改为“问题驱动（任务驱动）+数据先行”。
- 记录：矩阵统一采用**读者看到的印刷页码**，不混用 PDF 页；后续所有轮次沿用此口径。

## 改动

### 1. `references/02_英文论文结构与写作.md`

- 原：只给一套“问题驱动”骨架。
- 新：三种骨架（问题驱动/模型驱动/阶段驱动）+ 适用条件 + 证据论文 + 每类风险。
- 保留：25 页预算、Summary/Contents、英文写作规则。

### 2. `assets/paper_outline.md`

- 原：单套 outline。
- 新：先选骨架三选一；问题驱动型保留为默认，新增模型驱动/阶段驱动代码块。

### 3. 新增 `assets/structure_archetypes.md`

- 三种骨架的完整定义、证据页码、页数、优点、风险、反例。
- 选择规则：问题数 × 模型是否同构 × 证据链位置。

### 4. `SKILL.md` 第 4 节

- 标题改为“三选一 + 25 页预算”。
- 增加骨架选择表，默认仍给问题驱动型；指向 `assets/structure_archetypes.md`。
- 明确“每问必须有 mini-Conclusion；没有独立 Conclusion/优缺点章是缺口”。

### 5. `scripts/audit_shuwei_submission.py`

- 增加“Summary 单页”检查：p1 无 Keywords 时 WARN“可能跨页”，对应 C2551760 盲测发现。
- Summary 词数 300—450、数字 8—20。

### 5. 前置修正（本轮同时完成）

- 页数：论文总页数 ≤25（含 Summary/Contents），附录另计。
- 页眉：`Page x of y`，`y` 按模板 LastPage/实际总页数，不硬写 25；Summary 页可无页眉。
- `common/15`、`common/16` 的数维杯覆盖说明。
- audit 脚本 Summary 词数 300—450、数字 8—20。
- 样本口径统一 12 核心 + 2 held-out（A24 29348、C24 1029647）。

## 未改动

- 不改 `references/05` 的图规则（留给 R2）。
- 不改 `references/06` 的检验规则（留给 R3/R7）。
- 不改 `references/03` 的题型 playbook（留给 R9）。
- 不改任何论文/赛题文件。

## 回归

- 链接检查：SKILL 引用 17 个路径，0 缺失。
- `audit_shuwei_submission.py`：25 页样本 18 项，1 FAIL（旧论文作者元数据“鹤顶红”，非 skill 问题）；Summary 单页检查通过。
- C2551760 盲测：18 项，2 FAIL（第 2 页不是 Contents：Summary 跨到 p2；正文 44 页）；符合预期，说明“Summary 单页/总页数”门有效。
- 盲测：`ASHUWEICUP2541264`、`ASHUWEICUP2551760` 均能在 5 分钟内归入骨架并定位每问模型/检验；通过。
- 用户审阅：通过后才进入 R2。

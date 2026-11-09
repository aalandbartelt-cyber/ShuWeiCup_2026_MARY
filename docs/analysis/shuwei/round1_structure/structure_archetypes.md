# R1 结构骨架（8 篇证据）

> 结论：数维杯优秀论文不是一套固定目录，而是三种可复用骨架。选哪种由“问题数量 × 模型是否同构 × 证据链位置”决定；25 页预算按骨架微调。

---

## 1. 问题驱动型（最常见）

```text
p1 Summary
p2 Contents
1 Introduction
2 Problem Analysis（逐问 2.1—2.n）
3 Assumptions and Notations
4 Problem 1: Analysis → Model → Solution → Results → Validation
5 Problem 2: ...
...
N Test of the Models / Sensitivity Analysis
N+1 Strengths and Weaknesses
N+2 Conclusion
References
Appendix
```

**证据**：

- A24 `A2024110230228`：1 Intro p3 → 2 Problem analysis p3 → 3 Symbol/Assumptions p9 → 4 Model（4.1—4.4 逐问）p10—23 → 5 Strengths p24 → 6 Conclusion p24。
- B24 `B2024093029335`：1 Intro p2 → 2 Problem analysis p2 → 3 Symbol p4 → 4—7 逐问 Modeling p4—23 → 8 Sensitivity p25 → 9 Strengths p26 → 10 Conclusion p27。
- C25 `CSHUWEICUP2561635`：1 Intro p4 → 2 Assumptions/Notation p5 → 3—7 逐问 Construction+Solution p6—29（无独立 Conclusion，反面）。
- C25 `CSHUWEICUP2556821`：1 Intro p3 → 2 Problem analysis p3 → 3 Symbols p5 → 4 Model（4.1—4.5 逐问）p6—35 → 5 Strengths p36 → 6 Conclusion p37。
- D24 `D2024111231024`：1 Intro p3 → 2 Assumptions p4 → 3 Model Preparation p5 → 4—6 三个模型 p9—24 → 7 Proposals p25 → 8 Sensitivity p26 → 9 Evaluation p26。
- D24 `D2024091329034`（任务驱动+数据先行）：1 Introduce p1 → 2 Research ideas p2 → 3 Assumptions p3 → 4 Notations p4 → 5 Data preprocessing p4 → 6 Task 1 p11 → 7 Task 2 p15 → 8 Task 3 p17 → 9 Task 4 p20 → 10 Test p21 → 11 Evaluation p23。

**适用**：3—5 问、每问模型/数据/输出不同；C/D 题最常见。
**页数**：Intro 2—3；Problem Analysis 1—2；Assumptions/Notation 1—2；每问 3—5；Test/Sensitivity 1—2；Strengths/Conclusion 1—2；References 1；合计 ≤25。
**优点**：评委能按问题定位；每问闭环。
**风险**：问题多时每问只剩 2—3 页，容易变成“每问都浅”；必须提前定取舍。

---

## 2. 模型驱动型

```text
1 Introduction
2 Problem Analysis
3 Assumptions and Notation
4 Model I: Preparation → Building → Problem Solving
5 Model II: ...
6 Model III: ...
7 Sensitivity Analysis
8 Strengths and Weaknesses
9 Conclusion
References
Appendix
```

**证据**：

- C24 `C2024110930663`：3 Model I: TVF-EMD → 4 Model II: SA → 5 Model III: BDSK-SA；每个含 Model Preparation/Model Building/Problem Solving；6 Sensitivity → 7 Strengths → 8 Conclusion。
- 2025 C `CSHUWEICUP2556821` 的 4.1—4.5 也可视为“按模型/问题混合”的变体（p6—35）。

**适用**：3—4 个命名方法解决同一/相近任务；方法名需要全程复用（TVF-EMD/SA/BDSK-SA）。
**页数**：每个模型 4—6 页；Intro/Analysis/Assumptions 4—5；Sensitivity/Evaluation 2—3；合计 ≤25。
**优点**：方法主线清楚，适合“改进模型”叙事。
**风险**：问题本身多、模型只是工具时，容易让评委看不到“每题回答”；必须每模型末尾写“对应哪一问/什么结果”。

---

## 3. 阶段驱动型（A/B 仿真/优化常见）

```text
1 Introduction
2 Problem Restatement / Problem Analysis
3 Symbol and Assumptions
4 System Model Development
5 Problem 1: Model Development → Algorithm Design
6 Problem 2: ...
7 Problem 3: ...
8 Test the Models（数值/物理/实测/鲁棒）
9 Results Analysis（逐问 performance/error/convergence/stability/sensitivity）
10 Sensitivity Analysis
11 Conclusion（Summary + Advantages/Limitations）
References
Appendix
```

**证据**：

- A25 `ASHUWEICUP2537139`：1 Intro p4 → 2 Problem Restatement p5 → 3 Symbols p6 → 4 System Model p7 → 5—7 逐问 Model+Algorithm p9—12 → 8 Test the Models p12 → 9 Results Analysis p17 → 10 Sensitivity p28 → 11 Conclusion p29。
- B24 `B2024101029505`：1 Intro p2 → 2 Problem analysis p3 → 3 Symbols p9 → 4 Model p10 → 5 Test the Models p27 → 6 Sensitivity p28 → 7 Strengths p29。

**适用**：A/B 仿真、控制、优化；各问共享同一系统模型/算法链，只是阶段不同。
**页数**：Intro/Problem/Symbol 4—5；System Model 2—4；每问 2—4；Test/Sensitivity 3—5；Conclusion 1—2；合计 ≤25。
**优点**：模型链完整，检验章可以写深。
**风险**：容易把“每问”写成同一套流水线；必须每问有独立输出/图/结论，不能只换参数。

---

## 4. 选择规则

1. 先数列出“问题数、每问输出、模型是否同构”。
2. 每问输出不同、模型不同 → 问题驱动型。
3. 3—4 个命名方法解决相近任务 → 模型驱动型。
4. 同一系统模型/算法链分阶段 → 阶段驱动型。
5. 25 页不够时，优先合并“Assumptions/Notation”，不要合并“每问结果/验证”。
6. 不管哪种，**每问必须有 Analysis、Result、Validation、mini-Conclusion**；没有 Conclusion 章的论文（如 C25 2561635）是缺口，不是风格。

---

## 5. 反例

- 91 页的 A25 2537139：阶段骨架完整，但体量/中文小流程图不可照搬。
- C25 2561635：问题驱动清楚，但没有独立 Conclusion/优缺点章；不要学缺口。
- A24 30228：Strengths 段写成了 future work；骨架对，内容错位。
- C25 2556821：Objective/Core/Key 三行很好，但 3×3 九宫格图字太小；结构可学，图不可学。
- D24 1329034：技术路线/数据章很好，但 Summary 一整块 552 词、章号 3→6 跳号；不要学排版缺口。

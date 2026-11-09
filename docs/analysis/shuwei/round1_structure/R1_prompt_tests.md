# R1 真实提示测试（待用户逐条跑）

> 按 skill-creator：一次一条，与用户一起看输出/trace；不要并行评估。
> 目的：确认 description 能触发、三选一骨架能落地、25 页/页眉口径不冲突。

## Prompt 1：结构骨架

```text
用 mary-modeling-contest-pipeline-shuwei，给我一份数维杯 A 题 25 页英文论文的结构骨架和页数预算。
要求：先判断用问题驱动/模型驱动/阶段驱动哪种，再给章节与每章页数；不要写正文。
```

**期望**：
- 先选骨架；
- 给 25 页含 Summary/Contents 的预算；
- 点出每问必须有 Analysis→Result→Validation→mini-Conclusion；
- 不把 common/15 的 30 页/无页眉套进来。

## Prompt 2：PDF 审计

```text
用数维杯 skill 检查这篇 PDF：Summary/Contents/总页数/页眉/中文/字号/元数据。
PDF: <25页样本.pdf>
```

**期望**：
- 检查 p1 Summary/p2 Contents；
- 总页数≤25（含 summary/contents）；
- 页眉 `Team ASWBMCM... Page x of y`，不要求 y=25；
- 中文/身份/元数据；
- Summary 词数 300—450、数字 8—20；
- 旧论文元数据/旧页眉告警说明是样本差异。

## Prompt 3：结构总结

```text
读这 8 篇数维杯优秀论文的结构，归纳 3 种骨架，并说明各自适用条件/页数/反例；只输出证据页码，不要泛泛而谈。
```

**期望**：
- 问题驱动/模型驱动/阶段驱动；
- 每类有 ≥2 篇证据页；
- 指出 C25 2561635 无 Conclusion、D24 1329034 是任务驱动+数据先行等例外；
- 不把 91 页样本当页数标准。

## 状态

- [ ] Prompt 1
- [ ] Prompt 2
- [ ] Prompt 3
- [ ] 用户确认 R1 通过 → 进入 R2

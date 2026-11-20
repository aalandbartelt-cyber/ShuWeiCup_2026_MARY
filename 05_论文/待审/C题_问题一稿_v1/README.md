# 全文框架与第一问待审稿

入口：`C_Q1_review.pdf`；LaTeX 主文件：`main.tex`。该目录是待审候选，未写入正式论文或交付目录。

## 本轮用户要求

- 全文目录采用 Introduction → Problem Analysis → Model Assumptions and Notation → Problem 1 → Problem 2 → Problem 3 → Model Evaluation → Conclusion → References → Appendix。
- Introduction 和 Problem Analysis 是全文前置部分：背景覆盖整题；问题重述以 Problem 1/2/3 分段；分析以 Analysis of Problem 1/2/3 分节，说明各问任务、难点和衔接，不复述第一问的具体算法与结果。
- 第一问包含数据处理、模型建立、求解与验证；三级标题不进入目录，避免第一问的细节挤占全文目录。
- 第二、三问仅保留题意对应的章节位置和明确的待补说明，没有伪造结果。
- 摘要和关键词空白，三问完成后统一写。旧摘要不参与编译。
- 全文第 7 节 Model Evaluation 与第 8 节 Conclusion 只保留标题，正文及 7.1/7.2 暂时清空，三问完成后统一写；第一问的负结果和稀疏观测适用边界归入其对应结果段。
- 仅保留三个当前关键假设；Proposition 1（坐标一致性）和 Proposition 2（预测球嵌套性）均有独立 Proof 与规范证毕方框。
- 普通正文首行缩进 2 em，段间距为零；公式后的符号解释顶格。公式居中、编号靠右，多行按等号对齐。
- 正文六图、附录十一图（B.1–B.11）。多面板图宽为版心的 88%，简单对照图为 82%，均居中；按最终尺寸重排。图 1 保留来源/分配两个面板且图前解释 B；B.8 收其余数据诊断。B.9 收多步长误差及既有基础半径，B.10 收环境描述统计，B.11 收真实验证失败轨迹。
- 图题、表题只给简短名称，面板、单位和统计解释在正文。所有图内基础标注按最终插图尺寸核对不低于 8 pt，数学指数使用正常较小字号。
- 图表与可见正文不用内部迭代编号，统一使用实际方法名称。原始版本路径只在源文件索引和本地审计记录中追溯。
- 按用户要求删除赛题本身的参考文献；七个条目按首次引用排序，正文方括号编号与 reference_verification.json 一致。
- 求解部分明确 NumPy 线性代数、SciPy L-BFGS-B、PyTorch AdamW、LightGBM 分位数回归/二分类、RTX 5070 Ti Laptop GPU 与 CPU 分工。具体环境记录在本地 solver_environment.json。

## 证据与复现

正文以验证集/训练 OOF 为主要证据；验证集曾用于选择，不能称独立终检。测试已提前看过的事实、原冻结时间线保留；测试成绩只作附录补充复核。本轮没有重新训练或重跑测试。新增保持值结果仅使用冻结验证目标窗口，其他四个方法和原表最大差异 9.9e-17。失败轨迹仅以 CPU 复算冻结验证 checkpoint，误差差异不超过 3.06e-7；原视频四时刻 GPU/CPU 核验仍保留。

审稿补查入口为 scripts/audit_c_q1_review_repair.py：69 条记录的 UUID、字段/时间连续性、场景/操作者标识重叠及用途计数；test 仅检查原始质量/保留键，不计算预测成绩。手坐标单位和世界轴方向缺原文证据，不能由操作者身高的 m 字段推定。训练与测试共享 3 个操作者标识。脚本输出、源表、事件 bootstrap 和图例约定位于 artifacts/c_q1_review_repair/，逐条处理见 docs/collab/reviews/Q1_评审逐条核查与修复_20261120.md。

`figure_sources.json` 记录原证据图及其哈希、论文排版图及其哈希；`table_sources.json` 记录源数值表。原始实验图与冻结科学文件保持不变。`Q1_review_source.zip` 包含本次 LaTeX、排版图、源程序、配置和测试，是源文件快照；不包含原始比赛数据、外部权重或模型检查点，也不声称已在干净环境重跑整条科学流水线。

在当前项目根目录重建：

```bash
/home/mary/miniconda3/envs/mathorcup/bin/python scripts/prepare_c_q1_manuscript_tables.py
python3 scripts/build_c_q1_manuscript.py
/home/mary/miniconda3/envs/mathorcup/bin/python scripts/audit_c_q1_manuscript.py
```

历史绘图重排入口为 `scripts/reflow_c_q1_manuscript_figures.py`。本轮图 2–4 与 B.9–B.11 由 `scripts/plot_c_q1_review_repair.py` 覆盖渲染；重建图时应在历史重排之后运行该脚本，避免恢复旧测试主图。它输出至独立 `artifacts/c_q1_review_repair/`，不改冻结科学图。LaTeX 编译调用本机 Windows TeX Live；其他环境安装 Times New Roman 后，可在本目录运行 XeLaTeX 三次。

附录参照两篇自写终稿：Appendix A 当前保留已有源文件表，按用户最新要求暂不整理，留到最后；Appendix B 收补充表格与图片，每图保留简短说明。Draft provenance 段已按用户要求从论文删除，内部记录保留在本 README。

原始官方模板保留；本目录的副本调整了真实 Times New Roman、相配数学字体、英文标题、规范 proof 环境、段落及浮动体设置。原模板首页和目录不显示正文页码，正文从 1 开始；末页总数包含附录，PDF 物理页数另计。

本轮是 Codex 自动检查和视觉审阅，不代表用户或队员已完成独立审阅。最终提交仍须补齐第二、三问、全文摘要和关键词。

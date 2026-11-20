# C 题第一问结果 v1（2026-11-20）

第一问已形成可复现的开发结果：三套数据完成共同状态建模、记录与连续区间用途划分、因果预测对照、可靠性模型比较、参数敏感性和不确定性分析，并实际使用了全部训练与验证视频。**当前是验证集开发结果，尚未冻结为最终版本，未使用测试轨迹。**

## 1. 可直接核查的结论

1. 55 条训练记录和 7 条验证记录共 53,829 个数值帧。全部可读，无非有限手坐标、时间倒退或超过 1.5 倍记录中位采样间隔的缺口；未发现完整轨迹重复。并不意味着每帧观测准确。
2. 双手状态统一为 21 个点的几何中心与速度，双手相对向量由中心作差得到。三套数据不强行假定数字索引 0 都是腕部，也不把记录世界坐标拼成一个超市。
3. 多时间尺度线性模型在三套验证数据的短时预测上均优于两点线性基线。目标为不早于 0.1 秒的首个样本，实际跨度见第 4 节。具体增益见下表；这是有限验证环境内的比较。
4. 未来质量标记预测中，质量转移模型的整体 Brier 为 **0.015277**，训练数据集频率基线为 **0.020381**，相对下降 **25.04%**。数值提升树的排序指标较好，但概率误差尚未稳定优于质量转移模型。
5. 已处理训练/验证的 **166 段视频、24,750 个抽样帧**。当前 MediaPipe 多视角摘要加入模型后没有提高整体可靠性预测，不将这一负结果隐藏，也不据此否定更强的视觉模型。

## 2. 数据保留和区间用途

| dataset | split | records | environments | frames | qualified_hand_frames | total_hand_frames | retained_hand_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EgoDemo | train | 27 | 10 | 14369 | 28579 | 28738 | 99.447 |
| EgoDemo | validation | 2 | 2 | 1397 | 2794 | 2794 | 100.000 |
| EgoPro | train | 8 | 3 | 9652 | 19140 | 19304 | 99.150 |
| EgoPro | validation | 4 | 2 | 4853 | 9598 | 9706 | 98.887 |
| EgoStandard | train | 20 | 7 | 23025 | 45782 | 46050 | 99.418 |
| EgoStandard | validation | 1 | 1 | 533 | 1015 | 1066 | 95.216 |

分母是手—帧：一帧有左右手两个观测。参考资格要求坐标有限、原始质量未标坏、时间有效；本版所有开发记录时间均连续。标坏帧保留作低质量现象分析和降权状态更新，不能充当真实位置误差的参照。区间按左右手分别切分，边界在 `interval_uses.csv`，半开索引 `[start_index, stop_index_exclusive)`；持续时间以最后时间减最初时间加记录中位采样间隔近似，不声称精确遮挡时长。

![数据覆盖与质量分布](../../../artifacts/c_q1_v1/figures/q1_data_quality.png)

*Figure Q1-1. Development-set numerical coverage and raw quality-flag frequency. A frame is flagged if either hand is flagged. Counts do not establish physical sensor accuracy. Test trajectories are unopened.*

完整原始样例、字段与划分的独立审计见《C题读题_Codex.md》及 `audit_c_inputs_20261120.py`。总包的 69 条记录包括 7 条测试记录；上述模型开发只读 62 条训练/验证轨迹。训练/测试操作者有重叠，未来结论应主要称跨环境泛化。

## 3. 状态模型、公式和假设

对手 h、帧 t，观测中心 `c[h,t] = sum(j=1..21) p[h,j,t]/21`。共同状态为 `[cL,vL,cR,vR]`，维度 12；相对向量 `r=cR-cL` 为派生量。原生位置单位未独立核验，速度单位为原生位置单位/秒。

局部匀速状态转移为 `F(dt)=[[1,dt],[0,1]]`，三个坐标共享同一时间结构、两手分别更新。过程协方差为 `q*[[dt^4/4,dt^3/2],[dt^3/2,dt^2]]`，相当于一步内随机常加速度。观测噪声对标坏手乘倍数。训练估计 q、R 是时序自一致性的稳健代理，不能唯一辨识真正的传感器物理噪声。更新使用 Joseph 协方差形式保持数值稳定。

验证网格为 q 倍率 {0.1,1,10}、R 倍率 {1,10}、质量降权倍率 {1,10,100}，共 18 组；选择结果为 `{'q_scale': 10.0, 'r_scale': 1.0, 'bad_multiplier': 10.0, 'validation_normalized_error': 0.4130061407175598}`。q 倍率位于网格边界，应解释为本次候选范围内的选择，不能称连续参数全局最优。

增强的线性预测器用过去 1、3、6、12 帧形成四个平均速度，预测 `c(t+τ)=c(t)+τ*sum(a[k]*v[k](t))`。系数通过环境/记录等权的岭回归拟合，并向两点线性预测系数 `[1,0,0,0]` 收缩；三个坐标和左右手共享系数，因此保留平移、旋转等变性。0.1 秒系数为 `[0.9665423654269703, -0.0354356875100549, -0.062235368875674685, 0.01535281778945918]`。正则参数从 {0.001,0.01,0.1,1,10,100} 在验证集选择；拟合本身只用训练记录。

| 假设 | 依据与范围 | 失效时的影响/检查 |
| --- | --- | --- |
| 很短时间内运动可由局部速度描述 | 原始近 30 Hz 轨迹；预测跨度 0.1 秒 | 快速转向可能失配；保留误差尾部及标坏历史子集 |
| 三套数据共享状态定义和时间结构 | 都有两手 21×3 坐标 | 噪声和手尺度按数据集估计；不强行统一坐标原点 |
| 原始未标坏帧可作条件参考 | 官方允许合格记录作参照 | 未标坏不等于无误差；不能报告真实物理定位精度 |
| 标记历史可能预测后续标记 | 以训练记录估计转移 | 数据集/环境变化可能破坏校准；与运动、视频模型比较 |
| 阶段只用于事后解释 | 原始标注的片段三等分 | 不能在在线预测使用未来片段终点 |

## 4. 状态预测结果

| dataset | constant | linear | filter | multiscale_linear | gain_vs_two_point_percent |
| --- | --- | --- | --- | --- | --- |
| EgoDemo | 0.025904 | 0.013918 | 0.013470 | 0.012972 | 6.798452 |
| EgoPro | 0.031680 | 0.018043 | 0.018132 | 0.016950 | 6.058249 |
| EgoStandard | 0.040176 | 0.016405 | 0.015964 | 0.015523 | 5.380254 |

指标是先用截至 t 的信息预测，再在 **不早于 t+0.1 秒的首个真实样本**上求欧氏距离；不插值制造目标，也不会用当前更新后的状态和同一个输入比较来证明精度。受不规则时间步和浮点时间戳影响，“0.1s”是选择阈值，不是固定三帧或精确相同跨度。实际平均跨度如下，方法间使用完全相同的目标时刻；不能用跨数据集误差差异直接推断数据集难度。

| dataset | actual_horizon_mean_s |
| --- | --- |
| EgoDemo | 0.114178 |
| EgoPro | 0.127950 |
| EgoStandard | 0.100464 |

左右手先在记录内平均，再在环境内等权平均记录、环境间等权。`state_validation_by_record.csv` 保留实际预测跨度、每手样本量、90% 误差分位以及 recent_flag 子集。

![状态预测对照](../../../artifacts/c_q1_v1/figures/q1_state_forecast.png)

*Figure Q1-2. Mean Euclidean forecast error at the first existing sample at least 0.1 seconds after each origin, against unflagged held-out numerical references. Actual horizons vary with sampling timestamps and are tabulated separately. Hands are averaged within episodes, episodes within environments, then environments. Parameters were selected on validation; results are developmental. Coordinate units are not independently verified.*

`noise_parameter_bootstrap.csv` 给出 q、R、手尺度的训练环境 bootstrap 区间；`multiscale_coefficient_uncertainty.csv` 给出固定已选正则下的系数环境 bootstrap 区间。两者均未计入再次模型选择的不确定性。验证集仅 4 个环境，不能将数千帧当成同等数量的独立试验。

## 5. 可靠性模型与多视角比较

可靠性目标明确为“至少 0.1 秒后的下一合适样本是否任一手被原始质量标记标坏”。这既不是商品状态/作业成功率，也不是三维定位误差概率。

- 简单基线：训练数据集频率；按数据集和当前是否标坏估计的质量转移概率。
- 数值模型：当前及历史速度、速度变化、手几何尺度、双手距离、已有标记和数据集类别；去掉质量历史另做消融。
- 视觉增强：各视角最近的不晚于决策时刻的抽样图像，最大陈旧时间 0.25 秒，输入手检测数、二维中心、空间跨度、框面积、左右手分类分数与图像年龄。左右手分类分数没有解释为定位准确率。
- 学习器：逻辑回归和直方图提升树；权重为环境等权、环境内记录等权。Platt 校准仅拟合训练侧留环境预测；验证集选择原始概率或校准版本。
- 图像按每秒 5 帧抽样、IMAGE 无状态模式运行，未将视频时序追踪的隐藏帧带入特征。Demo 按容器 FPS 与片段起始时间近似对齐；Pro/Standard 使用 frame CSV 时间戳。

| method | brier | log_loss | average_precision | auc | ece |
| --- | --- | --- | --- | --- | --- |
| flag_transition | 0.015277 | 0.072815 | 0.302593 | 0.817890 | 0.006514 |
| numeric_boosting_raw | 0.015630 | 0.074225 | 0.328909 | 0.855447 | 0.008954 |
| multimodal_boosting_raw | 0.017882 | 0.090289 | 0.294129 | 0.806041 | 0.013980 |
| dataset_rate | 0.020381 | 0.102013 | 0.028103 | 0.624440 | 0.008449 |

![可靠性比较与校准](../../../artifacts/c_q1_v1/figures/q1_reliability.png)

*Figure Q1-3. Prediction of either hand being flagged at least 0.1 seconds later. Scores balance environments and episodes. The transition baseline conditions on the current flag. Numerical/video boosting uses raw probabilities; the full comparison also includes training-only out-of-fold calibration. Bins with fewer than five frames are omitted from the calibration plot.*

训练留操作者压力检验在 `quality_leave_operator_stress.csv`；它重新拟合训练子集上的模型，仅报告未校准版本，避免把包含被留操作者的全训练校准器带入检验。`quality_paired_vs_transition.csv` 按验证环境做固定模型配对 bootstrap；仅 4 个独立环境，区间为探索性证据。

重要失败/边界：EgoDemo 验证记录没有正质量标记，AUC/AP 不可定义，文件保留为空；此处低 Brier 不能证明检出能力。提升树/视频的更多参数未自动转化为更好的概率预测。当前未证明手中心估计的三维物理准确度；已准备 34 帧独立二维人工核验包，但尚未收到人工标签。

## 6. 任务、环境和阶段差异

![困难片段中的状态与质量](../../../artifacts/c_q1_v1/figures/q1_state_quality_example.png)

*Figure Q1-4. A deliberately difficult EgoStandard validation interval, chosen from the record with the highest flagged-frame fraction. A: centroid x. B: speed. C: future quality-flag probability. Dots show unflagged numerical observations, curves show causal state estimates, and orange spans mark raw bad-quality intervals. No true positional error is computed within the flagged spans. The probability curve predicts a future raw quality flag, not physical localization failure.*

上述时间线刻意选取验证集中标坏帧比例最高记录的一段，展示状态、速度与质量如何变化；橙色区间只作低质量诊断，不在其中计算真实位置误差。样例选取规则保存在 `example_selection.json`。

![阶段质量差异](../../../artifacts/c_q1_v1/figures/q1_retrospective_stages.png)

*Figure Q1-5. Offline association between annotated-interval thirds and raw quality flags, grouped by sampling family. A: training. B: validation. Segment endpoints are used for this retrospective description only. Task/environment confounding prevents a causal interpretation.*

这里用原始语义片段的前、中、后三分之一作事后阶段。P/A/L 是官方片段抽样类别，并不等同具体机器人技能。`retrospective_stage_statistics.csv` 与 `stage_summary.csv` 同时报阶段质量、清洁参考速度、双手距离和环境数。阶段/类别与环境的分布有混杂，当前结果只能说明关联；不可声称某类动作导致遮挡。未把阶段标签输入在线模型。

## 7. 第一问交付与后续接口

| 内容 | 位置 | 状态 |
| --- | --- | --- |
| 记录/区间用途与保留量 | `artifacts/c_q1_v1/record_audit.csv`、`interval_uses.csv`、`retention_summary.csv` | 已运行 |
| 12 维状态与派生相对向量 | `artifacts/c_q1_v1/states/*.npz`，62 条记录 | 已导出；Q2 后续统一接入该版本 |
| 模型与参数敏感性/不确定性 | `state_model.json`、`multiscale_model.json`、各参数 CSV | 已运行 |
| 质量模型、逐帧预测、对照/消融 | `quality_models.joblib`、`quality_validation_predictions.csv`、`quality_validation_metrics.csv` | 已运行 |
| 原始结果→关键数字→图 | `key_numbers.csv`、`figure_inventory.csv`、`figures/` | 同版生成 |
| 三维外部真值/人工二维核验 | `artifacts/c_video/human_2d/` | 独立三维真值缺失；二维人工回填待完成 |
| 测试集评价 | 官方 7 条测试记录 | 未启动；模型最终冻结后进行 |

复现顺序（工作目录为仓库根目录）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/run_c_q1.py --stage state
python scripts/prepare_c_video.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python scripts/run_c_q1.py --stage quality
OPENBLAS_NUM_THREADS=1 python scripts/extend_c_q1_dynamics.py
python scripts/report_c_q1.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_c_q1.py tests/test_c_protocol.py
```

数值依赖见 `requirements-c.txt`；视频环境另需 `requirements-c-vision.txt`。本地数值入口是现有 mathorcup Python，视觉入口是 `artifacts/c_vision_env/bin/python`。报告与模型均为 Codex 生成的可审阅工作产物；尚无队员独立复核结论。当前无 commit/push，也没有改写正式论文或正式提交包。

出图规范：[项目图表规范](../../../02_口径与约定/图表规范.md)。全部候选图使用 Times New Roman、外置图题/图注，保留 PDF/SVG/600 DPI PNG。

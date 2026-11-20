# Q2 稀疏校准与建议预测时长模块

日期：2026-11-20（北京时间）。执行：Codex 子智能体 `q1_figure1b_text`。状态：模块及测试完成；真实基线校准入口已接入、尚未启动，按主智能体要求交接释放并发槽。

## 所有权与接口约定

只维护 `src/shuwei_c/q2_calibration.py`、`tests/test_c_q2_calibration.py`、`scripts/analyze_c_q2_calibration.py`、`artifacts/c_q2/calibration/` 和本记录。主智能体负责真实逐时刻残差、模型选择及完整流水线；不改其他 Q2 模块、配置、根 handoff 或 Q1 冻结件。

输入 X 为仅当前可见特征；errors 为逐行的尺度归一化欧氏误差，尺度必须由点模型拟合环境确定，本模块不估计 scaler。meta 至少有 environment、episode、window_id、hand；scenario（b{cap}_d{delay}）及 phase 等已有实验列纳入独立窗口实验键，避免不同请求条件混合成同一窗口。评分按条件分别给出。

接口计划：`RadiusCalibrator.fit(X, errors, meta)`、`predict(X)`、`calibrate(X_cal, errors_cal, meta_cal)`。conditional 使用 CPU LightGBM 预测 log1p(error) 的 0.5/0.8/0.9/0.95 条件分位，static 为层级加权的静态分位。独立校准必须与拟合环境互斥；整环境 OOF 工具记录折边界。该工具只能验证传入数组的分组隔离，点预测、尺度和特征生成的外层隔离仍由调用方负责。

权重为等环境 → 环境内等记录 → 记录内等独立窗口实验 → 实验内等手 → 该手等时刻。主完整双手窗口与题目约定的等时间/双手一致。

建议时长仅由预测半径的时间上包络和给定阈值生成，不使用真实误差。逐点覆盖和整段最大误差的事后核验分开；不宣称 90% 同时覆盖。没有逐帧隐藏残差反馈入口，实际到达信息必须在上游严格构造为可见特征。模块不改变全局随机种子。

## 核验与后续

- `tests/test_c_q2_calibration.py`：15 passed（1.90 s，最后一轮）；两份 Python 源文件 `py_compile` 通过。测试覆盖实验场景/环境层级权重、独立校准环境隔离、OOF 不依赖本折标签、嵌套校准环境排除、异方差分位学习、全局 RNG 不污染、模型保存加载、特征顺序、当前直接可见样本的单独分母、时间上包络、零建议时长不算成功、逐点 90% 与整段 0% 覆盖反例、缺参照不认证、长表独立期望计数及重复时刻检查。
- 通用表格 CLI 已用明确标注的合成数据跑通：`artifacts/c_q2/calibration/synthetic_smoke/`，train 1600 行、独立 calibration 400 行、validation 400 行；来源、分布、种子在 inputs/provenance.json。此记录只验证接口/文件输出，不是比赛结果。后续接入真实 learning-data 的新增分支尚未实际运行，不能用先前合成 smoke 声称它已通过。
- 已读真实 learning-data 的布局/元信息：train 1,257,660 行，46 个 raw features，expert 8 = `damped_fast_0.2`；误差是 `norm(experts[:,:,8]-target)`，按每折拟合环境的 hand_scale 归一化，特征调用根模块 `transform_features`。未把官方验证残差用于拟合。
- **当前没有本任务真实训练进程，未启动长训练；无可恢复的活 session/PID。** 合成 smoke 与测试均已正常结束；进程查询未发现运行中的 calibration 脚本。不要把计划中的真实 probe 当作已启动或已完成。

## 已实现 API

```python
from shuwei_c.q2_calibration import (
    RadiusCalibrator, AgeBinnedRadius, hierarchical_weights,
    oof_predictions, evaluate_radii, evaluate_paths, evaluate_paths_long,
    suggest_horizon, evaluate_suggestions,
)
model = RadiusCalibrator('conditional', parameters={'num_leaves': 15, 'n_estimators': 150})
model.fit(X_fit, error_fit_normalized, meta_fit)
model.calibrate(X_cal, error_cal_normalized, meta_cal)  # optional; disjoint environments
radii = model.predict(X_eval)  # (n, 4), levels = .5/.8/.9/.95
model.save(directory)
loaded = RadiusCalibrator.load(directory)
```

- static 用同一 RadiusCalibrator 接口；AgeBinnedRadius.fit(ages, errors, meta)/predict(ages) 为年龄分层对照，固定年龄上界 `1e-9,.05,.1,.2,.5,1,2,inf`，不足 3 个拟合环境的 bin 回退全局分位。
- `oof_predictions` 使用现有 `training_environments/held_environments` 规则；可在每折指定 `calibration_environments`（training 的子集），自动从拟合集中排除。它不能证明上游点模型/scaler 的隔离，调用者须提供相应真实证据。
- `evaluate_radii` 给出每个 level 的覆盖/平均半径/pinball，可用 `group_by` 指定条件，并自动增加 `not_current_direct` 子集。
- `evaluate_paths_long` 要求 meta 额外有 `time_s`、独立窗口清单提供的 `expected_hand_times`；不能把遗漏的点直接当作完整窗口。输出是双手联合的整窗覆盖，不等同于逐点名义覆盖。
- `suggest_horizon(horizons, radii_at_one_level, tolerances)` 只读预测半径；两手共同建议应先对两手半径取最大。输出 0 表示没有支持的正预测时长。`evaluate_suggestions` 事后用真实误差检查所选区间，不给同时覆盖保证。

## 下一步直接运行（根接管）

真实 probe 分支已写在本任务拥有的 `scripts/analyze_c_q2_calibration.py`，固定默认 15 leaves / 150 trees / lr .04 / L2 10 / 2 CPU threads。将跑四个外层环境折及全 train → validation，对比 static、age_binned、conditional 四个 level；模型/逐时刻半径/独立窗口计数路径表/分条件环境指标都会保存。每个分区有 complete.json，允许同输入未完成作业恢复；identity 不同须另起目录。

```bash
/home/mary/miniconda3/envs/mathorcup/bin/python scripts/analyze_c_q2_calibration.py \
  --learning-data artifacts/c_q2/learning_data \
  --baseline damped_fast_0.2 \
  --out artifacts/c_q2/calibration/damped_probe \
  > /tmp/shuwei_q2_calibration_damped.log 2>&1
```

启动前先跑上面的 15 项测试，并注意新增真实分支尚未完整 smoke。主要输出：`point_summary.csv`、`path_summary.csv`、`suggestion_summary.csv` 及同名 environment_metrics；全部条件分开，环境等权。最终 manifest 只会在所有分区完成后写入。

建议时长探针**只使用 cap=0** 的同一起点固定可见前缀：未来每个时间点的特征在该固定前缀上可计算；两手半径取最大，再用独立的未来参照核验。不能从发生过返回更新的未来路径半径，倒推起点当时就知道的建议时长。主模型的实时建议时长仍需根将当前前缀外推到多个未来时刻再调用本模块。

真实 probe 的状态不是最终校准方案。当前尚未对它加入独立修正，也没有实际性能结论；它将揭示阻尼基线的稀疏条件覆盖/半径与路径差异，供根后续门控/策略模型校准参考。

# Q2 运动专家与基础系数专项

执行者：Codex 子智能体 `q1_review_repair`。根在冻结 Q1 迁移结果出现短窗优势后授权本任务。仅独立 `q2_rich_*` 模块/脚本、`artifacts/c_q2/rich_state/`、配套测试与本文档；不改 Q1 冻结、Q2 原专家库和根 handoff。

## 已确认的差异

同窗比较 `existing_matched_windows.csv`、`existing_matched_summary.csv` 对齐既有 mixture/residual/static/固定 50:50 与冻结 Q1 迁移。主延迟 0.1 s 下，Q1 在 0.2/0.5 s 的全部预算优于上述四个 Q2 候选；1 s 有返回观测时和 2 s 多数条件则是固定 50:50 更好。例如 0.2 s、预算0 的归一化误差为 Q1 .306724、mixture .336310、residual .338477、static .340801、50:50 .337001。不能声称新训练全面优于迁移。

Q2 原库缺少 Q1 的 2/30、.15、.3 s 速度和 13 点二次项；更关键的是，Q1 二次拟合含自由截距，而 Q2 原 5/9 点二次拟合强制穿过当前锚点。Q1 使用距离目标 L-BFGS-B 拟合基础系数，再训练有界残差；Q2 原 residual 固定从 damped_fast_.2 出发。`gate_oof_main` 的“main”仅指主跨度内层选择，拟合本身仍含 2 s。

这些差异提供待检验假设，尚不能仅凭源码认定原因。新方案不使用任何 full-train Q1 模型产生训练 OOF。

## 运行前固定的方案

特征库保留原 16 个专家及全部原特征为精确前缀，追加三个“最近披露样本”速度、锚定 13 点二次项、自由截距 5/9/13 点二次项，共 23 专家；再加入专家长度和速度方向余弦。使用与 Q2 相同的合格已披露历史，最多64个点；新位移在直接观测时为零。没有原始记录句柄或未来标签接口。三项常速精确性、刚体旋转/直接观测、原库前缀测试已通过。

基础系数按数据集与样本年龄的 7 个节点 `0,1/30,.1,.2,.5,1,2` 拟合，在 log(1+年龄) 上线性插值。节点系数限制 [-2,2]，距离目标正则 .001，同时平滑相邻节点；SciPy L-BFGS-B 最多180步，记录收敛标志。基础拟合只从相应拟合环境中按 episode×horizon×scenario×hand 固定随机优先级取最多16行；精确索引保存。合成可恢复线性运动检查中，原阻尼误差 .40115 降到 3.66e-7，求解器75步收敛。

神经修正保持两层48单元 SiLU，不增加网络深度；输出 `.5*tanh` 加在基础系数上，初始末层全零。AdamW lr .001、weight decay .01、梯度截断2、batch1024、最多800更新，每25更新检查主跨度内层误差，8次不改善停止，允许选择更新0。按真实更新次数比较，原点初始化也保存。

固定五个候选：

| 专家库 | 先验 | 拟合窗口 |
|---|---|---|
| 原16个 | 固定阻尼.2 | 含2s |
| 原16个 | 拟合年龄系数 | 含2s |
| 扩展23个 | 固定阻尼.2 | 含2s |
| 扩展23个 | 拟合年龄系数 | 含2s |
| 扩展23个 | 拟合年龄系数 | 仅≤1s |

每个外折4环境完全留出；剩余12中固定4作内层选择、8拟合。每个外折独立跑同一五格方案，避免把 fold0 的外部表现传给其他折。选中配置再以三个种子选更新数、在外层12环境拟合并评外层4；尺度和标准化各自只来自拟合环境。最终全train配置由四折内层平均名次确定，再用固定12/4训练分割选更新数。全train checkpoint 哈希冻结后才评新的官方 validation；test不评分。

GPU配额20%，CPU准备特征；训练集扩展缓存逐记录要求旧专家、旧特征、锚点精确一致，并逐键验证窗口/请求条件/手/时间顺序。新增特征与原目标的连接发生在预测接口之外。

## 校准接口和边界

保存每个内层 checkpoint 的拟合/留出环境、先验拟合行、全部训练行与内层预测行、噪声、完整主+压力预测及哈希。**内层主跨度标签参与早停与方案选择**，这些残差明确标为开发残差，不能冒充未参与选择的独立校准残差。外层4环境始终不参与基础拟合、标准化或选择。

根后续若建立整条流水线 OOF 校准，要在每个 outer12 内部再产生3×4留出残差，再拟合半径与路径倍率后仅评outer4。当前所存内层结果用于复用/核验，并不自动构成该完整校准。

CPU部署接口为 `src/shuwei_c/q2_rich_live.py::RichPredictor(models,dataset)`，与根回放预测器相同 `(history,now)→两手中心`。接受固定 checkpoint 列表，不访问目标；完成训练后再核对其与 GPU 批处理的数值一致性。

## 当前状态

准备入口 `scripts/prepare_c_q2_rich.py`，训练入口 `scripts/train_c_q2_rich.py`。训练特征、首折筛选、四个外折与全train三种子冻结均已完成，尚未读取新 validation。用户收回本智能体后，后续工作交给根，详见末尾交接。

## 首折内层与接口验收（已完成）

train 扩展缓存完成55记录、1,257,660行，431.37 s，全部旧前缀/锚点/目标键一致。首折8拟合、4内层留出环境的结果：

| 候选 | 初始误差 | 选中误差 | 选中更新 |
|---|---:|---:|---:|
| basic_fixed_all | .695132 | .673068 | 50 |
| basic_fitted_all | .710966 | .698271 | 75 |
| rich_fixed_all | .695132 | .660026 | 100 |
| rich_fitted_all | .717117 | .695447 | 175 |
| rich_fitted_main | .717158 | .698986 | 100 |

本折固定先验的补专家方案相对同训练器 basic 对照改善1.94%，拟合先验未改善。三个拟合先验均达到180迭代上限、未报告收敛；保留目标值、系数与收敛标志，不能称为最优解。五格矩阵保持不变，完整四外折/三种子已以 `--train-only` 启动，先冻结全train模型再评新validation。

首个真正外层 fold0 的主误差为 .825050，而既有 residual .816350、static .806583，说明内层优势尚未转为外层收益。必须等待完整分层结果，不把首折筛选数字作为最终改进。

实际训练内留出回放的 CPU 部署核验 `live_audit.json` 已PASS：三数据集 live CPU 对保存的 GPU 预测最大差分别为 3.67e-7、4.61e-7、6.43e-8，均小于坐标尺度相关 float32 舍入界。平移全部隐藏未来坐标后，首次返回前预测差严格为0，请求账本一致。接口检查不读官方validation或test。

## 用户收回子智能体时的交接

已按根转达的用户指令停止开启新实验，不再启动下一轮 main-only 控制。写交接期间已有进程自然结束；最后只读检查确认如下。

- **原进程 PID 924909 已结束；工具 session 63652**。命令：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/mary/miniconda3/envs/mathorcup/bin/python scripts/train_c_q2_rich.py --train-only`。
- 日志 `/tmp/shuwei_q2_rich_nested.log`；仅执行训练与全train冻结，**没有读取新validation**。没有rich validation特征缓存。
- 四外折已完成；`nested/fold{0,1,2,3}/ensemble_predictions.parquet` 与 `ensemble_summary.csv` 均存在。fold0/2/3选 `rich_fixed_all`，fold1选 `rich_fitted_main`。每折完整五格负结果在 `screen.csv` 和 `screen/<tag>/`。
- 全train按四内层平均名次选 `rich_fixed_all`；三个 `nested/full/seed2026112{0,1,2}.pt/.json` 及 `full/freeze.json` 已完成。日志结尾为 `Training and full model freeze complete; official validation not read`，本阶段381.97 s（首折筛选另59.16 s）。冻结哈希：种子20 `95308838c72456fcb827bcfb6a3edfe1b6ab9ada14daa807f267df05ad87d378`；21 `631cfd810acdfc459fc9a8db67eb2dfb2d207e24beaf443458b421abb0ddab95`；22 `e4518794a36e5759250e48f6c25d4bc12b4e04921659da3992eb69ec1cfc8fee`。
- `nested/manifest.json` 此时不存在属预期：脚本仅在后续官方validation评价完成时写最终manifest，不能以该文件不存在误判train-only失败。
- 若异常退出，先查日志/输出，再用**完全相同命令和源文件**续跑；已有同身份checkpoint会校验哈希后复用。不覆盖已有输出，不修改已运行脚本来绕过身份守卫。

文件所有权交回根：

- `src/shuwei_c/q2_rich_features.py`、`q2_rich_learning.py`、`q2_rich_live.py`。
- `scripts/prepare_c_q2_rich.py`、`scripts/train_c_q2_rich.py`；`tests/test_c_q2_rich_features.py`，3项通过。
- `artifacts/c_q2/rich_state/`：`data/train/`完整缓存、`nested/`完整训练分区/源快照/模型/预测；`compare_existing.py`及对应同窗表；`audit_live.py`、`live_audit.json`；训练损失分解CSV。
- 所有内层预测保存在 `nested/fold*/screen/<tag>/seed*_inner_predictions.parquet`，同行号在 `seed*_rows.npz`，先验抽样在 `prior_rows.npz`，噪声在模型/JSON。内层标签参与选择的事实已明确标注；其本身不是完整流水线 OOF 校准。

仅训练数据的损失诊断：既定原阻尼先验下，2 s 窗口占23.29%权重但贡献50.80%欧氏目标；0.2 s 占26.23%权重但只贡献5.48%。年龄≤0.2 s 累计54.95%权重、只贡献14.15%目标。数据在 `training_loss_by_horizon_s.csv` / `training_loss_by_age_bin.csv`，使用现有全train手尺度和 `q2_learning.row_weights`，不是独立性能结论。

根已指定的下一步（**尚未执行**）：

1. 核对已完成的train-only冻结哈希，汇总四折结果与短跨度差距。保留完整五格负结果和L-BFGS迭代上限状态。
2. 下一轮补 `rich_fixed_main` 和 `basic_fixed_main`，形成固定先验下的纯跨度对照。现矩阵只有 `rich_fitted_main`，不能单凭它排除“2 s拟合压低短时表现”的解释。
3. 可再在训练内检验固定年龄分段权重，避免近零误差相除；评分继续原始欧氏宏平均。保留含2 s训练候选作为压力比较。**不要先读新validation**。
4. 候选确定后，新的官方validation只作开发比较；若采用，再补每个outer12内3×4留出残差的完整流水线校准，最后仅评outer4。

Q1仍保持已验收PDF哈希 `c2a615a0f8ebc259d5cc780ac1af20ea37b8c6837995900bd72fd9cfa91ff31b`，本轮没有再改Q1稿件。

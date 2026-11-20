# Codex 交接：2026 数维杯 C 题数据与环境

> 日期：2026-11-20（比赛期）
> 交接人：ZCode（Windows）
> 接手：Codex（WSL，`~/shuwei2026`）
> 目的：把已完成的文件搬迁、C 题数据解密/解压、校验结果、当前 git 状态和待确认问题一次说清。

## 0. 一句话

C 题数据已从 Windows 搬到 WSL，并完成外层解密、内层解压和三套 Raw_Data 展开；当前可直接读取 `EgoDemo/EgoPro/EgoStandard` 与 `Restocking_Tasks_and_Constraints`。但题目引用的部分数据说明文档在发布包中未找到，**Q2 的正式实验协议在确认前不要自行编造**。

## 1. 比赛与题目

- 比赛：2026 年第十二届数维杯（秋季赛），北京时间 **2026-11-20 09:00—11-24 09:00**；论文提交截止 **11-24 10:00**。
- C 题：**无人超市补货作业的状态估计与决策优化**。
- **C 题已由用户于 2026-11-20 选定**（`docs/collab/decisions.md` D-008）。选题/建模路线见 Cursor 的 `docs/collab/C题选题交接_20261120.md`；本文补充数据、环境、校验与 git 状态，并更新其“原始包尚未解压”的过时表述。
- 本队题号 `SWBMCM2619167`；C 题文件名按 `CSWBMCM2619167.pdf` / `CSWBMCM2619167cns.pdf` / `CSWBMCM2619167fj.zip` 准备，最终以报名系统为准。
- 三个问题：
  1. **Q1 作业状态与观测可靠性建模**：读三套原始记录，检查时间对齐、坐标约定、重复记录、质量标记；定义状态变量；划分可用记录/区间；建立状态与观测可靠性模型；做基线、验证与失效分析。
  2. **Q2 观测缺失期间的状态预测与再观测**：人工遮蔽/回放，预测误差随时间增长、何时请求新观测；比较预测误差、成本、等待时间。
  3. **Q3 不确定条件下的补货作业调度**：用原始记录估计作业时长及不确定性；在 S1/S2/S3 三种情景下决定任务顺序、预留服务时长、付费查阅、等待/停止；报告完成率、准时率、拖期、成本、超时和闲置。
- Q3 的硬条件：S1/S2/S3 分别 8/12/16 个任务，H=100/100/125 秒，成本预算 32/34/40，查阅上限 4/6/8；标称映射 `p = αd + β`，α=1、β=0.5；敏感性 α∈{0.75,1.5}（β=0.5）与 β∈{0,1}（α=1）；30 个固定种子。

## 2. 已完成的搬迁

Windows 源目录已删除，文件已在 WSL：

| 内容 | WSL 路径 |
|---|---|
| 翻译（A—D 题中文译文等） | `/home/mary/shuwei2026/00_赛题与规则/正式发布/翻译/` |
| 官方赛题包（A—D 题 PDF、承诺书、Word/LaTeX 模板、格式规范、提交流程） | `/home/mary/shuwei2026/00_赛题与规则/正式发布/2026年第十二届数维杯全国大学生数学建模挑战赛（秋季赛）赛题/` |
| C 题外层加密包 | `/home/mary/shuwei2026/00_赛题与规则/正式发布/C_Problem_Data.zip` |
| C 题解压根目录 | `/home/mary/shuwei2026/00_赛题与规则/正式发布/C_Problem_Data/` |

Windows 访问前缀：

```text
\\wsl.localhost\Ubuntu-22.04\home\mary\shuwei2026\00_赛题与规则\正式发布\
```

## 3. C 题数据当前结构

```text
C_Problem_Data/
└── Restocking_Problem_and_Data_EN/
    ├── Raw_Data/
    │   ├── EgoDemo/                 # 31 个 session，297 文件，约 1.8G
    │   ├── EgoPro/                  # 14 个 session，308 文件，约 1.9G
    │   ├── EgoStandard/             # 24 个 session，384 文件，约 2.1G
    │   ├── EgoDemo.zip
    │   ├── EgoPro.zip
    │   └── EgoStandard.zip
    └── Restocking_Tasks_and_Constraints/
        ├── tasks.csv                # 36 个 job
        ├── record_split.csv         # episode/environment 划分
        ├── source_segment_index.csv # 349 个源标注区间
        ├── common_conditions.json
        ├── job_families.csv
        ├── service_zones.csv
        └── Problem_3_Requirements.pdf
```

- `C_Problem_Data` 共 **1000 个文件，约 17G**（含保留的三套 Raw_Data 原始 zip）。
- `C_Problem_Data.zip` 外层 **5.5G**，`Restocking_Problem_and_Data_EN.zip` 内层 **5.8G**，均保留。
- 三套 Raw_Data 的 `unzip -t` 均返回 `No errors detected`。
- 密码由用户提供，**没有写入任何文件、git 或本交接文档**；如果必须重新解压，向用户索取，不要猜。

## 4. 数据统计（已核）

| 数据集 | 记录/session | 文件数 | 大小 |
|---|---:|---:|---:|
| EgoDemo | 31 | 297 | 1.8G |
| EgoPro | 14 | 308 | 1.9G |
| EgoStandard | 24 | 384 | 2.1G |
| 合计 | 69 | 989 | 5.8G（未含 zip 与任务文件） |

- 题面 Table 1：69 条记录、24 个环境、66,070 个时间戳；与 `record_split.csv` 的 69 行一致。
- `record_split.csv` 是 **episode 级 55/7/7**；按 `environment_id` 聚合后正好是 **train 16 / validation 4 / test 4 个环境**。两者不要混用。
- `source_segment_index.csv` 349 行（表头+349 区间），题面说 343 个满足 ≥1.2 秒、6 个仅索引不抽样。
- `tasks.csv` 36 行数据（37 行含表头），覆盖 S1/S2/S3。
- `record_split.csv` 带 BOM；读取用 `encoding="utf-8-sig"`。

## 5. 重要差异 / 需要确认

### 5.1 题面引用但发布包中未见到的文档

题面明确引用：

```text
Data_Documentation/record_index.csv
Data_Documentation/missing_observation_experiments.md
Restocking_Tasks_and_Constraints/tasks_and_evaluation.md
file_manifest.csv
```

实际解压后 **没有找到** 这些文件。现有替代/相邻文件：

- `Restocking_Tasks_and_Constraints/record_split.csv`
- `Restocking_Tasks_and_Constraints/source_segment_index.csv`
- `Restocking_Tasks_and_Constraints/Problem_3_Requirements.pdf`

处理建议：

1. 先按现有文件做数据 QC 和探索性分析；
2. **Q2 的正式人工遮蔽/回放协议不要自行编造**，因为 `missing_observation_experiments.md` 缺失；先向用户/组委会确认是否有补充包或说明。
3. 不要把 `source_segment_index.csv` 直接当成题面所说 `record_index.csv` 的同义文件；列名和语义要先核对。
4. `Problem_3_Requirements.pdf` 内容很完整，Q3 可先按它推进。

### 5.2 数据格式与题面表述不一致

题面说：

- EgoDemo：JSON、Parquet、MP4；
- EgoPro / EgoStandard：内嵌消息模式定义的 MCAP 容器。

实际解压后：

- EgoDemo：124 个 JSON、93 个 Parquet、80 个 MP4；
- EgoPro：252 个 CSV、56 个 MP4；
- EgoStandard：336 个 CSV、48 个 MP4；
- **没有发现 `.mcap` 文件**。

这可能是官方把 MCAP 转成了 CSV，或发布的是转换后的本地子集。不要假设 CSV 与 MCAP 字段完全等价；先读 `session_metadata.csv`、`tables/*.csv`、`videos/*_frames.csv` 的实际列名，再决定解析器。

### 5.3 数据集内部结构

- EgoDemo 是 LeRobot 风格：`EgoDemo/EXXX/meta/episodes/chunk-000/file-000.parquet`、`info.json`、`stats.json`、`tasks.parquet`；`videos/observation.images.*/chunk-000/file-000.mp4`。
- EgoPro/EgoStandard 每个 session 是 `session_metadata.csv` + `tables/*.csv` + `videos/*.mp4` + `videos/*_frames.csv`。
- 不同记录的世界坐标原点各自定义，不能据此还原统一门店布局；腕部相机世界外参未提供，不能假定四路视频已统一标定。
- 原始质量标记只针对人员手部，不能直接推断商品是否被遮挡或作业是否成功。

## 6. Codex 建议接手顺序

1. 读 `翻译/2026数维杯C题_中文译文.pdf` 和 `赛题/C题/2026 “ShuWei Cup” C_Problem.pdf`，确认 Q1—Q3 与提交要求。
2. 读 `AGENTS.md`、`docs/collab/CURSOR_HANDOFF.md`、skill `SKILL.md` 和 `references/01`。
3. 做数据 QC：
   - 按 `record_split.csv` 聚合 `environment_id`，确认 16/4/4；
   - 统计每个数据集的记录、时间戳、字段、缺失率、质量标记；
   - 检查 EgoDemo 的 Parquet/JSON 与 EgoPro/Standard 的 CSV 时间/单位/坐标；
   - 先落盘 `数据对账.md`，不要直接开始建模。
4. Q1：定义手部代表点/状态变量，建立基线（常值/简单线性），按环境划分拟合/验证/测试，报告失效案例。
5. Q2：先确认缺失的 `missing_observation_experiments.md`；在协议确认前，只做探索性回放，不把自定义协议写成官方口径。
6. Q3：按 `Problem_3_Requirements.pdf` + `tasks.csv` + `common_conditions.json` 实现情景模拟器、基线策略、预留时长与查阅策略；用 30 个种子做配对比较。
7. 正式论文/图/交付包仍由 Codex 写；ZCode 只做辅助与复核。

## 7. Git 状态

- 比赛 public 仓库：`https://github.com/aalandbartelt-cyber/ShuWeiCup_2026_MARY`
  - 远端 `main`：`0066667 chore(git): ignore local official release data`
  - 本地 `main` 已对齐 `origin/main`，工作区干净。
- skill 私有仓库：`https://github.com/aalandbartelt-cyber/mary-modeling-contest-pipeline-shuwei`
  - 远端 `main`：`2c3d50c docs(submission): enforce 2026-11-17 three-slot rules`
  - 本地 `~/shuwei_skill_release` 已对齐，工作区干净。
- `00_赛题与规则/正式发布/` 已加入 `.gitignore`；C 数据、官方 PDF、翻译、raw 视频均不入库。
- WSL 的 `.bashrc`/`.profile` 已改成“7890 端口可达才启用代理，否则直连”；Windows `.gitconfig` 里失效的 GitHub 专用 7892 代理已移除。

## 8. 快速检查命令

```bash
BASE="$HOME/shuwei2026/00_赛题与规则/正式发布"
export BASE

# 目录
find "$BASE/C_Problem_Data/Restocking_Problem_and_Data_EN" -maxdepth 2 -type d

# 三套数据文件数
for d in "$BASE"/C_Problem_Data/Restocking_Problem_and_Data_EN/Raw_Data/Ego*; do
  [ -d "$d" ] && printf '%s: %s files\n' "$(basename "$d")" "$(find "$d" -type f | wc -l)"
done

# 划分：episode 与 environment 两个口径
python3 - <<'PY'
import csv, os, collections
p=os.path.expanduser("~/shuwei2026/00_赛题与规则/正式发布/C_Problem_Data/Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints/record_split.csv")
rows=list(csv.DictReader(open(p, encoding="utf-8-sig")))
for s in sorted(set(r["split"] for r in rows)):
    sub=[r for r in rows if r["split"]==s]
    print(s, "episodes", len(sub), "envs", len(set(r["environment_id"] for r in sub)))
PY

# 原始 zip 完整性
for z in "$BASE"/C_Problem_Data/Restocking_Problem_and_Data_EN/Raw_Data/*.zip; do
  unzip -tq "$z" >/dev/null && echo "OK $(basename "$z")"
done
```

## 9. 风险与纪律

- 数据总量约 23G（5.5G 外层 + 5.8G 内层 + 5.8G Raw zip + 5.8G 展开），WSL 剩余约 774G；不要重复解压。
- 不要把 `C_Problem_Data`、`正式发布/` 或任何 raw 视频/表格提交进 public 仓库。
- 不要写死密码；不要猜测缺失文档的内容；Q2 协议缺失是当前唯一明确阻塞。
- 任何“看起来像官方文档”的自拟协议，都要在论文/交接中标注为团队假设，不能冒充官方要求。

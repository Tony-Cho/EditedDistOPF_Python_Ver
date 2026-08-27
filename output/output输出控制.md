# KNN 训练/预测结果输出格式约定（v1）

本文档定义 KNN 代理模型的输出格式，覆盖两个程序：
- **train_knn.py**：基于 MC 抽样 OPF 结果训练 KNN 回归模型（min/max 两个方向），训练产物见第一部分；
- **scenario_dataset_mc.py**：用已训练模型对场景做概率化预测，预测输出见第二部分。

所有输出统一保存在 `output/` 目录下。示例以训练集 `training_dataset_default`、场景 `output_scenario_default` 的实际输出为基准。

**通用约定**：
- **训练产物**：`knn_lib/{训练集文件夹名}/`，训练集文件夹名取自 `training_dataset/` 下训练集所在目录名（如 `training_dataset/training_dataset_default/csv/` → `knn_lib/training_dataset_default/`）；模型与使用场景解耦（先训练、后选场景用）。
- **预测输出**：`output/{场景名}/`，场景名对应 `scenario/` 下场景子文件夹名；该场景使用的 KNN 模型**不复制**到场景下，由场景配置 `output_mc_config.csv` 的 `model` 键索引到总库 `knn_lib/{训练集文件夹名}/`（见第二部分 1.1 节）。
- `sense` 列：`min` = 最小化根节点注入模型，`max` = 最大化根节点注入模型。
- **列名单位与值单位严格一致**：功率列值即 MW / Mvar（根节点注入 `p_sub_mw`、`q_sub_mvar` 来自 OPF 结果 training_dataset_system.csv）；R²/误差等指标无量纲。
- 文件编码 UTF-8，表头行存在，列顺序与本文档一致。

---

# 第一部分：训练输出（train_knn.py）

## 1. 输出目录结构

```
knn_lib/{训练集文件夹名}/
├── knn_config.csv                  # 模型训练参数 (读入, knn_* 键; 可选, 见 1.1 节)
├── knn_params.csv                  # 模型参数 + 数据规模 (每 sense 一行)
├── knn_metrics.csv                 # 评估指标 R²/MAE/RMSE (每 sense×目标 一行)
├── knn_predictions.csv             # 测试集逐样本预测对比 (每 sense×样本×目标 一行)
├── knn_model_{min,max}.joblib      # KNN 回归模型 (KNeighborsRegressor)
├── knn_scaler_{min,max}.joblib     # StandardScaler (训练集拟合)
├── knn_feature_names.json          # 输入特征列名 (默认算例 45, 顺序与模型一致)
└── knn_target_names.json           # 输出目标列名 (18, 与 y 列一一对应)
```

示例（默认训练集）：

```
knn_lib/training_dataset_default/
├── knn_params.csv
├── knn_metrics.csv
├── knn_predictions.csv
├── knn_model_min.joblib / knn_model_max.joblib
├── knn_scaler_min.joblib / knn_scaler_max.joblib
├── knn_feature_names.json
└── knn_target_names.json
```

> `knn_config.csv` 为可选文件：默认训练集未提供时，train_knn.py 直接使用内置默认参数（或 CLI 覆盖）。

### 1.1 模型训练参数（knn_config.csv，读入）

放在 `knn_lib/{训练集文件夹名}/knn_config.csv`（KNN 参数已从 training_dataset 的 output_mc_config.csv 统一挪入），`name,value` 两列，train_knn.py 训练时读取。示例：

| name | value |
|---|---|
| knn\_k | 5 |
| knn\_weights | distance |
| knn\_test\_size | 0.2 |
| knn\_seed | 42 |
| knn\_metric |  |
| knn\_p |  |
| knn\_algorithm |  |
| knn\_leaf\_size |  |
| knn\_n\_jobs |  |

- 键含义与可选值见 [修改进度7.md](修改进度7.md) 第 5 节；留空 = sklearn 默认；
- 优先级：CLI（`--k/--weights/--test-size/--seed`） > knn_config.csv 非空值 > 内置默认值；
- 首次训练前若 `knn_lib/{训练集文件夹名}/` 不存在，可先手动创建该文件（train_knn 不生成，缺省则用默认参数）。

## 2. knn_params.csv（模型参数 + 数据规模）

每 **sense** 一行，记录该模型实际生效的全部参数（来源优先级：CLI 显式 > knn_config.csv 非空值 > 内置默认值）。

字段定义：

| 字段 | 类型 | 说明 |
|---|---|---|
| sense | str | 目标函数方向: min/max |
| k | int | KNN 邻居数（n_neighbors） |
| weights | str | 权重方式: uniform（等权平均）/ distance（距离倒数加权） |
| metric | str | 距离度量: minkowski / euclidean / manhattan / chebyshev / cosine |
| p | float | minkowski 范数指数（p=2 欧氏、p=1 曼哈顿） |
| algorithm | str | 最近邻搜索算法: auto / brute / ball_tree / kd_tree |
| leaf_size | int | 树索引叶节点大小（仅 ball_tree/kd_tree 生效） |
| n_jobs | int/空 | 搜索并行核数（-1 = 全部核；留空 = sklearn 默认） |
| test_size | float | 测试集占比（0~1，其余为训练集） |
| seed | int | 训练/测试划分随机种子（保证可复现） |
| n_train | int | 训练样本数 |
| n_test | int | 测试样本数 |
| n_features | int | 输入特征维数 |
| n_targets | int | 输出目标个数（多输出回归） |
| data_csv | str | 训练数据源目录（training_dataset/ 下 CSV 目录） |

示例：

| sense | k | weights | metric | p | algorithm | leaf_size | n_jobs | test_size | seed | n_train | n_test | n_features | n_targets | data_csv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| max | 5 | distance | minkowski | 2 | auto | 30 |  | 0.2 | 42 | 15360 | 3840 | 45 | 18 | training_dataset\training_dataset_default\csv |
| min | 5 | distance | minkowski | 2 | auto | 30 |  | 0.2 | 42 | 15360 | 3840 | 45 | 18 | training_dataset\training_dataset_default\csv |

## 3. knn_metrics.csv（评估指标）

每 **sense × 目标** 一行，在测试集上评估。目标由训练数据集动态生成，共 **18 个**：

| 目标类别 | target 值 |
|---|---|
| 根节点注入 | `p_sub_mw`（有功 MW）、`q_sub_mvar`（无功 Mvar） |
| 储能出力 | `BESS_Bus18_p_net_mw` / `BESS_Bus18_q_mvar` / `BESS_Bus18_se_mwh` |
| 光伏出力 | 各 PV 的 `{name}_p_out_mw` / `{name}_q_out_mvar` |
| 可调度负荷实际有功 | `EV_Bus19_p_out_mw` / `EV_Bus7_p_out_mw` / `AC_Bus2_p_out_mw` |

字段定义：

| 字段 | 类型 | 说明 |
|---|---|---|
| sense | str | 目标函数方向: min/max |
| target | str | 预测目标列（见上表，与 knn_predictions 的 target 一致） |
| r2 | float | 决定系数 R²（越接近 1 越好） |
| mae | float | 平均绝对误差（与 target 同单位） |
| rmse | float | 均方根误差（与 target 同单位） |

示例：

| sense | target | r2 | mae | rmse |
|---|---|---|---|---|
| max | p_sub_mw | 0.995317 | 0.034368 | 0.043705 |
| max | q_sub_mvar | 0.838167 | 0.024406 | 0.032917 |
| max | BESS_Bus18_p_net_mw | 1.000000 | 0.000000 | 0.000000 |
| max | BESS_Bus18_q_mvar | 0.993239 | 0.003929 | 0.009113 |
| max | BESS_Bus18_se_mwh | 0.938047 | 0.164769 | 0.214900 |

## 4. knn_predictions.csv（测试集逐样本预测对比）

每 **sense × 测试样本 × 目标** 一行（N_test = n_test，总行数 = 2 × N_test × 18 目标）。可用于误差分布可视化（散点图/箱线图）与 OPF 真值逐样本对比。

字段定义：

| 字段 | 类型 | 说明 |
|---|---|---|
| sense | str | 目标函数方向: min/max |
| sample_id | int | 样本编号（对应 training_dataset_sample.csv 的样本编号，每断面内从 1 起） |
| time_slot | str | 时间断面（H:MM，对应 MC 抽样的断面时刻） |
| target | str | 预测目标列（18 个，见第 3 节目标表） |
| y_true | float | OPF 求解真值（与 target 同单位 MW/Mvar/MWh） |
| y_pred | float | KNN 模型预测值（与 target 同单位） |
| abs_err | float | 绝对误差 = \|y_true − y_pred\|（与 target 同单位） |

示例：

| sense | sample_id | time_slot | target | y_true | y_pred | abs_err |
|---|---|---|---|---|---|---|
| max | 35 | 4:00 | p_sub_mw | 5.237343 | 5.226541 | 0.010802 |
| max | 35 | 4:00 | q_sub_mvar | 1.432585 | 1.548355 | 0.115770 |
| max | 60 | 11:45 | p_sub_mw | 5.422613 | 5.379468 | 0.043145 |

## 5. 约定汇总

1. 三张 CSV 为同一次训练的两类表达：`knn_params` 记录"怎么训的"，`knn_metrics`/`knn_predictions` 记录"训得怎么样"（汇总指标 + 逐样本明细）。
2. `knn_predictions` 以 `sense + sample_id + time_slot + target` 定位，可与 `training_dataset/{场景文件夹名}/csv/training_dataset_system.csv`（OPF 真值）按 `sample_id + time_slot + sense` 交叉核对。
3. 参数来源优先级：**CLI 显式 > knn_config.csv 非空值 > 内置默认值**；配置留空的键不写入 knn_params.csv 的取值逻辑，采用 sklearn 默认并如实记录在列中（如 `n_jobs` 留空）。
4. 每次运行**直接覆盖** `knn_lib/{训练集文件夹名}/` 下同名文件（无历史备份）。

---

# 第二部分：预测输出（scenario_dataset_mc.py）

## 1. 输出目录结构

```
output/{场景名}/
├── output_mc_config.csv                  # 场景预测配置 (输入, model 索引 + scenario + 抽样配置; 见 1.1 节)
├── output_sample.csv              # 抽样输入场景表 (每 断面×样本 一行, 45 特征列)
├── output_system.csv              # 根节点注入 (sense + p_sub_mw/q_sub_mvar)
├── output_storage.csv             # 储能出力 (p_net/q/se)
├── output_pvs.csv                 # 光伏出力 (p_out/q_out)
└── output_loads.csv               # 可调度负荷实际有功 (ev/ac 的 p_out)
```

示例（默认场景）：

```
output/output_scenario_default/
├── output_mc_config.csv                  # scenario=output_scenario_default, model=training_dataset_default + 抽样配置
├── output_sample.csv
├── output_system.csv
├── output_storage.csv
├── output_pvs.csv
└── output_loads.csv
```

> 场景下**不保存模型副本**：模型由 output_mc_config.csv 的 `model` 键索引到总库 `knn_lib/{model}/`，运行时直接加载。

### 1.1 场景预测配置（output_mc_config.csv，读入）

放在 `output/{场景名}/output_mc_config.csv`，**结构与 training_dataset 下的 training_dataset_mc_config.csv 一致**（全局运行参数 + 组件分布，抽样参数在组件行内单独设置），额外多一个 `model` 键索引模型库。由 `--config` 显式指定，或自动查找（顺序：`--scenario` 目录 → 默认算例 `output_scenario_default` → output/ 下唯一一份配置；多份且无 `--scenario` 时必须 `--config`）。示例（默认场景实际配置）：

| name | value | 参数列 1 | 参数列 2 | 备注 |
|---|---|---|---|---|
| model | training\_dataset\_default | - | - | 模型名 = 训练集文件夹名（索引 knn\_lib/{model}/） |
| scenario | output\_scenario\_default | - | - | 场景名（scenario/{scenario}/ 或去除 output\_ 前缀回退） |
| n\_samples | 200 | - | - | - |
| seed | 42 | - | - | - |
| start\_time | 0:00 | - | - | - |
| end\_time | 23:45 | - | - | - |
| EV\_Bus19\_lb | truncated\_normal | mu:EV\_Bus19\_lb\_mu | sigma:EV\_Bus19\_lb\_sigma | - |
| EV\_Bus19\_ub | truncated\_normal | mu:EV\_Bus19\_ub\_mu | sigma:EV\_Bus19\_ub\_sigma | - |
| EV\_Bus7\_lb | truncated\_normal | mu:EV\_Bus7\_lb\_mu | sigma:EV\_Bus7\_lb\_sigma | - |
| EV\_Bus7\_ub | truncated\_normal | mu:EV\_Bus7\_ub\_mu | sigma:EV\_Bus7\_ub\_sigma | - |
| AC\_Bus2\_lb | truncated\_normal | mu:AC\_Bus2\_lb\_mu | sigma:AC\_Bus2\_lb\_sigma | - |
| AC\_Bus2\_ub | truncated\_normal | mu:AC\_Bus2\_ub\_mu | sigma:AC\_Bus2\_ub\_sigma | - |

- 全局键：`model`（模型索引）/ `scenario`（场景名）/ `n_samples` / `seed` / `start_time` / `end_time`；
- 组件分布：EV/AC 的 `lb/ub` 等抽样量单独设置分布参数——`mu:曲线名`/`sigma:曲线名` 逐断面取场景 shapes 曲线的 μ 与 σ（默认算例写法），或 `cv:数值` 常数方式（σ=cv×μ，兼容旧配置）；**未配置的抽样量固定为曲线值（方案A）**——预测场景默认只配置 EV/AC 的 lb/ub；
- 优先级：CLI 显式（`--config/--scenario/--model/--n/--seed/--start-time/--end-time`） > output_mc_config.csv > 内置默认。

## 2. 抽样与预测规则

- 输入特征 45 维，由场景曲线 + 抽样量构成：
  - **固定（曲线值）**：32 个负荷当前功率（`Load1~32` 取场景 shapes 的 `Load*_cur` 曲线）、5 个 PV 辐照度（`PV_Bus*_irr` 曲线）、储能初始能量（`BESS_Bus18_en` 曲线 × 能量窗口上限）/ 初始功率 p\_init = 0；
  - **抽样（截断正态）**：EV/AC 的 `lb/ub` 共 6 个，μ/σ 逐断面取场景 shapes 曲线（`*_lb_mu`/`*_lb_sigma` 等），ub 先抽（截断 [0,1]），lb 后抽并截断于 [0, ub]（保序）；
- 每个断面独立抽样 N 次，min/max 共用同一批抽样；
- 输出行数 = 断面数 × N × 2（sense）。

## 3. 组件表字段（与 training_dataset 输出对齐，前缀 training_dataset → output）

- `output_system.csv`：`sample_id, time_slot, sense, p_sub_mw, q_sub_mvar`（KNN 预测值）；
- `output_storage.csv`：`sample_id, time_slot, sense, name, bus, p_net_mw, q_mvar, se_mwh`（正=放电）；
- `output_pvs.csv`：`sample_id, time_slot, sense, name, bus, p_out_mw, q_out_mvar`；
- `output_loads.csv`：`sample_id, time_slot, sense, name, bus, type, p_out_mw`（仅 ev/ac 可调度负荷）；
- **不含 training_dataset_buses / training_dataset_lines**：KNN 未训练电压/潮流目标，无法预测。

## 4. 约定汇总

1. `output_{*}.csv` 为 KNN 模型预测值（非 OPF 求解真值），`p_sub` 等精度取决于模型 R²（见对应训练集 knn_metrics.csv）。
2. 模型加载自 `knn_lib/{训练集文件夹名}/` 总库，由 output_mc_config.csv 的 `model` 键索引；场景下不保存模型副本。
3. 每次运行**直接覆盖** `output/{场景名}/` 下同名 CSV（无历史备份）；output_mc_config.csv 为输入配置，保留不被覆盖。

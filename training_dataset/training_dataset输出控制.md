# CSV 配电网 OPF 输出格式约定（v1）

本文档定义程序运行结果的输出格式，覆盖两个程序：

- **main.py**：单断面 VPP OPF，输出见第一部分；
- **training\_dataset\_mc.py**：蒙特卡洛抽样 OPF，输出见第二部分。

所有输出统一保存在 `training_dataset/` 目录下。示例以默认算例 `default`（模型 `model_default`，输出目录 `training_dataset/training_dataset_default/`）的实际输出为基准。
数值均为**工程单位**：功率 MW、无功 Mvar、能量 MWh、电压 pu（由 pu 值 × base\_mva 换算，base\_mva 见输入 model\_circuit.csv）。

**通用约定**：

- `sense` 列：`min` = 最小化根节点注入，`max` = 最大化根节点注入。
- **列名单位与值单位严格一致**：列名带 `_mw`/`_mvar`/`_mwh` 时，值即 MW/Mvar/MWh；无量纲比例（辐照度、挂载比例、z、pct）列名不加单位后缀。
- 文件编码 UTF-8，表头行存在，列顺序与本文档一致。

***

# 第一部分：main 输出（main.py）

## 1. 输出目录结构

```
training_dataset/{scenario}/
├── {scenario}.txt      # 概览文件: 模型摘要 + min/max 两个场景的完整结果
└── csv/                # 组件结果 CSV
    ├── training_dataset_system.csv      # 系统级指标
    ├── training_dataset_buses.csv       # 节点电压 + 净注入功率
    ├── training_dataset_lines.csv       # 支路潮流 + 损耗 + 热极限
    ├── training_dataset_loads.csv       # 负荷 (满载/当前/实际功率 + 调节因子 z)
    ├── training_dataset_pvs.csv         # 光伏出力
    └── training_dataset_storage.csv     # 储能充放/能量状态
```

## 2. training\_dataset\_system.csv（系统级指标）

字段定义：

| 字段             | 类型    | 说明               |
| -------------- | ----- | ---------------- |
| sense          | str   | 场景: min/max      |
| objective\_mw  | float | 目标函数值（根节点注入 MW）  |
| p\_sub\_mw     | float | 根节点 (slack) 有功注入 |
| q\_sub\_mvar   | float | 根节点 (slack) 无功注入 |
| p\_loss\_mw    | float | 全网有功损耗           |
| q\_loss\_mvar  | float | 全网无功损耗           |
| solve\_time\_s | float | 求解时间 (s)         |
| status         | str   | 求解状态             |

示例：

| sense | objective\_mw | p\_sub\_mw | q\_sub\_mvar | p\_loss\_mw | q\_loss\_mvar | solve\_time\_s | status  |
| ----- | ------------- | ---------- | ------------ | ----------- | ------------- | -------------- | ------- |
| min   | 2.20455       | 2.20455    | 2.438891     | 0.137985    | 0.105347      | 0.1064         | OPTIMAL |
| max   | 5.37204       | 5.37204    | 2.185812     | 0.579267    | 0.519306      | 0.0291         | OPTIMAL |

## 3. training\_dataset\_buses.csv（节点电压 + 净注入）

字段定义：

| 字段           | 类型    | 说明                |
| ------------ | ----- | ----------------- |
| bus          | str   | 节点名（按编号排序）        |
| sense        | str   | 场景: min/max       |
| vm\_pu       | float | 节点电压 (pu)         |
| p\_inj\_mw   | float | 节点**净注入**有功（注入为正） |
| q\_inj\_mvar | float | 节点**净注入**无功（注入为正） |

**净注入定义**：slack 注入 + PV 出力 + 储能放电 − 负荷消耗 − 储能充电，为负表示该节点为净消费。

示例：

| bus  | sense | vm\_pu   | p\_inj\_mw | q\_inj\_mvar |
| ---- | ----- | -------- | ---------- | ------------ |
| bus1 | min   | 1.05     | 2.20455    | 2.438891     |
| bus2 | min   | 1.048111 | -0.12      | -0.07        |

## 4. training\_dataset\_lines.csv（支路潮流 + 损耗 + 热极限）

字段定义：

| 字段                 | 类型    | 说明             |
| ------------------ | ----- | -------------- |
| name / fbus / tbus | str   | 线路名 / 首端 / 末端  |
| sense              | str   | 场景: min/max    |
| p\_mw              | float | 支路首端有功潮流       |
| q\_mvar            | float | 支路首端无功潮流       |
| s\_max\_mva        | float | 支路热极限容量        |
| p\_loss\_mw        | float | 该支路有功损耗 (r·I²) |
| q\_loss\_mvar      | float | 该支路无功损耗 (x·I²) |

示例：

| name     | fbus | tbus | sense | p\_mw   | q\_mvar  | s\_max\_mva | p\_loss\_mw | q\_loss\_mvar |
| -------- | ---- | ---- | ----- | ------- | -------- | ----------- | ----------- | ------------- |
| L1\_1\_2 | bus1 | bus2 | min   | 2.20455 | 2.438891 | 9.209661    | 0.005639    | 0.002875      |

## 5. training\_dataset\_loads.csv（负荷）

字段定义：

| 字段                 | 类型    | 说明                                   |
| ------------------ | ----- | ------------------------------------ |
| name / bus         | str   | 负荷名 / 挂接母线                           |
| type               | str   | 类型: fixed / ev / ac / fixed\_extra   |
| sense              | str   | 场景: min/max                          |
| p\_full\_mw        | float | **满载**有功（上报容量）                       |
| q\_full\_mvar      | float | **满载**无功                             |
| z                  | float | 调节因子 z（z=1 保持当前挂载）                   |
| p\_cur\_mw         | float | 当前实际挂载功率 = 满载 × 当前挂载比例               |
| p\_out\_mw         | float | 优化后实际功率 = p\_cur × z                 |
| pct\_of\_full\_pct | float | 实际功率占满载百分比（可调度负荷 = z × 当前挂载比例 × 100） |
| q\_out\_mvar       | float | 优化后实际无功 = q\_cur × z                 |

示例：

| name      | bus   | type  | sense | p\_full\_mw | q\_full\_mvar | z   | p\_cur\_mw | p\_out\_mw | pct\_of\_full\_pct | q\_out\_mvar |
| --------- | ----- | ----- | ----- | ----------- | ------------- | --- | ---------- | ---------- | ------------------ | ------------ |
| Load1     | bus2  | fixed | min   | 0.1         | 0.06          | 1.0 | 0.1        | 0.1        | 100.0              | 0.06         |
| EV\_Bus19 | bus19 | ev    | min   | 0.4         | 0.1           | 0.6 | 0.2        | 0.12       | 30.0               | 0.03         |

## 6. training\_dataset\_pvs.csv（光伏出力）

字段定义：

| 字段                  | 类型    | 说明           |
| ------------------- | ----- | ------------ |
| name / bus          | str   | 光伏名 / 母线     |
| sense               | str   | 场景: min/max  |
| p\_avail\_mw        | float | 当前辐照度下可用出力上限 |
| p\_out\_mw          | float | 优化后实际有功出力    |
| q\_out\_mvar        | float | 优化后实际无功出力    |
| pct\_of\_avail\_pct | float | 实际出力占可用出力百分比 |

示例：

| name      | bus   | sense | p\_avail\_mw | p\_out\_mw | q\_out\_mvar | pct\_of\_avail\_pct |
| --------- | ----- | ----- | ------------ | ---------- | ------------ | ------------------- |
| PV\_Bus6  | bus6  | min   | 0.195        | 0.195      | 0.039596     | 100.0               |
| PV\_Bus19 | bus19 | max   | 0.195        | 0.0        | 0.0          | 0.0                 |

## 7. training\_dataset\_storage.csv（储能）

字段定义：

| 字段         | 类型    | 说明                                   |
| ---------- | ----- | ------------------------------------ |
| name / bus | str   | 储能名 / 母线                             |
| sense      | str   | 场景: min/max                          |
| p\_ch\_mw  | float | 充电功率（0\~上限）                          |
| p\_dis\_mw | float | 放电功率（0\~上限）                          |
| p\_net\_mw | float | 净功率 = p\_dis − p\_ch，**正=放电**（向电网送电） |
| q\_mvar    | float | 无功出力                                 |
| se\_mwh    | float | 当前能量状态                               |
| soc\_pct   | float | SOC = se / 额定能量 × 100（相对额定容量）        |

示例：

| name        | bus   | sense | p\_ch\_mw | p\_dis\_mw | p\_net\_mw | q\_mvar   | se\_mwh  | soc\_pct |
| ----------- | ----- | ----- | --------- | ---------- | ---------- | --------- | -------- | -------- |
| BESS\_Bus18 | bus18 | min   | 0.0       | 1.343434   | 1.343434   | -0.004914 | 0.835859 | 16.7172  |
| BESS\_Bus18 | bus18 | max   | 1.631523  | 0.783752   | -0.847771  | 0.82171   | 2.974945 | 59.4989  |

## 8. 约定汇总

1. `p_inj` 为节点净注入（注入为正）；`p_net`（储能）为净放电（放电为正）。
2. `pct_of_full_pct`（负荷）= 实际功率 / 满载 × 100；`pct_of_avail_pct`（光伏）= 实际出力 / 可用出力 × 100。
3. 概览 txt 与 CSV 为同一运行的两份表达：txt 面向阅读，CSV 面向数据处理。

***

# 第二部分：sample 输出（training\_dataset\_mc.py）

## 1. 输出目录结构

```
training_dataset/training_dataset_{scenario}/
├── training_dataset_mc_config.csv# 全局控制文件 (读入; 组件分布配置见第 5 节)
├── training_dataset_sample.csv# 抽样场景表 (每 断面×样本 一行, min/max 共用同一批抽样)
├── logs/                     # txt 日志子文件夹
│   ├── 00_00_0001_{scenario}.txt   # 每 断面×样本 一个: {HH_MM}_{序号:04d}_{场景名}.txt
│   ├── 00_00_0002_{scenario}.txt   # 序号每断面内从 0001 起独立编号, 补齐四位
│   └── ...
└── csv/                      # 结果 CSV
    ├── training_dataset_system.csv# 组件结果表 (与 main 输出同字段, 首列 sample_id, 次列 time_slot)
    ├── training_dataset_buses.csv
    ├── training_dataset_lines.csv
    ├── training_dataset_loads.csv
    ├── training_dataset_pvs.csv
    └── training_dataset_storage.csv
```

## 2. training\_dataset\_sample.csv（抽样场景表）

每 断面×样本 一行；第一列 `sample_id`（样本编号，**每断面内从 1 起独立编号**），第二列 `time_slot`（时间断面 `H:MM`，标识该行属于哪个断面），后接各抽样量列。
**列名 = 组件名\_指标，列名单位与值单位严格一致。**

字段定义：

| 列名                                             | 说明                                                      | 单位      |
| ---------------------------------------------- | ------------------------------------------------------- | ------- |
| sample\_id                                     | 样本编号（每断面内从 1 起）                                         | -       |
| time\_slot                                     | 时间断面（如 10:00），由 mc\_config 的 start\_time/end\_time 决定范围 | -       |
| Load1\_mw / Load2\_mw / …                      | 各固定负荷抽样后满载有功（默认算例 Load1~32 共 32 个）                      | MW      |
| fixed\_total\_mw                               | 固定负荷总抽样功率（聚合）                                           | MW      |
| PV\_Bus6\_irr / …                              | 各光伏辐照度抽样                                                | 0-1 无量纲 |
| BESS\_Bus18\_se\_init\_mwh / …                 | 各储能初始能量抽样                                               | MWh     |
| BESS\_Bus18\_p\_init\_mw / …                   | 各储能初始功率抽样（正=放电，负=充电）                                    | MW      |
| EV\_Bus19\_cur / EV\_Bus19\_lb / EV\_Bus19\_ub | 各可调度负荷基准值（当前挂载）/ 可调下限 / 可调上限                            | 比例无量纲   |

示例（默认算例实际输出，省略中间列）：

| sample\_id | time\_slot | Load1\_mw | Load2\_mw | … | fixed\_total\_mw | PV\_Bus6\_irr | … | BESS\_Bus18\_se\_init\_mwh | BESS\_Bus18\_p\_init\_mw | … | EV\_Bus19\_cur | EV\_Bus19\_lb | EV\_Bus19\_ub |
| ---------- | ---------- | --------- | --------- | - | ---------------- | ------------- | - | -------------------------- | ----------------------- | - | -------------- | ------------- | ------------- |
| 1          | 0:00       | 0.0422    | 0.0327    | … | 1.5181           | 0.0           | … | 1.4116                     | -1.8012                 | … | 0.4200         | 0.2688        | 0.4730        |
| 2          | 0:00       | 0.0434    | 0.0359    | … | 1.4439           | 0.0           | … | 1.0823                     | -2.0618                 | … | 0.3919         | 0.2742        | 0.5225        |

## 3. 组件结果表（system/buses/lines/loads/pvs/storage）

- 字段与 main 输出完全一致（见第一部分第 2\~7 节）；
- 额外在**第一列插入** **`sample_id`**（样本编号，每断面内从 1 起），**第二列为** **`time_slot`**（时间断面），**第三列为** **`sense`**，其余列不变；
- 所有 断面×样本 结果堆叠：N 个样本 × K 个断面 → 每表 N × K × 2（min/max）组行。

示例（training\_dataset\_system.csv）：

| sample\_id | time\_slot | sense | objective\_mw | p\_sub\_mw | q\_sub\_mvar | p\_loss\_mw | q\_loss\_mvar | solve\_time\_s | status  |
| ---------- | ---------- | ----- | ------------- | ---------- | ------------ | ----------- | ------------- | -------------- | ------- |
| 1          | 10:00      | min   | 2.2046        | 2.2046     | 2.4389       | 0.1380      | 0.1053        | 0.03           | OPTIMAL |
| 1          | 10:00      | max   | 5.3720        | 5.3720     | 2.1858       | 0.5793      | 0.5193        | 0.03           | OPTIMAL |
| 1          | 10:15      | min   | …             | …          | …            | …           | …             | …              | …       |

## 4. txt 日志（logs/{HH\_MM}_{序号:04d}_{scenario}.txt）

- 每 断面×样本 一个日志文件，文件名 `{HH_MM}_{序号:04d}_{场景名}.txt`（如 `00_00_0001_default.txt`）：`HH_MM` 为时间断面前缀（补零两位），序号为该断面内样本编号（补齐四位）；
- 内容：断面与样本编号、该样本的抽样值汇总、OPF 结果摘要（目标值/网损/节点电压/光伏/储能/可调度负荷 z，min 与 max 各一块）。

## 5. 全局控制文件（training\_dataset\_mc\_config.csv，读入）

放在场景文件夹根目录 `training_dataset/training_dataset_{scenario}/training_dataset_mc_config.csv`，由 `--config` 显式指定或自动读取。
表头只需前两列 `name,value`；第 3 列起为参数列，每格一个 `参数名:值`，**按键名读取**（不同分布携带不同参数，不依赖列名）。

| name                                                   | value | 参数列（每格 `参数名:值`）                            |
| ------------------------------------------------------ | ----- | ------------------------------------------ |
| model / scenario / n\_samples / seed / start\_time / end\_time | 全局参数值 | -                                          |
| 组件名（固定负荷名 / 光伏名 / 储能名 / `{可调度名}_cur\|_lb\|_ub`）        | 分布名   | `mu:曲线名` / `sigma:曲线名` / `cv` / `lo` / `hi` |

全局参数说明：

- `model`（网络模型目录名，如 `model_default`）：加载 `data/csv_case33/{model}/`，与输出目录名**解耦**（同一模型可生成多个训练集）；
- `scenario`（输出目录名，如 `default`）：训练集输出至 `training_dataset/training_dataset_{scenario}/`；
- `start_time` / `end_time`（`H:MM`，默认 `0:00` / `23:45`）：抽样断面范围，15min 一步、含端点；每断面内样本编号独立（从 1 起），输出带 `time_slot` 列、日志文件名带 `HH_MM` 断面前缀。

抽样参数说明（`truncated_normal`，**逐断面解析**）：

- `mu:曲线名` → μ = shapes 表中该曲线在**当前断面的值**（96 点逐时段变化）；
- `sigma:曲线名` → σ = 该曲线在当前断面的值；`cv:数值` → σ = cv×μ（常数方式，兼容旧配置）；
- `lo` / `hi` → 截断边界（可省略，由程序按组件类型默认处理）。

支持分布（见 sampling.py）：`truncated_normal`（mu/sigma 逐断面或 cv 常数折算）、`uniform`（lo/hi）。

示例（默认算例 training\_dataset\_default 实际配置，节选）：

```csv
name,value,
model,model_default,
scenario,default,
n_samples,200,
seed,42,
start_time,0:00,
end_time,23:45,
Load1,truncated_normal,mu:Load1_cur_mu,sigma:Load1_cur_sigma,
PV_Bus6,truncated_normal,mu:PV_Bus6_irr_mu,sigma:PV_Bus6_irr_sigma,
BESS_Bus18,truncated_normal,mu:BESS_Bus18_en_mu,sigma:BESS_Bus18_en_sigma,
EV_Bus19_cur,truncated_normal,mu:EV_Bus19_cur_mu,sigma:EV_Bus19_cur_sigma,
```

- 未在配置表中列出的组件：**固定为断面基线，不抽样（方案 A：配置即抽样，未配置即固定）**。
- mu/sigma 曲线名指向模型 `model_shapes.csv` 的曲线（94 条 = 47 变量 × mu/sigma，由 `history/history_60_days_sample.csv` 60 天历史数据提取）。
- CLI 参数（`--n/--seed`、场景名）覆盖配置文件中对应项。

## 6. 约定汇总

1. `training_dataset_sample.csv` 记录抽样值，min/max 两个优化方向共用同一批抽样（抽样序列一致）；每断面内样本编号从 1 起，`time_slot` 列标识断面。
2. 组件结果表以 `sample_id` + `time_slot` + `sense` 定位：同一编号、同断面、同一 sense 的行构成一个样本的一个结果。
3. 运行前仅清空 `csv/` 与 `logs/` 子目录，场景根下的 `training_dataset_mc_config.csv` 保留。


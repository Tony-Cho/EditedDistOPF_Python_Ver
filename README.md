# EditedDistOPF — 配电网 VPP 最优潮流与 KNN 代理模型工具

基于 LinDistFlow SOCP 松弛 + Gurobi 求解的配电网最优潮流（OPF）计算工具，通过蒙特卡洛抽样生成训练数据集，训练 KNN 回归模型作为 OPF 的代理模型，实现快速场景评估。

***

## 数据流总览

项目由四个功能模块组成，数据按以下流程流转：

- **单断面 OPF 验证**：网络模型 → `main.py` → `demo_result/`
- **KNN 方法全流程**：网络模型 + 场景曲线 + 抽样配置 → `training_dataset_mc.py` → `train_knn.py` → `scenario_dataset_mc.py` → `output/`
- **OPF 方法概率化验证**：场景曲线 + 抽样配置 → `scenario_dataset_mc_OPF.py` → `output/`
- **结果可视化**：场景曲线 + 输出结果 → `plot_scenario.py` → 图表

各模块的详细数据流见下方对应章节。

***

## 目录结构

```
EditedDistOPF/
├── data/
│   ├── csv_case33/             ← 网络模型 CSV 目录 (前缀 model_xxx)
│   ├── opendss_case33/         ← OpenDSS 原始模型
│   └── json_case33/            ← 中间格式 (JSON)
├── scenario/                   ← 场景时序曲线 (scenario_xxx/)
├── training_dataset/           ← 训练集输出目录
├── output/                     ← 场景预测/验证输出目录
├── demo_result/                ← main.py 单断面输出
├── knn_lib/                    ← KNN 模型库
├── history/                    ← 历史数据 (PV/负荷归一化曲线)
├── main.py                     ← 单断面 VPP OPF
├── training_dataset_mc.py      ← 蒙特卡洛抽样生成训练集
├── train_knn.py                ← KNN 代理模型训练
├── scenario_dataset_mc.py      ← KNN 预测场景数据集
├── scenario_dataset_mc_OPF.py  ← OPF 真值场景数据集 (验证)
├── plot_scenario.py            ← 场景结果可视化
├── load_network.py             ← 网络模型加载
├── opf_model.py                ← OPF 建模与求解
├── parse_dss.py                ← OpenDSS 文件解析
├── parse_csv.py                ← CSV 网络模型解析
├── save_results.py             ← OPF 结果导出
├── save_output.py              ← KNN 模型结果导出
├── sampling.py                 ← 抽样函数库
├── requirements.txt            ← Python 依赖
└── README.md                   ← 本文件
```

***

## 输入数据格式

### 1. 网络模型 — `data/csv_case33/model_xxx/`

CSV 格式的配电网模型文件，由 `parse_csv.py` 解析。包含：

| 文件                  | 内容                   |
| ------------------- | -------------------- |
| `model_circuit.csv` | 电路参数 (基准容量等)         |
| `model_buses.csv`   | 母线列表                 |
| `model_lines.csv`   | 支路参数 (阻抗、热极限)        |
| `model_loads.csv`   | 负荷参数 (类型、满载功率、可调度标记) |
| `model_pvs.csv`     | 光伏参数 (容量、功率因数)       |
| `model_storage.csv` | 储能参数 (功率、容量、效率)      |

详细格式见 `data/csv_case33/输入控制.md`。

### 2. 场景时序曲线 — `scenario/scenario_xxx/`

定义 96 个断面 (15min 间隔，0:00\~23:45) 的时序曲线：

| 文件                    | 内容                     |
| --------------------- | ---------------------- |
| `scenario_loads.csv`  | 负荷基准功率 (kw)            |
| `scenario_pvs.csv`    | 光伏容量 (pmpp\_kw) 与辐照度曲线 |
| `scenario_shapes.csv` | 各负荷/PV 的归一化时序曲线 (96 点) |
| `scenario_lines.csv`  | 线路热极限 (可选)             |

### 3. 抽样配置

根据文件所在目录不同，文件名不同：
- `training_dataset/` 下子文件夹 → `training_dataset_mc_config.csv`
- `output/` 下子文件夹 → `output_mc_config.csv`

格式一致，均为两列 CSV：

```csv
name,value
model,model_storage_bus18          # 输入模型名
scenario,output_scenario_trail_1   # 场景名
n_samples,200                      # 每断面样本数
seed,42                            # 随机种子
start_time,0:00                    # 起始断面
end_time,23:45                     # 结束断面
EV_Bus19_lb,truncated_normal,cv:0.10   # 组件分布: 分布名, 参数
EV_Bus19_ub,truncated_normal,cv:0.10
AC_Bus2_lb,truncated_normal,cv:0.10
AC_Bus2_ub,truncated_normal,cv:0.10
```

- 全局参数：`model` / `scenario` / `n_samples` / `seed` / `start_time` / `end_time`
- 组件分布：`truncated_normal` (参数 `cv`/`sigma`/`lo`/`hi`) 或 `uniform` (参数 `lo`/`hi`)
- 组件名规则：`{可调度负荷名}_cur`|`_lb`|`_ub` / 固定负荷名 / 光伏名 / 储能名
- **未配置的组件固定为断面基线值**（方案A：配置即抽样，未配置即固定）

***

## 程序模块

### 1. 单断面 OPF 方法全流程验证 — `main.py`

```
data/csv_case33/model_{model_name}/     ← 网络模型 CSV
         │
         ▼
┌─ main.py ───────────────────────────────────────┐
│  加载网络 → OPF (min/max) → 输出结果           
│  → demo_result/demo_result_{model_name}/                 
└─────────────────────────────────────────────────┘
```

加载配电网模型，运行两个 OPF 场景（最小化/最大化根节点注入功率），输出结果。用于验证单断面 OPF 求解的正确性与完整性。

```bash
python main.py                                  # 默认场景 scenario_all
python main.py model_storage_bus18              # 指定模型
python main.py data/csv_case33/model_xxx/       # 直接指定路径
```

**输出** → `demo_result/demo_result_xxx/`：

- `demo_result_xxx.txt` — 概览文件
- `csv/training_dataset_system.csv` — 系统级指标
- `csv/training_dataset_buses.csv` — 节点电压 + 净注入
- `csv/training_dataset_lines.csv` — 支路潮流 + 损耗
- `csv/training_dataset_loads.csv` — 负荷结果
- `csv/training_dataset_pvs.csv` — 光伏出力
- `csv/training_dataset_storage.csv` — 储能充放/能量状态

---

### 2. 含概率化表征的 KNN 方法全流程验证

```
data/csv_case33/model_{model_name}/      ← 网络模型 CSV
scenario/scenario_{scenario_name}/       ← 场景时序曲线
training_dataset_mc_config.csv            ← 抽样配置
         │
         ▼
┌─ training_dataset_mc.py ─────────────────────────────────┐
│  加载网络 + 场景形状 + training_dataset_mc_config 抽样配置       
│  → 蒙特卡洛抽样 → 逐样本 OPF 求解              
│  → training_dataset/training_dataset_{model_name}_sample/ 
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌─ train_knn.py ──────────────────────────────────┐
│  读取训练集 + knn_config 训练配置
│  → 训练 KNN 模型 (min/max)          
│  → knn_lib/training_dataset_{model_name}_sample/
│  (模型 + 标准化器 + 特征列名)            
└─────────────────────────────────────────────────┘
         │
         │  scenario/scenario_{scenario_name}/ ← 场景时序曲线
         │  output_mc_config.csv                ← 抽样配置
         │               │
         ▼               ▼
┌─ scenario_dataset_mc.py ────────────────────────┐
│  加载场景曲线 + KNN 模型 + output_mc_config 抽样配置                         
│  → 抽样可调度负荷上下限 → KNN 预测根节点注入    
│  → output/output_{scenario_name}/                           
└─────────────────────────────────────────────────┘
```

由三个子模块构成流水线：蒙特卡洛抽样生成训练集 → 训练 KNN 代理模型 → KNN 预测场景数据集。

#### 2.1 训练集生成 — `training_dataset_mc.py`

对每个断面按 `training_dataset_mc_config.csv` 配置做蒙特卡洛抽样，对每个样本求解 OPF，生成训练数据集（含可调度负荷上下限的概率化表征）。

```bash
python training_dataset_mc.py model_storage_bus18 --n 500 --seed 42
python training_dataset_mc.py --config training_dataset_mc_config.csv
```

**输出** → `training_dataset/training_dataset_{model}_sample/`：

- `training_dataset_sample.csv` — 抽样输入场景表 (每样本一行)
- `csv/training_dataset_system.csv` — 系统级指标 (带 sample\_id)
- `csv/training_dataset_buses.csv` — 节点结果
- `csv/training_dataset_lines.csv` — 支路结果
- `csv/training_dataset_loads.csv` — 负荷结果
- `csv/training_dataset_pvs.csv` — 光伏结果
- `csv/training_dataset_storage.csv` — 储能结果
- `logs/{编号}_{model}.txt` — 每样本日志

**不可行样本处理**：OPF 求解不可行的样本记录为 `nan`，`status` 列标记求解状态，程序继续运行不中断。

#### 2.2 KNN 代理模型训练 — `train_knn.py`

读取训练集 CSV，训练两个 KNN 回归模型（min/max 场景），预测根节点注入有功/无功，作为 OPF 的替代模型。

```bash
python train_knn.py                                          # 默认读取 knn_config.csv
python train_knn.py --config knn_lib/{训练集文件夹名}/knn_config.csv
python train_knn.py --k 7 --weights uniform --test_size 0.2  # CLI 覆盖
```

**特征** (48 维)：固定负荷功率 (35) + 储能 p\_net/q (2×N\_st) + PV p\_out/q\_out (2×N\_pv) + EV/AC 负荷 p\_out (N\_evac)

**输出** → `knn_lib/{训练集文件夹名}/`：

- `knn_model_min.joblib` / `knn_model_max.joblib` — KNN 模型
- `knn_scaler_min.joblib` / `knn_scaler_max.joblib` — 标准化器
- `knn_feature_names.json` — 输入特征列名
- `knn_target_names.json` — 输出目标列名
- `knn_config.csv` — 训练配置 (含数据集路径)
- `训练集名_predictions.csv` — 测试集预测结果

#### 2.3 KNN 真值场景计算概率化表征上下界 — `scenario_dataset_mc.py`

加载场景时序曲线，对可调度负荷上下限做截断正态抽样（概率化表征），其余量固定为曲线值，通过 KNN 模型预测根节点注入。

```bash
python scenario_dataset_mc.py --config output/output_scenario_trail_1/output_mc_config.csv
python scenario_dataset_mc.py --scenario output_scenario_trail_1 --model training_dataset_storage_bus18_sample --n 500
```

**输出** → `output/output_{场景名}/`：

- `output_sample.csv` — 抽样输入场景表 (48 维特征)
- `output_system.csv` — 根节点注入预测 (min/max)
- `output_storage.csv` — 储能出力预测
- `output_pvs.csv` — 光伏出力预测
- `output_loads.csv` — 可调度负荷实际有功

---

### 3. 含概率化表征的 OPF 方法全流程验证 — `scenario_dataset_mc_OPF.py`

```
scenario/scenario_{scenario_name}/ ← 场景时序曲线
output_mc_config.csv                  ← 抽样配置
         │
         ▼
┌─ scenario_dataset_mc_OPF.py ─────────────────────────────┐
│  加载场景曲线 + OPF 模型 + output_mc_config 抽样配置
│  → 抽样可调度负荷上下限 → 实际 OPF 求解真值    
│  → output/output_{scenario_name}_opf/ (用于验证 KNN 精度)    
└──────────────────────────────────────────────────────────┘
```

与 `scenario_dataset_mc.py` 使用同一抽样配置和同一随机种子，对可调度负荷上下限做相同的截断正态抽样，但输出通过实际求解 OPF 获得，用于验证 KNN 预测精度，同时作为 OPF 方法的概率化表征。

```bash
python scenario_dataset_mc_OPF.py --config output/output_scenario_trail_1/output_mc_config.csv
python scenario_dataset_mc_OPF.py --scenario output_scenario_trail_1 --n 500 --seed 42
```

**输出** → `output/output_{场景名}_opf/`：格式与 KNN 版一致，字段含义相同。

---

### 4. 结果可视化模块 — `plot_scenario.py`

```
output/output_{scenario_name}(_opf)/output_system.csv
scenario/scenario_{scenario_name}/ ← 场景时序曲线
         │
         ▼
┌─ plot_scenario.py ───────────────────────────────────────┐
│  读取 KNN 预测 + OPF 真值 → 绘图对比分析        
│  → output/output_{scenario_name}/plot_{scenario_name}.png               
└──────────────────────────────────────────────────────────┘
```

读取场景曲线和 OPF/KNN 结果，绘制根节点基线功率与上/下调边界对比图，直观展示不同方法的差异。

```bash
python plot_scenario.py --scenario output_scenario_trail_1
```

**输出** → `output/output_{场景名}/plot_{场景名}.png`

绘制内容：

- 固定负荷基线 (Load1\~32 + Fixed\_Bus\*)
- 可调度负荷基线 (EV/AC)
- 概率边界 (10%\~90% 分位区间)
- 理论边界 (EV/AC 全削去/全满发 + PV 满发/不出力 + 储能满功率)

***

## OPF 数学模型

OPF 模型基于二阶锥规划（SOCP）松弛的 LinDistFlow，使用 Gurobi 求解。

### 变量定义

| 符号 | 含义 | 单位(pu) |
|:---|:----|:--------|
| $U_i = V_i^2$ | 节点 $i$ 电压平方 | pu² |
| $l_{ij} = I_{ij}^2$ | 支路 $(i,j)$ 电流平方 | pu² |
| $P_{ij}, Q_{ij}$ | 支路 $(i,j)$ 首端有功/无功潮流 | pu |
| $p_{\text{sub}}, q_{\text{sub}}$ | 根节点（slack bus）注入有功/无功 | pu |
| $p_i^{\text{pv}}, q_i^{\text{pv}}$ | 光伏 $i$ 有功/无功出力 | pu |
| $p_i^{\text{ch}}, p_i^{\text{dis}}$ | 储能 $i$ 充电/放电功率 | pu |
| $z_i$ | 可调度负荷 $i$ 的削减因子 | — |

### 目标函数

$$ \text{Min/Max} \quad p_{\text{sub}} $$

最小化（Min）对应根节点注入最小场景（如光伏大发、负荷低谷），最大化（Max）对应根节点注入最大场景（如负荷高峰、光伏出力不足）。

### 约束条件

#### ① DistFlow 支路方程（含线损）

$$ U_j = U_i - 2(r_{ij}P_{ij} + x_{ij}Q_{ij}) + (r_{ij}^2 + x_{ij}^2)l_{ij} $$

其中 $r_{ij}, x_{ij}$ 为支路 $(i,j)$ 的电阻和电抗。

#### ② SOCP 松弛（线损建模）

$$ l_{ij} \cdot U_i \geq P_{ij}^2 + Q_{ij}^2 $$

将非凸的潮流等式松弛为二阶锥约束，使问题变为凸优化。

#### ③ 热极限约束

热极限通过**线性箱式约束**实现，直接限定支路有功/无功功率的上下界：

$$ |P_{ij}| \leq S_{ij,\max}, \quad |Q_{ij}| \leq S_{ij,\max} $$

此外，添加 SOCP 链式约束作为冗余一致性校验：

$$ S_{ij,\max}^2 \geq l_{ij} \cdot U_i $$

热极限的完整约束链为： $S_{\max}^2 \geq l \cdot U_i \geq P^2 + Q^2$ ，通过两条独立的二次约束表达，但两条约束**不共享 $P,Q$ 变量**，避免了 Hessian 冲突。$P,Q$ 的箱式约束与 SOCP 链式约束共同确保热极限的有效性。

#### ④ 节点功率平衡（含线损）

对于每个母线 $i \in \mathcal{N}$ ，流入功率减去流出功率等于该母线的净消耗功率（含线损）。

**有功平衡：**

$$
\sum_{(k,i) \in \mathcal{E}} P_{ki} - \sum_{(i,j) \in \mathcal{E}} P_{ij} =
\begin{cases}
\sum_{\ell \in \mathcal{L}_i} p_{\ell}^0 z_\ell - \sum_{g \in \mathcal{G}_i} p_g + \sum_{s \in \mathcal{S}_i} (p_{\text{ch},s} - p_{\text{dis},s}) + \sum_{(k,i) \in \mathcal{E}} R_{ki} l_{ki}, & i \neq \text{slack} \\
P_{\text{sub}} - \sum_{(i,j) \in \mathcal{E}} P_{ij} = \sum_{\ell \in \mathcal{L}_i} p_{\ell}^0 z_\ell - \sum_{g \in \mathcal{G}_i} p_g + \sum_{s \in \mathcal{S}_i} (p_{\text{ch},s} - p_{\text{dis},s}) + \sum_{(k,i) \in \mathcal{E}} R_{ki} l_{ki}, & i = \text{slack}
\end{cases}
$$

**无功平衡：**

$$
\sum_{(k,i) \in \mathcal{E}} Q_{ki} - \sum_{(i,j) \in \mathcal{E}} Q_{ij} =
\begin{cases}
\sum_{\ell \in \mathcal{L}_i} q_\ell^0 z_\ell - \sum_{g \in \mathcal{G}_i} q_g - \sum_{c \in \mathcal{C}_i} q_c + \sum_{(k,i) \in \mathcal{E}} X_{ki} l_{ki}, & i \neq \text{slack} \\
Q_{\text{sub}} - \sum_{(i,j) \in \mathcal{E}} Q_{ij} = \sum_{\ell \in \mathcal{L}_i} q_\ell^0 z_\ell - \sum_{g \in \mathcal{G}_i} q_g - \sum_{c \in \mathcal{C}_i} q_c + \sum_{(k,i) \in \mathcal{E}} X_{ki} l_{ki}, & i = \text{slack}
\end{cases}
$$

其中：
- $p_\ell^0$ 、 $q_\ell^0$ = 原始负荷有功/无功（pu）
- $q_c$ = 电容器注入无功（正值，pu）
- $\mathcal{L}_i$ 、 $\mathcal{G}_i$ 、 $\mathcal{S}_i$ 、 $\mathcal{C}_i$ 分别为挂接在母线 $i$ 上的负荷、光伏、储能、电容器集合
- 新增的 $R_{ki} l_{ki}$ 和 $X_{ki} l_{ki}$ 项即为**以该母线为末端的支路线损**，使功率平衡中包含了线路发热损耗，物理上更完整

#### ⑤ 节点电压约束

$$ 0.95 \leq V_i \leq 1.07 \ \text{pu}, \quad V_{\text{slack}} = 1.05 \ \text{pu} $$

#### ⑥ 光伏出力约束

$$ 0 \leq p_i^{\text{pv}} \leq P_i^{\text{mpp}}, \quad |q_i^{\text{pv}}| \leq p_i^{\text{pv}} \cdot \tan(\arccos(\text{pf}_i)) $$

光伏出力在 $0$ 到满辐照额定功率之间连续可调，无功按功率因数限制。

#### ⑦ 储能约束（单时段，无时序耦合）

$$ p_i^{\text{ch}} \geq 0, \quad p_i^{\text{dis}} \geq 0 $$

单时段模型不考虑时序耦合，充电和放电功率非负。

#### ⑧ 可调度负荷约束

$$ z_i^{\text{lb}} \leq z_i \leq z_i^{\text{ub}} $$

| 负荷类型 | $z$ 范围 | 说明 |
|:--------|:--------|:----|
| EV（电动汽车） | $[0.3, 1.0]$ | 可调范围 30%~100% |
| AC（空调/柔性负荷） | $[0, 1.0]$ | 可调范围 0%~100% |
| Fixed（固定负荷） | $z = 1.0$ | 不可调 |

***

详细输出格式约定见：

- `output/输出控制.md` — 场景预测/验证输出格式
- `training_dataset/输出控制.md` — 训练集输出格式

所有数值均为**工程单位**：功率 MW、无功 Mvar、能量 MWh、电压 pu。

### 通用约定

- `sense` 列：`min` = 最小化根节点注入，`max` = 最大化根节点注入
- 列名带 `_mw`/`_mvar`/`_mwh` 时，值即 MW/Mvar/MWh
- 无量纲比例（辐照度、挂载比例、z、pct）列名不加单位后缀
- 文件编码 UTF-8，表头行存在

***

## 使用示例

### 全链路示例

```bash
# 1. 生成训练集 (model_storage_bus18, 500 样本/断面)
python training_dataset_mc.py model_storage_bus18 --n 500 --seed 42

# 2. 训练 KNN 模型
python train_knn.py

# 3. 场景预测 (KNN)
python scenario_dataset_mc.py --scenario output_scenario_trail_1 --model training_dataset_storage_bus18_sample --n 200

# 4. 场景验证 (OPF 真值)
python scenario_dataset_mc_OPF.py --scenario output_scenario_trail_1 --n 200 --seed 42

# 5. 可视化对比
python plot_scenario.py --scenario output_scenario_trail_1
```

### 单断面 OPF 示例

```bash
python main.py model_storage_bus18
```

***

## 依赖环境

- Python 3.13+
- [Gurobi](https://www.gurobi.com/) 13.0+ (商业求解器，需许可证)
- numpy 2.0+
- scikit-learn 1.6+
- matplotlib (仅 `plot_scenario.py` 需要)
- joblib (模型序列化)

```bash
pip install -r requirements.txt
```

***

## 网络模型加载路径规则

`resolve_model_path(scenario)` 按以下顺序查找模型：

1. `data/csv_case33/model_{scenario}/` — CSV 模型目录
2. `scenario/{scenario}/` — 场景目录
3. `data/opendss_case33/{scenario}/` — OpenDSS 目录
4. `{scenario}` — 直接作为路径

支持前缀兼容：`scenario_` ↔ `model_` ↔ `output_` (自动去除 `output_` 前缀回退到原场景名)。

***

## 项目背景

本项目从 Julia + PowerModelsDistribution + Ipopt (AC OPF) 重构为 Python + Gurobi (LinDistFlow SOCP)：

| 项目   | 原版                      | 重构版                 |
| ---- | ----------------------- | ------------------- |
| 语言   | Julia                   | Python              |
| 求解器  | Ipopt (NLP)             | Gurobi (SOCP)       |
| 模型   | AC OPF (ACPUPowerModel) | LinDistFlow SOCP 松弛 |
| 系统   | 三相非平衡                   | 三相平衡单相等值            |
| 代理模型 | —                       | KNN 回归              |


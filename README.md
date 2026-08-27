# EditedDistOPF — 配电网 VPP 最优潮流与 KNN 代理模型工具

基于 LinDistFlow SOCP 松弛 + Gurobi 求解的配电网最优潮流（OPF）计算工具，通过蒙特卡洛抽样生成训练数据集，训练 KNN 回归模型作为 OPF 的代理模型，实现快速场景评估。

***

## 数据流总览

项目由四个功能模块组成，数据按以下流程流转：

- **单断面 OPF 验证**：网络模型 → `main.py` → `demo_result/`
- **KNN 方法全流程**：网络模型 + model\_shapes mu/sigma 曲线 + 抽样配置 → `training_dataset_mc.py` → `train_knn.py` → `scenario_dataset_mc.py` → `output/`
- **OPF 方法概率化验证**：场景曲线 + 抽样配置 → `scenario_dataset_mc_OPF.py` → `output/`
- **结果可视化**：场景曲线 + 输出结果 → `plot_scenario.py` → 图表

**默认算例（default case）**：模型 `model_default` + 场景 `scenario_default` + 训练集 `training_dataset_default` + KNN 库 `knn_lib/training_dataset_default/`，验证结果输出至 `output/output_scenario_default/`（KNN）与 `output/output_scenario_default_opf/`（OPF 真值），最终由 `plot_scenario.py --mode real` 绘制七线对比图。

各模块的详细数据流见下方对应章节。

***

## 目录结构

```
EditedDistOPF/
├── data/
│   └── csv_case33/
│       ├── model_default/          ← 默认网络模型 CSV (含 model_shapes.csv: 47 变量 × mu/sigma 曲线)
│       └── 输入控制.md              ← 输入格式约定文档
├── scenario/                       ← 场景时序曲线 (scenario_default 等)
├── training_dataset/               ← 训练集输出目录 (training_dataset_default/ 等)
├── output/                         ← 场景预测/验证输出目录 (output_scenario_default(_opf)/ 等)
├── demo_result/                    ← main.py 单断面输出
├── knn_lib/                        ← KNN 模型库 (training_dataset_default/ 等)
├── history/
│   └── history_60_days_sample.csv  ← 60 天历史数据 (mu/sigma 曲线的提取来源)
├── main.py                         ← 单断面 VPP OPF
├── training_dataset_mc.py          ← 蒙特卡洛抽样生成训练集
├── train_knn.py                    ← KNN 代理模型训练
├── scenario_dataset_mc.py          ← KNN 预测场景数据集
├── scenario_dataset_mc_OPF.py      ← OPF 真值场景数据集 (验证)
├── plot_scenario.py                ← 场景结果可视化 (prob/real 两种模式)
├── load_network.py                 ← 网络模型加载
├── opf_model.py                    ← OPF 建模与求解
├── parse_csv.py                    ← CSV 网络模型解析
├── parse_dss.py                    ← OpenDSS 文件解析 (历史遗留, 对应数据已移除)
├── save_results.py                 ← OPF 结果导出
├── save_output.py                  ← KNN 模型结果导出
├── sampling.py                     ← 抽样函数库
├── requirements.txt                ← Python 依赖
└── README.md                       ← 本文件
```

***

## 输入输出格式

各模块的输入模型、场景时序曲线、抽样配置与输出结果的**详细字段约定**统一由以下文档控制，此处不再展开：

| 文档 | 内容 |
| ---- | ---- |
| `data/csv_case33/输入控制.md` | 网络模型输入格式（CSV 模型文件） |
| `output/output输出控制.md` | 场景预测/验证输出格式（KNN / OPF） |
| `training_dataset/training_dataset输出控制.md` | 训练集输出格式 |

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
python main.py model_default                    # 默认模型
python main.py data/csv_case33/model_default/   # 直接指定路径
```

**输出** → `demo_result/demo_result_{model_name}/`：

- `demo_result_{model_name}.txt` — 概览文件
- `csv/training_dataset_system.csv` — 系统级指标
- `csv/training_dataset_buses.csv` — 节点电压 + 净注入
- `csv/training_dataset_lines.csv` — 支路潮流 + 损耗
- `csv/training_dataset_loads.csv` — 负荷结果
- `csv/training_dataset_pvs.csv` — 光伏出力
- `csv/training_dataset_storage.csv` — 储能充放/能量状态

---

### 2. 含概率化表征的 KNN 方法全流程验证

```
data/csv_case33/model_{model_name}/      ← 网络模型 CSV (含 model_shapes.csv mu/sigma 曲线)
training_dataset_mc_config.csv            ← 抽样配置 (model 键指定网络模型)
         │
         ▼
┌─ training_dataset_mc.py ─────────────────────────────────┐
│  加载网络 + model_shapes mu/sigma 曲线 + 抽样配置
│  → 逐断面蒙特卡洛抽样 → 逐样本 OPF 求解
│  → training_dataset/training_dataset_{scenario}/
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌─ train_knn.py ──────────────────────────────────┐
│  读取训练集 + knn_config 训练配置
│  → 训练 KNN 模型 (min/max)          
│  → knn_lib/{训练集文件夹名}/
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

抽样配置支持两类写法：
- **mu/sigma 曲线**（当前默认算例）：参数格填 `mu:曲线名`、`sigma:曲线名`，逐断面从模型 `model_shapes.csv`（或场景 shapes）取该时间点的 μ 与 σ，μ/σ 曲线由 `history/history_60_days_sample.csv` 60 天历史数据提取；
- **常数参数**：`cv`（σ=cv×μ，μ=组件基线）/ `sigma` 等数值参数，各断面相同。

```bash
python training_dataset_mc.py --config training_dataset/training_dataset_default/training_dataset_mc_config.csv
python training_dataset_mc.py default --n 500 --seed 42
```

**输出** → `training_dataset/training_dataset_{scenario}/`：

- `training_dataset_mc_config.csv` — 抽样配置副本 (保留)
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
python train_knn.py --csv training_dataset/training_dataset_default/csv      # 默认算例训练集
python train_knn.py --config knn_lib/{训练集文件夹名}/knn_config.csv           # 显式配置
python train_knn.py --k 7 --weights uniform --test-size 0.2  # CLI 覆盖
```

**特征** (默认算例 45 维)：固定负荷功率 (32) + 储能初始状态 p\_init/se\_init (2) + PV 辐照度 (5) + EV/AC 可调上下限 lb/ub (6)

**输出** → `knn_lib/{训练集文件夹名}/`：

- `knn_config.csv` — 训练配置 (含数据集路径)
- `knn_params.csv` — 模型参数 + 数据规模
- `knn_metrics.csv` — 评估指标 (R²/MAE/RMSE)
- `knn_predictions.csv` — 测试集逐样本预测对比
- `knn_model_min.joblib` / `knn_model_max.joblib` — KNN 模型
- `knn_scaler_min.joblib` / `knn_scaler_max.joblib` — 标准化器
- `knn_feature_names.json` — 输入特征列名
- `knn_target_names.json` — 输出目标列名

#### 2.3 KNN 真值场景计算概率化表征上下界 — `scenario_dataset_mc.py`

加载场景时序曲线，对可调度负荷上下限做截断正态抽样（概率化表征），其余量固定为曲线值，通过 KNN 模型预测根节点注入。

```bash
python scenario_dataset_mc.py --config output/output_scenario_default/output_mc_config.csv
python scenario_dataset_mc.py --scenario output_scenario_default --model training_dataset_default --n 200
```

**输出** → `output/output_{scenario_name}/`：

- `output_mc_config.csv` — 抽样配置副本 (保留)
- `output_sample.csv` — 抽样输入场景表 (默认算例 45 维特征)
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
python scenario_dataset_mc_OPF.py --config output/output_scenario_default/output_mc_config.csv
python scenario_dataset_mc_OPF.py --scenario output_scenario_default --n 200 --seed 42
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

读取场景曲线和 OPF/KNN 结果，绘制根节点基线功率与上/下调边界对比图，直观展示不同方法的差异。支持两种模式：

**prob 模式（默认）** — 概率边界图：

```bash
python plot_scenario.py --scenario scenario_trail_1
```

**输出** → `output/{scenario}/plot_{scenario}.png`

绘制内容：

- 固定负荷基线 (Load1\~32)
- 可调度负荷基线 (EV/AC)
- 概率边界 (10%\~90% 分位区间)
- 理论边界 (EV/AC 全削去/全满发 + PV 满发/不出力 + 储能满功率)

**real 模式（默认算例验证图）** — 七条线单图 (`--mode real`)：

```bash
python plot_scenario.py --mode real --scenario scenario_default
```

读取 `output/output_{scenario}/`（KNN 预测）与 `output/output_{scenario}_opf/`（OPF 真值）的 output\_system.csv，单图绘制七条线（自上而下）：

1. 理论上界 (不考虑安全约束)
2. OPF P10 (上调上界真值)
3. KNN P10 (上调上界代理)
4. 负荷基线
5. KNN P90 (下调下界代理)
6. OPF P90 (下调下界真值)
7. 理论下界 (不考虑安全约束)

理论上下界之间、OPF P10\~P90 之间、KNN P10\~P90 之间均加背景半透明填充，并输出排序校验统计。

**输出** → `output/output_{scenario}/plot_{scenario}_real.png`

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
| $q_i^{\text{st}}$ | 储能 $i$ 无功出力 | pu |
| $e_i$ | 储能 $i$ 能量状态 | pu·h |
| $z_i$ | 可调度负荷 $i$ 的调节因子（乘当前挂载，可削减/增荷） | — |

### 目标函数

$$ \text{Min/Max} \quad p_{\text{sub}} $$

最小化（Min）对应根节点注入最小场景（如光伏大发、负荷低谷），最大化（Max）对应根节点注入最大场景（如负荷高峰、光伏出力不足）。

### 约束条件

#### ① DistFlow 支路方程（含线损）

$$ U_j = U_i - 2(r_{ij}P_{ij} + x_{ij}Q_{ij}) + (r_{ij}^2 + x_{ij}^2)l_{ij} $$

其中 $r_{ij}, x_{ij}$ 为支路 $(i,j)$ 的电阻和电抗。

**推导（无近似，精确）：**

**(1) 支路电压降（相量形式）。** 对支路 $(i,j)$，阻抗 $z_{ij}=r_{ij}+jx_{ij}$，电流从 $i$ 流向 $j$：

$$ \tilde{V}_j = \tilde{V}_i - z_{ij}\,\tilde{I}_{ij} $$

**(2) 用首端复功率反解电流。** 支路首端复功率 $S_{ij}=P_{ij}+jQ_{ij}=\tilde{V}_i\tilde{I}_{ij}^*$，即 $\tilde{I}_{ij}=\dfrac{P_{ij}-jQ_{ij}}{\tilde{V}_i^*}$。

**(3) 代入并取模平方（消去相角）。**

$$ |V_j|^2 = \left| V_i - \frac{rP+xQ + j(xP-rQ)}{V_i} \right|^2 = V_i^2 - 2(rP+xQ) + \frac{(rP+xQ)^2+(xP-rQ)^2}{V_i^2} $$

交叉项 $2rxPQ$ 恰好抵消，分子合并为 $(r^2+x^2)(P^2+Q^2)$。

**(4) 引入 $U$ 与 $l$ 化为线性。** 定义 $U_i=V_i^2$，电流平方 $l_{ij}=|I_{ij}|^2=\dfrac{P_{ij}^2+Q_{ij}^2}{V_i^2}=\dfrac{P^2+Q^2}{U_i}$（由 $|S|=|V||I|$ 得出），代入得①式。

**各项物理含义：**

- $U_i - U_j = 2(rP+xQ)$：线路电压降落（P 经电阻、Q 经电抗的投影叠加）；
- $(r^2+x^2)l$：电流流过阻抗产生的附加压降，即**线损项**，仅当 $l\neq0$（电流非零）时存在；
- $l_{ij}$ 的引入使方程对 $(U,P,Q,l)$ 为**线性**，非线性被"藏"进 $l$ 的定义式——这是后续 SOCP 松弛的前提。

整条推导只取模平方消去相角，**无任何近似假设**，故①式对辐射状配电网是精确的；LinDistFlow 则额外假设电压接近 1 pu 并忽略 $(r^2+x^2)l$ 项（LP），SOCP 版保留该项（凸 SOCP），更接近全 AC 潮流精度。

#### ② SOCP 松弛（线损建模）

$$ l_{ij} \cdot U_i \geq P_{ij}^2 + Q_{ij}^2 $$

**为什么要把潮流等式松弛为二阶锥？**

精确 DistFlow 中电流平方的定义为**分式等式**：

$$ l_{ij} = \frac{P_{ij}^2 + Q_{ij}^2}{U_i} $$

该式对 $(P,Q,U,l)$ 是**非线性非凸**的。若作为等式原样保留，问题退化为非凸 NLP：Gurobi 无法求解一般非凸问题，即使能解也不保证全局最优。

而电压降方程 $U_j = U_i - 2(r_{ij}P_{ij} + x_{ij}Q_{ij}) + (r_{ij}^2 + x_{ij}^2)l_{ij}$ 对 $(U,P,Q,l)$ 是**线性**的——非线性只藏在 $l$ 的定义式中。因此将等式松弛为不等式：

$$ l_{ij} \cdot U_i \geq P_{ij}^2 + Q_{ij}^2 \iff \left\| \begin{pmatrix} 2P_{ij} \\ 2Q_{ij} \\ l_{ij} - U_i \end{pmatrix} \right\|_2 \leq l_{ij} + U_i $$

不等式左端即**二阶锥**（SOC），是凸约束。松弛后整个问题由非凸 NLP 变为凸 SOCP，Gurobi 原生支持二阶锥约束，求解高效且保证全局最优。

**为什么松弛后仍然是精确的（紧性）？**

对于**辐射状（树状）配电网**，该松弛在最优解处**通常取等号**（松弛"紧"），即 $l_{ij}U_i = P_{ij}^2+Q_{ij}^2$。直觉上， $l_{ij}$ 代表支路电流平方，其增大只会抬高线损（电压降与功率平衡中的 $R_{ij}l_{ij}$、 $X_{ij}l_{ij}$ 项随之增大），而目标函数对根节点注入单调，优化没有动机让 $l$ 虚高，因此不会产生"伪潮流"解。故松弛前后最优解一致，等效于精确 DistFlow 线损模型。

**与 LinDistFlow 的对比：**

| 项目 | LinDistFlow（原版，忽略线损） | SOCP 松弛（当前版） |
|:----|:----|:----|
| 电压方程 | $U_i - U_j = 2(rP+xQ)$ | $U_j = U_i - 2(rP+xQ) + (r^2+x^2)l$ |
| 线损项 $(r^2+x^2)l$ | ❌ 忽略 | ✅ 保留 |
| 松弛紧性 | — | 辐射网下紧 |
| 凸性 | LP | SOCP（凸） |

#### ③ 热极限约束

热极限通过 **SOCP 链式约束**实现（作为冗余一致性校验），不再需要额外的箱式上下界：

$$ S_{ij,\max}^2 \geq l_{ij} \cdot U_i $$

**完整约束链与物理含义：**

热极限的完整约束链为 $S_{\max}^2 \geq l \cdot U_i \geq P^2 + Q^2$ ，通过两条独立的二次约束表达，但两条约束**不共享 $P,Q$ 变量**，避免了 Hessian 冲突。链式约束独立保证热极限的有效性，支路功率变量 $P,Q$ 在全实数域取值。

- **$lU$ 的物理身份**： $l_{ij}=|I_{ij}|^2$、 $U_i=|V_i|^2$，故 $l_{ij}U_i = |V_i|^2|I_{ij}|^2 = |S_{ij}|^2 = P_{ij}^2+Q_{ij}^2$。当 SOCP 松弛取紧（ $lU=P^2+Q^2$）时，链式约束 $S^2\ge lU$ 就退化为物理热极限 $P^2+Q^2\le S^2$——它并非人为强加，而是借助电流变量 $l$ 把"视在功率不超额定"这一物理约束无损地编码进锥模型；
- **为何恒成立且不"误伤"**： $P^2+Q^2\le lU$（SOCP 松弛）与 $lU\le S^2$（链式）均为显式约束，可行域内任意解同时满足，故 $P^2+Q^2\le lU\le S^2$ 恒成立，热极限始终被夹住。即使松弛不紧（ $lU>P^2+Q^2$），由于目标函数对 $l$ 单调（ $l$ 仅以线损形式出现），优化器会把 $l$ 压到最小值、自动取紧，链式只在物理极限真正越界时才起作用，不会提前限制 $P,Q$；
- **为何不需要箱式约束**： $|P|\le S,\ |Q|\le S$ 蕴含于 $P^2+Q^2\le S^2$，而后者已由链式约束精确保证，故 $P,Q$ 无需额外的箱式上下界即可满足热极限。

**$S_{ij,\max}$ 的计算：**

1. 线路阻抗由 Ω 折算为 pu： $z_{\text{base}} = V_{\text{base}}^2 / S_{\text{base}} = 12.66^2 / 10 \approx 16.03\ \Omega$， $r_{ij}^{\text{pu}} = R_{ij}/z_{\text{base}}$， $x_{ij}^{\text{pu}} = X_{ij}/z_{\text{base}}$。
2. 若数据给出 `normamps`（额定电流 A/相，来自 OpenDSS），则优先使用：

$$ S_{ij,\max} = \frac{\sqrt{3}\, V_{\text{base}}\, I_{\text{rated}}}{S_{\text{base}}} $$

3. 否则按阻抗反比例分配热限额。先计算所有启用线路的平均阻抗模值 $z_{\text{avg}} = \frac{1}{|\mathcal{E}|}\sum_{(i,j)\in\mathcal{E}}\sqrt{(r_{ij}^{\text{pu}})^2+(x_{ij}^{\text{pu}})^2}$，再：

$$ S_{ij,\max} = \operatorname{clamp}\!\left(2\cdot\frac{z_{\text{avg}}}{\max(z_{ij}^{\text{pu}},\,0.001)},\ 0.3,\ 2.5\right) \cdot P_{\text{load}}^{\text{total}} $$

其中 $P_{\text{load}}^{\text{total}}$ 为全网总有功负荷（pu）。含义：**阻抗越小（越靠近变电站），线路分配到的热限额越大**。

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
- $p_\ell^0$ 、 $q_\ell^0$ = 负荷**当前实际挂载**的有功/无功（pu，= 满载 × `LoadShape.mult[0]`）；实际消费 = $p_\ell^0 z_\ell$，固定负荷 $z_\ell \equiv 1$
- $q_c$ = 电容器注入无功（正值，pu）
- $\mathcal{L}_i$ 、 $\mathcal{G}_i$ 、 $\mathcal{S}_i$ 、 $\mathcal{C}_i$ 分别为挂接在母线 $i$ 上的负荷、光伏、储能、电容器集合
- 新增的 $R_{ki} l_{ki}$ 和 $X_{ki} l_{ki}$ 项即为**以该母线为末端的支路线损**，使功率平衡中包含了线路发热损耗，物理上更完整

#### ⑤ 节点电压约束

$$ 0.95 \leq V_i \leq 1.07 \ \text{pu} \ \Longleftrightarrow \ 0.9025 \leq U_i \leq 1.1449 \ \text{pu}^2, \quad V_{\text{slack}} = 1.05 \ \text{pu} $$

#### ⑥ 光伏出力约束

$$ 0 \leq p_i^{\text{pv}} \leq P_i^{\text{mpp}}, \quad |q_i^{\text{pv}}| \leq p_i^{\text{pv}} \cdot \tan(\arccos(\text{pf}_i)) $$

光伏出力在 $0$ 到满辐照额定功率之间连续可调，无功按功率因数限制。

#### ⑦ 储能约束（含能量状态与 PCS 容量）

充电/放电功率非负；储能无功为**独立箱式约束**（允许待机/纯无功运行，区别于光伏）：

$$ 0 \leq p_i^{\text{ch}} \leq p_{\text{ch,ub}}, \qquad 0 \leq p_i^{\text{dis}} \leq p_{\text{dis,ub}}, \qquad |q_i^{\text{st}}| \leq q_{\max} $$

**能量状态约束**（ $\Delta t = 1$ h， $\eta_{\text{ch}}=\eta_{\text{dis}}=0.95$）：

$$ e_i = e_{\text{init},i} + \Delta t \left( p_i^{\text{ch}} \eta_{\text{ch}} - \frac{p_i^{\text{dis}}}{\eta_{\text{dis}}} \right) $$

能量上下限通过变量边界 $e_i \in [e_{\min}, e_{\max}]$ 实现（由储能容量与当前时段能量窗口决定）。

**15 分钟支撑约束**（放电 15 min 后剩余能量仍 ≥ 下限、充电 15 min 不溢出）：

$$ e_i - 0.25\, p_i^{\text{dis}} \geq e_{\min}, \qquad e_i + 0.25\, p_i^{\text{ch}} \leq e_{\max} $$

**PCS 视在功率运行范围**（防止有功无功同时满发超出变流器容量）：

$$ (p_i^{\text{dis}} - p_i^{\text{ch}})^2 + (q_i^{\text{st}})^2 \leq S_{\text{pcs},i}^2 $$

其中 $S_{\text{pcs},i}$ 为储能 PCS 额定容量（pu）。

说明：未显式施加充放电互斥约束（ $p^{\text{ch}}\cdot p^{\text{dis}} = 0$）。由于目标函数对根节点注入单调，同时充放电只会造成能量浪费，最优解自然不同时充放电。

#### ⑧ 可调度负荷约束（z 为当前挂载的调节因子）

$$ z_i^{\text{lb}} \leq z_i \leq z_i^{\text{ub}}, \qquad \text{实际消费} = z_i \times \text{当前挂载}_i $$

$z$ 不再乘在**满载容量**上（不再是"直接削减"的硬比例），而是作用于**当前实际挂载**、既可削减也可**增荷**的软性调节因子：

- **当前挂载** `base_ratio` 取自 OpenDSS 官方 `LoadShape.mult[0]`：实际功率 = 满载 × mult（[parse_dss.py](parse_dss.py) 解析，无自定义字段）；
- **z 可大于 1**：增荷指令允许负荷从当前挂载提升到满载；也可削减至可调下限；
- **z 上下限**由可调范围（相对满载）与当前挂载推导：

$$ z_i^{\text{lb}} = \frac{\text{mult}_i^{\text{lb}}}{\text{mult}_i^{\text{cur}}}, \qquad z_i^{\text{ub}} = \frac{\text{mult}_i^{\text{ub}}}{\text{mult}_i^{\text{cur}}} $$

其中 $\text{mult}^{\text{cur}}$ 为当前挂载比例， $\text{mult}^{\text{lb/ub}}$ 为可调范围（相对满载，默认 [0, 1]，显式配置见 [parse_dss.py](parse_dss.py) `LOAD_MULT_LIMITS`）。

**可调范围配置（相对满载）：**

| 负荷类型 | mult 范围 | 当前挂载 | z 范围 | 说明 |
|:--------|:--------|:--------|:--------|:----|
| EV_Bus19 / EV_Bus20 | [0.3, 1.0] | 0.4 / 0.6 | [0.75, 2.5] / [0.5, 1.667] | EV 最低充电需求 = 满载 30% |
| EV_Bus7 | [0.1, 1.0] | 0.25 | [0.4, 4.0] | 可削减至满载 10% |
| AC_Bus2（空调） | [0.1, 1.0] | 0.5 | [0.2, 2.0] | 最低保持满载 10% |
| Fixed（固定负荷） | [1.0, 1.0] | 1.0 | z = 1.0 | 不可调 |

***

详细输出格式约定见：

- `output/output输出控制.md` — 场景预测/验证输出格式
- `training_dataset/training_dataset输出控制.md` — 训练集输出格式

所有数值均为**工程单位**：功率 MW、无功 Mvar、能量 MWh、电压 pu。

### 通用约定

- `sense` 列：`min` = 最小化根节点注入，`max` = 最大化根节点注入
- 列名带 `_mw`/`_mvar`/`_mwh` 时，值即 MW/Mvar/MWh
- 无量纲比例（辐照度、挂载比例、z、pct）列名不加单位后缀
- 文件编码 UTF-8，表头行存在

***

## 使用示例

### 全链路示例（流水线）

```bash
# 1. 生成训练集 (默认算例 default, 200 样本/断面)
python training_dataset_mc.py --config training_dataset/training_dataset_default/training_dataset_mc_config.csv

# 2. 训练 KNN 模型
python train_knn.py --csv training_dataset/training_dataset_default/csv

# 3. 场景预测 (KNN)
python scenario_dataset_mc.py --scenario output_scenario_default --model training_dataset_default --n 200

# 4. 场景验证 (OPF 真值)
python scenario_dataset_mc_OPF.py --scenario output_scenario_default --n 200 --seed 42

# 5. 可视化对比 (七线图)
python plot_scenario.py --mode real --scenario scenario_default
```

### 单断面 OPF 示例

```bash
python main.py model_default
```

### 各程序参数与优先级

**通用优先级**：CLI 显式参数 > 配置文件(config CSV) 非空值 > 内置默认值。下表"覆盖"即指覆盖 config 中对应键的值；config 未提供或未找到时该键使用内置默认值（表中加粗数值/字符串）。

#### main.py — 单断面 OPF（无配置文件）

| 参数 | 作用 | 默认值 |
| ---- | ---- | ---- |
| `scenario`（位置参数） | 模型名（`model_xxx`）或模型路径；原默认 `scenario_all` 依赖的 OpenDSS 数据已移除，需显式指定 | 无（需显式指定，如 `model_default`） |

#### training_dataset_mc.py — 训练集生成（config: `training_dataset_mc_config.csv`）

| 参数 | 作用 | 优先级 | 默认值 |
| ---- | ---- | ---- | ---- |
| `scenario`（位置） | 场景名（输出目录 `training_dataset/training_dataset_{scenario}/`）；网络模型由 config 的 `model` 键单独指定 | 给了即**覆盖** config 的 `scenario`；缺省读 config | 无内置默认 |
| `--config` | 全局控制 CSV 路径 | 显式指定优先；缺省自动找 `training_dataset/training_dataset_{scenario}/training_dataset_mc_config.csv` | — |
| `--n` | 抽样次数 | **覆盖** config `n_samples` | **500** |
| `--seed` | 随机种子 | **覆盖** config `seed` | **42** |
| `--sense` | `min` / `max` / `both` | 无 config 对应键 | **`both`** |

> config 全局键：`model`（网络模型目录名，如 `model_default`）、`scenario`（输出目录名，如 `default`）、`n_samples`、`seed`、`start_time`（默认 **`0:00`**）、`end_time`（默认 **`23:45`**）；组件行支持 `mu:曲线名`/`sigma:曲线名`（逐断面取 shapes 曲线值）或 `cv`/`sigma` 常数参数。

#### train_knn.py — KNN 模型训练（config: `knn_config.csv`）

| 参数 | 作用 | 优先级 | 默认值 |
| ---- | ---- | ---- | ---- |
| `--config` | knn_config.csv 路径 | 显式指定；缺省按 `--csv`/默认训练集名推导 `knn_lib/{训练集名}/knn_config.csv` | — |
| `--csv` | 训练集 CSV 目录 | **覆盖** config `dataset_csv` | 无 config 时旧默认 `training_dataset/training_dataset_storage_bus18_sample/csv`（建议显式指定，如 `training_dataset/training_dataset_default/csv`） |
| `--k` | KNN 邻居数 | **覆盖** config `knn_k` | **5** |
| `--weights` | `uniform` / `distance` | **覆盖** config `knn_weights` | **`distance`** |
| `--test-size` | 测试集比例 | **覆盖** config `knn_test_size` | **0.2** |
| `--seed` | 随机种子 | **覆盖** config `knn_seed` | **42** |

> config 其余键（无 CLI，留空即用内置默认）：`knn_metric`（默认 **`minkowski`**）、`knn_p`（默认 **2**）、`knn_algorithm`（默认 **`auto`**）、`knn_leaf_size`（默认 **30**）、`knn_n_jobs`（默认空 → sklearn 默认 **1**）。

#### scenario_dataset_mc.py — KNN 场景预测（config: `output_mc_config.csv`）

| 参数 | 作用 | 优先级 | 默认值 |
| ---- | ---- | ---- | ---- |
| `--config` | output_mc_config.csv 路径 | 显式指定优先；缺省自动查找 `output/` 下唯一一份配置（多份时必须 `--config`） | — |
| `--scenario` | 场景名 | **覆盖** config `scenario` | 无内置默认（读 config） |
| `--model` | 模型名（训练集文件夹名） | **覆盖** config `model` | 无内置默认（读 config） |
| `--model-dir` | 模型目录 | **覆盖** | `knn_lib/{model}/` |
| `--n` | 每断面抽样次数 | **覆盖** config `n_samples` | **200** |
| `--seed` | 随机种子 | **覆盖** config `seed` | **42** |
| `--start-time` / `--end-time` | 起始/结束断面 | **覆盖** config 对应键 | **`0:00` / `23:45`** |

#### scenario_dataset_mc_OPF.py — OPF 真值验证（config: `output_mc_config.csv`）

与 `scenario_dataset_mc.py` 规则相同（`--config` / `--scenario` / `--n` / `--seed` / `--start-time` / `--end-time`，默认 **200 / 42 / `0:00` / `23:45`**），无 `--model`（实际求解 OPF，config 的 `model` 键被忽略）。

#### plot_scenario.py — 可视化（无配置文件）

| 参数 | 作用 | 默认值 |
| ---- | ---- | ---- |
| `--scenario` | 场景名（`scenario/{name}/`） | `scenario_trail_1` |
| `--mode` | `prob`（概率边界图）/ `real`（七线验证图） | **`prob`** |
| `--knn-dir` | real 模式：KNN 结果目录 | `output/output_{scenario}/` |
| `--opf-dir` | real 模式：OPF 结果目录 | `output/output_{scenario}_opf/` |
| `--out` | 图片输出完整路径 | 见各模式输出说明 |
| `--out-dir` | 图片输出目录 | `output/{scenario}/`（prob）/ KNN 结果目录（real） |

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
3. `data/opendss_case33/{scenario}/` — OpenDSS 目录（数据已移除，此规则仅代码保留）
4. `{scenario}` — 直接作为路径

支持前缀兼容：`scenario_` ↔ `model_` ↔ `output_`（自动去除 `output_` 前缀回退到原场景名，如 `output_scenario_default` → `scenario_default`）。

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


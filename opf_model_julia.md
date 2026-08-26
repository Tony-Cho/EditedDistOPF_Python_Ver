# VPP OPF 优化模型 —— Julia 版（AC OPF）

## 概述

- **方法**：AC OPF（完整交流潮流，`ACPUPowerModel`）
- **求解器**：Ipopt（内点法，NLP 非线性规划）
- **系统**：三相非平衡显式建模，基准容量 10 MVA
- **框架**：PowerModelsDistribution.jl + JuMP.jl
- **网络**：IEEE 33 节点辐射状配电网 + VPP 资源（光伏/储能/柔性负荷/EV）

---

## 1. 符号说明

### 索引

| 符号 | 含义 |
|------|------|
| $i \in \mathcal{N}$ | 母线索引 |
| $(i,j) \in \mathcal{E}$ | 支路索引（首端 $i$ → 末端 $j$） |
| $g \in \mathcal{G}$ | 发电机索引（含主网电源和光伏） |
| $s \in \mathcal{S}$ | 储能索引 |
| $\ell \in \mathcal{L}$ | 负荷索引 |
| $c \in \{1,2,3\}$ | 相别（A/B/C 三相） |

### 参数

| 符号 | 含义 |
|------|------|
| $V_{\min} = 0.80$ pu | 电压幅值下限 |
| $V_{\max} = 1.15$ pu | 电压幅值上限 |
| $PF = 0.95$ | 光伏功率因数 |
| $z_{\min}^{EV} = 0.3$ | EV 最小充电比例 |
| $\Delta t = 1$ h | 时间步长 |
| $S_{\text{base}} = 10$ MVA | 基准容量 |
| $Y_{ij}$ | 支路导纳矩阵（由 R, X, B 计算） |
| $N_{ij}$ | 变压器变比 |

---

## 2. 决策变量

### 电压变量（三相）

$$V_i^c \in [V_{\min}, V_{\max}], \quad \forall i \in \mathcal{N}, \forall c \in \{1,2,3\}$$
$$\theta_i^c \in [-\pi, \pi], \quad \forall i \in \mathcal{N}, \forall c \in \{1,2,3\}$$

与 Python LinDistFlow 版的**关键区别**：Julia 版直接使用电压幅值 $V_i^c$ 和相角 $\theta_i^c$ 作为变量，**不**使用平方 $U=V^2$。每条母线每相都有一个独立的电压幅值和相角（三相非平衡）。

### 支路功率变量（三相）

$$P_{ij}^c \in \mathbb{R}, \quad Q_{ij}^c \in \mathbb{R}, \quad \forall (i,j) \in \mathcal{E}, \forall c$$

与 Python 版不同，Julia 版的支路功率是**三相各自独立**的。

### 主网电源（slack）注入（三相）

$$P_{\text{sub}}^c \in \mathbb{R}, \quad Q_{\text{sub}}^c \in \mathbb{R}, \quad \forall c$$

通过识别所有发电机中 `pmin = -∞`、`pmax = +∞` 的那个来确定 slack 发电机。

### 光伏出力（三相）

$$p_g^c \in [0, p_{\max}^c], \quad q_g^c \in [-q_{\max}^c, q_{\max}^c], \quad \forall g \in \mathcal{G}, \forall c$$

每相独立取值。$q_{\max}^c$ 基于该相的最大有功和功率因数计算：

$$q_{\max}^c = p_{\max}^c \cdot \tan(\arccos(0.95))$$

### 储能功率（三相）

$$p_s^c \in \mathbb{R}, \quad \forall s \in \mathcal{S}, \forall c$$

储能变量用**净功率** $p_s^c$（正=充电，负=放电）表示，**不**拆分为充电/放电两个变量。通过互补约束防止同时充放电。

### 储能能量状态

$$e_s \in [e_{\min}, e_{\max}], \quad \forall s \in \mathcal{S}$$

标量（非三相），单位 kWh。

### 负荷削减因子

$$z_\ell \in [z_{\min}, z_{\max}], \quad \forall \ell \in \mathcal{L}$$

负荷削减比例（连续变量 `relax=true`，非二进制）。不同类型取值与 Python 版相同：

| 负荷类型 | $z_{\min}$ | $z_{\max}$ |
|---------|-----------|-----------|
| 固定负荷（非可调度） | 1.0 | 1.0 |
| 可调负荷（AC_*） | 0.0 | 1.0 |
| EV 充电桩（EV*） | 0.3 | 1.0 |

---

## 3. 约束条件

### 3.1 参考母线相角

$$\theta_{\text{ref}}^c = 0, \quad \forall c$$

slack 母线的相角固定为零。

### 3.2 发电机出力限值

$$P_{\min,g}^c \leq p_g^c \leq P_{\max,g}^c, \quad \forall g, \forall c$$
$$Q_{\min,g}^c \leq q_g^c \leq Q_{\max,g}^c, \quad \forall g, \forall c$$

### 3.3 交流潮流方程（Ohm 定律）——与 LinDistFlow 的核心区别

`ACPUPowerModel` 使用**极坐标形式**，电压变量为幅值 $V_i^c$ 和相角 $\theta_i^c$。每条支路 $(i,j)$ 的首端和末端分别有 Ohm 定律约束。

**函数调用**（[run_vpp_opf.jl#L204-L205](file:///d:/广东电网/深圳电网/EditedDistOPF%20-%20副本/src/run_vpp_opf.jl#L204-L205)）：

```julia
PMD.constraint_mc_ohms_yt_from(pm, i)   # 首端，调用 PM 的 constraint_ohms_yt_from 逐相执行
PMD.constraint_mc_ohms_yt_to(pm, i)     # 末端
```

函数名中的 **yt** 表示 Y（导纳）和 T（变比）以**直角坐标形式**传入（`tr`, `ti` 为变压器变比的实部和虚部）。

**实际源码中的极坐标展开式**（来自 [PowerModels.jl `src/form/acp.jl`](https://github.com/lanl-ansi/PowerModels.jl/blob/master/src/form/acp.jl)）：

#### 首端（from）— 每相 $c$：

$$P_{ij}^c = \frac{G_{ij}+G_{ij}^\text{fr}}{T_{ij}^2} \cdot (V_i^c)^2 + \frac{-G_{ij}\cdot \text{tr} + B_{ij}\cdot \text{ti}}{T_{ij}^2} \cdot V_i^c V_j^c \cos(\theta_i^c - \theta_j^c) + \frac{-B_{ij}\cdot \text{tr} - G_{ij}\cdot \text{ti}}{T_{ij}^2} \cdot V_i^c V_j^c \sin(\theta_i^c - \theta_j^c)$$

$$Q_{ij}^c = -\frac{B_{ij}+B_{ij}^\text{fr}}{T_{ij}^2} \cdot (V_i^c)^2 - \frac{-B_{ij}\cdot \text{tr} - G_{ij}\cdot \text{ti}}{T_{ij}^2} \cdot V_i^c V_j^c \cos(\theta_i^c - \theta_j^c) + \frac{-G_{ij}\cdot \text{tr} + B_{ij}\cdot \text{ti}}{T_{ij}^2} \cdot V_i^c V_j^c \sin(\theta_i^c - \theta_j^c)$$

#### 对于普通线路（无变压器）

此时 $\text{tr}=1,\ \text{ti}=0,\ T_{ij}=1$，方程简化为：

$$P_{ij}^c = (G_{ij} + G_{ij}^\text{fr}) \cdot (V_i^c)^2 - G_{ij} \cdot V_i^c V_j^c \cos(\theta_i^c - \theta_j^c) - B_{ij} \cdot V_i^c V_j^c \sin(\theta_i^c - \theta_j^c)$$

$$Q_{ij}^c = -(B_{ij} + B_{ij}^\text{fr}) \cdot (V_i^c)^2 + B_{ij} \cdot V_i^c V_j^c \cos(\theta_i^c - \theta_j^c) - G_{ij} \cdot V_i^c V_j^c \sin(\theta_i^c - \theta_j^c)$$

#### 变量说明

| 符号 | 对应代码 | 含义 |
|------|---------|------|
| $V_i^c$ | `vm[bus][c]` | 母线 $i$ 第 $c$ 相电压幅值 |
| $\theta_i^c$ | `va[bus][c]` | 母线 $i$ 第 $c$ 相电压相角 |
| $G_{ij}$ | `g` | 支路 $(i,j)$ 串联电导 |
| $B_{ij}$ | `b` | 支路 $(i,j)$ 串联电纳 |
| $G_{ij}^\text{fr}$ | `g_fr` | 首端并联电导（π 型模型的接地支路） |
| $B_{ij}^\text{fr}$ | `b_fr` | 首端并联电纳（π 型模型的接地支路） |
| $\text{tr},\text{ti}$ | `tr`, `ti` | 变压器变比直角坐标分量（无变压器时为 1 和 0） |
| $T_{ij}$ | `tm` | 变压器变比幅值（无变压器时为 1） |

#### 非线性来源

方程中包含 **两项非线性**：

1. **双线性项** $V_i^c \cdot V_j^c$ — 两个变量相乘
2. **三角函数** $\cos(\theta_i^c - \theta_j^c)$, $\sin(\theta_i^c - \theta_j^c)$ — 超越函数

两者结合使得整个问题的约束是非线性、非凸的，Ipopt 只能找到局部最优解。

**末端（to）** 的方程形式类似，将首端换为末端、$\text{fr}$ 换为 $\text{to}$ 即可。

#### 与此对比 — Python LinDistFlow

$$U_i - U_j = 2(R_{ij}P_{ij} + X_{ij}Q_{ij})$$

全是变量的一次方，没有乘积也没有三角函数，完全是线性的。

### 3.4 支路相角差限值

$$|\theta_i^c - \theta_j^c| \leq \Theta_{\max}, \quad \forall (i,j), \forall c$$

### 3.5 支路热极限和载流量

$$\sqrt{(P_{ij}^c)^2 + (Q_{ij}^c)^2} \leq S_{\max,ij}^c, \quad \forall (i,j), \forall c$$

（二阶锥约束，Gurobi 可处理，但 Ipopt 处理为一般非线性约束）

### 3.6 节点功率平衡（含负荷削减）

对于每个母线 $i \in \mathcal{N}$，每相 $c$：

**有功平衡：**

$$\sum_{g \in \mathcal{G}_i} p_g^c + \sum_{s \in \mathcal{S}_i} p_s^c + \sum_{(k,i) \in \mathcal{E}} P_{ki}^c - \sum_{(i,j) \in \mathcal{E}} P_{ij}^c = \sum_{\ell \in \mathcal{L}_i} z_\ell \cdot p_\ell^{0,c} - \sum_{k \in \mathcal{K}_i} q_k^c$$

**无功平衡：**

$$\sum_{g \in \mathcal{G}_i} q_g^c + \sum_{(k,i) \in \mathcal{E}} Q_{ki}^c - \sum_{(i,j) \in \mathcal{E}} Q_{ij}^c = \sum_{\ell \in \mathcal{L}_i} z_\ell \cdot q_\ell^{0,c} - \sum_{k \in \mathcal{K}_i} q_k^c$$

这里使用的是 `constraint_mc_power_balance_shed`，它与标准功率平衡的区别是负荷项乘了 $z_\ell$。

### 3.7 储能约束

**能量状态演化（与 Python 版相同）：**

$$e_s = e_{\text{init},s} + \Delta t \cdot \left( p_{\text{ch},s} \cdot \eta_{\text{ch},s} - \frac{p_{\text{dis},s}}{\eta_{\text{dis},s}} \right)$$

**互补约束——与 Python 版的区别：**

$$p_{\text{ch},s} \cdot p_{\text{dis},s} = 0$$

这是**非线性互补约束**（`constraint_storage_complementarity_nl`），防止储能同时充电和放电。在 Julia 版中，充放电被显式拆分为两个变量 $p_{\text{ch}}$ 和 $p_{\text{dis}}$，互补约束确保两者不同时为正。

Python LinDistFlow 版**省略**了此约束，因为线性规划无法表达 $x \cdot y = 0$ 这类非线性的互补关系。

### 3.8 变压器约束

$$\begin{pmatrix} P_{ij}^c \\ Q_{ij}^c \\ P_{ji}^c \\ Q_{ji}^c \end{pmatrix} = f_{\text{transformer}}(V_i^c, V_j^c, \theta_i^c, \theta_j^c, N_{ij})$$

变压器约束涉及变比 $N_{ij}$ 和相移，是完整的非线性模型。

### 3.9 开关约束

开关闭合时：$V_i^c = V_j^c$（首末端电压相等）
开关断开时：$P_{ij}^c = Q_{ij}^c = 0$（无功率流过）

---

## 4. 目标函数

### 场景 1：最小化根节点注入

$$\min \sum_{c=1}^{3} P_{\text{sub}}^c$$

### 场景 2：最大化根节点注入

$$\max \sum_{c=1}^{3} P_{\text{sub}}^c$$

与 Python 版相同的目标，但 Python 版是**单相**变量 $P_{\text{sub}}$，Julia 版是三相求和。

---

## 5. 模型性质

| 性质 | Julia 版 | Python 版 |
|------|----------|-----------|
| **潮流模型** | AC OPF（完整交流潮流） | LinDistFlow（线性近似） |
| **系统建模** | 三相非平衡 | 三相平衡单相等值 |
| **求解器** | Ipopt | Gurobi |
| **问题类型** | NLP（非线性规划） | LP（线性规划） |
| **凸性** | ❌ 非凸（局部最优） | ✅ 凸（全局最优） |
| **约束类型** | 非线性 + 二阶锥 | 纯线性 |
| **线路损耗** | ✅ 显式建模 | ❌ 忽略 |
| **电压相角** | ✅ 显式建模 | ❌ 不考虑 |
| **支路热极限** | ✅ 有 | ❌ 无 |
| **变压器/开关** | ✅ 有 | ❌ 无 |
| **储能互补** | ✅ 有（非线性） | ❌ 无 |
| **求解速度** | 慢（数秒~数十秒） | 快（< 1 秒） |
| **精度** | 高（完整物理模型） | 较高（线损误差 < 1%） |

---

## 6. 与 Python 版的关键差异详解

### 6.1 三相 vs 单相等值

```
Julia 版：
  V_bus2 = [0.982, 0.981, 0.983]    ← 每相独立
  P_line_L1 = [0.33, 0.34, 0.33]    ← 每相独立

Python 版：
  V_bus2 = 0.982                     ← 单相（三相平衡假设）
  P_line_L1 = 1.0                    ← 三相总和
  结果输出时：V_bus2/3 填入三相      ← 人为展开
```

### 6.2 储能建模差异

| 方面 | Julia 版 | Python 版 |
|------|----------|-----------|
| 变量 | $p_{\text{ch}}$、$p_{\text{dis}}$ 拆分为两个 | $p_{\text{ch}}$、$p_{\text{dis}}$ 拆分相同 |
| 互补 | $p_{\text{ch}} \cdot p_{\text{dis}} = 0$（非线性） | 无互补约束 |
| 净功率 | $p_s = p_{\text{ch}} - p_{\text{dis}}$（受互补约束） | $p_s = p_{\text{ch}} - p_{\text{dis}}$（可能同时 > 0） |
| 效率 | $se = se_0 + \Delta t(\eta_{\text{ch}} p_{\text{ch}} - p_{\text{dis}} / \eta_{\text{dis}})$ | 相同 |

### 6.3 光伏无功处理

- **Julia 版**：从 OpenDSS 解析后，逐相设置 `qg_lb` 和 `qg_ub`
- **Python 版**：在 `parse_dss.py` 中用 `PV_Q_FACTOR` 整体计算，三相共用一个限值

### 6.4 电压约束实现

| Julia 版 | Python 版 |
|----------|-----------|
| 通过 JuMP 手动添加 `@constraint(v .>= 0.80)` | 通过变量上下界 `lb=U_MIN, ub=U_MAX` 隐式实现 |
| 约束在求解器内部可见 | 变量边界由求解器直接处理 |
| 可灵活地逐个母线差异化设置 | 所有母线统一上下限 |

---

## 7. 代码结构对照

```
Julia 版（main.jl）                          Python 版（main.py）
────────────────────────                     ────────────────────────
main.jl                                      main.py
  └─ run_vpp_demo()                            └─ run_vpp_demo()
       ├─ PMD.parse_file(dss_path)                  ├─ parse_dss(dss_path)
       ├─ add_vpp_resources!()                      ├─ show_vpp_resources()
       ├─ build_and_solve_opf(:Min)                 ├─ build_and_solve_opf("min")
       │    ├─ transform_data_model()               │    ├─ 直接构建 Gurobi 模型
       │    ├─ instantiate_model()                  │    ├─ 添加变量/约束
       │    │    └─ build_vpp_opf()                 │    └─ model.optimize()
       │    │         ├─ variable_mc_bus_voltage    │
       │    │         ├─ variable_mc_branch_power   │
       │    │         ├─ constraint_mc_ohms_yt      │ ← 这一步在 Python 中替换为
       │    │         │                              │    LinDistFlow 线性方程
       │    │         └─ constraint_mc_power_balance │
       │    └─ optimize_model!(Ipopt)               │
       └─ build_and_solve_opf(:Max)                 └─ build_and_solve_opf("max")
```

核心区别：
- Julia 版依赖 PMD 库提供模块化的 `variable_*` 和 `constraint_*` 函数
- Python 版从头手写所有变量和线性约束方程

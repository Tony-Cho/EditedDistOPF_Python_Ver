# VPP OPF 优化模型 —— 数学描述

## 概述

- **方法**：SOCP 松弛（精确 DistFlow 方程的二阶锥松弛，含线损）
- **求解器**：Gurobi（SOCP 二阶锥规划）
- **系统**：三相平衡单相等值，基准容量 10 MVA（标幺值）
- **网络**：IEEE 33 节点辐射状配电网

---

## 1. 符号说明

### 集合

| 符号 | 含义 |
|------|------|
| $\mathcal{N}$ | 所有母线集合 |
| $\mathcal{E}$ | 所有有效支路集合（从 slack BFS 遍历得到） |
| $\mathcal{G}$ | 所有光伏集合 |
| $\mathcal{S}$ | 所有储能集合 |
| $\mathcal{L}$ | 所有负荷集合 |

### 参数

| 符号 | 值 | 含义 |
|------|-----|------|
| $V_{\min}$ | 0.95 pu | 电压幅值下限 |
| $V_{\max}$ | 1.10 pu | 电压幅值上限 |
| $U_{\min}=V_{\min}^2$ | 0.64 pu² | 电压平方下限 |
| $U_{\max}=V_{\max}^2$ | 1.3225 pu² | 电压平方上限 |
| $PF$ | 0.95 | 光伏功率因数 |
| $z_{\min}^{EV}$ | 0.3 | EV 最小充电比例 |
| $\Delta t$ | 1.0 h | 时间步长 |
| $S_{\text{base}}$ | 10 MVA | 基准容量 |

---

## 2. 决策变量

### 电压变量

$$U_i \in [U_{\min}, U_{\max}], \quad \forall i \in \mathcal{N}$$

$U_i = V_i^2$，即电压幅值的平方。用平方代替幅值是为了使 LinDistFlow 约束保持线性。

### 支路功率变量

$$P_{ij} \in \mathbb{R}, \quad \forall (i,j) \in \mathcal{E}$$
$$Q_{ij} \in \mathbb{R}, \quad \forall (i,j) \in \mathcal{E}$$

从节点 $i$ 流向节点 $j$ 的有功和无功功率（pu）。

### 支路电流平方

$$l_{ij} \geq 0, \quad \forall (i,j) \in \mathcal{E}$$

$l_{ij}$ 是支路电流幅值的平方（pu），满足 $l_{ij} = \frac{P_{ij}^2 + Q_{ij}^2}{U_i}$（通过 SOCP 松弛实现）。

### 变电站（slack）注入

$$P_{\text{sub}} \in \mathbb{R}, \quad Q_{\text{sub}} \in \mathbb{R}$$

从变电站（根节点 bus1）注入配电网的有功和无功功率。

### 光伏出力

$$p_g \in [0, p_{\max}], \quad \forall g \in \mathcal{G}$$
$$q_g \in [-q_{\max}, q_{\max}], \quad \forall g \in \mathcal{G}$$

光伏有功出力（0 到最大功率点之间）和无功出力（由功率因数 0.95 限制）。

其中：
- $p_{\max} = P_{\text{pmpp}} / S_{\text{base}}$（标幺化最大功率）
- $q_{\max} = p_{\max} \cdot \tan(\arccos(0.95)) = p_{\max} \cdot \frac{\sqrt{1-0.95^2}}{0.95} \approx 0.3287 \cdot p_{\max}$

### 储能

$$p_{\text{ch}} \in [0, p_{\text{ch,ub}}], \quad \forall s \in \mathcal{S}$$
$$p_{\text{dis}} \in [0, p_{\text{dis,ub}}], \quad \forall s \in \mathcal{S}$$
$$e_s \in [e_{\min}, e_{\max}], \quad \forall s \in \mathcal{S}$$

储能充电功率、放电功率（均非负），以及能量状态。

其中：
- $p_{\text{ch,ub}} = \frac{\%\text{charge}}{100} \cdot \frac{P_{\text{rated}}}{S_{\text{base}}}$
- $p_{\text{dis,ub}} = \frac{\%\text{discharge}}{100} \cdot \frac{P_{\text{rated}}}{S_{\text{base}}}$
- $e_{\max} = \frac{E_{\text{rated}}}{S_{\text{base}}}$（pu·h）
- $e_{\min} = 0.1 \cdot e_{\max}$

### 负荷削减因子

$$z_\ell \in [z_{\min}, z_{\max}], \quad \forall \ell \in \mathcal{L}$$

负荷削减比例。不同负荷类型取值不同：

| 负荷类型 | $z_{\min}$ | $z_{\max}$ | 说明 |
|---------|-----------|-----------|------|
| 固定负荷 | 1.0 | 1.0 | 不可削减 |
| 可调负荷（非 EV） | 0.0 | 1.0 | 0~100% 可调 |
| EV 充电桩 | 0.3 | 1.0 | 最低充电需求 30% |

实际负荷有功/无功消耗 = 原始负荷 $\times$ $z_\ell$。

---

## 3. 约束条件

### 3.1 Slack 电压固定

$$U_{\text{slack}} = 1.0 \quad (\text{即 } V_{\text{slack}} = 1.0 \text{ pu})$$

变电站母线电压固定在额定值。

### 3.2 SOCP 电压降（含线损）

引入支路电流平方 $l_{ij} = \dfrac{P_{ij}^2 + Q_{ij}^2}{U_i}$，精确 DistFlow 电压降方程可写为：

$$U_j = U_i - 2(R_{ij}P_{ij} + X_{ij}Q_{ij}) + (R_{ij}^2 + X_{ij}^2)l_{ij}$$

此方程对 $U$, $P$, $Q$, $l$ 是**线性**的。$l_{ij}$ 的定义式是非线性的，但可以松弛为二阶锥约束。

#### SOCP 松弛

$$l_{ij} \cdot U_i \geq P_{ij}^2 + Q_{ij}^2, \quad \forall (i,j) \in \mathcal{E}$$

或写为标准二阶锥形式：

$$\left\| \begin{pmatrix} 2P_{ij} \\ 2Q_{ij} \\ l_{ij} - U_i \end{pmatrix} \right\|_2 \leq l_{ij} + U_i$$

对于**辐射状配电网**，此松弛在最优解处**通常是紧的**（即 $l_{ij} \cdot U_i = P_{ij}^2 + Q_{ij}^2$），因此等效于精确的 DistFlow 线损模型。

#### 对比 LinDistFlow

| 项目 | LinDistFlow（原 Python 版） | SOCP 松弛（当前版） |
|------|--------------------------|-------------------|
| 电压方程 | $U_i - U_j = 2(rP+xQ)$ | $U_j = U_i - 2(rP+xQ) + (r^2+x^2)l$ |
| 线损项 $(r^2+x^2)l$ | ❌ 忽略 | ✅ 保留 |
| 松弛紧性 | — | 辐射网下紧 |
| 凸性 | LP | SOCP（凸） |

### 3.3 节点功率平衡（含线损）

对于每个母线 $i \in \mathcal{N}$：

**有功平衡：**

$$
\sum_{(k,i) \in \mathcal{E}} P_{ki} \;-\!\! \sum_{(i,j) \in \mathcal{E}} P_{ij} =
\begin{cases}
\displaystyle\sum_{\ell \in \mathcal{L}_i} p_{\ell}^0 z_\ell \;-\!\! \sum_{g \in \mathcal{G}_i} p_g +\!\! \sum_{s \in \mathcal{S}_i} (p_{\text{ch},s} - p_{\text{dis},s}) +\!\! \sum_{(k,i) \in \mathcal{E}} R_{ki} l_{ki}, & i \neq \text{slack} \\[10pt]
P_{\text{sub}} \;-\!\! \displaystyle\sum_{(i,j) \in \mathcal{E}} P_{ij} = \displaystyle\sum_{\ell \in \mathcal{L}_i} p_{\ell}^0 z_\ell \;-\!\! \sum_{g \in \mathcal{G}_i} p_g +\!\! \sum_{s \in \mathcal{S}_i} (p_{\text{ch},s} - p_{\text{dis},s}) +\!\! \sum_{(k,i) \in \mathcal{E}} R_{ki} l_{ki}, & i = \text{slack}
\end{cases}
$$

**无功平衡：**

$$
\sum_{(k,i) \in \mathcal{E}} Q_{ki} \;-\!\! \sum_{(i,j) \in \mathcal{E}} Q_{ij} =
\begin{cases}
\displaystyle\sum_{\ell \in \mathcal{L}_i} q_\ell^0 z_\ell \;-\!\! \sum_{g \in \mathcal{G}_i} q_g \;-\!\! \sum_{c \in \mathcal{C}_i} q_c +\!\! \sum_{(k,i) \in \mathcal{E}} X_{ki} l_{ki}, & i \neq \text{slack} \\[10pt]
Q_{\text{sub}} \;-\!\! \displaystyle\sum_{(i,j) \in \mathcal{E}} Q_{ij} = \displaystyle\sum_{\ell \in \mathcal{L}_i} q_\ell^0 z_\ell \;-\!\! \sum_{g \in \mathcal{G}_i} q_g \;-\!\! \sum_{c \in \mathcal{C}_i} q_c +\!\! \sum_{(k,i) \in \mathcal{E}} X_{ki} l_{ki}, & i = \text{slack}
\end{cases}
$$

新增的 $R_{ki} l_{ki}$ 和 $X_{ki} l_{ki}$ 项即为**以该母线为末端的支路线损**。与原 LinDistFlow 相比，现在的功率平衡中包含了线路发热损耗，物理上更完整。

#### 对比

| 版本 | 有功平衡 | 物理含义 |
|------|---------|---------|
| 原始 DistFlow | $\sum P_{\text{in}} - \sum P_{\text{out}} = \text{净消费} + \sum R l$ | 入流减去出流还要扣除线损 |
| LinDistFlow（原 Python） | $\sum P_{\text{in}} - \sum P_{\text{out}} = \text{净消费}$ | 入流减去出流刚好等于消费（忽略线损） |
| SOCP（当前版） | 同上，但 $l$ 是变量且满足 SOCP 松弛 | 保留了线损项 |

其中：
- $p_\ell^0$、$q_\ell^0$ = 原始负荷有功/无功（pu）
- $q_c$ = 电容器注入无功（正值，pu）
- $\mathcal{L}_i$、$\mathcal{G}_i$、$\mathcal{S}_i$、$\mathcal{C}_i$ 分别为挂接在母线 $i$ 上的负荷、光伏、储能、电容器集合

含义：**流入母线的功率 - 流出母线的功率 = 该母线净消耗功率**。

### 3.4 储能能量状态约束

$$\forall s \in \mathcal{S}: \quad e_s = e_{\text{init},s} + \Delta t \cdot \left( p_{\text{ch},s} \cdot \eta_{\text{ch},s} - \frac{p_{\text{dis},s}}{\eta_{\text{dis},s}} \right)$$

其中：
- $e_{\text{init},s} = \text{SOC}_0 \cdot e_{\max}$ 初始能量
- $\eta_{\text{ch}}$、$\eta_{\text{dis}}$ = 充放电效率（默认 0.95）
- 能量上下限通过 $e_s$ 的变量边界 $[e_{\min}, e_{\max}]$ 实现

### 3.5 储能互补性说明

代码中未显式施加充放电互斥约束（$p_{\text{ch}} \cdot p_{\text{dis}} = 0$）。在实际优化中，由于目标函数是最小化/最大化 $P_{\text{sub}}$，且充放电同时进行会造成能量浪费，因此优化结果自然不会同时充放电。

### 3.6 支路热极限约束（SOCP）

每条有效支路（非开关、启用的线路）受到视在功率额定值限制：

$$P_{ij}^2 + Q_{ij}^2 \leq (S_{\max,ij})^2, \quad \forall (i,j) \in \mathcal{E}$$

$S_{\max,ij}$ 的计算分三步：

**第 1 步：将线路阻抗从 Ω 转为 pu**

$$z_{\text{base}} = \frac{V_{\text{base}}^2}{S_{\text{base}}} = \frac{12.66^2}{10} = 16.0276 \ \Omega$$

$$r_{ij}^{\text{pu}} = \frac{R_{ij}}{z_{\text{base}}}, \quad x_{ij}^{\text{pu}} = \frac{X_{ij}}{z_{\text{base}}}$$

**第 2 步：计算所有启用线路的平均阻抗模值**

$$z_{ij}^{\text{pu}} = \sqrt{(r_{ij}^{\text{pu}})^2 + (x_{ij}^{\text{pu}})^2}$$

$$z_{\text{avg}} = \frac{1}{|\mathcal{E}_{\text{active}}|} \sum_{(i,j) \in \mathcal{E}_{\text{active}}} z_{ij}^{\text{pu}}$$

IEEE 33 节点结果：

```
L1_1_2:   r=0.00575, x=0.00293 → z=0.00645
L2_2_3:   r=0.03075, x=0.01567 → z=0.03452
L3_3_4:   r=0.02283, x=0.01163 → z=0.02562
...
L17_17_18: r=0.00780, x=0.00747 → z=0.01080
────────────────────────────────────────────
z_avg = 所有 32 条线路 z 的算术平均 ≈ 0.03
```

**第 3 步：按阻抗反比例分配热限额**

$$S_{\max,ij} = \max\left(0.3,\ \min\left(4.0,\ 2.0 \cdot \frac{z_{\text{avg}}}{\max(z_{ij}^{\text{pu}}, 0.001)}\right)\right) \cdot P_{\text{load}}^{\text{total}}$$

其中 $P_{\text{load}}^{\text{total}}$ 为全网总有功负荷（pu）。

含义：**阻抗越小（越靠近变电站），线路分配到的热限额越大。**

```
以 z_avg = 0.03, total_p_load = 0.37 pu 为例

L1_1_2 (z=0.00645):
  ratio = 2 × 0.03 / 0.00645 = 9.3
  → 上限 4.0 截断
  smax = 4.0 × 0.37 = 1.48 pu (14.8 MVA)    ← 首端大容量

L16_16_17 (z=0.0479):
  ratio = 2 × 0.03 / 0.0479 = 1.25
  → 在 [0.3, 4.0] 范围内
  smax = 1.25 × 0.37 = 0.46 pu (4.6 MVA)    ← 末端小容量
```

如果 OpenDSS 数据中指定了 `normamps`（额定电流 A/相），则优先使用：

$$S_{\max,ij} = \frac{\sqrt{3} \cdot V_{\text{base}} \cdot I_{\text{rated}}}{S_{\text{base}}}$$

添加此约束后，模型从 **LP（线性规划）** 变为 **SOCP（二阶锥规划）**。Gurobi 原生支持二阶锥约束，求解效率仍然很高。

**与 Julia 版的区别**：Julia AC OPF 中也包含热极限约束（`constraint_mc_thermal_limit_from/to`），形式相同，但 Julia 版逐相施加，Python 版是单相等值。

---

## 4. 目标函数

### 场景 1：最小化根节点注入

$$\min \; P_{\text{sub}}$$

含义：尽可能利用本地分布式资源（光伏 + 储能 + 负荷调节）满足负荷需求，减少从电网购电。对应 VPP 的"自消纳"模式。

### 场景 2：最大化根节点注入

$$\max \; P_{\text{sub}}$$

含义：尽可能将本地多余功率送回电网。相当于储能放电 + 光伏满发，同时负荷尽可能削减，最大限度地反向送电。

---

## 5. 模型性质

| 性质 | 说明 |
|------|------|
| **类型** | 二阶锥规划（SOCP） |
| **凸性** | 凸（全局最优） |
| **求解器** | Gurobi |
| **线损** | ✅ SOCP 松弛精确建模 |
| **支路热极限** | ✅ $P_{ij}^2+Q_{ij}^2 \leq S_{\max}^2$ |
| **变量数** | 约 $|\mathcal{N}| + 3|\mathcal{E}| + 2|\mathcal{G}| + 3|\mathcal{S}| + |\mathcal{L}| + 2$ |
| **约束数** | $1 + 2|\mathcal{E}| + 2|\mathcal{N}| + |\mathcal{S}| + |\mathcal{E}_{\text{active}}|$（含 SOCP + 热极限） |

对于 IEEE 33 节点：
- 母线数 33，有效支路 32
- 新增 $l_{ij}$ 变量：32 个
- 光伏 3 个，储能 2 个，负荷 38 个
- 总变量约 212 个，总约束约 165 个
- Gurobi SOCP 求解时间通常 < 1 秒

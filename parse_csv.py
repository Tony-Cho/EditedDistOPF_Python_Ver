# -*- coding: utf-8 -*-
"""
parse_csv.py
CSV 配电网数据解析器

读取 data/csv_case33/{model}/ 目录下的 CSV 数据文件
(格式约定见 data/csv_case33/输入控制.md), 构建 Network 对象
(含 finalize 派生量), 供 LinDistFlow VPP OPF 求解使用。

与 parse_dss.py 相互独立: 本文件自带数据模型与派生量计算逻辑,
与 parse_dss 的数据类/算法保持一致 (同一套约定), 二者可并行使用。

用法:
    from parse_csv import parse_csv
    net = parse_csv("data/csv_case33/model_storage_bus18")

约定:
- 三相平衡单相等值, per-unit on model_circuit.csv 的 base_mva
- 资源接入 = model_circuit.csv 类别开关 (*_enabled) 且 元件 enabled 均有效
- loads/pvs/storage 的 shape 列引用 model_shapes.csv 曲线, npts/interval 由组件表给出
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =====================================================================
# 配置常量 (与 parse_dss.py 保持一致)
# =====================================================================

# 可调度负荷 mult 可调范围配置: mult = 当前挂载比例 (相对满载, 由 LoadShape.mult[0] 给出)
# 可调范围 [mult_lb, mult_ub] 决定 z 的求解范围: z ∈ [mult_lb/当前, mult_ub/当前]
# 未在表中显式配置的负荷, 默认 mult ∈ [0.0, 1.0]
LOAD_MULT_LIMITS: Dict[str, Tuple[float, float]] = {
    "EV_Bus19": (0.3, 0.8),    # EV 最小充电需求 = 满载 30% (含原 EV_Bus20, 已合并); 上限 = 历史均值
    "EV_Bus7":  (0.1, 0.8),    # 当前挂载 0.25, 允许削减至满载 10%; 上限 = 历史均值
    "AC_Bus2":  (0.1, 0.8),    # 空调最低保持满载 10%; 上限 = 历史均值
}

# 储能容量上下限配置: 值 = (能量下限比例, 能量上限比例), 均相对额定能量容量 (kWh)
# 未在表中显式配置的储能, 默认能量比例 ∈ [0.1, 1.0]
STORAGE_ENERGY_LIMITS: Dict[str, Tuple[float, float]] = {
    "BESS_Bus18": (0.1, 0.9),   # 容量下限 = 10% 额定, 上限 = 90% 额定 (SOC 运行窗口)
}

# 光伏功率因数统一约定 (工程约定 0.98): pvs.csv 的 pf 列按此值写入/兜底
PV_PF_UNIFIED = 0.98


def pf_to_q_factor(pf: float) -> float:
    """功率因数 → 无功/有功比 |Q|/P = tan(acos(PF))"""
    return (1.0 - pf ** 2) ** 0.5 / pf


# =====================================================================
# 数据类定义 (与 parse_dss.py 保持一致)
# =====================================================================

@dataclass
class Bus:
    """母线"""
    name: str
    base_kv: float = 12.66
    vm: float = 1.0  # 初始电压 (pu)


@dataclass
class Line:
    """线路 (三相)"""
    name: str
    fbus: str
    tbus: str
    r_ohm: float          # 单相电阻 (ohm)
    x_ohm: float          # 单相电抗 (ohm)
    phases: int = 3
    enabled: bool = True
    is_switch: bool = False
    normamps: float = 0.0 # 额定电流 (A/相)，来自 OpenDSS normamps
    # 派生量 (在 Network.finalize 中计算)
    r_pu: float = 0.0
    x_pu: float = 0.0
    smax_pu: float = 0.0  # 热极限视在功率 (pu)


@dataclass
class Load:
    """负荷 (三相, 恒功率 model=1)"""
    name: str
    bus: str
    kw: float
    kvar: float
    kv: float = 12.66
    type: Optional[str] = None    # 显式类型 (CSV): fixed/ev/ac/fixed_extra; None=DSS 按命名约定识别
    dispatchable: bool = False
    is_ev: bool = False
    shape: str = ""           # 绑定的 LoadShape 名 (daily/duty/yearly, 提供当前挂载比例)
    # 派生量
    p_pu: float = 0.0         # 满载有功 (pu)
    q_pu: float = 0.0         # 满载无功 (pu)
    base_ratio: float = 1.0   # 当前挂载比例 = LoadShape.mult[0] (仅可调度负荷有意义)
    mult_lb: float = 0.0      # mult 可调下限 (相对满载); 常数或绑定时变曲线当前断面值
    mult_ub: float = 1.0      # mult 可调上限 (相对满载, 默认=满载)
    mult_lb_shape: str = ""   # mult_lb 绑定的时变曲线名 (model_shapes.csv); 空=常数不随时间变化
    mult_ub_shape: str = ""   # mult_ub 绑定的时变曲线名 (model_shapes.csv); 空=常数不随时间变化
    p_cur_pu: float = 0.0     # 当前实际挂载有功 (pu)
    q_cur_pu: float = 0.0     # 当前实际挂载无功 (pu)
    z_lb: float = 0.0         # z 下限 = mult_lb / 当前挂载比例
    z_ub: float = 1.0         # z 上限 = mult_ub / 当前挂载比例


@dataclass
class LoadShape:
    """负荷形状: 实际功率 = 元件额定 × mult[i]"""
    name: str
    npts: int = 1
    interval: float = 1.0
    mult: List[float] = field(default_factory=list)   # 每点功率乘数


@dataclass
class PVSystem:
    """光伏"""
    name: str
    bus: str
    pmpp_kw: float       # 最大功率点 (kW)
    kw: float = 0.0      # 当前输出 (kW)
    pf: float = PV_PF_UNIFIED
    irradiance: float = 1.0   # 辐照度兜底值 (仅未绑定 LoadShape 时生效; 绑定后由 mult 替代)
    shape: str = ""           # 绑定的 LoadShape 名 (daily/duty/yearly, 全天辐照度曲线)
    # 派生量
    p_max_pu: float = 0.0     # 额定 (满辐照) 有功上限 (pu)
    p_avail_pu: float = 0.0   # 当前辐照度下的可用出力上限 (pu)
    q_max_pu: float = 0.0     # |Q| 上限 (由 pf 字段计算, 见 pf_to_q_factor)


@dataclass
class Storage:
    """储能 (BESS)"""
    name: str
    bus: str
    kw: float            # 额定功率 (kW)
    kwh: float           # 能量容量 (kWh)
    state_of_charge: float = 0.5
    pct_charge: float = 100.0
    pct_discharge: float = 100.0
    charge_eff: float = 0.95
    discharge_eff: float = 0.95
    pf: float = 0.95            # 储能功率因数 (数据驱动 q_max, 替代硬编码 0.95)
    shape: str = ""           # 绑定的 LoadShape 名 (daily/duty/yearly, 可用能量上限比例曲线)
    energy_lb_ratio: Optional[float] = None  # 能量运行下限比例 (CSV 显式); None=查 STORAGE_ENERGY_LIMITS
    energy_ub_ratio: Optional[float] = None  # 能量运行上限比例 (CSV 显式); None=查 STORAGE_ENERGY_LIMITS
    # 派生量
    charge_ub_pu: float = 0.0
    discharge_ub_pu: float = 0.0
    energy_ub_pu: float = 0.0     # 额定能量上限 (pu·h, on base_mva)
    energy_lb_pu: float = 0.0
    energy_init_pu: float = 0.0
    energy_ratio: float = 1.0     # 当前时段可用能量上限比例 = LoadShape.mult[0]
    energy_ub_cur_pu: float = 0.0 # 当前时段能量上限 = 额定 × energy_ratio
    energy_lb_cur_pu: float = 0.0
    energy_init_cur_pu: float = 0.0
    p_init_cur_pu: float = 0.0        # 初始功率 (pu, 抽样记录用; OPF 中为优化变量)
    q_max_pu: float = 0.0         # 无功容量上限 (|Q|, 与 P 无耦合约束)


@dataclass
class Capacitor:
    """并联电容器组"""
    name: str
    bus: str
    kvar: float
    kv: float = 12.66
    # 派生量
    q_pu: float = 0.0    # 注入无功 (pu, 正值)


@dataclass
class Circuit:
    """系统"""
    name: str
    bus1: str             # 根节点 (slack)
    base_kv: float = 12.66
    phases: int = 3
    frequency: float = 50.0


@dataclass
class Network:
    """配电网模型"""
    circuit: Optional[Circuit] = None
    base_mva: float = 10.0
    buses: Dict[str, Bus] = field(default_factory=dict)
    lines: Dict[str, Line] = field(default_factory=dict)
    loads: Dict[str, Load] = field(default_factory=dict)
    shapes: Dict[str, LoadShape] = field(default_factory=dict)
    pvs: Dict[str, PVSystem] = field(default_factory=dict)
    storages: Dict[str, Storage] = field(default_factory=dict)
    capacitors: Dict[str, Capacitor] = field(default_factory=dict)

    # 拓扑结构 (在 finalize 中计算)
    slack_bus: str = ""
    children: Dict[str, List[str]] = field(default_factory=dict)      # children[bus] = [tbus...]
    parent: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # parent[bus] = (fbus, line_name)
    active_branches: List[Tuple[str, str, str]] = field(default_factory=list)  # (line_name, fbus, tbus)

    @property
    def z_base_ohm(self) -> float:
        """基准阻抗 (ohm)"""
        return (self.circuit.base_kv ** 2) / self.base_mva

    def finalize(self):
        """完成解析后：计算派生量，构建拓扑 (逻辑与 parse_dss.Network.finalize 一致)"""
        # 1. 添加根节点 (slack)
        if self.circuit is None:
            raise ValueError("Circuit 未定义")
        self.slack_bus = self.circuit.bus1
        if self.slack_bus not in self.buses:
            self.buses[self.slack_bus] = Bus(self.slack_bus, base_kv=self.circuit.base_kv)

        # 2. 注册所有出现过的母线
        for line in self.lines.values():
            for b in (line.fbus, line.tbus):
                if b not in self.buses:
                    self.buses[b] = Bus(b, base_kv=self.circuit.base_kv)
        for ld in self.loads.values():
            if ld.bus not in self.buses:
                self.buses[ld.bus] = Bus(ld.bus, base_kv=self.circuit.base_kv)
        for pv in self.pvs.values():
            if pv.bus not in self.buses:
                self.buses[pv.bus] = Bus(pv.bus, base_kv=self.circuit.base_kv)
        for st in self.storages.values():
            if st.bus not in self.buses:
                self.buses[st.bus] = Bus(st.bus, base_kv=self.circuit.base_kv)
        for cap in self.capacitors.values():
            if cap.bus not in self.buses:
                self.buses[cap.bus] = Bus(cap.bus, base_kv=self.circuit.base_kv)

        # 3. 计算线路阻抗 per-unit 及热极限
        z_base = self.z_base_ohm
        s_base_kva = self.base_mva * 1000.0
        total_p_load = sum(ld.kw for ld in self.loads.values()) / s_base_kva

        for line in self.lines.values():
            line.r_pu = line.r_ohm / z_base
            line.x_pu = line.x_ohm / z_base

        active_lines = [l for l in self.lines.values() if l.enabled and not l.is_switch]
        z_vals = [(l.r_pu**2 + l.x_pu**2)**0.5 for l in active_lines]
        z_avg = sum(z_vals) / len(z_vals) if z_vals else 0.01

        for line in active_lines:
            if line.normamps > 0:
                # S_max = sqrt(3) * V_nom * I_rated / S_base (三相)
                line.smax_pu = (3**0.5) * self.circuit.base_kv * line.normamps / s_base_kva
            else:
                # 默认热极限：以平均阻抗为基准，反比例分配
                z_pu = (line.r_pu**2 + line.x_pu**2)**0.5
                ratio = 2.0 * z_avg / max(z_pu, 0.001)
                line.smax_pu = min(max(ratio, 0.3), 2.5) * total_p_load

        # 4. 计算负荷 per-unit
        for ld in self.loads.values():
            ld.p_pu = ld.kw / s_base_kva
            ld.q_pu = ld.kvar / s_base_kva

        # 5. 计算 PV per-unit
        for pv in self.pvs.values():
            pv.p_max_pu = pv.pmpp_kw / s_base_kva
            # |Q|_max = P_max * tan(acos(PF))，PF 取自数据字段 pv.pf (数据驱动)
            pv.q_max_pu = pv.p_max_pu * pf_to_q_factor(pv.pf)
            # 辐照度: 绑定 shape 时 mult[0] 替代 irradiance; 否则用兜底值
            irrad = pv.irradiance
            if pv.shape:
                shape = self.shapes.get(pv.shape)
                if shape and shape.mult:
                    irrad = shape.mult[0]   # 单断面: 只取第一个点
                else:
                    print(f"  警告: 光伏 {pv.name} 引用的 LoadShape {pv.shape!r} "
                          f"未定义或无 mult, 按 irradiance={irrad:.3f} 处理")
            pv.irradiance = irrad
            pv.p_avail_pu = pv.p_max_pu * irrad

        # 6. 计算储能 per-unit
        for st in self.storages.values():
            kwrated = st.kw
            st.charge_ub_pu = (st.pct_charge / 100.0) * kwrated / s_base_kva
            st.discharge_ub_pu = (st.pct_discharge / 100.0) * kwrated / s_base_kva
            rated_energy_pu = st.kwh / (s_base_kva * 1.0)   # 额定能量上限 (pu·h)
            # 容量上下限: CSV 显式字段优先, 未显式 (None) 时查配置表 STORAGE_ENERGY_LIMITS
            lb_ratio = st.energy_lb_ratio
            ub_ratio = st.energy_ub_ratio
            if lb_ratio is None or ub_ratio is None:
                lb_ratio, ub_ratio = STORAGE_ENERGY_LIMITS.get(st.name, (0.1, 1.0))
            st.energy_ub_pu = ub_ratio * rated_energy_pu
            st.energy_lb_pu = lb_ratio * rated_energy_pu
            st.energy_init_pu = st.state_of_charge * st.energy_ub_pu
            # 当前时段可用能量上限比例: 绑定的 LoadShape.mult[0] (即能量比例本身)
            st.energy_ratio = 1.0
            if st.shape:
                shape = self.shapes.get(st.shape)
                if shape and shape.mult:
                    st.energy_ratio = shape.mult[0]   # 单断面: 只取第一个点
                else:
                    print(f"  警告: 储能 {st.name} 引用的 LoadShape {st.shape!r} "
                          f"未定义或无 mult, 按能量比例 1.0 处理")
            # 当前时段能量窗口 (额定 × 比例)
            st.energy_ub_cur_pu = st.energy_ub_pu * st.energy_ratio
            st.energy_lb_cur_pu = st.energy_lb_pu * st.energy_ratio
            st.energy_init_cur_pu = st.state_of_charge * st.energy_ub_cur_pu
            # 储能无功容量: 按 model_storage.csv 的 pf 字段数据驱动折算 (与有功 P 无耦合约束)
            q_factor = pf_to_q_factor(st.pf)
            st.q_max_pu = kwrated / s_base_kva * q_factor

        # 7. 计算电容器 per-unit (无功注入)
        for cap in self.capacitors.values():
            cap.q_pu = cap.kvar / s_base_kva

        # 8. 识别可调度负荷: CSV 显式 type 优先, 否则按命名约定 (AC_Bus* 或 EV*)
        for ld in self.loads.values():
            if ld.type is not None:
                ld.dispatchable = ld.type in ("ev", "ac")
                ld.is_ev = (ld.type == "ev")
                continue
            name_lower = ld.name.lower()
            if re.match(r"^(ac)[-_]", name_lower) or re.match(r"^ev(?:[_\d])", name_lower):
                ld.dispatchable = True
                ld.is_ev = bool(re.match(r"^ev(?:[_\d])", name_lower))

        # 8b. 计算当前实际挂载与可调范围派生量
        # 当前挂载比例 = 绑定的 LoadShape.mult[0]; 固定负荷当前 = 满载
        # 可调范围 [mult_lb, mult_ub] 来自 CSV 显式值或 LOAD_MULT_LIMITS
        # z 求解范围 = [mult_lb/当前, mult_ub/当前], z=1 表示保持当前挂载
        for ld in self.loads.values():
            base_ratio = 1.0
            if ld.dispatchable and ld.shape:
                shape = self.shapes.get(ld.shape)
                if shape and shape.mult:
                    base_ratio = shape.mult[0]   # 单断面: 只取第一个点
                else:
                    print(f"  警告: 负荷 {ld.name} 引用的 LoadShape {ld.shape!r} "
                          f"未定义或无 mult, 按当前=满载处理")
            ld.base_ratio = base_ratio
            ld.p_cur_pu = ld.p_pu * base_ratio
            ld.q_cur_pu = ld.q_pu * base_ratio
            if ld.dispatchable:
                # CSV 已显式给出 mult_lb/mult_ub (常数或绑定时变曲线) 时保留;
                # 保持默认 (0.0, 1.0) 且未绑定曲线时查配置表
                if (not ld.mult_lb_shape and not ld.mult_ub_shape
                        and ld.mult_lb == 0.0 and ld.mult_ub == 1.0):
                    limits = LOAD_MULT_LIMITS.get(ld.name)
                    if limits is not None:
                        ld.mult_lb, ld.mult_ub = limits
                if ld.base_ratio < ld.mult_lb or ld.base_ratio > ld.mult_ub:
                    print(f"  警告: 负荷 {ld.name} 当前挂载比例 {base_ratio:.3f} "
                          f"超出可调范围 [{ld.mult_lb:.3f}, {ld.mult_ub:.3f}]")
                ld.z_lb = ld.mult_lb / base_ratio if base_ratio > 0 else 0.0
                ld.z_ub = ld.mult_ub / base_ratio if base_ratio > 0 else 1e6

        # 9. 构建拓扑 (从 slack_bus 出发, 使用启用的非开关线路)
        self._build_topology()

    def _build_topology(self):
        """从 slack 节点出发构建树 (忽略 disabled 和 switch 线路)"""
        self.children = {b: [] for b in self.buses}
        self.parent = {}
        self.active_branches = []

        # 收集所有启用且非开关的线路, 建立邻接表
        adj: Dict[str, List[Tuple[str, str]]] = {b: [] for b in self.buses}
        for line in self.lines.values():
            if not line.enabled or line.is_switch:
                continue
            adj[line.fbus].append((line.tbus, line.name))
            adj[line.tbus].append((line.fbus, line.name))

        # BFS 构建树
        visited = {self.slack_bus}
        queue = [self.slack_bus]
        while queue:
            cur = queue.pop(0)
            for nbr, lname in adj[cur]:
                if nbr in visited:
                    continue
                visited.add(nbr)
                self.children[cur].append(nbr)
                self.parent[nbr] = (cur, lname)
                self.active_branches.append((lname, cur, nbr))
                queue.append(nbr)

        if len(visited) != len(self.buses):
            unreachable = set(self.buses.keys()) - visited
            print(f"  警告: {len(unreachable)} 个母线不可达: {sorted(unreachable)}")


# =====================================================================
# CSV 读取辅助
# =====================================================================

# 场景文件前缀: 新场景目录使用 scenario_*, 兼容旧目录的 model_*
FILE_PREFIXES = ("scenario_", "model_")


def _file_path(csv_dir: str, base: str) -> str:
    """返回目录中存在的场景文件路径 (scenario_ 前缀优先, 退回 model_); 都不存在返回默认路径"""
    for prefix in FILE_PREFIXES:
        p = os.path.join(csv_dir, prefix + base)
        if os.path.exists(p):
            return p
    return os.path.join(csv_dir, FILE_PREFIXES[0] + base)


REQUIRED_FILES = ["circuit.csv", "lines.csv", "loads.csv"]
OPTIONAL_FILES = ["pvs.csv", "storage.csv", "capacitors.csv", "shapes.csv"]

# load type → model_circuit.csv 接入开关 的映射 (None = 基础负荷, 始终接入)
_TYPE_SWITCH = {
    "ev": "ev_enabled",
    "ac": "flexible_enabled",
    "fixed": None,
    "fixed_extra": None,
}


def _read_rows(path: str) -> List[dict]:
    """读取 CSV, 返回 dict 行列表 (跳过全空行)"""
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if any((v or "").strip() for v in row.values())]


def _to_bool(v, default: bool = True) -> bool:
    s = (v or "").strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    return default


def _to_float(v, default: float = 0.0) -> float:
    s = (v or "").strip()
    return float(s) if s else default


def _to_float_or_shape(v, default: float):
    """mult_lb/mult_ub 单元格: 数字 → (数值, ''); 曲线名(字符串) → (None, 曲线名)"""
    s = (v or "").strip()
    if not s:
        return default, ""
    try:
        return float(s), ""
    except ValueError:
        return None, s


def _bind_mult_shape(name, net):
    """校验 mult_lb/mult_ub 引用的曲线存在; 存在返回曲线名, 否则 '' (按常数处理)"""
    name = (name or "").strip()
    if not name:
        return ""
    if name not in net.shapes:
        print(f"  警告: mult 曲线引用的 LoadShape {name!r} 未在 model_shapes.csv 中定义, 按常数处理")
        return ""
    return name


def _to_int(v, default: int = 1) -> int:
    s = (v or "").strip()
    return int(float(s)) if s else default


def _parse_mult(s) -> List[float]:
    """mult 序列: 分号/空格/逗号宽容切分"""
    text = (s or "").strip()
    if not text:
        return []
    parts = [p for p in text.replace(",", ";").replace(" ", ";").split(";") if p.strip()]
    return [float(p) for p in parts]


# =====================================================================
# 主解析函数
# =====================================================================

def parse_csv(csv_dir: str) -> Network:
    """读取 CSV 场景目录, 返回已 finalize 的 Network"""
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"CSV 场景目录不存在: {csv_dir}")
    for base in REQUIRED_FILES:
        if not os.path.exists(_file_path(csv_dir, base)):
            raise FileNotFoundError(f"缺少必填数据表: {base} (场景目录: {csv_dir})")

    print(f"加载 CSV 配电网模型: {csv_dir}")
    net = Network()

    # 1. circuit (系统参数 + 资源接入开关)
    c = _read_rows(_file_path(csv_dir, "circuit.csv"))[0]
    net.circuit = Circuit(
        name=c["name"].strip(),
        bus1=c["slack_bus"].strip(),
        base_kv=_to_float(c.get("base_kv"), 12.66),
        phases=_to_int(c.get("phases"), 3),
        frequency=_to_float(c.get("frequency"), 50.0),
    )
    net.base_mva = _to_float(c.get("base_mva"), 10.0)
    sw: Dict[str, bool] = {k: _to_bool(c.get(k), True) for k in
                           ("pv_enabled", "storage_enabled", "ev_enabled",
                            "flexible_enabled", "capacitor_enabled")}

    # 2. shapes (先读, 供组件表引用)
    # 注意: 曲线行可能用逗号把点序列展开在多列中 (表头仅 name,mult),
    # 用 DictReader 只会取到 mult 第一列, 故改用 csv.reader 合并整行数据列
    shapes_path = _file_path(csv_dir, "shapes.csv")
    if os.path.exists(shapes_path):
        with open(shapes_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)                      # 跳过表头
            for row in reader:
                cells = [c.strip() for c in row]
                if not any(cells) or not cells[0]:
                    continue                        # 空行 / 无曲线名 (遗留行) 跳过
                name = cells[0]
                mults = _parse_mult(";".join(cells[1:]))
                net.shapes[name] = LoadShape(name=name, npts=len(mults), interval=1.0, mult=mults)

    def _bind_shape(shape_name, npts, interval) -> str:
        """绑定 LoadShape 并按组件表设置 npts/interval; 返回 shape 名 (未绑定返回 '')"""
        name = (shape_name or "").strip()
        if not name:
            return ""
        shape = net.shapes.get(name)
        if shape is None:
            print(f"  警告: 引用的 LoadShape {name!r} 未在 model_shapes.csv 中定义, 按未绑定处理")
            return ""
        shape.interval = _to_float(interval, 1.0)
        n = _to_int(npts, len(shape.mult) if shape.mult else 1)
        shape.npts = n
        if shape.mult and len(shape.mult) != n:
            print(f"  警告: LoadShape {name} 的 mult 点数 ({len(shape.mult)}) 与 npts ({n}) 不一致")
        return name

    # 3. lines
    for row in _read_rows(_file_path(csv_dir, "lines.csv")):
        if not _to_bool(row.get("enabled"), True):
            continue
        net.lines[row["name"].strip()] = Line(
            name=row["name"].strip(),
            fbus=row["fbus"].strip(),
            tbus=row["tbus"].strip(),
            r_ohm=_to_float(row.get("r_ohm")),
            x_ohm=_to_float(row.get("x_ohm")),
            phases=_to_int(row.get("phases"), 3),
            enabled=True,
            is_switch=_to_bool(row.get("is_switch"), False),
            normamps=_to_float(row.get("normamps"), 0.0),
        )

    # 4. loads (type 显式标识; 应用 ev/flexible 类别开关)
    for row in _read_rows(_file_path(csv_dir, "loads.csv")):
        if not _to_bool(row.get("enabled"), True):
            continue
        typ = (row.get("type") or "").strip().lower()
        switch = _TYPE_SWITCH.get(typ)
        if switch and not sw.get(switch, True):
            continue
        name = row["name"].strip()
        # mult_lb/mult_ub: 常数(数字) 或 时变曲线名(字符串, 绑定 shapes 曲线, 单断面取 mult[0])
        lb_val, lb_raw = _to_float_or_shape(row.get("mult_lb"), 0.0)
        ub_val, ub_raw = _to_float_or_shape(row.get("mult_ub"), 1.0)
        lb_shape = _bind_mult_shape(lb_raw, net)
        ub_shape = _bind_mult_shape(ub_raw, net)
        if lb_shape:
            lb_val = net.shapes[lb_shape].mult[0] if net.shapes[lb_shape].mult else 0.0
        elif lb_val is None:
            lb_val = 0.0          # 写了曲线名但未定义 → 回退默认
        if ub_shape:
            ub_val = net.shapes[ub_shape].mult[0] if net.shapes[ub_shape].mult else 1.0
        elif ub_val is None:
            ub_val = 1.0
        net.loads[name] = Load(
            name=name,
            bus=row["bus"].strip(),
            kw=_to_float(row.get("kw")),
            kvar=_to_float(row.get("kvar")),
            kv=_to_float(row.get("kv"), 12.66),
            type=typ if typ else None,
            shape=_bind_shape(row.get("shape"), row.get("npts"), row.get("interval")),
            mult_lb=lb_val,
            mult_ub=ub_val,
            mult_lb_shape=lb_shape,
            mult_ub_shape=ub_shape,
        )

    # 4b. 一致性检查: mult_lb/mult_ub 时变曲线点数应与主挂载曲线一致
    for ld in net.loads.values():
        main_shape = net.shapes.get(ld.shape) if ld.shape else None
        if main_shape is None or not main_shape.mult:
            continue
        for tag, sname in (("mult_lb", ld.mult_lb_shape), ("mult_ub", ld.mult_ub_shape)):
            s = net.shapes.get(sname) if sname else None
            if s and s.mult and len(s.mult) != len(main_shape.mult):
                print(f"  警告: 负荷 {ld.name} 的 {tag} 曲线 {sname} 点数 ({len(s.mult)}) "
                      f"与主挂载曲线 {ld.shape} 点数 ({len(main_shape.mult)}) 不一致")

    # 5. pvs (类别开关 pv_enabled)
    if sw.get("pv_enabled", True):
        pvs_path = _file_path(csv_dir, "pvs.csv")
        if os.path.exists(pvs_path):
            for row in _read_rows(pvs_path):
                if not _to_bool(row.get("enabled"), True):
                    continue
                name = row["name"].strip()
                pmpp = _to_float(row.get("pmpp_kw"))
                net.pvs[name] = PVSystem(
                    name=name,
                    bus=row["bus"].strip(),
                    pmpp_kw=pmpp,
                    kw=_to_float(row.get("kw"), pmpp),
                    pf=_to_float(row.get("pf"), PV_PF_UNIFIED),  # 默认 0.98 (统一约定)
                    irradiance=_to_float(row.get("irradiance"), 1.0),
                    shape=_bind_shape(row.get("shape"), row.get("npts"), row.get("interval")),
                )

    # 6. storage (类别开关 storage_enabled)
    if sw.get("storage_enabled", True):
        st_path = _file_path(csv_dir, "storage.csv")
        if os.path.exists(st_path):
            for row in _read_rows(st_path):
                if not _to_bool(row.get("enabled"), True):
                    continue
                name = row["name"].strip()
                lb_raw = (row.get("energy_lb_ratio") or "").strip()
                ub_raw = (row.get("energy_ub_ratio") or "").strip()
                net.storages[name] = Storage(
                    name=name,
                    bus=row["bus"].strip(),
                    kw=_to_float(row.get("kw")),
                    kwh=_to_float(row.get("kwh")),
                    state_of_charge=_to_float(row.get("state_of_charge"), 0.5),
                    pct_charge=_to_float(row.get("pct_charge"), 100.0),
                    pct_discharge=_to_float(row.get("pct_discharge"), 100.0),
                    charge_eff=_to_float(row.get("charge_eff"), 0.95),
                    discharge_eff=_to_float(row.get("discharge_eff"), 0.95),
                    pf=_to_float(row.get("pf"), 0.95),
                    shape=_bind_shape(row.get("shape"), row.get("npts"), row.get("interval")),
                    energy_lb_ratio=_to_float(lb_raw, 0.1) if lb_raw else None,
                    energy_ub_ratio=_to_float(ub_raw, 1.0) if ub_raw else None,
                )

    # 7. capacitors (类别开关 capacitor_enabled, 默认关闭)
    if sw.get("capacitor_enabled", False):
        cap_path = _file_path(csv_dir, "capacitors.csv")
        if os.path.exists(cap_path):
            for row in _read_rows(cap_path):
                if not _to_bool(row.get("enabled"), False):
                    continue
                net.capacitors[row["name"].strip()] = Capacitor(
                    name=row["name"].strip(),
                    bus=row["bus"].strip(),
                    kvar=_to_float(row.get("kvar")),
                    kv=_to_float(row.get("kv"), 12.66),
                )

    # 8. 计算派生量 + 拓扑
    net.finalize()

    # 摘要
    print(f"\nCSV 模型数据结构:")
    print(f"   - 母线数: {len(net.buses)}")
    print(f"   - 线路数: {len(net.lines)} (启用: {sum(1 for l in net.lines.values() if l.enabled and not l.is_switch)})")
    print(f"   - 负荷数: {len(net.loads)} (可调度: {sum(1 for l in net.loads.values() if l.dispatchable)})")
    print(f"   - 光伏数: {len(net.pvs)}")
    print(f"   - 储能数: {len(net.storages)}")
    print(f"   - 电容器: {len(net.capacitors)}")
    print(f"   - Slack 母线: {net.slack_bus}")
    print(f"   - 基准: S_base={net.base_mva} MVA, V_base={net.circuit.base_kv} kV, Z_base={net.z_base_ohm:.4f} Ω")
    return net


# =====================================================================

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/csv_case33/model_storage_bus18"
    net = parse_csv(path)
    print("\n储能参数:")
    for s in net.storages.values():
        print(f"  {s.name}: bus={s.bus}, charge_ub={s.charge_ub_pu:.4f} pu, "
              f"discharge_ub={s.discharge_ub_pu:.4f} pu, energy_ub={s.energy_ub_pu:.4f} pu·h")
    print("\n光伏参数:")
    for p in net.pvs.values():
        print(f"  {p.name}: bus={p.bus}, pf={p.pf}, p_max={p.p_max_pu:.4f} pu, "
              f"q_max={p.q_max_pu:.4f} pu")

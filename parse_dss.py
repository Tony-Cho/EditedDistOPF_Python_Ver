# -*- coding: utf-8 -*-
"""
parse_dss.py
OpenDSS 文件解析器：解析 IEEE 33 配电网 + VPP 资源 (PV/Storage/EV/柔性负荷/电容)

将 OpenDSS 文本模型解析为 Python 数据结构 (Network)，
供后续 LinDistFlow OPF 求解使用。

约定：
- 系统为三相平衡，采用单相等值 (3-phase, per-unit on 10 MVA base)
- 阻抗单位：欧姆 → 在 Network 中转为 per-unit
- 功率单位：kW/kvar → 在 Network 中转为 per-unit (on 10 MVA)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =====================================================================
# 可调度负荷 mult 可调范围配置 (Python 侧数据化)
# =====================================================================
# mult = 当前挂载比例 (相对满载, 由 LoadShape.mult[0] 给出)
# 可调范围 [mult_lb, mult_ub] 决定 z 的求解范围: z ∈ [mult_lb/当前, mult_ub/当前]
# mult_ub 为根据过去历史数据得到的历史均值 (实际调度很少拉到满载, 默认取 0.8)
# 未在表中显式配置的负荷, 默认 mult ∈ [0.0, 1.0]
LOAD_MULT_LIMITS: Dict[str, Tuple[float, float]] = {
    "EV_Bus19": (0.3, 0.8),    # EV 最小充电需求 = 满载 30% (含原 EV_Bus20, 已合并); 上限 = 历史均值
    "EV_Bus7":  (0.1, 0.8),    # 当前挂载 0.25, 允许削减至满载 10%; 上限 = 历史均值
    "AC_Bus2":  (0.1, 0.8),    # 空调最低保持满载 10%; 上限 = 历史均值
}

# 储能容量上下限配置 (参考 LOAD_MULT_LIMITS 写法, 按储能名索引, 可供外部调用):
# 值 = (能量下限比例, 能量上限比例), 均相对额定能量容量 (kWh)
# 未在表中显式配置的储能, 默认能量比例 ∈ [0.1, 1.0] (下限 10% / 上限 100%)
STORAGE_ENERGY_LIMITS: Dict[str, Tuple[float, float]] = {
    "BESS_Bus18": (0.1, 0.9),   # 容量下限 = 10% 额定, 上限 = 90% 额定 (SOC 运行窗口)
}

# 光伏功率因数统一约定 (工程约定 0.98):
# - CSV 路径: pvs.csv 的 pf 列由 dss2csv 按此值写入
# - DSS 路径: 忽略 DSS 中的 PF 字段 (其值 1.0 仅表示未配置无功能力), 统一按此值
PV_PF_UNIFIED = 0.98


def pf_to_q_factor(pf: float) -> float:
    """功率因数 → 无功/有功比 |Q|/P = tan(acos(PF))"""
    return (1.0 - pf ** 2) ** 0.5 / pf


# =====================================================================
# 数据类定义
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
    mult_lb: float = 0.0      # mult 可调下限 (相对满载)
    mult_ub: float = 1.0      # mult 可调上限 (相对满载, 默认=满载)
    p_cur_pu: float = 0.0     # 当前实际挂载有功 (pu)
    q_cur_pu: float = 0.0     # 当前实际挂载无功 (pu)
    z_lb: float = 0.0         # z 下限 = mult_lb / 当前挂载比例
    z_ub: float = 1.0         # z 上限 = mult_ub / 当前挂载比例


@dataclass
class LoadShape:
    """负荷形状 (OpenDSS LoadShape): 实际功率 = Load.kw × mult[i]"""
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
    pf: float = 1.0
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
    kw: float            # 额定功率 (kW/相) - 在 OpenDSS 中是每相? 实际是总功率按相分配
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
    energy_ub_pu: float = 0.0     # 额定能量上限 (pu·h, on 10 MVA base)
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
    """系统 (来自 Circuit 命令)"""
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
    # children[bus] = [tbus for active lines from bus]
    children: Dict[str, List[str]] = field(default_factory=dict)
    # parent[bus] = (fbus, line_name) for each non-slack bus
    parent: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    # active_branches: list of (line_name, fbus, tbus)
    active_branches: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def z_base_ohm(self) -> float:
        """基准阻抗 (ohm)"""
        return (self.circuit.base_kv ** 2) / self.base_mva

    def finalize(self):
        """完成解析后：计算派生量，构建拓扑"""
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
        s_base_kva = self.base_mva * 1000.0  # 10000 kVA
        total_p_load = sum(ld.kw for ld in self.loads.values()) / s_base_kva

        # 先算完所有线路的 pu 阻抗，再算平均阻抗
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
                # 基准限额 = total_p_load * 2.0 (所有线路可承受约 2 倍总负荷)
                # 阻抗越小(靠近根节点)分配越大
                z_pu = (line.r_pu**2 + line.x_pu**2)**0.5
                # smax 范围: 0.3*total_p_load ~ 2.5*total_p_load
                ratio = 2.0 * z_avg / max(z_pu, 0.001)
                line.smax_pu = min(max(ratio, 0.3), 2.5) * total_p_load

        # 4. 计算负荷 per-unit
        for ld in self.loads.values():
            ld.p_pu = ld.kw / s_base_kva
            ld.q_pu = ld.kvar / s_base_kva

        # 5. 计算 PV per-unit
        for pv in self.pvs.values():
            pv.p_max_pu = pv.pmpp_kw / s_base_kva
            # |Q|_max = P_max * tan(acos(PF))，PF 取自数据字段 pv.pf
            # (与 opf_model.py 约束口径一致, 数据驱动)
            pv.q_max_pu = pv.p_max_pu * pf_to_q_factor(pv.pf)
            # 光伏辐照度: 绑定了 LoadShape 时, mult[0] 直接作为辐照度 (替代 irradiance 字段);
            # 未绑定 shape 时, 用 irradiance 字段兜底 (默认 1.0 = 满辐照)
            irrad = pv.irradiance
            if pv.shape:
                shape = self.shapes.get(pv.shape)
                if shape and shape.mult:
                    irrad = shape.mult[0]   # 单断面: 只取第一个点, mult 即辐照度
                else:
                    print(f"  警告: 光伏 {pv.name} 引用的 LoadShape {pv.shape!r} "
                          f"未定义或无 mult, 按 irradiance={irrad:.3f} 处理")
            pv.irradiance = irrad
            pv.p_avail_pu = pv.p_max_pu * irrad

        # 6. 计算储能 per-unit
        for st in self.storages.values():
            # OpenDSS 中 kW 为额定功率 (总功率? 还是每相?) — 这里按 PMD 解析: ps 每相
            # 原 Julia 代码: kw_per_phase = abs(first(ps)); charge_ub = kw_per_phase (默认)
            # OpenDSS Storage.dss: kW=100 表示额定功率, PMD 解析为 ps=[kW/3, kW/3, kW/3]
            # 所以单相等值下, 总额定功率 = kW (DSS 中给定值)
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
            # 储能无功容量: 按 DSS storage 的 pf 字段数据驱动折算 (与有功 P 无耦合约束)
            # (光伏: |q| ≤ p*tan(acos(PF)), 储能: |q| ≤ q_max 独立箱式约束)
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
        # 当前挂载比例 = 绑定的 LoadShape.mult[0] (官方字段); 固定负荷当前 = 满载
        # 可调范围 [mult_lb, mult_ub] 来自 LOAD_MULT_LIMITS (默认 EV:[0.3,1], 其他:[0,1])
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
                # CSV 已显式给出 mult_lb/mult_ub 时保留; 保持默认 (0.0, 1.0) 时查配置表
                if ld.mult_lb == 0.0 and ld.mult_ub == 1.0:
                    limits = LOAD_MULT_LIMITS.get(ld.name)
                    if limits is not None:
                        ld.mult_lb, ld.mult_ub = limits
                # 未显式配置的可调度负荷: 保持默认 mult ∈ [0.0, 1.0]
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
# OpenDSS 文本解析
# =====================================================================

def _strip_comments(line: str) -> str:
    """移除 ! 注释"""
    idx = line.find("!")
    if idx >= 0:
        line = line[:idx]
    return line.strip()


def _parse_kv_pairs(text: str) -> Dict[str, str]:
    """解析 'key1=val1 key2=val2 %key=val' 形式的键值对"""
    result: Dict[str, str] = {}
    # 匹配 key=val, key 可能以 % 开头
    for m in re.finditer(r"([\w%]+)\s*=\s*([\w.+-]+)", text):
        key = m.group(1).lower()
        val = m.group(2)
        result[key] = val
    return result


def _parse_element_line(line: str) -> Optional[Tuple[str, str, str, Dict[str, str]]]:
    """
    解析 'New Element.Type.Name key=val ...' 行
    返回 (element_type, element_name, full_command_text_for_kv) 或 None
    """
    # 匹配 New Type.Name (Type 例如 Line/Load/PVSystem/Storage/Capacitor/Circuit)
    m = re.match(r"^\s*New\s+(\w+)\.([\w.-]+)\s*(.*)$", line, re.IGNORECASE)
    if not m:
        return None
    etype = m.group(1).lower()
    ename = m.group(2)
    rest = m.group(3)
    return etype, ename, rest


def _read_dss_file(path: str) -> List[str]:
    """读取 DSS 文件, 合并续行 (~), 移除注释, 返回逻辑行列表"""
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    logical_lines: List[str] = []
    current = ""
    for raw in raw_lines:
        line = _strip_comments(raw)
        if not line:
            continue
        if line.startswith("~"):
            # 续行
            current += " " + line[1:].strip()
        else:
            if current:
                logical_lines.append(current)
            current = line
    if current:
        logical_lines.append(current)
    return logical_lines


def _process_dss_file(
    path: str,
    network: Network,
    visited: Optional[set] = None,
    depth: int = 0,
):
    """递归处理 DSS 文件 (含 Redirect)"""
    if visited is None:
        visited = set()
    abs_path = os.path.abspath(path)
    if abs_path in visited:
        return
    visited.add(abs_path)

    if not os.path.exists(abs_path):
        print(f"  警告: 文件不存在: {path}")
        return

    if depth == 0:
        print(f"  解析: {path}")
    else:
        print(f"    Redirect → {os.path.basename(path)}")

    dss_dir = os.path.dirname(abs_path)
    logical_lines = _read_dss_file(abs_path)

    for line in logical_lines:
        low = line.lower()
        # 处理 Redirect
        if low.startswith("redirect "):
            target = line[len("redirect "):].strip()
            target_path = os.path.join(dss_dir, target) if not os.path.isabs(target) else target
            _process_dss_file(target_path, network, visited, depth + 1)
            continue

        # 跳过非 New 命令 (Clear, Set, Solve, Show, CalcVoltageBases)
        if not low.startswith("new "):
            continue

        parsed = _parse_element_line(line)
        if parsed is None:
            continue
        etype, ename, rest = parsed
        kv = _parse_kv_pairs(rest)

        if etype == "circuit":
            # New Circuit.case33bw_vpp bus1=bus1.1.2.3 basekv=12.66 ...
            bus1_full = kv.get("bus1", "bus1.1.2.3")
            bus1_name = bus1_full.split(".")[0]
            network.circuit = Circuit(
                name=ename,
                bus1=bus1_name,
                base_kv=float(kv.get("basekv", 12.66)),
                phases=int(kv.get("phases", 3)),
                frequency=float(kv.get("frequency", 50.0)),
            )
        elif etype == "line":
            bus1_full = kv.get("bus1", "")
            bus2_full = kv.get("bus2", "")
            fbus = bus1_full.split(".")[0]
            tbus = bus2_full.split(".")[0]
            is_switch = kv.get("switch", "no").lower() in ("yes", "true", "1")
            enabled = kv.get("enabled", "yes").lower() in ("yes", "true", "1")
            normamps = float(kv.get("normamps", 0.0))
            network.lines[ename] = Line(
                name=ename,
                fbus=fbus,
                tbus=tbus,
                r_ohm=float(kv.get("r1", 0.0)),
                x_ohm=float(kv.get("x1", 0.0)),
                phases=int(kv.get("phases", 3)),
                enabled=enabled,
                is_switch=is_switch,
                normamps=normamps,
            )
        elif etype == "load":
            bus1_full = kv.get("bus1", "")
            bus = bus1_full.split(".")[0]
            # 绑定的 LoadShape (官方字段 daily/duty/yearly, 提供当前挂载比例)
            shape_name = kv.get("daily", kv.get("duty", kv.get("yearly", "")))
            network.loads[ename] = Load(
                name=ename,
                bus=bus,
                kw=float(kv.get("kw", 0.0)),
                kvar=float(kv.get("kvar", 0.0)),
                kv=float(kv.get("kv", 12.66)),
                shape=shape_name,
            )
        elif etype == "loadshape":
            # mult 为括号数组 mult=(v1, v2, ...), 需专门提取 (现有 kv 正则不认括号)
            mults: List[float] = []
            m_mult = re.search(r"mult\s*=\s*\(([^)]*)\)", rest, re.IGNORECASE)
            if m_mult:
                mults = [float(x) for x in m_mult.group(1).replace(",", " ").split()]
            network.shapes[ename] = LoadShape(
                name=ename,
                npts=int(kv.get("npts", 1)),
                interval=float(kv.get("interval", 1.0)),
                mult=mults,
            )
        elif etype == "pvsystem":
            bus1_full = kv.get("bus1", "")
            bus = bus1_full.split(".")[0]
            # 绑定的 LoadShape (官方 daily/duty/yearly, 提供全天辐照度曲线)
            shape_name = kv.get("daily", kv.get("duty", kv.get("yearly", "")))
            network.pvs[ename] = PVSystem(
                name=ename,
                bus=bus,
                pmpp_kw=float(kv.get("pmpp", kv.get("kw", 0.0))),
                kw=float(kv.get("kw", 0.0)),
                pf=PV_PF_UNIFIED,   # 光伏功率因数统一 0.98 (不随 DSS 的 PF 字段, 见 PV_PF_UNIFIED)
                irradiance=float(kv.get("irradiance", 1.0)),
                shape=shape_name,
            )
        elif etype == "storage":
            bus1_full = kv.get("bus1", "")
            bus = bus1_full.split(".")[0]
            # 绑定的 LoadShape (官方 daily/duty/yearly, 可用能量上限比例曲线)
            shape_name = kv.get("daily", kv.get("duty", kv.get("yearly", "")))
            network.storages[ename] = Storage(
                name=ename,
                bus=bus,
                kw=float(kv.get("kw", 0.0)),
                kwh=float(kv.get("kwh", 0.0)),
                state_of_charge=float(kv.get("stateofcharge", 0.5)),
                pct_charge=float(kv.get("%charge", 100.0)),
                pct_discharge=float(kv.get("%discharge", 100.0)),
                charge_eff=float(kv.get("chargeeff", 0.95)),
                discharge_eff=float(kv.get("dischargeeff", 0.95)),
                pf=float(kv.get("pf", 0.95)),
                shape=shape_name,
            )
        elif etype == "capacitor":
            bus1_full = kv.get("bus1", "")
            bus = bus1_full.split(".")[0]
            network.capacitors[ename] = Capacitor(
                name=ename,
                bus=bus,
                kvar=float(kv.get("kvar", 0.0)),
                kv=float(kv.get("kv", 12.66)),
            )


# =====================================================================
# 主入口
# =====================================================================

def parse_dss(dss_path: str) -> Network:
    """解析 OpenDSS Master.dss (或任意 .dss 入口), 返回 Network"""
    print(f"加载 OpenDSS 模型: {dss_path}")
    network = Network()
    _process_dss_file(dss_path, network)
    network.finalize()

    # 打印摘要
    print(f"\n原始模型数据结构:")
    print(f"   - 母线数: {len(network.buses)}")
    print(f"   - 线路数: {len(network.lines)} (启用: {sum(1 for l in network.lines.values() if l.enabled and not l.is_switch)})")
    print(f"   - 负荷数: {len(network.loads)} (可调度: {sum(1 for l in network.loads.values() if l.dispatchable)})")
    print(f"   - 光伏数: {len(network.pvs)}")
    print(f"   - 储能数: {len(network.storages)}")
    print(f"   - 电容器: {len(network.capacitors)}")
    print(f"   - Slack 母线: {network.slack_bus}")
    print(f"   - 基准: S_base={network.base_mva} MVA, V_base={network.circuit.base_kv} kV, Z_base={network.z_base_ohm:.4f} Ω")
    return network


if __name__ == "__main__":
    net = parse_dss("data/opendss_case33/Master.dss")
    print("\n储能参数:")
    for s in net.storages.values():
        print(f"  {s.name}: bus={s.bus}, charge_ub={s.charge_ub_pu:.4f} pu, "
              f"discharge_ub={s.discharge_ub_pu:.4f} pu, energy_ub={s.energy_ub_pu:.4f} pu·h")
    print("\n光伏参数:")
    for p in net.pvs.values():
        print(f"  {p.name}: bus={p.bus}, p_max={p.p_max_pu:.4f} pu, q_max={p.q_max_pu:.4f} pu")

# -*- coding: utf-8 -*-
"""
opf_model.py
基于 SOCP 松弛的 VPP OPF 模型 (Gurobi 求解)

模型说明
--------
原 Julia 代码使用 Ipopt 求解非线性 AC OPF (ACPUPowerModel)。
Gurobi 只能求解 LP/QP/MIP/SOCP，因此采用 SOCP 松弛 (DistFlow + 线损) —
保留线损项 (r²+x²)·l 在电压降中，并用 SOCP 松弛 l ≥ (P²+Q²)/U，
同时将线损 r·l、x·l 计入功率平衡。对 IEEE 33 网络精度与全 AC OPF 相当。

系统为三相平衡，采用单相等值 (3-phase per-unit on 10 MVA base)。

变量 (均为 per-unit)
- U[i]        : 母线电压幅值的平方 (V²)，slack=1.0
- P[i,j],Q[i,j]: 支路有功/无功 (i→j 为正方向)
- l[i,j]      : 支路电流幅值的平方 (I²)
- p_sub, q_sub: 根节点 (变电站) 注入功率
- p_pv, q_pv  : 光伏有功/无功
- p_ch, p_dis : 储能充电/放电功率 (均非负)
- q_st        : 储能无功 (独立箱式约束, 与 P 无耦合)
- z_demand[l] : 负荷调节因子 (乘当前实际挂载; 固定=1,
               可调度 ∈ [mult_lb/当前mult, mult_ub/当前mult], z=1 保持现状)

约束
- SOCP 电压降: U_j = U_i - 2(rP+xQ) + (r²+x²)·l
- SOCP 松弛:  l·U_i ≥ P²+Q²
- 节点功率平衡 (含线损): 入流 - 出流 = 净消费 + Σ(r·l)
- 电压上下限: 0.95 ≤ V ≤ 1.10 pu  → 0.64 ≤ U ≤ 1.3225
- 支路热极限: P_ij² + Q_ij² ≤ S_max²
- PV: 0 ≤ p_pv ≤ P_avail (= Pmpp × 当前辐照度); |q_pv| ≤ p_pv * tan(acos(PF))  (Q 与 P 耦合, PF 取自数据字段 pv.pf)
- 储能: 0 ≤ p_ch ≤ charge_ub; 0 ≤ p_dis ≤ discharge_ub (放松互补);
       |q_st| ≤ q_max (独立箱式, 与 P 无耦合, 区别于光伏);
       能量窗口 se ∈ [0.1, 1.0] × 额定kWh × 日曲线能量比例 (mult[0]);
       15 分钟约束: p_dis × 0.25 ≤ se - lb (放电 15min 后仍 ≥ 下限);
                  se + p_ch × 0.25 ≤ energy_ub_cur (充电 15min 不溢出)
- 负荷: z_demand 上下限

目标: min/max p_sub (最小化/最大化根节点注入)
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import gurobipy as gp
from gurobipy import GRB

from parse_dss import Network, pf_to_q_factor


# 电压上下限
V_MIN = 0.95
V_MAX = 1.07
U_MIN = V_MIN * V_MIN   # 0.9025
U_MAX = V_MAX * V_MAX   # 1.1449


# =====================================================================
# 模型构建与求解
# =====================================================================

def build_and_solve_opf(network: Network, sense: str, verbose: bool = True) -> Tuple[dict, dict]:
    """
    构建 VPP OPF 模型并求解。

    Parameters
    ----------
    network : Network
        已解析并 finalize 的配电网模型
    sense : str
        "min" 或 "max" — 最小化/最大化根节点注入功率
    verbose : bool
        是否打印求解过程

    Returns
    -------
    result : dict
        结果字典 (兼容 Julia 输出格式)
    var_values : dict
        原始变量值 (供打印使用)
    """
    if sense.lower() not in ("min", "max"):
        raise ValueError("sense 必须为 'min' 或 'max'")

    if verbose:
        print("\n转换数据模型...")
        print("构建优化模型 (LinDistFlow VPP OPF)...")

    # 建立母线索引 (按 bus 名称)
    bus_names = sorted(network.buses.keys())

    # 建立每个母线下挂的负荷列表、PV、储能、电容
    loads_at: Dict[str, list] = {b: [] for b in bus_names}
    for ld in network.loads.values():
        loads_at[ld.bus].append(ld)

    pvs_at: Dict[str, list] = {b: [] for b in bus_names}
    for pv in network.pvs.values():
        pvs_at[pv.bus].append(pv)

    storages_at: Dict[str, list] = {b: [] for b in bus_names}
    for st in network.storages.values():
        storages_at[st.bus].append(st)

    caps_at: Dict[str, list] = {b: [] for b in bus_names}
    for cap in network.capacitors.values():
        caps_at[cap.bus].append(cap)

    # ---------- 创建模型 ----------
    model = gp.Model("VPP_OPF")
    model.Params.TimeLimit = 60
    if not verbose:
        model.Params.OutputFlag = 0
    # 其余参数全部使用 Gurobi 默认设置

    # ---------- 变量 ----------
    # U[i] = V_i^2 (pu)
    U = {}
    for b in bus_names:
        U[b] = model.addVar(lb=U_MIN, ub=U_MAX, name=f"U_{b}")

    # 支路功率 P, Q
    P = {}   # P[(fbus, tbus, line_name)]
    Q = {}
    for lname, fbus, tbus in network.active_branches:
        line = network.lines[lname]
        smax = line.smax_pu if line.smax_pu > 0 else GRB.INFINITY
        P[(fbus, tbus, lname)] = model.addVar(lb=-smax, ub=smax, name=f"P_{lname}")
        Q[(fbus, tbus, lname)] = model.addVar(lb=-smax, ub=smax, name=f"Q_{lname}")

    # 变电站注入
    p_sub = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="p_sub")
    q_sub = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="q_sub")

    # PV
    p_pv = {}   # p_pv[pv_name]
    q_pv = {}
    for pv in network.pvs.values():
        # 有功上限 = 当前辐照度下的可用出力 (MPPT 决定 P_avail, 0~P_avail 连续可调)
        p_pv[pv.name] = model.addVar(lb=0.0, ub=pv.p_avail_pu, name=f"p_pv_{pv.name}")
        q_pv[pv.name] = model.addVar(lb=-pv.q_max_pu, ub=pv.q_max_pu, name=f"q_pv_{pv.name}")
        # 光伏发无功需有有功输出: |q| ≤ p * tan(acos(PF))，PF 取自数据字段 pv.pf
        # 当 p=0 时 q 只能为 0，不允许纯发无功
        pv_qfac = pf_to_q_factor(pv.pf)
        model.addConstr(q_pv[pv.name] <= p_pv[pv.name] * pv_qfac, name=f"pvq_ub_{pv.name}")
        model.addConstr(q_pv[pv.name] >= -p_pv[pv.name] * pv_qfac, name=f"pvq_lb_{pv.name}")

    # 储能
    p_ch = {}
    p_dis = {}
    q_st = {}
    se = {}   # 储能能量状态 (pu·h)
    for st in network.storages.values():
        p_ch[st.name] = model.addVar(lb=0.0, ub=st.charge_ub_pu, name=f"p_ch_{st.name}")
        p_dis[st.name] = model.addVar(lb=0.0, ub=st.discharge_ub_pu, name=f"p_dis_{st.name}")
        # 储能无功: 独立箱式约束 |q| ≤ q_max, 与有功 P 无耦合
        # (不同于光伏的 |q| ≤ p*tan(acos(PF)), 允许待机/纯无功运行)
        q_st[st.name] = model.addVar(lb=-st.q_max_pu, ub=st.q_max_pu, name=f"q_st_{st.name}")
        # 能量状态上下限 = 当前时段能量窗口 (额定 × 日曲线能量比例)
        se[st.name] = model.addVar(lb=st.energy_lb_cur_pu, ub=st.energy_ub_cur_pu, name=f"se_{st.name}")

    # 负荷调节因子 z_demand (乘当前实际挂载)
    z_demand = {}
    for ld in network.loads.values():
        lb = 0.0
        ub = ld.z_ub
        if not ld.dispatchable:
            lb = ub = 1.0      # 固定负荷: z=1
        else:
            # 可调度负荷: z ∈ [mult_lb/当前, mult_ub/当前] (parse_dss 已算好)
            lb = ld.z_lb
        z_demand[ld.name] = model.addVar(lb=lb, ub=ub, name=f"z_{ld.name}")

    model.update()

    # ---------- 变量 (续) ----------
    # 5. 支路电流平方 l_ij (用于 SOCP 线损)
    l_var = {}
    for lname, fbus, tbus in network.active_branches:
        l_var[lname] = model.addVar(lb=0.0, ub=GRB.INFINITY, name=f"l_{lname}")

    model.update()

    # ---------- 约束 ----------
    # 1. Slack 电压固定 = 1.05
    model.addConstr(U[network.slack_bus] == 1.05 ** 2, name="slack_voltage")

    # 2. SOCP 电压降 (含线损): U_j = U_i - 2(rP+xQ) + (r²+x²)·l
    for lname, fbus, tbus in network.active_branches:
        line = network.lines[lname]
        z2 = line.r_pu**2 + line.x_pu**2
        model.addConstr(
            U[tbus] == U[fbus]
            - 2.0 * (line.r_pu * P[(fbus, tbus, lname)]
                     + line.x_pu * Q[(fbus, tbus, lname)])
            + z2 * l_var[lname],
            name=f"vdrop_{lname}"
        )

    # 2b. SOCP 松弛: l ≥ (P²+Q²)/U_fbus  ⇔  l·U_fbus ≥ P²+Q²
    for lname, fbus, tbus in network.active_branches:
        model.addConstr(
            l_var[lname] * U[fbus]
            >= P[(fbus, tbus, lname)]**2 + Q[(fbus, tbus, lname)]**2,
            name=f"socp_{lname}"
        )

    # 2c. SOCP 松弛: smax^2 ≥ P²+Q²
    for lname, fbus, tbus in network.active_branches:
        line = network.lines[lname]
        smax = line.smax_pu if line.smax_pu > 0 else GRB.INFINITY
        model.addConstr(
            smax ** 2
            >= l_var[lname] * U[fbus],
            name=f"socp_{lname}"
        )

    # 3. 节点功率平衡
    # 收集每个母线的下游支路 (出流)
    out_branches: Dict[str, list] = {b: [] for b in bus_names}
    in_branches: Dict[str, list] = {b: [] for b in bus_names}
    for lname, fbus, tbus in network.active_branches:
        out_branches[fbus].append((fbus, tbus, lname))
        in_branches[tbus].append((fbus, tbus, lname))

    for b in bus_names:
        # 该母线的净注入 (消费为正)
        # 有功消费 = sum(load_p * z) + sum(storage charge) - sum(storage discharge) - sum(pv)
        # 无功消费 = sum(load_q * z) - sum(cap q) - sum(pv q)
        p_load_total = gp.LinExpr()
        q_load_total = gp.LinExpr()
        for ld in loads_at[b]:
            # 消费 = z × 当前实际挂载 (可调度负荷可削减/增荷)
            p_load_total += ld.p_cur_pu * z_demand[ld.name]
            q_load_total += ld.q_cur_pu * z_demand[ld.name]
        p_gen_total = gp.LinExpr()
        q_gen_total = gp.LinExpr()
        for pv in pvs_at[b]:
            p_gen_total += p_pv[pv.name]
            q_gen_total += q_pv[pv.name]
        p_st_total = gp.LinExpr()
        for st in storages_at[b]:
            p_st_total += p_ch[st.name] - p_dis[st.name]
        q_st_total = gp.LinExpr()
        for st in storages_at[b]:
            q_st_total += q_st[st.name]   # 储能无功注入 (正=发出)
        q_cap_total = gp.LinExpr()
        for cap in caps_at[b]:
            q_cap_total += cap.q_pu   # 电容注入无功 (正值)

        # 线损项: 以母线 b 为末端的线路损耗
        p_loss = gp.quicksum(network.lines[l].r_pu * l_var[l]
                             for (f, t, l) in in_branches[b])
        q_loss = gp.quicksum(network.lines[l].x_pu * l_var[l]
                             for (f, t, l) in in_branches[b])

        # 净消费 = 负荷 - 光伏 - 储能无功 + 储能净充电 - 电容 + 线损
        p_net = p_load_total - p_gen_total + p_st_total + p_loss
        q_net = q_load_total - q_gen_total - q_st_total - q_cap_total + q_loss

        # 入流 - 出流 = 净消费
        p_in = gp.quicksum(P[(f, t, l)] for (f, t, l) in in_branches[b])
        q_in = gp.quicksum(Q[(f, t, l)] for (f, t, l) in in_branches[b])
        p_out = gp.quicksum(P[(f, t, l)] for (f, t, l) in out_branches[b])
        q_out = gp.quicksum(Q[(f, t, l)] for (f, t, l) in out_branches[b])

        if b == network.slack_bus:
            # 变电站注入 - 出流 = 净消费
            model.addConstr(p_sub - p_out == p_net, name=f"p_bal_{b}")
            model.addConstr(q_sub - q_out == q_net, name=f"q_bal_{b}")
        else:
            model.addConstr(p_in - p_out == p_net, name=f"p_bal_{b}")
            model.addConstr(q_in - q_out == q_net, name=f"q_bal_{b}")

    # 4. 储能能量状态约束 (对应 Julia constraint_storage_state)
    # se_final = energy_init + (p_ch * charge_eff - p_dis / discharge_eff) * dt  (dt=1h)
    # energy_lb_cur ≤ se_final ≤ energy_ub_cur (已通过变量上下限实现)
    DT = 1.0   # 时间步长 (1小时)
    for st in network.storages.values():
        model.addConstr(
            se[st.name] == st.energy_init_cur_pu
            + DT * (p_ch[st.name] * st.charge_eff - p_dis[st.name] / st.discharge_eff),
            name=f"se_state_{st.name}"
        )
        # 5. 15 分钟支撑约束: 放电 15 分钟后剩余能量仍 ≥ 容量下限
        # se - p_dis × 0.25 ≥ energy_lb_cur, 即 p_dis × 0.25 ≤ se - energy_lb_cur
        model.addConstr(se[st.name] - p_dis[st.name] * 0.25 >= st.energy_lb_cur_pu,
                        name=f"se_15min_dis_{st.name}")
        # 6. 15 分钟充电容量约束: 当前能量 + 15 分钟充电量 ≤ 储能容量上限
        # se + p_ch × 0.25 ≤ energy_ub_cur, 防止充电溢出
        model.addConstr(se[st.name] + p_ch[st.name] * 0.25 <= st.energy_ub_cur_pu,
                        name=f"se_15min_ch_{st.name}")
        # 7. 储能 PCS 视在功率运行范围: (p_dis - p_ch)² + q² ≤ S_pcs²
        # S_pcs = 额定功率 (kw), 防止有功无功同时满发超出 PCS 容量
        s_pcs_pu = st.kw / (network.base_mva * 1000.0)
        model.addConstr(
            (p_dis[st.name] - p_ch[st.name]) ** 2 + q_st[st.name] ** 2 <= s_pcs_pu ** 2,
            name=f"pcs_cap_{st.name}")

    if verbose:
        thermal_count = sum(1 for _, f, t in network.active_branches if network.lines[_].smax_pu > 0)
        print(f"   支路热极限: {thermal_count} 条 (SOCP 链式: P²+Q² ≤ S_max²)")
        print(f"   模型类型: SOCP (含线损)")

    # ---------- 目标函数 ----------
    if sense.lower() == "min":
        model.setObjective(p_sub, GRB.MINIMIZE)
        if verbose:
            print("   目标函数: 最小化根节点注入功率 (Min P_bus1)")
    else:
        model.setObjective(p_sub, GRB.MAXIMIZE)
        if verbose:
            print("   目标函数: 最大化根节点注入功率 (Max P_bus1)")

    # ---------- 求解 ----------
    if verbose:
        print(f"   发电机: 1 (slack bus)")
        print(f"   电压约束: {V_MIN} ≤ V ≤ {V_MAX} pu")
        print("   求解中...")

    t0 = time.time()
    model.optimize()
    solve_time = time.time() - t0

    # ---------- 提取结果 ----------
    if model.Status == GRB.OPTIMAL:
        status_str = "OPTIMAL"
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        status_str = "TIME_LIMIT_FEASIBLE"
    elif model.Status == GRB.TIME_LIMIT:
        status_str = "TIME_LIMIT"
    elif model.Status == GRB.INFEASIBLE:
        status_str = "INFEASIBLE"
        if verbose:
            print("   模型不可行，正在计算 IIS (冲突约束集)...")
            model.computeIIS()
            for c in model.getConstrs():
                if c.IISConstr == 1:
                    print(f"     冲突约束: {c.VarName}")
    elif model.Status == GRB.UNBOUNDED:
        status_str = "UNBOUNDED"
    else:
        status_str = f"STATUS_{model.Status}"

    obj_val = model.ObjVal if model.SolCount > 0 else float("nan")

    # 收集变量值
    if model.SolCount > 0:
        var_values = {
            "p_sub": p_sub.X,
            "q_sub": q_sub.X,
            "U": {b: U[b].X for b in bus_names},
            "V": {b: (U[b].X ** 0.5) for b in bus_names},
            "branch": {
                lname: {
                    "P": P[(f, t, lname)].X,
                    "Q": Q[(f, t, lname)].X,
                } for lname, f, t in network.active_branches
            },
            "p_pv": {name: p_pv[name].X for name in network.pvs},
            "q_pv": {name: q_pv[name].X for name in network.pvs},
            "p_ch": {name: p_ch[name].X for name in network.storages},
            "p_dis": {name: p_dis[name].X for name in network.storages},
            "q_st": {name: q_st[name].X for name in network.storages},
            "se": {name: se[name].X for name in network.storages},
            "z_demand": {ld.name: z_demand[ld.name].X for ld in network.loads.values()},
            "l_var": {lname: l_var[lname].X for lname, _, _ in network.active_branches},
        }
        # 储能净功率 (正=充电, 负=放电)
        var_values["p_storage"] = {
            name: var_values["p_ch"][name] - var_values["p_dis"][name]
            for name in network.storages
        }
        # 网损
        p_loss_pu = sum(network.lines[l].r_pu * var_values["l_var"][l]
                        for l, _, _ in network.active_branches)
        q_loss_pu = sum(network.lines[l].x_pu * var_values["l_var"][l]
                        for l, _, _ in network.active_branches)
    else:
        var_values = {
            "p_sub": float("nan"),
            "q_sub": float("nan"),
            "U": {b: float("nan") for b in bus_names},
            "V": {b: float("nan") for b in bus_names},
            "branch": {},
            "p_pv": {},
            "q_pv": {},
            "p_ch": {},
            "p_dis": {},
            "q_st": {},
            "se": {},
            "z_demand": {},
            "l_var": {},
            "p_storage": {},
        }
        p_loss_pu = float("nan")
        q_loss_pu = float("nan")

    # 转换为 Julia 兼容的输出格式
    if model.SolCount > 0:
        bus_solution = {}
        for b in bus_names:
            v_val = var_values["V"][b]
            bus_solution[b] = {
                "vm": [v_val, v_val, v_val],   # 三相平衡
                "va": [0.0, -2 * 3.141592653589793 / 3, 2 * 3.141592653589793 / 3],  # 仅占位
            }

        gen_solution = {}
        # slack gen
        gen_solution["1"] = {
            "pg": [var_values["p_sub"] / 3.0] * 3,
            "qg": [var_values["q_sub"] / 3.0] * 3,
        }
        # PV gens (按顺序编号)
        gen_idx = 2
        for pv in network.pvs.values():
            gen_solution[str(gen_idx)] = {
                "pg": [var_values["p_pv"][pv.name] / 3.0] * 3,
                "qg": [var_values["q_pv"][pv.name] / 3.0] * 3,
                "source_id": pv.name,
            }
            gen_idx += 1

        storage_solution = {}
        for idx, st in enumerate(network.storages.values(), start=1):
            p_net_st = var_values["p_storage"][st.name]
            q_st_val = var_values["q_st"][st.name]
            storage_solution[str(idx)] = {
                "ps": [p_net_st / 3.0] * 3,
                "qs": [q_st_val / 3.0] * 3,
                "source_id": st.name,
            }

        solution = {
            "bus": bus_solution,
            "gen": gen_solution,
            "storage": storage_solution,
            "z_demand": var_values["z_demand"],
        }
    else:
        solution = {
            "bus": {},
            "gen": {},
            "storage": {},
            "z_demand": {},
        }

    result = {
        "solve_time": solve_time,
        "optimizer": "Gurobi",
        "termination_status": status_str,
        "objective": obj_val,
        "solution": solution,
        # 网损信息
        "_p_loss_pu": p_loss_pu,
        "_q_loss_pu": q_loss_pu,
        # 额外信息 (便于打印)
        "_var_values": var_values,
        "_sense": sense,
    }

    return result, var_values


# =====================================================================
# 结果打印
# =====================================================================

def print_result_summary(result: dict, network: Network, var_values: dict):
    """打印结果摘要 (兼容 Julia 输出格式)"""
    sense = result.get("_sense", "?")

    print(f"   - 终止状态: {result['termination_status']}")
    obj_pu = result["objective"]
    obj_mw = obj_pu * network.base_mva
    print(f"   - 目标函数值: {round(obj_pu, 6)} pu ({round(obj_mw, 4)} MW)")

    # 网损
    l_vals = var_values.get("l_var", {})
    if l_vals:
        p_loss_pu = sum(network.lines[l].r_pu * l_vals[l] for l, _, _ in network.active_branches)
        q_loss_pu = sum(network.lines[l].x_pu * l_vals[l] for l, _, _ in network.active_branches)
        p_loss_mw = p_loss_pu * network.base_mva
        q_loss_mvar = q_loss_pu * network.base_mva
        p_loss_obj_pct = (p_loss_pu / obj_pu * 100) if obj_pu != 0 else 0.0
        print(f"   - 全网网损: P_loss = {round(p_loss_mw, 4)} MW ({round(p_loss_obj_pct, 3)}% of 目标函数值), "
              f"Q_loss = {round(q_loss_mvar, 4)} Mvar")
    else:
        print("   - 网损: 未计入")

    # 全部节点电压 (带资源标注)
    print()
    print("  全部节点电压:")
    # 构建 bus → 资源名 映射
    bus_resources = {}
    for pv in network.pvs.values():
        bus_resources.setdefault(pv.bus, []).append(pv.name)
    for ld in network.loads.values():
        if ld.dispatchable:
            bus_resources.setdefault(ld.bus, []).append(ld.name)
    # 按编号排序
    bus_numbers = sorted(
        [b for b in network.buses if b.startswith("bus")],
        key=lambda x: int(x.replace("bus", ""))
    )
    for b in bus_numbers:
        v = var_values.get("V", {}).get(b, float("nan"))
        if not (v != v):
            tag = ""
            if b in bus_resources:
                tag = " (" + ", ".join(bus_resources[b]) + ")"
            print(f"   Bus {b.replace('bus','').rjust(2)}: Vm = {round(v, 4)} pu{tag}")

    # 储能出力
    print()
    print("  储能出力:")
    p_storage = var_values.get("p_storage", {})
    q_storage = var_values.get("q_st", {})
    for st in network.storages.values():
        p_net = p_storage.get(st.name, 0.0)
        q_net = q_storage.get(st.name, 0.0)
        p_total_mw = p_net * network.base_mva
        q_total_mvar = q_net * network.base_mva
        se_val = var_values.get("se", {}).get(st.name, float("nan"))
        se_mwh = se_val * network.base_mva
        if p_net < -1e-6:
            action = f"放电 P = {abs(round(p_total_mw, 4))} MW"
        elif p_net > 1e-6:
            action = f"充电 P = {round(p_total_mw, 4)} MW"
        else:
            action = "待机 P = 0 MW"
        print(f"   {st.name} (额定 {st.kw} kW / {st.kwh} kWh, 能量比例 {st.energy_ratio:.2f}): "
              f"{action} (能量: {round(se_mwh, 4)} MWh / "
              f"{round(st.energy_ub_cur_pu * network.base_mva, 4)} MWh), "
              f"Q = {round(q_total_mvar, 4)} Mvar")

    # 光伏出力
    print()
    print("  光伏出力:")
    p_pv = var_values.get("p_pv", {})
    q_pv = var_values.get("q_pv", {})
    for pv in network.pvs.values():
        p = p_pv.get(pv.name, float("nan"))
        q = q_pv.get(pv.name, float("nan"))
        p_mw = p * network.base_mva
        q_mvar = q * network.base_mva
        ratio = p / pv.p_avail_pu * 100 if pv.p_avail_pu > 0 else 0
        print(f"   {pv.name} (额定 {pv.pmpp_kw} kW, 辐照 {pv.irradiance:.2f}, "
              f"可用 {pv.p_avail_pu * network.base_mva:.4f} MW): "
              f"P = {round(p_mw, 4)} MW ({ratio:.1f}% of 可用), Q = {round(q_mvar, 4)} Mvar")

    # 可调度负荷
    print()
    print("  可调度负荷 (z_demand, z=1 表示保持当前挂载):")
    z_demand = var_values.get("z_demand", {})
    for ld in network.loads.values():
        if ld.dispatchable:
            z = z_demand.get(ld.name, float("nan"))
            cur_kw = ld.kw * ld.base_ratio   # 当前实际挂载
            actual_kw = z * cur_kw           # 实际消费
            pct_full = z * ld.base_ratio * 100.0   # 相对满载的 %
            marker = "★" if ld.dispatchable else " "
            print(f"   {marker} {ld.name} (满载 {ld.kw} kW, 当前挂载 {cur_kw:.0f} kW): "
                  f"z = {round(z, 4)} → {round(actual_kw, 1)} kW ({pct_full:.1f}% of 满载)")

    # 固定负荷（仅打印总线损级汇总已在前面给出，此处可简要展示不可调度负荷的总量）
    fixed_loads = [ld for ld in network.loads.values() if not ld.dispatchable]
    if fixed_loads:
        total_fixed_kw = sum(ld.kw for ld in fixed_loads)
        z_fixed = sum(z_demand.get(ld.name, 1.0) * ld.kw for ld in fixed_loads)
        print(f"   固定负荷 ({len(fixed_loads)} 个, 合计额定 {total_fixed_kw} kW): z=1 (不可调)")

# -*- coding: utf-8 -*-
"""
save_results.py
OPF 结果输出导出模块 (完全独立, 不依赖 main / parse_dss / parse_csv)

提供两个导出入口 (输出格式约定见 training_dataset/输出控制.md):
1. save_case_results     : 单断面 (main.py)  — 6 张组件 CSV, sense 列区分 min/max
2. save_sampled_results  : 抽样 (training_dataset_mc.py) — 6 张组件 CSV, 第一列 sample_id,
                            第二列 sense, 所有样本结果堆叠

参数约定 (鸭子类型, 与 opf_model 的返回值结构一致):
    network    : 已 finalize 的配电网模型 (提供 base_mva/buses/slack_bus/pvs/
                 loads/storages/lines/active_branches)
    result     : OPF 结果字典 (objective/_p_loss_pu/_q_loss_pu/
                 solve_time/termination_status)
    var_values : 变量值字典 (p_sub/q_sub/V/branch/p_pv/q_pv/
                 p_ch/p_dis/q_st/se/p_storage/z_demand/l_var)

数值均为工程单位 (MW/Mvar/MWh, 电压 pu)。
"""

from __future__ import annotations

import csv
import os
import re


# 组件 CSV 表名 (与 training_dataset/输出控制.md 对应)
CSV_FILES = ["training_dataset_system.csv", "training_dataset_buses.csv", "training_dataset_lines.csv", "training_dataset_loads.csv", "training_dataset_pvs.csv", "training_dataset_storage.csv"]


def _infer_type(ld) -> str:
    """负荷类型: 显式 type 优先, 否则按命名约定推导 (与 parse_dss 一致)"""
    if ld.type is not None:
        return ld.type
    low = ld.name.lower()
    if low.startswith("fixed_"):
        return "fixed_extra"
    if re.match(r"^ev(?:[_\d])", low):
        return "ev"
    if re.match(r"^(ac)[-_]", low):
        return "ac"
    return "fixed"


def save_case_results(network, result_min, var_min, result_max, var_max,
                      out_dir: str) -> None:
    """单断面模式: 在 out_dir 下保存 6 张组件 CSV (sense 列区分 min/max)"""
    entries = [
        (None, None, "min", result_min, var_min, None),
        (None, None, "max", result_max, var_max, None),
    ]
    _write_tables(network, entries, out_dir)


def save_sampled_results(network, records, out_dir: str) -> None:
    """抽样模式: 在 out_dir 下保存 6 张组件 CSV。

    records: list of (time_slot:str, sample_id:int, sense:str, result, var_values, ctx)
    其中 ctx 为每样本抽样上下文 (loads/pvs/buses 的计算基准):
        {"base_ratio": {负荷名: 当前挂载比例},
         "p_cur_pu":   {负荷名: 当前挂载有功 pu},
         "q_cur_pu":   {负荷名: 当前挂载无功 pu},
         "p_avail_pu": {光伏名: 可用出力 pu}}
    每行前两列为 sample_id / time_slot (断面), 次列为 sense, 其余列与单断面模式一致。
    """
    _write_tables(network, list(records), out_dir)


# =====================================================================
# 内部实现
# =====================================================================

def _write_tables(network, entries, out_dir: str) -> None:
    """entries: list of (sample_id|None, sense, result, var_values)"""
    if not entries:
        return
    os.makedirs(out_dir, exist_ok=True)
    tables = _build_tables(network, entries)
    for fname, (header, rows) in tables.items():
        with open(os.path.join(out_dir, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


def _build_tables(network, entries) -> dict:
    """生成 {表名: (表头, 行列表)}
    entries: list of (time_slot|None, sample_id|None, sense, result, var_values, ctx|None)
    entries 元素 sample_id 非 None 时为抽样模式 (ctx 提供每样本抽样上下文)"""
    s_base = network.base_mva  # 1 pu = s_base MW
    mw = lambda v: v * s_base
    with_id = entries[0][1] is not None
    tables: dict = {}

    def finish(header, row, ts, sid, sense):
        """单断面: 原样返回; 抽样: 首列 sample_id, 次列 time_slot, 次次列 sense, 去重原 sense 列"""
        if not with_id:
            return header, row
        i = header.index("sense")
        h = ["sample_id", "time_slot", "sense"] + [c for c in header if c != "sense"]
        r = [sid, ts, sense] + [v for j, v in enumerate(row) if j != i]
        return h, r

    # ---- 节点净注入 (注入为正): slack + PV + 储能 - 负荷 - 储能充电 ----
    def node_injection(var, ctx):
        p_inj = {b: 0.0 for b in network.buses}
        q_inj = {b: 0.0 for b in network.buses}
        p_inj[network.slack_bus] += var["p_sub"]
        q_inj[network.slack_bus] += var["q_sub"]
        for pv in network.pvs.values():
            p_inj[pv.bus] += var["p_pv"][pv.name]
            q_inj[pv.bus] += var["q_pv"][pv.name]
        for ld in network.loads.values():
            z = var["z_demand"][ld.name]
            p_cur_pu = ctx["p_cur_pu"][ld.name] if ctx else ld.p_cur_pu
            q_cur_pu = ctx["q_cur_pu"][ld.name] if ctx else ld.q_cur_pu
            p_inj[ld.bus] -= p_cur_pu * z
            q_inj[ld.bus] -= q_cur_pu * z
        for st in network.storages.values():
            p_inj[st.bus] -= var["p_storage"][st.name]  # p_storage 正=充电(取电)
            q_inj[st.bus] += var["q_st"][st.name]
        return p_inj, q_inj

    # ---- training_dataset_system.csv ----
    header = ["sense", "objective_mw", "p_sub_mw", "q_sub_mvar",
              "p_loss_mw", "q_loss_mvar", "solve_time_s", "status"]
    h_final, rows = None, []
    for ts, sid, sense, res, var, ctx in entries:
        row = [sense,
               round(mw(res["objective"]), 6),
               round(mw(var["p_sub"]), 6),
               round(mw(var["q_sub"]), 6),
               round(mw(res["_p_loss_pu"]), 6),
               round(mw(res["_q_loss_pu"]), 6),
               round(res["solve_time"], 4),
               res["termination_status"]]
        h, r = finish(header, row, ts, sid, sense)
        h_final = h if h_final is None else h_final
        rows.append(r)
    tables["training_dataset_system.csv"] = (h_final, rows)

    # ---- training_dataset_buses.csv ----
    def bus_key(b):
        return int(b.replace("bus", "")) if b.startswith("bus") else 10 ** 9
    bus_ids = sorted(network.buses, key=bus_key)
    header = ["bus", "sense", "vm_pu", "p_inj_mw", "q_inj_mvar"]
    h_final, rows = None, []
    for ts, sid, sense, res, var, ctx in entries:
        p_inj, q_inj = node_injection(var, ctx)
        for b in bus_ids:
            row = [b, sense, round(var["V"][b], 6),
                   round(mw(p_inj[b]), 6), round(mw(q_inj[b]), 6)]
            h, r = finish(header, row, ts, sid, sense)
            h_final = h if h_final is None else h_final
            rows.append(r)
    tables["training_dataset_buses.csv"] = (h_final, rows)

    # ---- lines.csv ----
    header = ["name", "fbus", "tbus", "sense", "p_mw", "q_mvar",
              "s_max_mva", "p_loss_mw", "q_loss_mvar"]
    h_final, rows = None, []
    for ts, sid, sense, res, var, ctx in entries:
        for lname, fbus, tbus in network.active_branches:
            line = network.lines[lname]
            p = var["branch"][lname]["P"]
            q = var["branch"][lname]["Q"]
            l = var["l_var"][lname]
            row = [lname, fbus, tbus, sense,
                   round(mw(p), 6), round(mw(q), 6),
                   round(mw(line.smax_pu), 6),
                   round(mw(line.r_pu * l), 6),
                   round(mw(line.x_pu * l), 6)]
            h, r = finish(header, row, ts, sid, sense)
            h_final = h if h_final is None else h_final
            rows.append(r)
    tables["training_dataset_lines.csv"] = (h_final, rows)

    # ---- training_dataset_loads.csv ----
    header = ["name", "bus", "type", "sense", "p_full_mw", "q_full_mvar",
              "z", "p_cur_mw", "p_out_mw", "pct_of_full_pct", "q_out_mvar"]
    h_final, rows = None, []
    for ts, sid, sense, res, var, ctx in entries:
        for ld in network.loads.values():
            p_full = ld.kw / 1000.0
            q_full = ld.kvar / 1000.0
            z = var["z_demand"][ld.name]
            base_ratio = ctx["base_ratio"][ld.name] if ctx else ld.base_ratio
            p_cur = p_full * base_ratio
            p_out = p_cur * z
            q_out = q_full * base_ratio * z
            pct = (p_out / p_full * 100) if p_full else 0.0
            row = [ld.name, ld.bus, _infer_type(ld), sense,
                   round(p_full, 6), round(q_full, 6),
                   round(z, 6), round(p_cur, 6),
                   round(p_out, 6), round(pct, 4), round(q_out, 6)]
            h, r = finish(header, row, ts, sid, sense)
            h_final = h if h_final is None else h_final
            rows.append(r)
    tables["training_dataset_loads.csv"] = (h_final, rows)

    # ---- pvs.csv ----
    header = ["name", "bus", "sense", "p_avail_mw", "p_out_mw", "q_out_mvar",
              "pct_of_avail_pct"]
    h_final, rows = None, []
    for ts, sid, sense, res, var, ctx in entries:
        for pv in network.pvs.values():
            p_avail = mw(ctx["p_avail_pu"][pv.name]) if ctx else mw(pv.p_avail_pu)
            p = mw(var["p_pv"][pv.name])
            q = mw(var["q_pv"][pv.name])
            pct = (p / p_avail * 100) if p_avail else 0.0
            row = [pv.name, pv.bus, sense,
                   round(p_avail, 6), round(p, 6), round(q, 6), round(pct, 4)]
            h, r = finish(header, row, ts, sid, sense)
            h_final = h if h_final is None else h_final
            rows.append(r)
    tables["training_dataset_pvs.csv"] = (h_final, rows)

    # ---- training_dataset_storage.csv ----
    # p_net = p_dis - p_ch (正=放电); SOC = se / 额定kWh × 100
    header = ["name", "bus", "sense", "p_ch_mw", "p_dis_mw", "p_net_mw",
              "q_mvar", "se_mwh", "soc_pct"]
    h_final, rows = None, []
    for ts, sid, sense, res, var, ctx in entries:
        for st in network.storages.values():
            kwh_rated = st.kwh / 1000.0
            p_ch = mw(var["p_ch"][st.name])
            p_dis = mw(var["p_dis"][st.name])
            p_net = p_dis - p_ch          # 正=放电
            se = mw(var["se"][st.name])
            soc = (se / kwh_rated * 100) if kwh_rated else 0.0
            row = [st.name, st.bus, sense,
                   round(p_ch, 6), round(p_dis, 6), round(p_net, 6),
                   round(mw(var["q_st"][st.name]), 6),
                   round(se, 6), round(soc, 4)]
            h, r = finish(header, row, ts, sid, sense)
            h_final = h if h_final is None else h_final
            rows.append(r)
    tables["training_dataset_storage.csv"] = (h_final, rows)

    return tables

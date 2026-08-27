# -*- coding: utf-8 -*-
"""
scenario_dataset_mc_OPF.py — 场景数据集蒙特卡洛 (OPF 真值求解版, 验证 KNN 预测)

整体运行逻辑与 scenario_dataset_mc.py 一致 (同一 output_mc_config.csv + 同一抽样):
  - 只对 mc_config 中配置的 EV/AC 可调度负荷 lb/ub 做截断正态抽样 (σ=cv×μ 或 sigma)
  - 其余量全部作为实际值: 固定负荷功率 (曲线), PV 辐照度 (曲线),
    储能初始能量 (曲线, = 能量窗口上限) / 初始功率 0, EV/AC 当前挂载 (曲线)
  - 断面按 15min 步进 (start_time ~ end_time, 含端点), 每断面 N 个样本

区别: 输出不是 KNN 预测, 而是对每个样本实际求解 OPF (min/max 根节点注入,
LinDistFlow SOCP, Gurobi), 得到真值用于验证 KNN 预测精度。

输入:
  - 场景配置 (output/{场景}/output_mc_config.csv): scenario + n_samples/seed/start_time/end_time
    + EV/AC 的 lb/ub 抽样分布 (cv/sigma; 未配置的 lb/ub 固定曲线值)
  - 场景网络 (scenario/{scenario}/ 或 data/csv_case33/model_{scenario}/)

输出 (output/{场景}_opf/, 格式参照 scenario_dataset_mc, 前缀 output):
  - output_mc_config.csv       场景预测配置 (读入, 保留副本)
  - output_sample.csv   抽样输入场景表 (每 断面×样本 一行, 与 KNN 版完全一致)
  - output_system.csv   根节点注入真值 (sense + p_sub/q_sub)
  - output_storage.csv  储能出力真值 (p_net/q/se, p_net 正=放电, 与 training_dataset_storage 一致)
  - output_pvs.csv      光伏出力真值 (p_out/q_out)
  - output_loads.csv    可调度负荷实际有功真值 (ev/ac)

用法:
  python scenario_dataset_mc_OPF.py --config output/output_scenario_default/output_mc_config.csv
  python scenario_dataset_mc_OPF.py --scenario output_scenario_default --n 200 --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import shutil
import sys
import time

import numpy as np

from load_network import load_network, resolve_model_path
from sampling import truncated_normal_vec, resolve_mu_sigma
from opf_model import build_and_solve_opf


# =====================================================================
# Gurobi 求解输出抑制
# =====================================================================

class _suppress_output:
    """临时静默 stdout+stderr (sys 对象 + fd 1/2), 抑制 Gurobi 每次求解的
    'Set parameter ...' 刷屏 (Gurobi 默认写到 stdout)。仅包住 build_and_solve_opf
    调用, 求解循环外的进度打印与异常堆栈不受影响。
    """
    def __enter__(self):
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._saved_fds = {}
        for fd in (1, 2):
            self._saved_fds[fd] = os.dup(fd)
            os.dup2(self._devnull, fd)
        self._saved_out, self._saved_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._saved_out, self._saved_err
        for fd in (1, 2):
            os.dup2(self._saved_fds[fd], fd)
            os.close(self._saved_fds[fd])
        os.close(self._devnull)
        return False


def solve_opf_silent(network, sense):
    """调用 build_and_solve_opf (verbose=False) 并抑制 Gurobi 求解输出"""
    with _suppress_output():
        return build_and_solve_opf(network, sense, verbose=False)


# =====================================================================
# 断面辅助 (与 scenario_dataset_mc / training_dataset_mc 一致: 15min 步进, 含端点)
# =====================================================================

def _parse_time(s, default="0:00"):
    s = (s or "").strip()
    try:
        h, m = s.split(":")
        return int(h), int(m)
    except (ValueError, AttributeError):
        h, m = default.split(":")
        return int(h), int(m)


def _slots(start, end):
    h0, m0 = _parse_time(start, "0:00")
    h1, m1 = _parse_time(end, "23:45")
    t0 = max(0, min(h0 * 4 + m0 // 15, 95))
    t1 = max(0, min(h1 * 4 + m1 // 15, 95))
    return [(h, rem * 15, t) for t in range(t0, t1 + 1)
            for h, rem in (divmod(t, 4),)]


def _slot_label(h, m):
    return f"{h}:{m:02d}"


# output_mc_config.csv 全局键 (value 列为参数值); 其余为组件分布配置
# model 键在 KNN 版用于索引模型库, 本版实际求解 OPF 不需要, 但保留以便复用同一配置
MC_GLOBAL_KEYS = {"model", "scenario", "n_samples", "seed", "start_time", "end_time"}


def load_mc_config(path: str):
    """读取 output/{场景}/output_mc_config.csv → (全局参数 dict, 组件配置 {组件名: (分布名, 参数字典)})

    结构与 scenario_dataset_mc.load_mc_config 完全一致:
      name,value
      model,training_dataset_default                 # KNN 版索引模型库用, 本版忽略
      scenario,output_scenario_default              # 场景名
      n_samples,200
      seed,42
      start_time,0:00
      end_time,23:45
      EV_Bus19_lb,truncated_normal,cv:0.10   # 组件分布: 仅 EV/AC 的 lb/ub 抽样
      EV_Bus19_ub,truncated_normal,cv:0.10
    """
    global_params, comps = {}, {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            cells = [c.strip() for c in row]
            if not cells or not cells[0] or cells[0].lower() == "name":
                continue
            name, val = cells[0], cells[1] if len(cells) > 1 else ""
            if not val:
                continue
            if name in MC_GLOBAL_KEYS:
                global_params[name] = val
                continue
            params = {}
            for cell in cells[2:]:
                if not cell:
                    continue
                k, _, v = cell.partition(":")
                k, v = k.strip(), v.strip()
                try:
                    params[k] = float(v)        # 数字参数 (cv:0.10 / sigma:0.05)
                except ValueError:
                    params[k] = v               # shape 名 (mu:<曲线> / sigma:<曲线>, 96 点)
            comps[name] = (val, params)
    return global_params, comps


# =====================================================================
# 断面曲线应用 (与 training_dataset_mc._apply_slot 一致; 初始能量 = 窗口上限)
# =====================================================================

def _apply_slot(network, t):
    """将网络组件当前值设为第 t 断面的曲线值 (mult[t]), 重算派生量与可调区间 clamp:
    - 负荷: base_ratio = 主曲线[t]; mult_lb/ub = 绑定曲线[t]; lb/ub 相对 cur clamp;
      重算 p_cur/q_cur/z 上下限
    - 光伏: irradiance = 曲线[t]; p_avail = p_max × irradiance
    - 储能: energy_ratio = 曲线[t]; 能量窗口 = 额定 × ratio;
      初始能量 = 窗口上限 (与 KNN 特征 se_init 一致), 初始功率 0
    曲线越界时取末点。
    """
    for ld in network.loads.values():
        if ld.shape:
            s = network.shapes.get(ld.shape)
            if s and s.mult:
                ld.base_ratio = s.mult[min(t, len(s.mult) - 1)]
        else:
            ld.base_ratio = 1.0
        if ld.mult_lb_shape:
            s = network.shapes.get(ld.mult_lb_shape)
            if s and s.mult:
                ld.mult_lb = s.mult[min(t, len(s.mult) - 1)]
        if ld.mult_ub_shape:
            s = network.shapes.get(ld.mult_ub_shape)
            if s and s.mult:
                ld.mult_ub = s.mult[min(t, len(s.mult) - 1)]
        # 可调区间 clamp (与 load_network._preprocess_dispatchable 一致): lb ≤ cur ≤ ub
        if ld.dispatchable:
            if ld.mult_lb > ld.base_ratio:
                ld.mult_lb = ld.base_ratio
            if ld.mult_ub < ld.base_ratio:
                ld.mult_ub = ld.base_ratio
        ld.p_cur_pu = ld.p_pu * ld.base_ratio
        ld.q_cur_pu = ld.q_pu * ld.base_ratio
        ld.z_lb = ld.mult_lb / ld.base_ratio if ld.base_ratio > 0 else 0.0
        ld.z_ub = ld.mult_ub / ld.base_ratio if ld.base_ratio > 0 else 1e6
    for pv in network.pvs.values():
        irrad = pv.irradiance
        if pv.shape:
            s = network.shapes.get(pv.shape)
            if s and s.mult:
                irrad = s.mult[min(t, len(s.mult) - 1)]
        pv.irradiance = irrad
        pv.p_avail_pu = pv.p_max_pu * irrad
    for st in network.storages.values():
        st.energy_ratio = 1.0
        if st.shape:
            s = network.shapes.get(st.shape)
            if s and s.mult:
                st.energy_ratio = s.mult[min(t, len(s.mult) - 1)]
        st.energy_ub_cur_pu = st.energy_ub_pu * st.energy_ratio
        st.energy_lb_cur_pu = st.energy_lb_pu * st.energy_ratio
        st.energy_init_cur_pu = st.energy_ub_cur_pu   # 初始能量 = 窗口上限 (未配置储能, 方案A)
        st.p_init_cur_pu = 0.0


# =====================================================================
# 特征构建 (与 scenario_dataset_mc.build_slot_features 完全一致)
# =====================================================================

def build_slot_features(feature_names, net, shapes, t, rng, comp_cfg, n):
    """构建第 t 断面的 n 个样本的 len(feature_names) 维特征矩阵 (X: n×F)

    固定项: 负荷当前功率 (曲线), PV 辐照度 (曲线), 储能初始能量 (曲线)/功率 0
    抽样项: comp_cfg 中配置的 EV/AC lb/ub (截断正态, ub 先抽 [0,1], lb 截断 [0, ub]);
            未配置的 lb/ub 固定为曲线值 (方案A)
    """
    base_mva = net.base_mva
    X = np.zeros((n, len(feature_names)))
    ub_saved = {}
    for j, f in enumerate(feature_names):
        if f.endswith("_mw"):
            ld = net.loads.get(f[:-3])
            if ld is None:                       # 未知特征名兜底 0
                continue
            full_mw = ld.p_pu * base_mva
            mult = shapes[ld.shape].mult[t] if (ld.shape and ld.shape in shapes) else 1.0
            X[:, j] = full_mw * mult
        elif f.endswith("_irr"):
            X[:, j] = shapes[f].mult[t] if f in shapes else 0.0
        elif f.endswith("_se_init_mwh"):
            st = net.storages.get(f[: -len("_se_init_mwh")])
            er = shapes["BESS_Bus18_en"].mult[t] if "BESS_Bus18_en" in shapes else 1.0
            X[:, j] = (st.energy_ub_pu * base_mva * er) if st is not None else 0.0
        elif f.endswith("_p_init_mw"):
            X[:, j] = 0.0
        elif f.endswith("_ub"):
            name = f[:-3]
            mu = shapes[f].mult[t] if f in shapes else 1.0
            if f in comp_cfg:                    # 配置 → 抽样 (μ/σ 可来自 shape 曲线)
                mu, sigma = resolve_mu_sigma(comp_cfg[f], shapes, t, mu)
                ub = truncated_normal_vec(np.full(n, mu), np.full(n, sigma),
                                          np.zeros(n), np.ones(n), rng)
            else:                                # 未配置 → 固定曲线值 (方案A)
                ub = np.full(n, mu)
            X[:, j] = ub
            ub_saved[name] = ub
        elif f.endswith("_lb"):
            name = f[:-3]
            mu = shapes[f].mult[t] if f in shapes else 0.0
            hi = ub_saved.get(name)
            hi = hi if hi is not None else np.ones(n)
            if f in comp_cfg:                    # 配置 → 抽样 (截断于 [0, 对应 ub])
                mu, sigma = resolve_mu_sigma(comp_cfg[f], shapes, t, mu)
                lb = truncated_normal_vec(np.full(n, mu), np.full(n, sigma),
                                          np.zeros(n), hi, rng)
            else:                                # 未配置 → 固定曲线值 (方案A)
                lb = np.full(n, mu)
            X[:, j] = lb
    return X


# =====================================================================
# 输出 (格式与 scenario_dataset_mc 一致; 数值取 OPF 真值)
# =====================================================================

def save_outputs(out_dir, records, feature_names, net):
    """records: list of (slot_label, sample_id, sense, rec, X_row, cur_vals, disp_p_cur)

    rec: OPF 求解结果快照 (status/objective_pu/p_sub/q_sub/
         st:{名:(p_ch,p_dis,q_st,se) pu} / pv:{名:(p_out,q_out) pu} / z:{可调度名:z})
    X_row 与 feature_names 一一对应; cur_vals: {EV/AC 名: 断面曲线 cur};
    disp_p_cur: {可调度负荷名: 求解时当前挂载有功 pu} (p_out 计算基准)。
    """
    os.makedirs(out_dir, exist_ok=True)
    base_mva = net.base_mva
    mw = lambda v: v * base_mva

    # ---- output_sample.csv: 抽样输入场景表 (与 scenario_dataset_mc 完全一致, min/max 共用同一批抽样) ----
    load_feats = [f for f in feature_names if f.endswith("_mw") and not f.endswith("_p_init_mw")]
    irr_feats = [f for f in feature_names if f.endswith("_irr")]
    # 储能初始状态: 固定 se_init 在前、p_init 在后 (与 training_dataset_sample 列序一致)
    bess_feats = ([f for f in feature_names if f.endswith("_se_init_mwh")]
                  + [f for f in feature_names if f.endswith("_p_init_mw")])
    lbub_feats = [f for f in feature_names if f.endswith("_lb") or f.endswith("_ub")]
    disp_names = [ld.name for ld in net.loads.values() if ld.dispatchable]  # 可调度组件 (cur 固定曲线值)
    disp_header = []
    for n in disp_names:                                     # 与 training_dataset_sample 交替顺序 cur/lb/ub
        disp_header += [f"{n}_cur"] + [f for f in lbub_feats if f.startswith(n)]
    sample_header = (["sample_id", "time_slot"] + load_feats + ["fixed_total_mw"]
                     + irr_feats + bess_feats + disp_header)
    load_idx = [feature_names.index(f) for f in load_feats]

    sample_rows = [r for r in records if r[2] == "min"]
    with open(os.path.join(out_dir, "output_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(sample_header)
        for ts, sid, sense, rec, xrow, cur_vals, _ in sample_rows:
            fixed_total = sum(xrow[i] for i in load_idx)
            vals = [sid, ts] + [f"{xrow[i]:.6f}" for i in load_idx] + [f"{fixed_total:.6f}"]
            vals += [f"{xrow[feature_names.index(f)]:.6f}" for f in irr_feats]
            vals += [f"{xrow[feature_names.index(f)]:.6f}" for f in bess_feats]
            for n in disp_names:
                vals.append(f"{cur_vals.get(n, 0.0):.6f}")          # cur 固定曲线值
                vals += [f"{xrow[feature_names.index(f)]:.6f}" for f in lbub_feats if f.startswith(n)]
            w.writerow(vals)
    print(f"  已保存: {os.path.join(out_dir, 'output_sample.csv')}")

    # ---- output_system.csv: 根节点注入真值 ----
    with open(os.path.join(out_dir, "output_system.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "time_slot", "sense", "p_sub_mw", "q_sub_mvar"])
        for ts, sid, sense, rec, _, _, _ in records:
            w.writerow([sid, ts, sense,
                        f"{mw(rec['p_sub']):.6f}", f"{mw(rec['q_sub']):.6f}"])
    print(f"  已保存: {os.path.join(out_dir, 'output_system.csv')}")

    def write_component(fname, header, rows):
        with open(os.path.join(out_dir, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"  已保存: {os.path.join(out_dir, fname)}")

    # ---- output_storage.csv: 储能出力真值 (p_net 正=放电, 与 training_dataset_storage 一致) ----
    st_rows, st_headers = [], ["sample_id", "time_slot", "sense", "name", "bus",
                               "p_net_mw", "q_mvar", "se_mwh"]
    for ts, sid, sense, rec, _, _, _ in records:
        for st in net.storages.values():
            p_ch, p_dis, q_st, se = rec["st"][st.name]
            vals = [mw(p_dis - p_ch), mw(q_st), mw(se)]
            st_rows.append([sid, ts, sense, st.name, st.bus] + [f"{v:.6f}" for v in vals])
    write_component("output_storage.csv", st_headers, st_rows)

    # ---- output_pvs.csv: 光伏出力真值 ----
    pv_rows, pv_headers = [], ["sample_id", "time_slot", "sense", "name", "bus",
                               "p_out_mw", "q_out_mvar"]
    for ts, sid, sense, rec, _, _, _ in records:
        for pv in net.pvs.values():
            p_out, q_out = rec["pv"][pv.name]
            vals = [mw(p_out), mw(q_out)]
            pv_rows.append([sid, ts, sense, pv.name, pv.bus] + [f"{v:.6f}" for v in vals])
    write_component("output_pvs.csv", pv_headers, pv_rows)

    # ---- output_loads.csv (仅 ev/ac 可调度负荷, p_out = z × 当前挂载) ----
    ld_rows, ld_headers = [], ["sample_id", "time_slot", "sense", "name", "bus", "type",
                               "p_out_mw"]
    for ts, sid, sense, rec, _, _, disp_p_cur in records:
        for ld in net.loads.values():
            if not ld.dispatchable:
                continue
            p_out = rec["z"][ld.name] * disp_p_cur[ld.name] * base_mva
            ld_rows.append([sid, ts, sense, ld.name, ld.bus, ld.type, f"{p_out:.6f}"])
    write_component("output_loads.csv", ld_headers, ld_rows)


# =====================================================================
# 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="场景数据集蒙特卡洛 OPF 真值求解 (验证 KNN 预测)")
    parser.add_argument("--config", default=None,
                        help="场景配置 output_mc_config.csv (output/{场景}/output_mc_config.csv)")
    parser.add_argument("--scenario", default=None, help="场景名 (scenario/{scenario}/), 覆盖配置")
    parser.add_argument("--n", type=int, default=None, help="每断面抽样次数 (覆盖配置)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖配置)")
    parser.add_argument("--start-time", default=None, help="起始断面 (覆盖配置)")
    parser.add_argument("--end-time", default=None, help="结束断面 (覆盖配置)")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    # 0. 场景配置 (output_mc_config.csv): CLI --config 显式 > --scenario 目录内查找
    #    > 默认算例 (output_scenario_default) > output/ 下唯一配置
    config_path = args.config
    if config_path is None and args.scenario:
        dflt = os.path.join("output", args.scenario, "output_mc_config.csv")
        if os.path.exists(dflt):
            config_path = dflt
    if config_path is None:
        dflt = os.path.join("output", "output_scenario_default", "output_mc_config.csv")
        if os.path.exists(dflt):
            config_path = dflt
    if config_path is None:
        found = [os.path.join("output", d, "output_mc_config.csv")
                 for d in sorted(os.listdir("output"))
                 if os.path.isfile(os.path.join("output", d, "output_mc_config.csv"))]
        if len(found) == 1:
            config_path = found[0]
        elif len(found) > 1:
            raise ValueError(f"output/ 下存在多个 output_mc_config.csv ({found}), 请用 --config 指定")
    cfg_global, comp_cfg = {}, {}
    if config_path and os.path.exists(config_path):
        cfg_global, comp_cfg = load_mc_config(config_path)
        print(f"读取场景配置: {config_path} "
              f"(全局 {len(cfg_global)} 项, 组件分布 {len(comp_cfg)} 个)")
    elif args.config:
        print(f"  警告: 未找到配置文件 {args.config}")

    # 全局键: scenario 场景名; n_samples/seed/start/end 抽样运行参数
    # (model 键仅 KNN 版用于索引模型库, 本版实际求解 OPF, 忽略)
    scenario = args.scenario or cfg_global.get("scenario")
    if not scenario:
        raise ValueError("未指定场景: 请用 --scenario 或 output_mc_config.csv 的 scenario 键")
    n_samples = args.n if args.n is not None else int(cfg_global.get("n_samples", 200))
    seed = args.seed if args.seed is not None else int(cfg_global.get("seed", 42))
    start_time = args.start_time or cfg_global.get("start_time", "0:00")
    end_time = args.end_time or cfg_global.get("end_time", "23:45")

    # 1. 加载场景网络
    net = load_network(resolve_model_path(scenario))
    shapes = {name: sh for name, sh in net.shapes.items() if sh.mult}
    print(f"场景已加载: {scenario} (曲线 {len(shapes)} 条, base_mva={net.base_mva})")

    # 2. 特征列 (与 KNN 模型 knn_feature_names 同序:
    #    固定负荷 mw + 储能 p_init + PV irr + 储能 se_init + EV/AC lb/ub)
    fixed_loads = [ld for ld in net.loads.values() if not ld.dispatchable]
    disp_loads = [ld for ld in net.loads.values() if ld.dispatchable]
    pv_list = list(net.pvs.values())
    st_list = list(net.storages.values())
    feature_names = ([f"{ld.name}_mw" for ld in fixed_loads]
                     + [f"{st.name}_p_init_mw" for st in st_list]
                     + [f"{pv.name}_irr" for pv in pv_list]
                     + [f"{st.name}_se_init_mwh" for st in st_list]
                     + [f for ld in disp_loads for f in (f"{ld.name}_lb", f"{ld.name}_ub")])
    fidx = {f: i for i, f in enumerate(feature_names)}
    print(f"特征 {len(feature_names)} 维 (固定负荷 {len(fixed_loads)}, PV {len(pv_list)}, "
          f"储能 {len(st_list)}, 可调度 {[ld.name for ld in disp_loads]})")
    if comp_cfg:
        print(f"抽样组件: {', '.join(sorted(comp_cfg))}")
    else:
        print("抽样组件: 无 (全部固定曲线值)")

    # 3. 逐断面抽样 + OPF 真值求解
    slots = _slots(start_time, end_time)
    rng = np.random.default_rng(seed)
    records = []
    t0 = time.time()
    total = len(slots) * n_samples
    done = 0
    step = max(1, total // 10)
    print(f"断面: {_slot_label(*slots[0][:2])} ~ {_slot_label(*slots[-1][:2])} "
          f"共 {len(slots)} 个, N={n_samples}/断面, 每样本求解 min+max 两个 OPF")
    for h, m, t in slots:
        ts = _slot_label(h, m)
        # EV/AC 当前挂载 cur (固定曲线值, 仅用于 output_sample 记录)
        cur_vals = {ld.name: (shapes[f"{ld.name}_cur"].mult[t] if f"{ld.name}_cur" in shapes else 1.0)
                    for ld in disp_loads}
        # 曲线基线应用到网络 (负荷/PV/储能窗口; 其余 lb/ub 固定为曲线值)
        _apply_slot(net, t)
        disp_cur0 = {ld.name: ld.base_ratio for ld in disp_loads}
        # 抽样 (仅 comp_cfg 配置的 lb/ub; 与 KNN 版同一套抽样序列)
        X = build_slot_features(feature_names, net, shapes, t, rng, comp_cfg, n_samples)
        for i in range(n_samples):
            # 应用抽样 lb/ub → 可调度负荷 (cur 保持曲线实际值, 与 _preprocess_dispatchable 一致:
            # lb > cur 时松弛 lb 到 cur, ub < cur 时松弛 ub 到 cur, 保证 lb ≤ cur ≤ ub)
            for ld in disp_loads:
                cur0 = disp_cur0[ld.name]
                ub = X[i, fidx[f"{ld.name}_ub"]]
                lb = min(X[i, fidx[f"{ld.name}_lb"]], ub)   # 保序: lb ≤ ub
                lb = min(lb, cur0)                          # lb > cur → 松弛到 cur (不允许削减到低于当前)
                ub = max(ub, cur0)                          # ub < cur → 松弛到 cur (保证当前可达)
                ld.base_ratio = cur0
                ld.mult_lb = lb
                ld.mult_ub = ub
                ld.p_cur_pu = ld.p_pu * cur0
                ld.q_cur_pu = ld.q_pu * cur0
                ld.z_lb = lb / cur0 if cur0 > 0 else 0.0
                ld.z_ub = ub / cur0 if cur0 > 0 else 1e6
            disp_p_cur = {ld.name: ld.p_cur_pu for ld in disp_loads}
            for sense in ("min", "max"):
                result, var_values = solve_opf_silent(net, sense)
                # 不可行/无解样本记录 nan + 状态 (保留行, 便于与 KNN 版逐行对比)
                if result["termination_status"] in ("OPTIMAL", "TIME_LIMIT_FEASIBLE"):
                    rec = {
                        "status": result["termination_status"],
                        "objective_pu": result["objective"],
                        "p_sub": var_values["p_sub"],
                        "q_sub": var_values["q_sub"],
                        "st": {st.name: (var_values["p_ch"][st.name], var_values["p_dis"][st.name],
                                         var_values["q_st"][st.name], var_values["se"][st.name])
                               for st in st_list},
                        "pv": {pv.name: (var_values["p_pv"][pv.name], var_values["q_pv"][pv.name])
                               for pv in pv_list},
                        "z": {ld.name: var_values["z_demand"][ld.name] for ld in disp_loads},
                    }
                else:
                    rec = {"status": result["termination_status"],
                           "objective_pu": float("nan"),
                           "p_sub": float("nan"), "q_sub": float("nan"),
                           "st": {st.name: (float("nan"),) * 4 for st in st_list},
                           "pv": {pv.name: (float("nan"),) * 2 for pv in pv_list},
                           "z": {ld.name: float("nan") for ld in disp_loads}}
                records.append((ts, i + 1, sense, rec, X[i], cur_vals, disp_p_cur))
            done += 1
            if done % step == 0 or done == total:
                print(f"    {done}/{total} (断面 {ts} 样本 {i + 1}) 完成 ({time.time() - t0:.1f}s)")

    # 4. 输出 (output/{场景}_opf/; 与 KNN 版 output/{场景}/ 分离, 便于对比验证)
    out_dir = os.path.join("output", f"{scenario}_opf")
    save_outputs(out_dir, records, feature_names, net)
    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(out_dir, "output_mc_config.csv"))
        print(f"  已保存: {os.path.join(out_dir, 'output_mc_config.csv')} (配置副本)")

    # 5. 概览 (每 sense OPTIMAL 统计)
    for sense in ("min", "max"):
        rows_s = [r for r in records if r[2] == sense]
        ok = [r for r in rows_s if r[3]["status"] == "OPTIMAL"]
        print(f"  [{sense}] 有效 (OPTIMAL) 样本: {len(ok)}/{len(rows_s)}")
        if ok:
            psub = [r[3]["objective_pu"] * net.base_mva for r in ok]
            print(f"    p_sub_mw: min={min(psub):.4f} 中位={float(np.median(psub)):.4f} "
                  f"max={max(psub):.4f}")
    print(f"\nOPF 求解完成: {len(records)} 行 (断面×样本×sense), 已保存至 {out_dir}")


if __name__ == "__main__":
    main()

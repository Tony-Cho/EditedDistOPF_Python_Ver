# -*- coding: utf-8 -*-
"""
training_dataset_mc.py — 训练集蒙特卡洛抽样 (负荷抽样 + OPF 求解, 生成训练数据)

抽样配置由全局控制 CSV 驱动 (--config training_dataset_mc_config.csv), 每个抽样量在组件配置行内
单独设置分布参数 (如 cv:0.10); **未配置的组件固定为断面基线 (方案A: 配置即抽样, 未配置即固定)**:
   - 全局参数: name 列 = scenario/n_samples/seed/start_time/end_time, value 列 = 参数值
   - 组件分布: name 列 = 组件名, value 列 = 分布名, 其后每格一个 '参数名:值'
   - dist: truncated_normal (cv/lo/hi, μ=组件基线) | uniform (lo/hi)
   - 组件名: 固定负荷名 / 光伏名 / 储能名 / {可调度名}_cur|_lb|_ub

用法:
  python training_dataset_mc.py model_storage_bus18 --n 500 --seed 42
  python training_dataset_mc.py --config training_dataset_mc_config.csv
  python training_dataset_mc.py model_x --config training_dataset_mc_config.csv --n 200   # CLI 覆盖配置

输出 (格式见 training_dataset/输出控制.md 第二部分, 输出根目录 training_dataset/training_dataset_{model}/):
  - logs/{编号}_{model}.txt   每样本一个 txt 日志
  - training_dataset_sample.csv                   抽样场景表 (每样本一行)
  - system/buses/lines/loads/pvs/storage.csv   组件结果表
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import math
import os
import shutil
import time

import numpy as np

from opf_model import build_and_solve_opf, print_result_summary
from save_results import save_sampled_results
from load_network import load_network, resolve_model_path
from sampling import sample, truncated_normal_vec, resolve_mu_sigma


# =====================================================================
# 全局控制 CSV 解析
# =====================================================================

# 全局参数键 (value 列为参数值); 其余 name 行为组件 (value 列为分布名)
# 输入模型由 model 键指定 (data/csv_case33/{model}/); 兼容旧键 scenario
# 抽样参数 (cv/sigma 等) 均在各组件配置行内单独设置, 无全局 cv
GLOBAL_KEYS = {"model", "scenario", "n_samples", "seed", "start_time", "end_time"}


def load_mc_config(path: str):
    """读取全局控制 CSV → (全局参数 dict, 组件配置 {组件名: (分布名, 参数字典)})

    CSV 结构 (表头只需前两列 name,value; 第 3 列起为参数列, 每格 '参数名:值'):
      name,value
      model,model_storage_bus18
      n_samples,200
      seed,42
      start_time,0:00
      end_time,23:45
      Load1,truncated_normal,,,cv:0.10
      PV_Bus6,truncated_normal,lo:0.0,hi:1.0,cv:0.05
      BESS_Bus18,uniform,lo:0.5,hi:4.5
    - name 为全局参数名 → value 列为参数值 (无全局 cv; 抽样参数在组件行内单独设置)
    - name 为组件名     → value 列为分布名, 其后每格一个 '参数名:值'
      (参数按键名读取, 不依赖列名/表头, 不同分布可携带不同参数)
    """
    global_params, comps = {}, {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            cells = [c.strip() for c in row]
            if not cells or not cells[0]:
                continue                       # 空行
            if cells[0].lower() == "name":
                continue                       # 表头行
            name = cells[0]
            val = cells[1] if len(cells) > 1 else ""
            if name in GLOBAL_KEYS:
                global_params[name] = val
                continue
            if not val:
                raise ValueError(f"组件 {name} 未指定分布名 (value 列)")
            params = {}
            for cell in cells[2:]:
                if not cell:
                    continue
                if ":" not in cell:
                    raise ValueError(f"分布参数格式错误: {cell!r} (应为 参数名:值)")
                k, v = cell.split(":", 1)
                k, v = k.strip(), v.strip()
                try:
                    params[k] = float(v)        # 数字参数 (cv:0.10 / sigma:0.05)
                except ValueError:
                    params[k] = v               # shape 名 (mu:<曲线> / sigma:<曲线>, 96 点)
            comps[name] = (val, params)
    return global_params, comps


def _make_sampler(comp_cfg: dict, name: str, base_mu, default_lo, default_hi,
                  shapes: dict = None, t: int = 0):
    """按组件配置构建标量抽样器; 未配置返回 None (走默认回退逻辑)。

    截断正态: cv → σ=cv×μ(基线); 实验分支: mu:<shape>/sigma:<shape> 取槽 t 曲线值;
    mu/lo/hi 缺省由组件基线/默认边界补齐。
    """
    cfg = comp_cfg.get(name)
    if cfg is None:
        return None
    dist, params = cfg
    p = dict(params)
    if dist == "truncated_normal":
        if shapes is not None:
            # 实验分支: 从 shape 曲线按槽解析 μ/σ (兼容数字 mu/sigma 与 cv)
            p["mu"], p["sigma"] = resolve_mu_sigma(cfg, shapes, t, base_mu)
            p.pop("cv", None)
        else:
            if "cv" in p:
                # σ = cv×μ, μ 为 0 时用极小值兜底 (避免 σ=0 使拒绝采样死循环)
                p["sigma"] = p.pop("cv") * max(base_mu, 1e-6)
            p.setdefault("mu", base_mu)
        # lo/hi 一律取调用方默认边界, 配置中的 lo:/hi: 不再起作用
        p["lo"] = default_lo
        p["hi"] = default_hi
    elif dist == "uniform":
        p["lo"] = default_lo
        p["hi"] = default_hi
    return lambda rng: sample(dist, p, rng)


def _validate_components(comp_cfg: dict, fixed_loads, pv_list, st_list, dispatchable_loads):
    """配置了但程序不认识的组件 → 报错"""
    known = set()
    known |= {ld.name for ld in fixed_loads}
    known |= {pv.name for pv in pv_list}
    known |= {st.name for st in st_list}
    known |= {f"{ld.name}_{suf}" for ld in dispatchable_loads for suf in ("cur", "lb", "ub")}
    unknown = set(comp_cfg) - known
    if unknown:
        raise ValueError(f"配置了未知组件 (不在网络中): {sorted(unknown)}")


# =====================================================================
# 多时段断面辅助
# =====================================================================

def _parse_time(s, default="0:00"):
    """时间 'H:MM' → (h, m)"""
    s = (s or "").strip()
    try:
        h, m = s.split(":")
        return int(h), int(m)
    except (ValueError, AttributeError):
        h, m = default.split(":")
        return int(h), int(m)


def _slots(start, end):
    """起止时间 → [(h, m, t)] 断面列表 (15min 步进, 含端点; 0:00~23:45 共 96 点)"""
    h0, m0 = _parse_time(start, "0:00")
    h1, m1 = _parse_time(end, "23:45")
    t0 = max(0, min(h0 * 4 + m0 // 15, 95))
    t1 = max(0, min(h1 * 4 + m1 // 15, 95))
    return [(h, rem * 15, t) for t in range(t0, t1 + 1)
            for h, rem in (divmod(t, 4),)]


def _slot_label(h, m):
    """断面标签 'H:MM' (CSV 列显示)"""
    return f"{h}:{m:02d}"


def _slot_prefix(h, m):
    """日志文件前缀 'HH_MM'"""
    return f"{h:02d}_{m:02d}"


def _apply_slot(network, t):
    """将网络组件当前值设为第 t 个断面的曲线值 (mult[t]), 重算派生量与可调区间 clamp:
    - 负荷: base_ratio = 主曲线[t]; mult_lb/ub = 绑定曲线[t] (常数不变); lb/ub 相对 cur clamp
    - 光伏: irradiance = 曲线[t]; p_avail = p_max × irradiance
    - 储能: energy_ratio = 曲线[t]; 能量窗口 = 额定 × ratio
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


# =====================================================================
# 统计辅助
# =====================================================================

def summarize(values):
    """返回 (min, p5, mean, p50, p95, max, std)"""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return [float("nan")] * 7
    return [
        arr.min(),
        np.percentile(arr, 5),
        arr.mean(),
        np.percentile(arr, 50),
        np.percentile(arr, 95),
        arr.max(),
        arr.std(),
    ]


def print_stats(title, rows):
    """打印指标分布统计表"""
    print(f"\n  指标分布 ({title}):")
    print(f"    {'':<12}{'min':>10}{'p5':>10}{'mean':>10}{'p50':>10}{'p95':>10}{'max':>10}{'std':>10}")
    for name, vals in rows:
        s = summarize(vals)
        print(f"    {name:<12}" + "".join(f"{v:>10.4f}" for v in s))


def _loss_pct(result):
    """网损百分比 (目标为 0 或不可行时返回 nan)"""
    obj = result["objective"]
    pl = result["_p_loss_pu"]
    if math.isfinite(obj) and obj != 0.0 and math.isfinite(pl):
        return pl / obj * 100.0
    return float("nan")


# =====================================================================
# 蒙特卡洛主流程
# =====================================================================

def run_mc(scenario: str, n_samples: int, seed: int, senses: list,
           comp_cfg: dict = None, model: str = None,
           start_time: str = "0:00", end_time: str = "23:45"):
    """对指定场景执行多时段蒙特卡洛抽样并统计 OPF 结果分布 (可同时 min/max)

    scenario: 场景名 (输出目录 training_dataset/training_dataset_{scenario}/)
    model:    网络模型目录名 (data/csv_case33/{model}/); 缺省用 scenario 名。

    comp_cfg: {组件名: (分布名, 参数字典)}; 每个抽样量在配置行内单独设置分布参数
              (如 cv:0.10), 未列出的组件固定为断面基线 (方案A: 配置即抽样, 未配置即固定)。
    start_time/end_time: 抽样断面范围 ('H:MM', 15min 步进, 含端点), 默认 0:00 ~ 23:45。
    """
    comp_cfg = comp_cfg or {}
    model_path = resolve_model_path(model or scenario)   # CSV 场景目录优先, DSS 退回
    print(f"\n加载基线场景: {model_path}")
    network = load_network(model_path)

    fixed_loads = [ld for ld in network.loads.values() if not ld.dispatchable]
    dispatchable_loads = [ld for ld in network.loads.values() if ld.dispatchable]
    print(f"  固定负荷 {len(fixed_loads)} 个 + 可调度负荷 {len(dispatchable_loads)} 个 (均参与抽样)")
    if not fixed_loads:
        raise RuntimeError("该场景没有固定负荷")

    base_mva = network.base_mva
    pv_list = list(network.pvs.values())
    _validate_components(comp_cfg, fixed_loads, pv_list, list(network.storages.values()),
                         dispatchable_loads)

    slots = _slots(start_time, end_time)
    print(f"  多时段断面: {_slot_label(*slots[0][:2])} ~ {_slot_label(*slots[-1][:2])} "
          f"共 {len(slots)} 个 (15min/点), N={n_samples}/断面")

    rng = np.random.default_rng(seed)
    sense_label = {s: ("最小化根节点注入" if s == "min" else "最大化根节点注入") for s in senses}
    print(f"  蒙特卡洛抽样: 种子={seed}, senses={senses}")

    # 输出目录: training_dataset/training_dataset_{scenario}/
    # (清空 csv/ 与 logs/ 子目录, 保留场景根如 training_dataset_mc_config.csv)
    out_dir = os.path.join("training_dataset", f"training_dataset_{scenario}")
    csv_dir = os.path.join(out_dir, "csv")
    logs_dir = os.path.join(out_dir, "logs")
    shutil.rmtree(csv_dir, ignore_errors=True)
    shutil.rmtree(logs_dir, ignore_errors=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    sample_rows = []   # training_dataset_sample.csv 行 (每 断面×样本 一行)
    records = []       # (time_slot, sample_id, sense, result, var_values, ctx)
    t0 = time.time()
    total = len(slots) * n_samples
    done = 0
    step = max(1, total // 10)
    for h, m, t in slots:
        # 当前断面曲线值 → 组件 (base_ratio/lb/ub/irradiance/energy_ratio) + 可调区间 clamp
        _apply_slot(network, t)
        slot_label = _slot_label(h, m)
        slot_prefix = _slot_prefix(h, m)

        # ---- 该断面抽样基线 ----
        mu_pu = np.array([ld.p_cur_pu for ld in fixed_loads], dtype=float)
        ratio_q = np.array([ld.q_cur_pu / max(ld.p_cur_pu, 1e-9) for ld in fixed_loads], dtype=float)
        pv_irr0 = np.array([pv.irradiance for pv in pv_list], dtype=float)
        disp_info = [{"ld": ld, "cur0": ld.base_ratio, "lb0": ld.mult_lb, "ub0": ld.mult_ub}
                     for ld in dispatchable_loads]

        # ---- 逐组件抽样器 (配置优先, 每断面按当前值重建; 实验分支: mu/sigma 取 shape 曲线槽值) ----
        fixed_samplers = [
            _make_sampler(comp_cfg, ld.name, base_mu=ld.base_ratio,   # 按曲线单位(挂载比例)抽样
                          default_lo=0.0, default_hi=2.0 * ld.base_ratio,
                          shapes=network.shapes, t=t)
            for ld in fixed_loads
        ]
        pv_samplers = {
            pv.name: _make_sampler(comp_cfg, pv.name, base_mu=pv.irradiance,
                                   default_lo=0.0, default_hi=1.0,
                                   shapes=network.shapes, t=t)
            for pv in pv_list
        }
        st_samplers = {}
        for st in network.storages.values():
            st_samplers[st.name] = _make_sampler(
                comp_cfg, st.name, base_mu=st.energy_ratio,   # 按曲线单位(能量比例)抽样
                default_lo=st.energy_lb_ratio if st.energy_lb_ratio is not None else 0.1,
                default_hi=st.energy_ub_ratio if st.energy_ub_ratio is not None else 0.9,
                shapes=network.shapes, t=t)
        disp_samplers = {}
        for info in disp_info:
            ld = info["ld"]
            disp_samplers[f"{ld.name}_cur"] = _make_sampler(comp_cfg, f"{ld.name}_cur", base_mu=ld.base_ratio,
                                                            default_lo=ld.mult_lb, default_hi=ld.mult_ub,
                                                            shapes=network.shapes, t=t)
            disp_samplers[f"{ld.name}_lb"] = _make_sampler(comp_cfg, f"{ld.name}_lb", base_mu=ld.mult_lb,
                                                           default_lo=0.0, default_hi=ld.mult_ub,
                                                           shapes=network.shapes, t=t)
            disp_samplers[f"{ld.name}_ub"] = _make_sampler(comp_cfg, f"{ld.name}_ub", base_mu=ld.mult_ub,
                                                           default_lo=ld.mult_lb, default_hi=1.0,
                                                           shapes=network.shapes, t=t)

        for i in range(n_samples):
            # 1) 固定负荷: 配置组件抽样, 未配置组件固定为断面基线 (方案A); 按曲线单位(挂载比例)抽样
            for j, (ld, qr) in enumerate(zip(fixed_loads, ratio_q)):
                s = fixed_samplers[j]
                if s is not None:
                    r = s(rng)                          # 抽样值 = 挂载比例 (曲线单位)
                else:
                    r = mu_pu[j] / max(ld.p_pu, 1e-9)   # 未配置: 固定为断面比例
                ld.p_cur_pu = ld.p_pu * r
                ld.q_cur_pu = ld.p_cur_pu * qr

            # 2) 可调度负荷: 配置组件抽样 (lb ≤ cur ≤ ub), 未配置组件固定为断面基线 (方案A)
            for info in disp_info:
                ld = info["ld"]
                s_ub = disp_samplers[f"{ld.name}_ub"]
                s_lb = disp_samplers[f"{ld.name}_lb"]
                s_cur = disp_samplers[f"{ld.name}_cur"]
                if s_ub is not None:
                    ub = s_ub(rng)
                else:
                    ub = info["ub0"]           # 未配置: 固定
                if s_lb is not None:
                    lb = s_lb(rng)
                else:
                    lb = info["lb0"]
                if s_cur is not None:
                    cur = s_cur(rng)
                else:
                    cur = info["cur0"]
                # 配置值可能破坏顺序约束, 夹取保护: lb ≤ cur ≤ ub
                lb = min(lb, ub)
                cur = min(max(cur, lb), ub)
                ld.base_ratio = cur
                ld.mult_lb = lb
                ld.mult_ub = ub
                ld.p_cur_pu = ld.p_pu * cur
                ld.q_cur_pu = ld.q_pu * cur
                ld.z_lb = lb / cur if cur > 0 else 0.0
                ld.z_ub = ub / cur if cur > 0 else 1e6

            # 3) 光伏辐照度: 配置组件抽样, 未配置组件固定为断面基线 (方案A)
            for pv, irr_mu in zip(pv_list, pv_irr0):
                s = pv_samplers[pv.name]
                if s is not None:
                    irr = s(rng)
                else:
                    irr = irr_mu               # 未配置: 固定
                pv.irradiance = irr
                pv.p_avail_pu = pv.p_max_pu * irr

            # 4) 储能初始状态: 配置组件抽样, 未配置组件固定为曲线能量窗口上限 (方案A); 按曲线单位(能量比例)抽样
            for st in network.storages.values():
                s = st_samplers[st.name]
                if s is not None:
                    en = s(rng)                          # 抽样值 = 能量比例 (曲线单位)
                    st.energy_ratio = en
                    st.energy_ub_cur_pu = st.energy_ub_pu * en
                    st.energy_lb_cur_pu = st.energy_lb_pu * en
                    st.energy_init_cur_pu = st.energy_ub_cur_pu   # 初始能量 = 窗口上限
                    # 初始功率: 按 model_storage.csv 出力上限均匀抽样 p ∈ [-P_dis_max, +P_ch_max]
                    st.p_init_cur_pu = rng.uniform(-st.discharge_ub_pu, st.charge_ub_pu)
                else:
                    st.energy_init_cur_pu = st.energy_ub_cur_pu   # 未配置: 固定为曲线窗口上限
                    st.p_init_cur_pu = 0.0
                st.state_of_charge = st.energy_init_cur_pu / st.energy_ub_cur_pu if st.energy_ub_cur_pu > 0 else 0.5

            # ---------- 抽样值记录 (training_dataset_sample.csv) ----------
            fixed_total_mw = sum(ld.p_cur_pu for ld in fixed_loads) * base_mva
            rec = {"sample_id": i + 1, "time_slot": slot_label}
            for ld in fixed_loads:
                rec[f"{ld.name}_mw"] = ld.p_cur_pu * base_mva
            rec["fixed_total_mw"] = fixed_total_mw
            for pv in pv_list:
                rec[f"{pv.name}_irr"] = pv.irradiance
            for st in network.storages.values():
                rec[f"{st.name}_se_init_mwh"] = st.energy_init_cur_pu * base_mva
                rec[f"{st.name}_p_init_mw"] = st.p_init_cur_pu * base_mva
            for info in disp_info:
                ld = info["ld"]
                rec[f"{ld.name}_cur"] = ld.base_ratio
                rec[f"{ld.name}_lb"] = ld.mult_lb
                rec[f"{ld.name}_ub"] = ld.mult_ub
            sample_rows.append(rec)

            # 抽样上下文 (组件表 loads/pvs/buses 的每样本计算基准)
            ctx = {
                "base_ratio": {ld.name: ld.base_ratio for ld in network.loads.values()},
                "p_cur_pu": {ld.name: ld.p_cur_pu for ld in network.loads.values()},
                "q_cur_pu": {ld.name: ld.q_cur_pu for ld in network.loads.values()},
                "p_avail_pu": {pv.name: pv.p_avail_pu for pv in pv_list},
            }

            # ---------- 每个 sense 求解 ----------
            sample_results = []
            for sense in senses:
                result, var_values = build_and_solve_opf(network, sense, verbose=False)
                records.append((slot_label, i + 1, sense, result, var_values, ctx))
                sample_results.append((sense, result, var_values))

            # ---------- txt 日志 (每 断面×样本 一个, 文件名带断面前缀) ----------
            buf = io.StringIO()
            buf.write(f"Sample {i + 1} / {scenario} @ {slot_label}\n")
            buf.write("=" * 50 + "\n")
            buf.write("抽样值:\n")
            buf.write(f"  固定负荷总功率: {fixed_total_mw:.4f} MW\n")
            for pv in pv_list:
                buf.write(f"  {pv.name}_irr: {pv.irradiance:.6f}\n")
            for st in network.storages.values():
                buf.write(f"  {st.name}_se_init_mwh: {st.energy_init_cur_pu * base_mva:.6f}\n")
                buf.write(f"  {st.name}_p_init_mw: {st.p_init_cur_pu * base_mva:.6f}\n")
            for info in disp_info:
                ld = info["ld"]
                buf.write(f"  {ld.name}: cur={ld.base_ratio:.4f}, lb={ld.mult_lb:.4f}, ub={ld.mult_ub:.4f}\n")
            for sense, result, var_values in sample_results:
                buf.write(f"\n{'=' * 50}\nsense: {sense}\n")
                with contextlib.redirect_stdout(buf):
                    print_result_summary(result, network, var_values)
            log_name = f"{slot_prefix}_{i + 1:04d}_{scenario}.txt"
            with open(os.path.join(logs_dir, log_name), "w", encoding="utf-8") as f:
                f.write(buf.getvalue())

            done += 1
            if done % step == 0 or done == total:
                print(f"    {done}/{total} (断面 {slot_label} 样本 {i + 1}) 完成 ({time.time() - t0:.1f}s)")

    # ---------- 统计 (per sense, 仅 OPTIMAL 样本) ----------
    for sense in senses:
        sense_rows = [(sid, s, res, var) for (ts, sid, s, res, var, _) in records if s == sense]
        ok = [r for r in sense_rows if r[2]["termination_status"] == "OPTIMAL"]
        print(f"\n  [{sense_label[sense]}] 有效 (OPTIMAL) 样本: {len(ok)}/{len(sense_rows)}")
        if ok:
            stat_rows = [
                ("p_sub(MW)", [r[2]["objective"] * base_mva for r in ok]),
                ("P_loss(MW)", [r[2]["_p_loss_pu"] * base_mva for r in ok]),
                ("Q_loss(Mvar)", [r[2]["_q_loss_pu"] * base_mva for r in ok]),
                ("loss(%)", [_loss_pct(r[2]) for r in ok]),
            ]
            for info in disp_info:
                name = info["ld"].name
                stat_rows.append((f"z_{name}", [r[3]["z_demand"][name] for r in ok]))
            print_stats(sense_label[sense], stat_rows)

    # ---------- 输出 training_dataset_sample.csv (抽样场景表, sample_id 后为 time_slot) ----------
    fieldnames = ["sample_id", "time_slot"]
    fieldnames += [f"{ld.name}_mw" for ld in fixed_loads]
    fieldnames += ["fixed_total_mw"]
    fieldnames += [f"{pv.name}_irr" for pv in pv_list]
    fieldnames += [f"{st.name}_se_init_mwh" for st in network.storages.values()]
    fieldnames += [f"{st.name}_p_init_mw" for st in network.storages.values()]
    for info in disp_info:
        name = info["ld"].name
        fieldnames += [f"{name}_cur", f"{name}_lb", f"{name}_ub"]
    sample_path = os.path.join(csv_dir, "training_dataset_sample.csv")
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(sample_rows)
    print(f"  已保存: {sample_path}")

    # ---------- 输出组件结果表 ----------
    save_sampled_results(network, records, csv_dir)
    print(f"  组件表已保存至: {csv_dir}")
    print(f"  txt 日志已保存至: {logs_dir}")

    return records


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="负荷/光伏/储能蒙特卡洛抽样 + OPF 分布统计")
    parser.add_argument("scenario", nargs="?", help="场景名 (缺省从 --config 读取)")
    parser.add_argument("--config", default=None, help="全局控制 CSV 路径 (training_dataset_mc_config.csv)")
    parser.add_argument("--n", type=int, default=None, help="抽样次数 (覆盖配置文件)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖配置文件)")
    parser.add_argument("--sense", default="both", choices=["min", "max", "both"],
                        help="目标函数方向 (默认 both)")
    args = parser.parse_args()

    # 读取全局控制 CSV: --config 显式指定, 否则默认 training_dataset/training_dataset_{scenario}/training_dataset_mc_config.csv
    cfg_global, comp_cfg = {}, {}
    cfg_path = args.config
    if cfg_path is None and args.scenario:
        default_cfg = os.path.join("training_dataset", f"training_dataset_{args.scenario}", "training_dataset_mc_config.csv")
        if os.path.exists(default_cfg):
            cfg_path = default_cfg
    if cfg_path:
        cfg_global, comp_cfg = load_mc_config(cfg_path)
        print(f"读取全局控制: {cfg_path} "
              f"(全局参数 {len(cfg_global)} 项, 组件配置 {len(comp_cfg)} 个)")

    # 场景名与网络模型分离:
    #   scenario 键 = 场景名 (输出 training_dataset/training_dataset_{scenario}/, 如 default);
    #   model 键    = 网络模型目录名 (data/csv_case33/{model}/, 如 model_default)。
    scenario = args.scenario or cfg_global.get("scenario") or cfg_global.get("model")
    if not scenario:
        raise ValueError("未指定模型/场景: 请在命令行传模型名或配置文件提供 model")
    model = cfg_global.get("model") or scenario
    n_samples = args.n if args.n is not None else int(cfg_global.get("n_samples", 200))
    seed = args.seed if args.seed is not None else int(cfg_global.get("seed", 42))
    start_time = cfg_global.get("start_time", "0:00")
    end_time = cfg_global.get("end_time", "23:45")
    senses = ["min", "max"] if args.sense == "both" else [args.sense]

    run_mc(scenario, n_samples, seed, senses, comp_cfg, model=model,
           start_time=start_time, end_time=end_time)


if __name__ == "__main__":
    main()

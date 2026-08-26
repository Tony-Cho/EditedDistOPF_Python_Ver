# -*- coding: utf-8 -*-
"""
gen_dataset.py — 用合成整日轨迹生成 train/val/test 三套特征-真值数据集

数据来源：gen_synthetic.sample_day（日级因子 + 共同因子 + AR(1) 平滑噪声 + 钟形可控方差），
每"日"一条 96 点整日轨迹，天然保留时段相关性，且 train/val/test 同源（同一生成器不同随机日）→ 严格匹配。

流程：对每个"日" × 每个槽 t：
  1. 由该日曲线设定网络状态（负荷 cur/lb/ub、PV 辐照、储能能量窗口）；
  2. 求解 OPF min/max 真值；
  3. 记录特征特征（固定负荷 mw、PV irr、储能 se_init/p_init、可调度 cur/lb/ub）与真值
     （p_sub/q_sub、储能/PV/可调度出力）。

输出（每 split 一个目录，格式与 training_dataset_mc.py / train_knn.py 一致）：
  {out}/{split}/training_dataset_{sample,system,storage,pvs,loads}.csv

用法：
  python gen_dataset.py --train 120 --val 40 --test 40 --seed 42
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np

import gen_synthetic as g
from load_network import load_network
from opf_model import build_and_solve_opf

TIMES = g.TIMES  # 96 个时间标签


@contextlib.contextmanager
def _silence_fd1():
    """临时把 fd1(stdout) 重定向到 devnull，抑制 Gurobi 每次 solve 的 'Set parameter' 刷屏。"""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)
        sys.stdout.flush()


def set_slot(net, day, t):
    """将第 t 槽网络状态设为 day 曲线的值（负荷 cur/lb/ub、PV irr、储能窗口）"""
    for ld in net.loads.values():
        cur = float(day.get(f"{ld.name}_cur", np.full(g.NPT, 1.0))[t])
        lb = float(day.get(f"{ld.name}_lb", np.full(g.NPT, 0.0))[t])
        ub = float(day.get(f"{ld.name}_ub", np.full(g.NPT, 1.0))[t])
        if ld.dispatchable:
            if lb > cur:
                lb = cur
            if ub < cur:
                ub = cur
        ld.base_ratio = cur
        ld.mult_lb, ld.mult_ub = lb, ub
        ld.p_cur_pu = ld.p_pu * cur
        ld.q_cur_pu = ld.q_pu * cur
        ld.z_lb = lb / cur if cur > 0 else 0.0
        ld.z_ub = ub / cur if cur > 0 else 1e6
    for pv in net.pvs.values():
        pv.irradiance = float(day.get(f"{pv.name}_irr", np.ones(g.NPT))[t])
        pv.p_avail_pu = pv.p_max_pu * pv.irradiance
    for st in net.storages.values():
        en = float(day.get(f"{st.name}_en", np.full(g.NPT, 1.0))[t])
        st.energy_ratio = en
        st.energy_ub_cur_pu = st.energy_ub_pu * en
        st.energy_lb_cur_pu = st.energy_lb_pu * en
        st.energy_init_cur_pu = st.energy_ub_cur_pu   # 初始能量 = 窗口上限


def sample_slot_t(day, rng, base_shapes, net, p, t):
    """【slot 模式】按"每个时间断面独立"思路抽样槽 t 的值。

    各分量围绕各自 shape[t] 作正态抽样（cv=slot_cv），就地覆盖 day[name][t]
    （day 为 base_shapes 的副本，其余槽保持基准值）；可调度 lb/cur/ub 逐槽保序。
    无任何时段相关性 —— 与 gen_synthetic.sample_day（因子法）互斥的两套抽样。
    """
    cv = p.get("slot_cv", 0.10)
    for name, base in base_shapes.items():
        mu = base[t]
        v = rng.normal(mu, cv * max(mu, 1e-6))
        if name.endswith("_lb"):
            day[name][t] = max(v, 0.02)
        elif name.endswith("_ub"):
            day[name][t] = min(max(v, 0.02), 1.4)
        elif name.endswith("_irr") or name.endswith("_en"):
            day[name][t] = min(max(v, 0.0), 1.0)
        else:  # *_cur（固定/可调度当前挂载）
            day[name][t] = min(max(v, 0.02), p["hi_mult"])
    for ld in net.loads.values():
        if not ld.dispatchable:
            continue
        cur, lb, ub = (day[f"{ld.name}_cur"], day[f"{ld.name}_lb"], day[f"{ld.name}_ub"])
        lb[t] = min(lb[t], ub[t])
        cur[t] = min(max(cur[t], lb[t]), ub[t])
        ub[t] = max(ub[t], cur[t])


def build_schema(net):
    """返回表头列序与各列的槽取值函数（与 training_dataset_mc.py 对齐）"""
    base_mva = net.base_mva
    fixed = [ld for ld in net.loads.values() if not ld.dispatchable]
    disp = [ld for ld in net.loads.values() if ld.dispatchable]
    pvs = list(net.pvs.values())
    sts = list(net.storages.values())

    load_cols = [f"{ld.name}_mw" for ld in fixed]
    pv_cols = [f"{pv.name}_irr" for pv in pvs]
    st_cols = [f"{st.name}_se_init_mwh" for st in sts] + [f"{st.name}_p_init_mw" for st in sts]
    disp_cols = [c for ld in disp for c in (f"{ld.name}_cur", f"{ld.name}_lb", f"{ld.name}_ub")]

    sample_header = (["sample_id", "time_slot"] + load_cols + ["fixed_total_mw"]
                     + pv_cols + st_cols + disp_cols)

    def sample_row(sid, ts, net_vals):
        # net_vals: 快照（固定负荷 p_cur_pu / pv irr / 储能 se_init / 可调度 cur/lb/ub）
        row = [sid, ts]
        load_mw = [net_vals["loads"][ld.name] for ld in fixed]
        row += [f"{v:.6f}" for v in load_mw]
        row.append(f"{sum(load_mw):.6f}")
        row += [f"{net_vals['pv'][pv.name]:.6f}" for pv in pvs]
        for st in sts:
            row.append(f"{net_vals['se_init'][st.name]:.6f}")
        for st in sts:
            row.append("0.000000")  # p_init_mw = 0
        for ld in disp:
            row += [f"{net_vals['cur'][ld.name]:.6f}",
                    f"{net_vals['lb'][ld.name]:.6f}",
                    f"{net_vals['ub'][ld.name]:.6f}"]
        return row

    return fixed, disp, pvs, sts, base_mva, sample_header, sample_row


def gen_split(net, base_shapes, params, n_days, seed, split_name, out_root,
              max_attempts=50, margin_mw=0.10):
    """生成一个 split（训练/验证/测试）并写 CSV。

    可行性门（整日级）：一天内任一槽 t 的 min/max OPF 非 OPTIMAL，或基线总消费
    不落在 [pmin+margin, pmax-margin]（即基线不可行或无可调余量）→ 整日重抽。
    保证写出的所有样本均可行（KNN 训练不再混入不可行/NaN 标签）。
    """
    rng = np.random.default_rng(seed)
    fixed, disp, pvs, sts, base_mva, sample_header, sample_row = build_schema(net)

    os.makedirs(out_root, exist_ok=True)
    def wopen(name):
        return open(os.path.join(out_root, f"training_dataset_{name}.csv"),
                    "w", newline="", encoding="utf-8")

    f_sample = wopen("sample"); f_sys = wopen("system")
    f_st = wopen("storage"); f_pv = wopen("pvs"); f_ld = wopen("loads")
    import csv
    w_sample = csv.writer(f_sample); w_sample.writerow(sample_header)
    w_sys = csv.writer(f_sys); w_sys.writerow(["sample_id", "time_slot", "sense", "p_sub_mw", "q_sub_mvar", "status"])
    w_st = csv.writer(f_st); w_st.writerow(["sample_id", "time_slot", "sense", "name", "bus", "p_net_mw", "q_mvar", "se_mwh"])
    w_pv = csv.writer(f_pv); w_pv.writerow(["sample_id", "time_slot", "sense", "name", "bus", "p_out_mw", "q_out_mvar"])
    w_ld = csv.writer(f_ld); w_ld.writerow(["sample_id", "time_slot", "sense", "name", "bus", "type", "p_out_mw"])

    total = n_days * g.NPT
    done = 0
    n_rejected = 0
    for d in range(n_days):
        day_records = None
        for attempt in range(max_attempts):
            day = g.sample_day(rng, base_shapes, net, params)
            buf = []
            day_ok = True
            for t in range(g.NPT):
                ts = TIMES[t]
                sid = d * g.NPT + (t + 1)  # (日,槽) 复合样本号
                set_slot(net, day, t)

                # 特征快照
                nv = {
                    "loads": {ld.name: ld.p_cur_pu * base_mva for ld in fixed},
                    "pv": {pv.name: pv.irradiance for pv in pvs},
                    "se_init": {st.name: st.energy_init_cur_pu * base_mva for st in sts},
                    "cur": {ld.name: ld.base_ratio for ld in disp},
                    "lb": {ld.name: ld.mult_lb for ld in disp},
                    "ub": {ld.name: ld.mult_ub for ld in disp},
                }
                srow = sample_row(sid, ts, nv)
                disp_p_cur = {ld.name: ld.p_cur_pu for ld in disp}
                base_mw = sum(ld.p_cur_pu for ld in net.loads.values()) * base_mva

                per_sense = []
                for sense in ("min", "max"):
                    with _silence_fd1():
                        result, var_values = build_and_solve_opf(net, sense, verbose=False)
                    if result["termination_status"] in ("OPTIMAL", "TIME_LIMIT_FEASIBLE"):
                        status = "OPTIMAL"
                        p_sub = result["objective"]; q_sub = var_values["q_sub"]
                        st_vals = {st.name: (var_values["p_ch"][st.name], var_values["p_dis"][st.name],
                                             var_values["q_st"][st.name], var_values["se"][st.name])
                                   for st in sts}
                        pv_vals = {pv.name: (var_values["p_pv"][pv.name], var_values["q_pv"][pv.name])
                                   for pv in pvs}
                        zz = {ld.name: var_values["z_demand"][ld.name] for ld in disp}
                    else:
                        status = result["termination_status"]
                        p_sub = q_sub = float("nan")
                        st_vals = {st.name: (float("nan"),) * 4 for st in sts}
                        pv_vals = {pv.name: (float("nan"),) * 2 for pv in pvs}
                        zz = {ld.name: float("nan") for ld in disp}
                    per_sense.append((sense, status, p_sub, q_sub, st_vals, pv_vals, zz))

                # 可行性门: min/max 均 OPTIMAL, 且基线总消费留有可调余量
                if per_sense[0][1] != "OPTIMAL" or per_sense[1][1] != "OPTIMAL":
                    day_ok = False
                    break
                pmin_mw = per_sense[0][2] * base_mva
                pmax_mw = per_sense[1][2] * base_mva
                if not (pmin_mw <= base_mw - margin_mw and pmax_mw >= base_mw + margin_mw):
                    day_ok = False
                    break
                buf.append((sid, ts, srow, per_sense, disp_p_cur))
            if day_ok:
                day_records = buf
                break
        if day_records is None:
            n_rejected += 1
            print(f"    [{split_name}] 日 {d + 1}/{n_days} 重抽 {max_attempts} 次仍不满足可行性门, 已跳过",
                  file=sys.stderr)
            continue
        for sid, ts, srow, per_sense, disp_p_cur in day_records:
            for sense, status, p_sub, q_sub, st_vals, pv_vals, zz in per_sense:
                w_sys.writerow([sid, ts, sense, f"{p_sub * base_mva:.6f}", f"{q_sub * base_mva:.6f}", status])
                for st in sts:
                    p_ch, p_dis, q_st, se = st_vals[st.name]
                    w_st.writerow([sid, ts, sense, st.name, st.bus,
                                   f"{(p_dis - p_ch) * base_mva:.6f}", f"{q_st * base_mva:.6f}", f"{se * base_mva:.6f}"])
                for pv in pvs:
                    p_out, q_out = pv_vals[pv.name]
                    w_pv.writerow([sid, ts, sense, pv.name, pv.bus,
                                   f"{p_out * base_mva:.6f}", f"{q_out * base_mva:.6f}"])
                for ld in disp:
                    p_out = zz[ld.name] * disp_p_cur[ld.name] * base_mva
                    w_ld.writerow([sid, ts, sense, ld.name, ld.bus, ld.type, f"{p_out:.6f}"])
            w_sample.writerow(srow)
            done += 1
            if done % (total // 10 or 1) == 0 or done == total:
                print(f"    [{split_name}] {done}/{total} (样本 {sid})", file=sys.stderr)
    for f in (f_sample, f_sys, f_st, f_pv, f_ld):
        f.close()
    if n_rejected:
        print(f"[{split_name}] 跳过不可行日 {n_rejected}/{n_days} (可行性门)", file=sys.stderr)
    print(f"[{split_name}] 完成 {n_days} 天 × {g.NPT} 槽 = {total} 样本 → {out_root}")


def gen_split_slot(net, base_shapes, params, n_days, seed, split_name, out_root,
                   max_attempts=50, margin_mw=0.10):
    """生成一个 split：每个时间断面独立正态抽样（围绕各自 shape），单槽重抽门。

    与 gen_split（整日轨迹 + 整日门）的区别：
      - 抽样：sample_slot_t 逐槽独立（无时段相关），各分量 N(shape[t], cv×shape[t])；
      - 可行性门：槽级 —— 某槽不可行只重抽该槽，不丢弃其他槽。
    输出格式与 gen_split 完全一致。
    """
    rng = np.random.default_rng(seed)
    fixed, disp, pvs, sts, base_mva, sample_header, sample_row = build_schema(net)

    os.makedirs(out_root, exist_ok=True)
    def wopen(name):
        return open(os.path.join(out_root, f"training_dataset_{name}.csv"),
                    "w", newline="", encoding="utf-8")

    f_sample = wopen("sample"); f_sys = wopen("system")
    f_st = wopen("storage"); f_pv = wopen("pvs"); f_ld = wopen("loads")
    import csv
    w_sample = csv.writer(f_sample); w_sample.writerow(sample_header)
    w_sys = csv.writer(f_sys); w_sys.writerow(["sample_id", "time_slot", "sense", "p_sub_mw", "q_sub_mvar", "status"])
    w_st = csv.writer(f_st); w_st.writerow(["sample_id", "time_slot", "sense", "name", "bus", "p_net_mw", "q_mvar", "se_mwh"])
    w_pv = csv.writer(f_pv); w_pv.writerow(["sample_id", "time_slot", "sense", "name", "bus", "p_out_mw", "q_out_mvar"])
    w_ld = csv.writer(f_ld); w_ld.writerow(["sample_id", "time_slot", "sense", "name", "bus", "type", "p_out_mw"])

    total = n_days * g.NPT
    done = 0
    n_rejected = 0
    for d in range(n_days):
        day = {name: base.copy() for name, base in base_shapes.items()}  # 基准副本, 仅覆盖 [t]
        for t in range(g.NPT):
            ts = TIMES[t]
            sid = d * g.NPT + (t + 1)  # (日,槽) 复合样本号
            accepted = False
            for attempt in range(max_attempts):
                sample_slot_t(day, rng, base_shapes, net, params, t)
                set_slot(net, day, t)

                nv = {
                    "loads": {ld.name: ld.p_cur_pu * base_mva for ld in fixed},
                    "pv": {pv.name: pv.irradiance for pv in pvs},
                    "se_init": {st.name: st.energy_init_cur_pu * base_mva for st in sts},
                    "cur": {ld.name: ld.base_ratio for ld in disp},
                    "lb": {ld.name: ld.mult_lb for ld in disp},
                    "ub": {ld.name: ld.mult_ub for ld in disp},
                }
                srow = sample_row(sid, ts, nv)
                disp_p_cur = {ld.name: ld.p_cur_pu for ld in disp}
                base_mw = sum(ld.p_cur_pu for ld in net.loads.values()) * base_mva

                per_sense = []
                for sense in ("min", "max"):
                    with _silence_fd1():
                        result, var_values = build_and_solve_opf(net, sense, verbose=False)
                    if result["termination_status"] in ("OPTIMAL", "TIME_LIMIT_FEASIBLE"):
                        status = "OPTIMAL"
                        p_sub = result["objective"]; q_sub = var_values["q_sub"]
                        st_vals = {st.name: (var_values["p_ch"][st.name], var_values["p_dis"][st.name],
                                             var_values["q_st"][st.name], var_values["se"][st.name])
                                   for st in sts}
                        pv_vals = {pv.name: (var_values["p_pv"][pv.name], var_values["q_pv"][pv.name])
                                   for pv in pvs}
                        zz = {ld.name: var_values["z_demand"][ld.name] for ld in disp}
                    else:
                        status = result["termination_status"]
                        p_sub = q_sub = float("nan")
                        st_vals = {st.name: (float("nan"),) * 4 for st in sts}
                        pv_vals = {pv.name: (float("nan"),) * 2 for pv in pvs}
                        zz = {ld.name: float("nan") for ld in disp}
                    per_sense.append((sense, status, p_sub, q_sub, st_vals, pv_vals, zz))

                # 槽级可行性门: min/max 均 OPTIMAL 且基线留有可调余量
                if per_sense[0][1] != "OPTIMAL" or per_sense[1][1] != "OPTIMAL":
                    continue
                pmin_mw = per_sense[0][2] * base_mva
                pmax_mw = per_sense[1][2] * base_mva
                if pmin_mw <= base_mw - margin_mw and pmax_mw >= base_mw + margin_mw:
                    accepted = True
                    break
            if not accepted:
                n_rejected += 1
                print(f"    [{split_name}] 样本 {sid} ({ts}) 重抽 {max_attempts} 次仍不可行, 跳过",
                      file=sys.stderr)
                continue
            for sense, status, p_sub, q_sub, st_vals, pv_vals, zz in per_sense:
                w_sys.writerow([sid, ts, sense, f"{p_sub * base_mva:.6f}", f"{q_sub * base_mva:.6f}", status])
                for st in sts:
                    p_ch, p_dis, q_st, se = st_vals[st.name]
                    w_st.writerow([sid, ts, sense, st.name, st.bus,
                                   f"{(p_dis - p_ch) * base_mva:.6f}", f"{q_st * base_mva:.6f}", f"{se * base_mva:.6f}"])
                for pv in pvs:
                    p_out, q_out = pv_vals[pv.name]
                    w_pv.writerow([sid, ts, sense, pv.name, pv.bus,
                                   f"{p_out * base_mva:.6f}", f"{q_out * base_mva:.6f}"])
                for ld in disp:
                    p_out = zz[ld.name] * disp_p_cur[ld.name] * base_mva
                    w_ld.writerow([sid, ts, sense, ld.name, ld.bus, ld.type, f"{p_out:.6f}"])
            w_sample.writerow(srow)
            done += 1
            if done % (total // 10 or 1) == 0 or done == total:
                print(f"    [{split_name}] {done}/{total} (样本 {sid})", file=sys.stderr)
    for f in (f_sample, f_sys, f_st, f_pv, f_ld):
        f.close()
    if n_rejected:
        print(f"[{split_name}] 跳过不可行槽 {n_rejected} 个 (可行性门)", file=sys.stderr)
    print(f"[{split_name}] 完成 {n_days} 天 × {g.NPT} 槽 = {total} 样本 → {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/csv_case33/model_default")
    ap.add_argument("--train", type=int, default=120)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="dataset/model_default")
    ap.add_argument("--load-scale", type=float, default=0.65,
                    help="负荷基准曲线缩放（<1 压低负荷，保证生成日落在 OPF 可行域）")
    ap.add_argument("--sample-mode", choices=["day", "slot"], default="day",
                    help="day=整日轨迹(因子法, gen_synthetic.sample_day); slot=每断面独立正态(围绕 shape)")
    ap.add_argument("--slot-cv", type=float, default=0.10,
                    help="slot 模式抽样变异系数（σ = cv × shape 值）")
    args = ap.parse_args()

    net = load_network(args.model)
    base_shapes = g.build_default_shapes(net, load_scale=args.load_scale)
    params = dict(g.DEFAULT_PARAMS)
    params["slot_cv"] = args.slot_cv

    split_fns = (gen_split_slot if args.sample_mode == "slot" else gen_split)
    split_fns(net, base_shapes, params, args.train, args.seed, "train", os.path.join(args.out, "train"))
    split_fns(net, base_shapes, params, args.val, args.seed + 1, "val", os.path.join(args.out, "val"))
    split_fns(net, base_shapes, params, args.test, args.seed + 2, "test", os.path.join(args.out, "test"))


if __name__ == "__main__":
    main()
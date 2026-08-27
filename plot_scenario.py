# -*- coding: utf-8 -*-
"""
plot_scenario.py — 根节点基线功率与上/下调边界绘图

输入:
  - scenario/{scenario}/scenario_shapes.csv   (96 点曲线: Load*_cur / EV*_cur / AC_cur / PV*_irr)
  - scenario/{scenario}/scenario_loads.csv    (负荷基准功率 kw)
  - scenario/{scenario}/scenario_pvs.csv      (光伏容量 pmpp_kw)
  - scenario/{scenario}/scenario_storage.csv  (储能额定功率 kw)
  - output/{scenario}/output_system.csv       (min/max 各 N 样本的 p_sub)

绘制内容 (横轴 96 断面 0:00~23:45, 纵轴根节点功率 MW):
  1. 固定负荷基线: Σ (Load1~32 + Fixed_Bus*) kw/1000 × cur[t] (无曲线按满载)
  2. 可调度负荷基线: Σ EV/AC kw/1000 × cur[t]
  3. 概率边界: 每断面取 output_system 该时段的全部样本 (min+max 共 2N 个) p_sub,
     上界 = 从低到高 10% 分位, 下界 = 从低到高 90% 分位
  4. 理论边界: 同固定负荷基线
     - 下界: EV/AC 全部削去(0) + PV 按 irr 满发 + 储能满功率放电
     - 上界: EV/AC 全部满发 + PV 完全不出力(0) + 储能满功率充电

--mode real 真实模式七条线 (KNN vs OPF 真值, 场景曲线固定仅抽可调度 lb/ub):
  单图绘制七条线 (自上而下): 理论上界 > OPF P10 ≈ KNN P10 > 负荷基线
  > KNN P90 ≈ OPF P90 > 理论下界;
  理论上下界之间、OPF P10~P90 之间、KNN P10~P90 之间均加背景半透明填充。
  读取 output/output_{scenario}/ (KNN) 与 output/output_{scenario}_opf/ (OPF) 的
  output_system.csv, 输出 plot_{scenario}_real.png 及排序校验统计。

输出: output/{scenario}/plot_{scenario}.png  (--mode prob, 默认)
      output/output_{scenario}/plot_{scenario}_real.png  (--mode real)

用法:
  python plot_scenario.py --scenario scenario_trail_1
  python plot_scenario.py --mode real --scenario scenario_default
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 中文字体 (Windows)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 可调度负荷组件 (EV/AC)
DISPATCHABLE = ["EV_Bus19", "EV_Bus7", "AC_Bus2"]
SLOTS = [f"{h}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]


def _to_bool(v, default: bool = True) -> bool:
    """字符串 → bool (与 parse_csv 一致; 缺失/无法识别取默认)"""
    s = (v or "").strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    return default


def _slot_label(t: int) -> str:
    h, m = divmod(t * 15, 60)
    return f"{h}:{m:02d}"


def read_shapes(path: str) -> dict:
    """曲线表 → {曲线名: np.array(96)}"""
    curves = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            name = r["name"].strip()
            vals = [float(v) for k, v in r.items()
                    if k != "name" and v not in (None, "")]
            if vals:
                curves[name] = np.asarray(vals, dtype=float)
    return curves


def read_loads(path: str) -> dict:
    """负荷表 → {负荷名: kw} (仅 enabled=TRUE 的负荷; 缺失 enabled 默认启用)"""
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not _to_bool(r.get("enabled"), True):
                continue
            out[r["name"].strip()] = float(r["kw"])
    return out


def read_pvs(path: str) -> dict:
    """光伏表 → {光伏名: pmpp_kw} (仅 enabled=TRUE; 缺失 enabled 默认启用)"""
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not _to_bool(r.get("enabled"), True):
                continue
            out[r["name"].strip()] = float(r["pmpp_kw"])
    return out


def read_storage(path: str) -> float:
    """储能表 → 满功率充/放功率 (MW), 取第一个 enabled 储能"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if not _to_bool(r.get("enabled"), True):
                continue
            return float(r["kw"]) / 1000.0   # pct_charge/pct_discharge 视为 100%
    return 0.0


def read_output_system(path: str) -> dict:
    """output_system.csv → {time_slot: np.array(p_sub 全部样本)} (按断面数值排序)"""
    by_slot = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            by_slot.setdefault(r["time_slot"].strip(), []).append(float(r["p_sub_mw"]))

    def key(ts):
        h, _, m = ts.partition(":")
        return int(h) * 60 + int(m or 0)
    return {ts: np.asarray(v, dtype=float) for ts, v in
            sorted(by_slot.items(), key=lambda kv: key(kv[0]))}


def read_sys_by_sense(path: str) -> dict:
    """output_system.csv → {(time_slot, sense): np.array(p_sub_mw)} (仅有限值, real 模式用)"""
    by = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = float(r["p_sub_mw"])
            if not np.isfinite(v):
                continue
            by.setdefault((r["time_slot"].strip(), r["sense"].strip()), []).append(v)
    return {k: np.array(v) for k, v in by.items()}


def plot_real_mode(scenario: str, scen_dir: str, knn_dir: str, opf_dir: str,
                   out_path: str) -> None:
    """真实模式七条线 (KNN 预测 vs OPF 真值) 与排序校验。

    上调 (自上而下): theo_hi > OPF P10 ≈ KNN P10 > 基线
    下调 (自上而下): 基线 > OPF P90 ≈ KNN P90 > theo_lo
    """
    shapes = read_shapes(os.path.join(scen_dir, "scenario_shapes.csv"))
    loads = read_loads(os.path.join(scen_dir, "scenario_loads.csv"))
    pvs = read_pvs(os.path.join(scen_dir, "scenario_pvs.csv"))
    p_sto = read_storage(os.path.join(scen_dir, "scenario_storage.csv"))
    print(f"场景 {scenario}: 曲线 {len(shapes)}, 负荷 {len(loads)}, "
          f"光伏 {len(pvs)}, 储能充放功率 {p_sto:.3f} MW")

    fixed = {n: kw / 1000.0 for n, kw in loads.items() if n not in DISPATCHABLE}
    disp = {n: loads.get(n, 0.0) / 1000.0 for n in DISPATCHABLE}
    pv_pmpp = {n: pm / 1000.0 for n, pm in pvs.items()}
    evac_full = sum(disp.values())

    knn = read_sys_by_sense(os.path.join(knn_dir, "output_system.csv"))
    opf = read_sys_by_sense(os.path.join(opf_dir, "output_system.csv"))
    print(f"KNN 输出: {knn_dir}/output_system.csv ({sum(len(v) for v in knn.values())} 行)")
    print(f"OPF 输出: {opf_dir}/output_system.csv ({sum(len(v) for v in opf.values())} 行)")

    c = {"up_base": 0, "up_knn": 0, "up_theo": 0, "dn_base": 0, "dn_knn": 0, "dn_theo": 0}
    n = 96
    base = np.full(n, np.nan); theo_hi = np.full(n, np.nan); theo_lo = np.full(n, np.nan)
    op10 = np.full(n, np.nan); kn10 = np.full(n, np.nan)
    op90 = np.full(n, np.nan); kn90 = np.full(n, np.nan)

    print(f"储能 {p_sto:.3f} MW | 可调满载 {evac_full:.3f} MW")
    for t, ts in enumerate(SLOTS):
        fixed_cur = sum(kw * shapes.get(f"{n0}_cur", np.ones(n))[t] for n0, kw in fixed.items())
        disp_cur = sum(kw * shapes.get(f"{n0}_cur", np.ones(n))[t] for n0, kw in disp.items())
        base[t] = fixed_cur + disp_cur
        theo_hi[t] = fixed_cur + evac_full + p_sto
        pv_full = sum(pm * shapes.get(f"{n0}_irr", np.zeros(n))[t] for n0, pm in pv_pmpp.items())
        theo_lo[t] = fixed_cur - pv_full - p_sto
        op10[t] = np.percentile(opf.get((ts, "max"), [np.nan]), 10)
        kn10[t] = np.percentile(knn.get((ts, "max"), [np.nan]), 10)
        op90[t] = np.percentile(opf.get((ts, "min"), [np.nan]), 90)
        kn90[t] = np.percentile(knn.get((ts, "min"), [np.nan]), 90)

        c["up_base"] += bool(base[t] > kn10[t])
        c["up_knn"] += bool(op10[t] < kn10[t])
        c["up_theo"] += bool(op10[t] >= theo_hi[t])
        c["dn_base"] += bool(base[t] < op90[t])
        c["dn_knn"] += bool(op90[t] > kn90[t])
        c["dn_theo"] += bool(op90[t] < theo_lo[t])

    print(f"\n【上调】 基线>KNN P10 {c['up_base']}/96 | OPF P10<KNN P10 {c['up_knn']}/96 "
          f"(max {np.nanmax(kn10 - op10):.4f} MW) | OPF P10>=theo {c['up_theo']}/96")
    print(f"【下调】 基线<OPF P90 {c['dn_base']}/96 | OPF P90>KNN P90 {c['dn_knn']}/96 "
          f"(max {np.nanmax(op90 - kn90):.4f} MW) | OPF P90<theo_lo {c['dn_theo']}/96")

    t_idx = np.arange(n)
    fig, ax = plt.subplots(figsize=(14, 7))

    # 背景半透明填充: 理论可行区间最宽, OPF/KNN 概率区间叠其上
    ax.fill_between(t_idx, theo_lo, theo_hi, color="gray", alpha=0.12,
                    label="理论可行区间 (P 不受安全约束)")
    ax.fill_between(t_idx, op90, op10, color="tab:red", alpha=0.14,
                    label="OPF 概率区间 (P10~P90)")
    ax.fill_between(t_idx, kn90, kn10, color="tab:orange", alpha=0.14,
                    label="KNN 概率区间 (P10~P90)")

    # 七条线 (自上而下): theo_hi > OPF P10 ≈ KNN P10 > 基线 > KNN P90 ≈ OPF P90 > theo_lo
    ax.plot(t_idx, theo_hi, color="tab:purple", ls=":", lw=2.0, label="理论上界 (不考虑安全约束)")
    ax.plot(t_idx, op10, color="tab:red", ls="--", lw=1.6, label="OPF P10 (上调上界真值)")
    ax.plot(t_idx, kn10, color="tab:orange", ls="--", lw=1.6, label="KNN P10 (上调上界代理)")
    ax.plot(t_idx, base, color="tab:blue", lw=2.2, label="负荷基线")
    ax.plot(t_idx, kn90, color="tab:orange", ls="--", lw=1.6, label="KNN P90 (下调下界代理)")
    ax.plot(t_idx, op90, color="tab:red", ls="--", lw=1.6, label="OPF P90 (下调下界真值)")
    ax.plot(t_idx, theo_lo, color="tab:cyan", ls=":", lw=2.0, label="理论下界 (不考虑安全约束)")

    step = 4
    ax.set_xticks(t_idx[::step])
    ax.set_xticklabels([SLOTS[t] for t in t_idx[::step]], rotation=45)
    ax.set_xlim(0, n - 1)                # 0:00 与纵轴重合, 23:45 靠右边界

    ax.set_ylabel("根节点功率 (MW)")
    ax.set_xlabel("时间 (15min 断面)")
    ax.set_title(f"根节点功率：{scenario}", fontsize=12)
    ax.grid(alpha=0.3)
    # 图例放到图片下方 (figure 级, 5 列), 用户指定顺序: 基线居中于 P90 组
    handles, labels = ax.get_legend_handles_labels()
    # 绘制序: 0理论带 1OPF带 2KNN带 | 3theo_hi 4op10 5kn10 6base 7kn90 8op90 9theo_lo
    # 图例顺序: 理论下界与OPF P10对调, 再与新位置的OPF P10和负荷基线对调
    order = [3, 9, 5, 7, 4, 8, 6, 0, 1, 2]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, frameon=False, fontsize=9, columnspacing=1.8,
               handlelength=1.6)
    fig.tight_layout(rect=[0, 0.07, 1, 1])   # 底部为图例预留空间
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="根节点基线功率与上/下调边界绘图")
    parser.add_argument("--scenario", default="scenario_trail_1", help="场景名 (scenario/{name}/)")
    parser.add_argument("--mode", default="prob", choices=["prob", "real"],
                        help="prob=基线+概率带+理论带 (默认); real=真实模式七条线 (KNN vs OPF)")
    parser.add_argument("--knn-dir", default=None,
                        help="real 模式: KNN 输出目录 (默认 output/output_{scenario}/)")
    parser.add_argument("--opf-dir", default=None,
                        help="real 模式: OPF 输出目录 (默认 output/output_{scenario}_opf/)")
    parser.add_argument("--out", default=None,
                        help="图片输出路径 (prob 默认 output/{scenario}/plot_{scenario}.png; "
                             "real 默认 KNN 目录/plot_{scenario}_real.png)")
    parser.add_argument("--out-dir", default=None,
                        help="图片输出目录 (默认 output/{scenario}/)")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    scen_dir = os.path.join("scenario", args.scenario)

    if args.mode == "real":
        knn_dir = args.knn_dir or os.path.join("output", f"output_{args.scenario}")
        opf_dir = args.opf_dir or os.path.join("output", f"output_{args.scenario}_opf")
        out_path = args.out or os.path.join(args.out_dir or knn_dir,
                                            f"plot_{args.scenario}_real.png")
        plot_real_mode(args.scenario, scen_dir, knn_dir, opf_dir, out_path)
        return

    shapes = read_shapes(os.path.join(scen_dir, "scenario_shapes.csv"))
    loads = read_loads(os.path.join(scen_dir, "scenario_loads.csv"))
    pvs = read_pvs(os.path.join(scen_dir, "scenario_pvs.csv"))
    p_sto = read_storage(os.path.join(scen_dir, "scenario_storage.csv"))
    sys_out = read_output_system(os.path.join("output", args.scenario, "output_system.csv"))
    print(f"场景 {args.scenario}: 曲线 {len(shapes)}, 负荷 {len(loads)}, "
          f"光伏 {len(pvs)}, 储能充放功率 {p_sto:.3f} MW")

    n = 96
    t_idx = np.arange(n)
    fixed_base = np.zeros(n)     # Σ Load1~32 + Fixed_Bus* kw×cur
    disp_base = np.zeros(n)      # Σ EV/AC kw×cur
    for name, kw in loads.items():
        if name in DISPATCHABLE:                      # 可调度 EV/AC
            c = shapes.get(f"{name}_cur")
            disp_base += (kw / 1000.0) * (c if c is not None else np.ones(n))
        else:                                         # 固定负荷: Load1~32 + Fixed_Bus* (无曲线按满载)
            c = shapes.get(f"{name}_cur")
            fixed_base += (kw / 1000.0) * (c if c is not None else np.ones(n))
    total_base = fixed_base + disp_base                   # 根节点基线 = 固定 + 可调度

    # 理论边界: 同固定负荷基线
    pv_full = np.zeros(n)                                  # PV 按 irr 满发
    for pv_name, pmpp in pvs.items():
        irr = shapes.get(f"{pv_name}_irr")
        pv_full += (pmpp / 1000.0) * (irr if irr is not None else np.zeros(n))
    evac_full = sum(loads.get(d, 0.0) / 1000.0 for d in DISPATCHABLE)   # EV/AC 满载
    theo_low = fixed_base - pv_full - p_sto                # EV/AC 削去 + PV 满发 + 储能放电
    theo_high = fixed_base + evac_full + p_sto             # EV/AC 满发 + PV 0 + 储能充电

    # 概率边界: 每断面 2N 样本的 P10 / P90 (按用户口径: 上界=P10, 下界=P90)
    prob_low = np.zeros(n)                                 # 下界 = 90% 分位
    prob_high = np.zeros(n)                                # 上界 = 10% 分位
    for t in range(n):
        ts = _slot_label(t)
        arr = sys_out.get(ts)
        if arr is None or arr.size == 0:
            prob_low[t] = prob_high[t] = np.nan
            continue
        prob_high[t] = np.percentile(arr, 10)              # 上界 = P10
        prob_low[t] = np.percentile(arr, 90)               # 下界 = P90

    # ---- 绘图 (理论带先画、概率带后画, 两个填充区间均可见) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t_idx, total_base, color="tab:blue", lw=2.2, label="根节点基线 (固定+可调)")
    ax.plot(t_idx, fixed_base, color="tab:blue", ls=":", lw=1.2, label="固定负荷分量 (Load1~32)")
    ax.plot(t_idx, disp_base, color="tab:orange", ls=":", lw=1.2, label="可调度负荷分量 (EV/AC cur)")

    # 理论边界带 (先画, 范围较大)
    ax.fill_between(t_idx, theo_low, theo_high, color="orange", alpha=0.10,
                    label="理论可行区间")
    ax.plot(t_idx, theo_high, color="tab:purple", ls=":", lw=1.8, label="理论上界")
    ax.plot(t_idx, theo_low, color="tab:cyan", ls=":", lw=1.8, label="理论下界")

    # 概率边界带 (后画, 覆盖在上层)
    ax.fill_between(t_idx, prob_low, prob_high, color="gray", alpha=0.20,
                    label="概率波动区间")
    ax.plot(t_idx, prob_high, color="tab:red", ls="--", lw=1.5,
            label="概率上界 P10 (90%样本在其上方)")
    ax.plot(t_idx, prob_low, color="tab:green", ls="--", lw=1.5,
            label="概率下界 P90 (90%样本在其下方)")

    step = 4
    ax.set_xticks(t_idx[::step])
    ax.set_xticklabels([_slot_label(t) for t in t_idx[::step]], rotation=45)
    ax.set_ylabel("根节点功率 (MW)")
    ax.set_title(f"根节点功率: 基线 / 概率边界 / 理论边界 ({args.scenario})")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    # handles 顺序: 0根基线 1固定 2可调 3理论区间 4理论上 5理论下 6概率区间 7P10 8P90
    # 图例 5 列 × 2 行, 顺序 (row-major):
    #   行1: 固定分量 | 可调分量 | 概率上界 | 概率下界 | 理论上界
    #   行2: 理论下界 | 理论可行区间 | 概率波动区间 | 根节点基线 | (空)
    pad = Patch(facecolor="none", edgecolor="none")          # 空位占位
    order = [1, 2, 7, 8, 4, 5, 3, 6, 0, -1]                 # -1 = 占位
    h = [handles[i] if i >= 0 else pad for i in order]
    l = [labels[i] if i >= 0 else "" for i in order]
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=5, frameon=False, fontsize=8, columnspacing=2.0, handlelength=1.5)

    out_dir = args.out_dir or os.path.join("output", args.scenario)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"plot_{args.scenario}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(bottom=0.16)   # 底部为 figure 图例预留空间 (位于 xlabel 下方)
    fig.savefig(out_path, dpi=150)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()

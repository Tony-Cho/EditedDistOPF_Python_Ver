# -*- coding: utf-8 -*-
"""
verify_real_mode.py — "真实部署模式"验证：场景曲线固定，仅对可调度负荷 lb/ub 抽样，
对比 KNN 预测与 OPF 真值的上调/下调分位排序。

真实模式与训练/测试 split 的区别：
  - train/val/test（gen_dataset）是整日抽样，固定负荷/PV/储能随天变化；
  - 真实部署只对可调度负荷的 lb/ub 做截断正态抽样（其余量固定为当日曲线值），
    即 scenario_dataset_mc(_OPF).py 的运行方式。本脚本在同一个场景上对比
    KNN 版 (output/{scenario}/) 与 OPF 版 (output/{scenario}_opf/) 的输出。

预期排序（逐槽）：
  上调 (max，自上而下)：
    理论上界 theo_hi = 固定负荷(当前) + 可调度满载 + 储能满功率充电
    > OPF P10 (max p_sub 10% 分位) ≈ KNN P10 > 基线 = 固定 + 可调度当前挂载
  下调 (min，自上而下)：
    基线 > OPF P90 (min p_sub 90% 分位) ≈ KNN P90 > 理论下界 theo_lo = 固定 − PV满发 − 储能放电

注意（已知坑）：场景的 scenario_pvs.csv / scenario_storage.csv 必须把 shape 列
绑定到对应曲线（PV_BusX_irr / BESS_Bus18_en），否则 OPF 会用满辐照/满窗口求解，
与 KNN 特征（用曲线值）不一致，min 场景会出现反向潮流等失真。

用法：
  python verify_real_mode.py --scenario scenario_syn_day
  python verify_real_mode.py --scenario scenario_syn_day --out output/output_scenario_syn_day/plot_real_mode.png
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DISPATCHABLE = ["EV_Bus19", "EV_Bus7", "AC_Bus2"]
SLOTS = [f"{h}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]


def _read_shapes(path):
    shapes = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            shapes[r["name"]] = np.array([float(r[k]) for k in r
                                          if k != "name" and r[k] not in (None, "")])
    return shapes


def _read_enabled(path, key):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("enabled", "TRUE")).upper() in ("TRUE", "1", "YES", "ON"):
                out[r["name"]] = float(r[key])
    return out


def _read_sys(path):
    by = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = float(r["p_sub_mw"])
            if not np.isfinite(v):
                continue
            by.setdefault((r["time_slot"].strip(), r["sense"].strip()), []).append(v)
    return {k: np.array(v) for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, help="场景名 (scenario/{name}/)")
    ap.add_argument("--knn-dir", default=None, help="KNN 输出目录 (默认 output/output_{scenario}/)")
    ap.add_argument("--opf-dir", default=None, help="OPF 输出目录 (默认 output/output_{scenario}_opf/)")
    ap.add_argument("--out", default=None, help="图片输出路径 (默认 KNN 目录/plot_real_mode.png)")
    args = ap.parse_args()

    scen_dir = os.path.join("scenario", args.scenario)
    knn_dir = args.knn_dir or os.path.join("output", f"output_{args.scenario}")
    opf_dir = args.opf_dir or os.path.join("output", f"output_{args.scenario}_opf")

    shapes = _read_shapes(os.path.join(scen_dir, "scenario_shapes.csv"))
    loads = _read_enabled(os.path.join(scen_dir, "scenario_loads.csv"), "kw")
    pvs = _read_enabled(os.path.join(scen_dir, "scenario_pvs.csv"), "pmpp_kw")
    p_sto = 0.0
    with open(os.path.join(scen_dir, "scenario_storage.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("enabled", "TRUE")).upper() in ("TRUE", "1", "YES", "ON"):
                p_sto = float(r["kw"]) / 1000.0
                break

    fixed = {n: kw / 1000.0 for n, kw in loads.items() if n not in DISPATCHABLE}
    disp = {n: loads.get(n, 0.0) / 1000.0 for n in DISPATCHABLE}
    pv_pmpp = {n: pm / 1000.0 for n, pm in pvs.items()}
    evac_full = sum(disp.values())

    knn = _read_sys(os.path.join(knn_dir, "output_system.csv"))
    opf = _read_sys(os.path.join(opf_dir, "output_system.csv"))

    c = {"up_base": 0, "up_knn": 0, "up_theo": 0, "dn_base": 0, "dn_knn": 0, "dn_theo": 0}
    n = 96
    base = np.full(n, np.nan); theo_hi = np.full(n, np.nan); theo_lo = np.full(n, np.nan)
    op10 = np.full(n, np.nan); kn10 = np.full(n, np.nan)
    op90 = np.full(n, np.nan); kn90 = np.full(n, np.nan)

    print(f"储能 {p_sto:.3f} MW | 可调满载 {evac_full:.3f} MW")
    print(f"{'slot':>6} {'基线':>7} | {'theo_hi':>8} {'op10':>7} {'kn10':>7} | "
          f"{'op90':>7} {'kn90':>7} {'theo_lo':>8}")
    for t, ts in enumerate(SLOTS):
        fixed_cur = sum(kw * shapes.get(f"{n0}_cur", np.ones(1))[t] for n0, kw in fixed.items())
        disp_cur = sum(kw * shapes.get(f"{n0}_cur", np.ones(1))[t] for n0, kw in disp.items())
        base[t] = fixed_cur + disp_cur
        theo_hi[t] = fixed_cur + evac_full + p_sto
        pv_full = sum(pm * shapes.get(f"{n0}_irr", np.zeros(1))[t] for n0, pm in pv_pmpp.items())
        theo_lo[t] = fixed_cur - pv_full - p_sto
        op10[t] = np.percentile(opf.get((ts, "max"), [np.nan]), 10)
        kn10[t] = np.percentile(knn.get((ts, "max"), [np.nan]), 10)
        op90[t] = np.percentile(opf.get((ts, "min"), [np.nan]), 90)
        kn90[t] = np.percentile(knn.get((ts, "min"), [np.nan]), 90)

        c["up_base"] += base[t] > kn10[t]
        c["up_knn"] += op10[t] < kn10[t]
        c["up_theo"] += op10[t] >= theo_hi[t]
        c["dn_base"] += base[t] < op90[t]
        c["dn_knn"] += op90[t] > kn90[t]
        c["dn_theo"] += op90[t] < theo_lo[t]
        if ts.endswith((":00", ":30")):
            print(f"{ts:>6} {base[t]:7.3f} | {theo_hi[t]:8.3f} {op10[t]:7.3f} {kn10[t]:7.3f} | "
                  f"{op90[t]:7.3f} {kn90[t]:7.3f} {theo_lo[t]:8.3f}")

    print(f"\n【上调】 基线>KNN P10 {c['up_base']}/96 | OPF P10<KNN P10 {c['up_knn']}/96 "
          f"(max {np.nanmax(kn10 - op10):.4f} MW) | OPF P10>=theo {c['up_theo']}/96")
    print(f"【下调】 基线<OPF P90 {c['dn_base']}/96 | OPF P90>KNN P90 {c['dn_knn']}/96 "
          f"(max {np.nanmax(op90 - kn90):.4f} MW) | OPF P90<theo_lo {c['dn_theo']}/96")

    t_idx = np.arange(n)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(t_idx, theo_hi, color="tab:purple", ls=":", lw=2.0, label="理论上界 (不考虑安全约束)")
    ax1.plot(t_idx, op10, color="tab:red", ls="--", lw=1.8, label="OPF P10 (上调上界真值)")
    ax1.plot(t_idx, kn10, color="tab:orange", ls="--", lw=1.8, label="KNN P10 (上调上界代理)")
    ax1.plot(t_idx, base, color="tab:blue", lw=2.2, label="负荷基线")
    ax1.set_ylabel("根节点功率 (MW)")
    ax1.set_title(f"真实模式·上调: theo > OPF P10 ≈ KNN P10 > 基线   "
                  f"(OPF<KNN {c['up_knn']}/96, max {np.nanmax(kn10 - op10):.4f} MW)")
    ax1.grid(alpha=0.3); ax1.legend(loc="upper left", fontsize=9)
    ax2.plot(t_idx, base, color="tab:blue", lw=2.2, label="负荷基线")
    ax2.plot(t_idx, op90, color="tab:red", ls="--", lw=1.8, label="OPF P90 (下调下界真值)")
    ax2.plot(t_idx, kn90, color="tab:orange", ls="--", lw=1.8, label="KNN P90 (下调下界代理)")
    ax2.plot(t_idx, theo_lo, color="tab:cyan", ls=":", lw=2.0, label="理论下界 (不考虑安全约束)")
    ax2.set_ylabel("根节点功率 (MW)")
    ax2.set_title(f"真实模式·下调: 基线 > OPF P90 ≈ KNN P90 > theo   "
                  f"(OPF>KNN {c['dn_knn']}/96, max {np.nanmax(op90 - kn90):.4f} MW)")
    ax2.grid(alpha=0.3); ax2.legend(loc="upper left", fontsize=9)
    step = 4
    ax2.set_xticks(t_idx[::step])
    ax2.set_xticklabels([SLOTS[t] for t in t_idx[::step]], rotation=45)
    ax2.set_xlabel("时间 (15min 断面)")
    fig.suptitle(f"真实部署模式验证 (仅抽可调度 lb/ub): {args.scenario}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or os.path.join(knn_dir, "plot_real_mode.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()

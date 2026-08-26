# -*- coding: utf-8 -*-
"""
plot_ordering.py — 验证并绘制"上调/下调边界"预期排序（min 与 max 两个 sense）

预期排序：
  上调（max，自上而下）：
    1. 理论上界 theo_high = 固定负荷(当前) + 可调度满载 + 储能满功率充电（不考虑安全约束）
    2. OPF P10   = OPF 真值 max p_sub 的 10% 分位数（同源 test split，逐槽）
    3. KNN P10   = KNN 预测 max p_sub 的 10% 分位数
    4. 基线       = 固定 + 可调度当前挂载（逐槽均值）

  下调（min，自上而下）：
    1. 基线
    2. OPF P90   = OPF 真值 min p_sub 的 90% 分位数（"下调到 P90，90% 置信符合 OPF"）
    3. KNN P90   = KNN 预测 min p_sub 的 90% 分位数
    4. 理论下界 theo_low = 固定负荷(当前) − PV 满发 − 储能满功率放电

前提：train/val/test 由 gen_dataset.py 同源生成（同一生成器不同随机日），
KNN 在 train 全量拟合、test 上评估 —— 测试特征必须落在训练分布内，KNN 才不外推。

用法：
  python plot_ordering.py --train dataset/model_default/train --test dataset/model_default/test
  python plot_ordering.py --train ... --test ... --k 7 --out output/plot_ordering.png
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from train_knn import load_knn_dataset
from load_network import load_network

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DISPATCHABLE = ["EV_Bus19", "EV_Bus7", "AC_Bus2"]
DISP_KW = {"EV_Bus19": 400.0, "EV_Bus7": 600.0, "AC_Bus2": 200.0}
SLOTS = [f"{h}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
Q_UP = 10    # 上调: 保守分位 (P10)
Q_DN = 90    # 下调: 保守分位 (P90)


def _slot_idx(ts: str) -> int:
    h, _, m = ts.partition(":")
    return int(h) * 4 + int(m or 0) // 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="训练集目录 (含 training_dataset_*.csv)")
    ap.add_argument("--test", required=True, help="测试集目录 (含 training_dataset_*.csv)")
    ap.add_argument("--model", default="data/csv_case33/model_default", help="网络模型目录 (读储能/PV 功率)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None, help="图片输出路径 (默认 --test 上级目录/plot_ordering.png)")
    args = ap.parse_args()

    # ---- KNN: train 全量拟合, test 预测 (min/max) ----
    ds_train = load_knn_dataset(args.train)
    ds_test = load_knn_dataset(args.test)
    preds = {}
    truths = {}
    keys = None
    for sense in ("min", "max"):
        Xtr, ytr, _, targets, _ = ds_train[sense]
        scaler = StandardScaler().fit(Xtr)
        model = KNeighborsRegressor(n_neighbors=args.k, weights="distance").fit(scaler.transform(Xtr), ytr)
        idx = targets.index("p_sub_mw")
        Xte, yte, _, _, keys = ds_test[sense]
        preds[sense] = model.predict(scaler.transform(Xte))[:, idx]
        truths[sense] = yte[:, idx]

    # ---- 储能/PV 额定 (网络) ----
    net = load_network(args.model)
    p_sto = max((st.kw / 1000.0 for st in net.storages.values()), default=0.0)
    pv_pmpp = {pv.name: pv.pmpp_kw / 1000.0 for pv in net.pvs.values()}
    evac_full = sum(DISP_KW.values()) / 1000.0

    # ---- test sample: 基线 / 理论上界 / 理论下界 (逐 样本) ----
    sample = {}
    with open(os.path.join(args.test, "training_dataset_sample.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            fixed = float(r["fixed_total_mw"])
            disp_cur = sum(DISP_KW[n] / 1000.0 * float(r[f"{n}_cur"]) for n in DISPATCHABLE)
            pv_full = sum(pv_pmpp[name] * float(r[f"{name}_irr"]) for name in pv_pmpp)
            sample[(r["sample_id"], r["time_slot"])] = (fixed, disp_cur, pv_full)

    by_slot = {ts: {"base": [], "theo_hi": [], "theo_lo": [],
                    "opf_min": [], "knn_min": [], "opf_max": [], "knn_max": []} for ts in SLOTS}
    for k, (sid, ts) in enumerate(keys):
        fixed, disp_cur, pv_full = sample[(sid, ts)]
        d = by_slot[ts]
        d["base"].append(fixed + disp_cur)
        d["theo_hi"].append(fixed + evac_full + p_sto)
        d["theo_lo"].append(fixed - pv_full - p_sto)
        d["opf_min"].append(truths["min"][k])
        d["knn_min"].append(preds["min"][k])
        d["opf_max"].append(truths["max"][k])
        d["knn_max"].append(preds["max"][k])

    n = 96
    A = {key: np.full(n, np.nan) for key in
         ("base", "theo_hi", "theo_lo", "op10", "kn10", "op90", "kn90")}
    stats = {"c_up1": 0, "c_up2": 0, "c_up3": 0, "c_dn1": 0, "c_dn2": 0, "c_dn3": 0}
    for t, ts in enumerate(SLOTS):
        d = by_slot[ts]
        A["base"][t] = float(np.mean(d["base"]))
        A["theo_hi"][t] = float(np.max(d["theo_hi"]))
        A["theo_lo"][t] = float(np.min(d["theo_lo"]))
        A["op10"][t] = float(np.percentile(d["opf_max"], Q_UP))
        A["kn10"][t] = float(np.percentile(d["knn_max"], Q_UP))
        A["op90"][t] = float(np.percentile(d["opf_min"], Q_DN))
        A["kn90"][t] = float(np.percentile(d["knn_min"], Q_DN))
        # 上调: theo > OPF P10, OPF P10 >= KNN P10, 基线 < KNN P10
        stats["c_up1"] += A["op10"][t] >= A["theo_hi"][t]
        stats["c_up2"] += A["op10"][t] < A["kn10"][t]
        stats["c_up3"] += A["base"][t] > A["kn10"][t]
        # 下调: 基线 > OPF P90, OPF P90 <= KNN P90, 理论下界 < OPF P90
        stats["c_dn1"] += A["base"][t] < A["op90"][t]
        stats["c_dn2"] += A["op90"][t] > A["kn90"][t]
        stats["c_dn3"] += A["op90"][t] < A["theo_lo"][t]

    print(f"test 样本 {len(keys)}, 储能 {p_sto:.3f} MW, 可调度满载 {evac_full:.3f} MW")
    print(f"\n【上调 max】 theo_hi > OPF P10 > KNN P10 > 基线")
    print(f"  违反: OPF P10>=theo {stats['c_up1']}/96 | OPF P10<KNN P10 {stats['c_up2']}/96 "
          f"(max {np.nanmax(A['kn10']-A['op10']):.4f} MW) | 基线>KNN P10 {stats['c_up3']}/96")
    print(f"【下调 min】 基线 > OPF P90 > KNN P90 > theo_lo")
    print(f"  违反: 基线<OPF P90 {stats['c_dn1']}/96 | OPF P90>KNN P90 {stats['c_dn2']}/96 "
          f"(max {np.nanmax(A['op90']-A['kn90']):.4f} MW) | OPF P90<theo_lo {stats['c_dn3']}/96")
    print(f"\n{'slot':>6} {'基线':>7} | {'theo_hi':>8} {'op10':>7} {'kn10':>7} | "
          f"{'op90':>7} {'kn90':>7} {'theo_lo':>8}")
    for t, ts in enumerate(SLOTS):
        if ts.endswith((":00", ":30")):
            print(f"{ts:>6} {A['base'][t]:7.3f} | {A['theo_hi'][t]:8.3f} {A['op10'][t]:7.3f} "
                  f"{A['kn10'][t]:7.3f} | {A['op90'][t]:7.3f} {A['kn90'][t]:7.3f} {A['theo_lo'][t]:8.3f}")

    # ---- 绘图: 上下两个子图 ----
    t_idx = np.arange(n)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    # 上调
    ax1.plot(t_idx, A["theo_hi"], color="tab:purple", ls=":", lw=2.0, label="理论上界 (不考虑安全约束)")
    ax1.plot(t_idx, A["op10"], color="tab:red", ls="--", lw=1.8, label="OPF P10 (上调上界真值)")
    ax1.plot(t_idx, A["kn10"], color="tab:orange", ls="--", lw=1.8, label="KNN P10 (上调上界代理)")
    ax1.plot(t_idx, A["base"], color="tab:blue", lw=2.2, label="负荷基线")
    ax1.fill_between(t_idx, A["base"], A["kn10"], color="orange", alpha=0.10)
    ax1.fill_between(t_idx, A["kn10"], A["op10"], color="red", alpha=0.08)
    ax1.set_ylabel("根节点功率 (MW)")
    ax1.set_title(f"上调上界: theo > OPF P10 > KNN P10 > 基线   "
                  f"(违反: {stats['c_up2']}/96, max {np.nanmax(A['kn10']-A['op10']):.4f} MW)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    # 下调
    ax2.plot(t_idx, A["base"], color="tab:blue", lw=2.2, label="负荷基线")
    ax2.plot(t_idx, A["op90"], color="tab:red", ls="--", lw=1.8, label="OPF P90 (下调下界真值)")
    ax2.plot(t_idx, A["kn90"], color="tab:orange", ls="--", lw=1.8, label="KNN P90 (下调下界代理)")
    ax2.plot(t_idx, A["theo_lo"], color="tab:cyan", ls=":", lw=2.0, label="理论下界 (不考虑安全约束)")
    ax2.fill_between(t_idx, A["theo_lo"], A["kn90"], color="cyan", alpha=0.10)
    ax2.fill_between(t_idx, A["kn90"], A["op90"], color="red", alpha=0.08)
    ax2.set_ylabel("根节点功率 (MW)")
    ax2.set_title(f"下调下界: 基线 > OPF P90 > KNN P90 > theo   "
                  f"(违反: {stats['c_dn2']}/96, max {np.nanmax(A['op90']-A['kn90']):.4f} MW)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)
    step = 4
    ax2.set_xticks(t_idx[::step])
    ax2.set_xticklabels([SLOTS[t] for t in t_idx[::step]], rotation=45)
    ax2.set_xlabel("时间 (15min 断面)")
    fig.suptitle(f"上调/下调边界排序验证 (test split 同源, KNN k={args.k})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or os.path.join(os.path.dirname(args.test), "plot_ordering.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()

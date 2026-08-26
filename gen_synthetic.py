# -*- coding: utf-8 -*-
"""
gen_synthetic.py — 替 model_default 生成合成基准曲线（96 点/天，15min）

设计（阶段2已确认）：
  抽样单元 = 整日 96 点轨迹；基准曲线作为 shapes.csv 的"典型"曲线，
  反映历史小时均值的日内形状（负荷晚峰 / 光伏午峰），并约束到可行范围。

本脚本当前职责：
  1. 由 24 点小时均值锚点插值出 96 点（15min）基准曲线；
  2. 写入 data/csv_case33/model_default/model_shapes.csv；
  3. 可选 --check：逐断面跑 OPF，统计 min/max 可行率。

后续（整日轨迹抽样 + train/val/test）由同模块的 sample_day / gen_dataset 扩展。
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# 24 点小时均值锚点（来自 history/IEEE33_PV&Load_Data.mat 的 1461 天均值，mult 相对满载）
LOAD_H_MEAN = np.array([
    0.613, 0.585, 0.568, 0.568, 0.609, 0.735, 0.887, 0.981,
    1.078, 1.177, 1.220, 1.240, 1.237, 1.229, 1.230, 1.246,
    1.264, 1.274, 1.267, 1.235, 1.176, 1.048, 0.845, 0.688,
])
PV_H_MEAN = np.array([
    0.000, 0.000, 0.000, 0.000, 0.000, 0.007, 0.049, 0.141,
    0.268, 0.391, 0.493, 0.564, 0.594, 0.579, 0.522, 0.427,
    0.301, 0.168, 0.064, 0.012, 0.000, 0.000, 0.000, 0.000,
])

NPT = 96  # 15min 间隔


def hourly_to_96(h24: np.ndarray) -> np.ndarray:
    """24 点小时均值 -> 96 点 15min（分段线性插值，端点延拓）"""
    h24 = np.asarray(h24, dtype=float)
    xs = np.linspace(0, 24, NPT, endpoint=False)  # 0..23.75，每 15min
    return np.interp(xs, np.arange(24), h24)


def clamp01(x):
    return float(min(max(x, 0.0), 1.0))


def base_load_curve() -> np.ndarray:
    """固定负荷基准 mult 曲线（96 点）：历史日内均值形状"""
    return hourly_to_96(LOAD_H_MEAN)


def base_pv_curve() -> np.ndarray:
    """光伏辐照基准曲线（96 点）：午峰、夜零"""
    return np.clip(hourly_to_96(PV_H_MEAN), 0.0, 1.0)


def build_default_shapes(net, load_scale: float = 1.0) -> dict:
    """为 model_default 各组件生成基准曲线（96 点），返回 {曲线名: (npts, interval, mult)}。

    组件 -> 曲线名：
      固定负荷 Load1..Load32 -> LoadX_cur （= 基准 diurnal mult）
      可调度负荷              -> {名}_cur / _lb / _ub
      光伏                    -> {名}_irr
      储能                    -> BESS_Bus18_en（能量比例，限 [lb,ub] 窗内）

    load_scale：对负荷基准曲线整体缩放（<1 压低负荷水平）。设计算例时用于把
    生成日的总负荷压进 IEEE33 网络 OPF 可行域（电压/热极限有充足余量），
    避免出现"基线本身不可行 / max OPF 无解"（见 gen_dataset 可行性门）。
    """
    shapes = {}
    load_base = base_load_curve() * load_scale
    pv_base = base_pv_curve()

    # 固定负荷
    for ld in net.loads.values():
        if ld.dispatchable:
            continue
        shapes[f"{ld.name}_cur"] = load_base.copy()

    # 可调度负荷：cur 用基准形状，lb/ub 围绕 cur 给可调带
    for ld in net.loads.values():
        if not ld.dispatchable:
            continue
        cur = load_base.copy()
        # 可调带：相对 cur 偏移（EV 偏 evening，AC 偏白天可用同一基准；带宽固定）
        lb = cur - 0.10
        ub = cur + 0.10
        np.clip(lb, 0.02, None, out=lb)
        np.clip(ub, None, 1.20, out=ub)
        keep = lb < ub
        lb, ub = np.maximum(lb, 0.02), np.maximum(ub, lb + 0.02)
        shapes[f"{ld.name}_cur"] = cur
        shapes[f"{ld.name}_lb"] = lb
        shapes[f"{ld.name}_ub"] = ub

    # 光伏
    for pv in net.pvs.values():
        shapes[f"{pv.name}_irr"] = pv_base.copy()

    # 储能能量比例（限在能量窗 [energy_lb_ratio, energy_ub_ratio] 内平滑变化）
    for st in net.storages.values():
        lb_r = st.energy_lb_ratio if st.energy_lb_ratio is not None else 0.1
        ub_r = st.energy_ub_ratio if st.energy_ub_ratio is not None else 0.9
        lo = float(lb_r + 0.05)
        hi = float(ub_r - 0.05)
        # 以负荷形状驱动能量（夜晚谷底低、晚峰高），线性映射进 [lo, hi]
        m = load_base
        m = (m - m.min()) / max(m.max() - m.min(), 1e-9)
        shapes[f"{st.name}_en"] = hi * 0.3 + (hi - lo) * 0.7 * m

    return shapes


# 时间戳表头（96 个 15min 点）
TIMES = [f"{h}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]


def write_shapes_csv(path: str, shapes: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write("name," + ",".join(TIMES) + "\n")
        for name, mult in shapes.items():
            f.write(name + "," + ",".join(f"{v:.6f}" for v in np.asarray(mult, dtype=float)) + "\n")
    print(f"已写入 shapes: {path} ({len(shapes)} 条曲线, {NPT} 点)")


# =====================================================================
# 可行性检查（复用场景脚本的 _apply_slot + build_and_solve_opf）
# =====================================================================
def check_feasibility(model_dir: str):
    from load_network import load_network, resolve_model_path
    from opf_model import build_and_solve_opf

    net = load_network(model_dir)
    _apply_slot = None
    # 内联同款 _apply_slot（与 scenario_dataset_mc_OPF 一致）
    def apply_slot(network, t):
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

    counts = {"min": {"OPTIMAL": 0, "n": 0}, "max": {"OPTIMAL": 0, "n": 0}}
    for t in range(NPT):
        apply_slot(net, t)
        for sense in ("min", "max"):
            result, _ = build_and_solve_opf(net, sense, verbose=False)
            counts[sense]["n"] += 1
            if result["termination_status"] in ("OPTIMAL", "TIME_LIMIT_FEASIBLE"):
                counts[sense]["OPTIMAL"] += 1
    for sense in ("min", "max"):
        ok = counts[sense]["OPTIMAL"]
        print(f"  [{sense}] 可行 {ok}/{counts[sense]['n']} 断面")
    return counts


# =====================================================================
# 整日轨迹抽样（阶段2设计：日级因子 + 共同因子 + 平滑 AR(1) 噪声 + 钟形可控方差）
# =====================================================================

DEFAULT_PARAMS = {
    "sigma_day": 0.06,    # 日级等级因子（全负荷共同缩放）std
    "phi": 0.90,          # AR(1) 自相关（时段前后联系）
    "w_c": 0.05,          # 共同日内扰动的边际 std（全负荷共享）
    "sigma_fix": 0.03,    # 各固定负荷独立噪声的边际 std
    "sigma_day_pv": 0.08, # 光伏日级因子 std
    "w_pv_c": 0.06,       # 光伏共同日内扰动 std
    "sigma_pv": 0.04,     # 各光伏独立噪声 std
    "sigma_disp": 0.04,   # 可调度负荷 cur 独立噪声 std
    "band": 0.12,         # 可调度 lb/ub 相对 cur 的带宽
    "sigma_en": 0.03,     # 储能能量比例噪声 std
    "lo_mult": 1.30,      # 负荷 mult 上限
    "hi_mult": 1.50,      # 负荷 mult 上限
}


def _ar1(rng, n, phi, sigma, lo=-np.inf, hi=np.inf):
    """平稳 AR(1)：x[t] = phi*x[t-1] + sqrt(1-phi^2)*sigma*N(0,1)，边际 std=sigma。"""
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1.0 - phi ** 2) * sigma * rng.normal()
    return np.clip(x, lo, hi)


def sample_day(rng, base_shapes, net, p: dict) -> dict:
    """生成一天的整日 96 点轨迹 → {组件曲线名: (96,) 数组}。

    结构：
      - 日级因子 gam（全负荷共同）缩放各基准曲线；
      - 共同日内容扰动 c(t)（AR1），加权叠加到所有负荷；
      - 各组件独立 AR1 噪声；
      - 可调度 cur/lb/ub 保序（lb<cur<ub）；PV 夜间置零；储能落在能量窗内。
    """
    load_base = base_shapes  # {曲线名: 96 数组}
    gam = 1.0 + rng.normal(0.0, p["sigma_day"])      # 日级负荷因子
    gam_pv = 1.0 + rng.normal(0.0, p["sigma_day_pv"])
    c = _ar1(rng, NPT, p["phi"], p["w_c"])           # 共同日内扰动
    c_pv = _ar1(rng, NPT, p["phi"], p["w_pv_c"])

    out = {}
    fix_names = [f"{ld.name}_cur" for ld in net.loads.values() if not ld.dispatchable]
    for name in fix_names:
        e = _ar1(rng, NPT, p["phi"], p["sigma_fix"])
        mult = load_base[name] * gam + c + e
        out[name] = np.clip(mult, 0.02, p["hi_mult"])

    for ld in net.loads.values():
        if not ld.dispatchable:
            continue
        cur_name, lb_name, ub_name = f"{ld.name}_cur", f"{ld.name}_lb", f"{ld.name}_ub"
        e = _ar1(rng, NPT, p["phi"], p["sigma_disp"])
        cur = np.clip(load_base[cur_name] * gam + c + e, 0.03, p["lo_mult"])
        lb = np.clip(cur - p["band"], 0.02, None)
        ub = np.maximum(cur + p["band"], lb + 0.02)
        ub = np.minimum(ub, 1.4)
        out[cur_name] = cur
        out[lb_name] = lb
        out[ub_name] = ub

    for pv in net.pvs.values():
        name = f"{pv.name}_irr"
        e = _ar1(rng, NPT, p["phi"], p["sigma_pv"])
        base = load_base[name]
        day = base * gam_pv + (c_pv + e) * (base > 0.02)
        out[name] = np.clip(day, 0.0, 1.0)

    for st in net.storages.values():
        name = f"{st.name}_en"
        e = _ar1(rng, NPT, p["phi"], p["sigma_en"])
        lb_r = st.energy_lb_ratio if st.energy_lb_ratio is not None else 0.1
        ub_r = st.energy_ub_ratio if st.energy_ub_ratio is not None else 0.9
        out[name] = np.clip(load_base[name] + e, lb_r + 0.02, ub_r - 0.02)

    return out


def mc_check_feasibility(model_dir: str, n_days: int = 5, seed: int = 7):
    """抽样 n_days 天，逐槽跑 OPF，统计可考虑（验证采样参数不会引入不可行）。"""
    from load_network import load_network
    from opf_model import build_and_solve_opf

    network0 = load_network(model_dir)
    shapes = build_default_shapes(network0)
    params = dict(DEFAULT_PARAMS)
    rng = np.random.default_rng(seed)

    counts = {"min": {"OPTIMAL": 0, "n": 0}, "max": {"OPTIMAL": 0, "n": 0}}
    net = load_network(model_dir)
    for d in range(n_days):
        day = sample_day(rng, shapes, net, params)
        for t in range(NPT):
            for ld in net.loads.values():
                cur = day.get(f"{ld.name}_cur", np.full(NPT, 1.0))[t]
                ld.base_ratio = cur
                lb = day.get(f"{ld.name}_lb", np.full(NPT, 0.0))[t]
                ub = day.get(f"{ld.name}_ub", np.full(NPT, 1.0))[t]
                if ld.dispatchable:
                    if lb > cur:
                        lb = cur
                    if ub < cur:
                        ub = cur
                ld.mult_lb, ld.mult_ub = lb, ub
                ld.p_cur_pu = ld.p_pu * cur
                ld.q_cur_pu = ld.q_pu * cur
                ld.z_lb = lb / cur if cur > 0 else 0.0
                ld.z_ub = ub / cur if cur > 0 else 1e6
            for pv in net.pvs.values():
                pv.irradiance = day.get(f"{pv.name}_irr", np.ones(NPT))[t]
                pv.p_avail_pu = pv.p_max_pu * pv.irradiance
            for st in net.storages.values():
                st.energy_ratio = day.get(f"{st.name}_en", np.full(NPT, 1.0))[t]
                st.energy_ub_cur_pu = st.energy_ub_pu * st.energy_ratio
                st.energy_lb_cur_pu = st.energy_lb_pu * st.energy_ratio
            for sense in ("min", "max"):
                result, _ = build_and_solve_opf(net, sense, verbose=False)
                counts[sense]["n"] += 1
                if result["termination_status"] in ("OPTIMAL", "TIME_LIMIT_FEASIBLE"):
                    counts[sense]["OPTIMAL"] += 1
        print(f"  天 {d + 1}/{n_days} 完成"
              f" [min {counts['min']['OPTIMAL']}/{counts['min']['n']}  max {counts['max']['OPTIMAL']}/{counts['max']['n']}]")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="写曲线后逐断面跑 OPF 可行性")
    ap.add_argument("--mc-check", action="store_true", help="抽样 n 天逐槽跑 OPF 可行性 (默认参数)")
    ap.add_argument("--mc-days", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="data/csv_case33/model_default")
    args = ap.parse_args()

    from load_network import load_network
    net = load_network(args.model)
    shapes = build_default_shapes(net)
    out = os.path.join(args.model, "model_shapes.csv")
    write_shapes_csv(out, shapes)

    if args.check:
        print("逐断面 OPF 可行性检查：")
        check_feasibility(args.model)
    if args.mc_check:
        print(f"整日轨迹抽样 MC 可行性检查（{args.mc_days} 天）：")
        mc_check_feasibility(args.model, n_days=args.mc_days, seed=args.seed)


if __name__ == "__main__":
    main()
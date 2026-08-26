# -*- coding: utf-8 -*-
"""
scenario_dataset_mc.py — 场景数据集蒙特卡洛生成 (用已训练 KNN 模型预测场景输出)

输入:
  - 场景配置 (output/{场景名}/output_mc_config.csv): model 索引模型库 + scenario 场景名
    + 抽样运行参数 + 组件分布 (仅 EV/AC 的 lb/ub 抽样, 其余固定曲线值)
  - 模型目录 (knn_lib/{model}/): knn_model_{sense}.joblib +
    knn_scaler_{sense}.joblib + knn_feature_names.json + knn_target_names.json

流程 (对每个 15min 断面 × N 样本):
  1. 固定特征: Load1~32 + Fixed_Bus* 当前功率 (曲线), PV 辐照度 (曲线),
     储能初始能量 (曲线) / 初始功率 0
  2. 抽样特征: mc_config 中配置的 EV/AC lb/ub 截断正态 (σ=cv×μ 或 sigma,
     ub 先抽 [0,1], lb 截断 [0, ub] 保序); 未配置量固定曲线值 (方案A)
  3. 48 维特征 → scaler → KNN(min/max) → 18 维输出预测

输出 (output/{场景名}/, 格式参照 training_dataset/输出控制.md, 前缀 training_dataset → output):
  - output_mc_config.csv       场景预测配置 (读入, 保留)
  - output_sample.csv   抽样输入场景表 (每 断面×样本 一行)
  - output_system.csv   根节点注入 (sense + p_sub/q_sub)
  - output_storage.csv  储能出力 (p_net/q/se)
  - output_pvs.csv      光伏出力 (p_out/q_out)
  - output_loads.csv    可调度负荷实际有功 (ev/ac)
  模型不复制到场景下: 由 output_mc_config.csv 的 model 键索引到总库 knn_lib/{model}/
  (buses/lines 无 KNN 目标, 不输出)

用法:
  python scenario_dataset_mc.py --config output/output_scenario_trail_1/output_mc_config.csv
  python scenario_dataset_mc.py --scenario output_scenario_trail_1 --model training_dataset_storage_bus18_sample --n 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import joblib
import numpy as np

from load_network import load_network, resolve_model_path
from sampling import truncated_normal_vec, resolve_mu_sigma

# 断面辅助 (与 training_dataset_mc 一致: 15min 步进, 含端点, 全天 96 点)
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
MC_GLOBAL_KEYS = {"model", "scenario", "n_samples", "seed", "start_time", "end_time"}


def load_mc_config(path: str):
    """读取 output/{场景}/output_mc_config.csv → (全局参数 dict, 组件配置 {组件名: (分布名, 参数字典)})

    与 training_dataset_mc.load_mc_config 结构一致, 额外支持全局键 model (索引模型库):
      name,value
      model,training_dataset_storage_bus18_sample    # 模型名 = 训练集文件夹名
      scenario,output_scenario_trail_1              # 场景名
      n_samples,200
      seed,42
      start_time,0:00
      end_time,23:45
      EV_Bus19_lb,truncated_normal,cv:0.10   # 组件分布: 抽样参数单独设置
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
# 特征构建
# =====================================================================

def build_slot_features(feature_names, net, shapes, t, rng, comp_cfg, n):
    """构建第 t 断面的 n 个样本的 48 维特征矩阵 (X: n×48)

    固定项: 负荷当前功率 (曲线), PV 辐照度 (曲线), 储能初始能量 (曲线)/功率 0
    抽样项: comp_cfg 中配置的 EV/AC lb/ub (截断正态, ub 先抽 [0,1], lb 截断 [0,ub]);
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
# 输出
# =====================================================================

def save_outputs(out_dir, records, target_names, feature_names, net):
    """records: list of (slot_label, sample_id, sense, y_pred_row, X_row, cur_vals)

    y_pred_row 与 target_names 一一对应; X_row 与 feature_names 一一对应;
    cur_vals: {组件名: EV/AC 当前挂载曲线值} (output_sample 记录用, 预测时不抽样)。
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- output_sample.csv: 抽样输入场景表 (列与 training_dataset_sample 一致, min/max 共用同一批抽样) ----
    load_feats = [f for f in feature_names
                  if f.endswith("_mw") and not f.endswith("_p_init_mw")]  # 排除 BESS 初始功率列
    irr_feats = [f for f in feature_names if f.endswith("_irr")]
    # BESS 初始状态: 固定 se_init 在前、p_init 在后 (与 training_dataset_sample 列序一致)
    bess_feats = ([f for f in feature_names if f.endswith("_se_init_mwh")]
                  + [f for f in feature_names if f.endswith("_p_init_mw")])
    lbub_feats = [f for f in feature_names if f.endswith("_lb") or f.endswith("_ub")]
    disp_names = ["EV_Bus19", "EV_Bus7", "AC_Bus2"]          # 可调度组件 (cur 固定曲线值)
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
        for ts, sid, sense, row, xrow, cur_vals in sample_rows:
            fixed_total = sum(xrow[i] for i in load_idx)
            vals = [sid, ts] + [f"{xrow[i]:.6f}" for i in load_idx] + [f"{fixed_total:.6f}"]
            vals += [f"{xrow[feature_names.index(f)]:.6f}" for f in irr_feats]
            vals += [f"{xrow[feature_names.index(f)]:.6f}" for f in bess_feats]
            for n in disp_names:
                vals.append(f"{cur_vals.get(n, 0.0):.6f}")          # cur 固定曲线值
                vals += [f"{xrow[feature_names.index(f)]:.6f}" for f in lbub_feats if f.startswith(n)]
            w.writerow(vals)
    print(f"  已保存: {os.path.join(out_dir, 'output_sample.csv')}")

    # ---- output_system.csv ----
    system_cols = ["p_sub_mw", "q_sub_mvar"]
    with open(os.path.join(out_dir, "output_system.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "time_slot", "sense"] + system_cols)
        idx = [target_names.index(c) for c in system_cols]
        for ts, sid, sense, row, _, _ in records:
            w.writerow([sid, ts, sense] + [f"{row[i]:.6f}" for i in idx])
    print(f"  已保存: {os.path.join(out_dir, 'output_system.csv')}")

    def write_component(fname, header, rows):
        with open(os.path.join(out_dir, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"  已保存: {os.path.join(out_dir, fname)}")

    # ---- output_storage.csv ----
    st_rows, st_headers = [], ["sample_id", "time_slot", "sense", "name", "bus",
                               "p_net_mw", "q_mvar", "se_mwh"]
    for ts, sid, sense, row, _, _ in records:
        for st in net.storages.values():
            vals = [row[target_names.index(f"{st.name}_{c}")]
                    for c in ("p_net_mw", "q_mvar", "se_mwh")]
            st_rows.append([sid, ts, sense, st.name, st.bus] + [f"{v:.6f}" for v in vals])
    write_component("output_storage.csv", st_headers, st_rows)

    # ---- output_pvs.csv ----
    pv_rows, pv_headers = [], ["sample_id", "time_slot", "sense", "name", "bus",
                               "p_out_mw", "q_out_mvar"]
    for ts, sid, sense, row, _, _ in records:
        for pv in net.pvs.values():
            vals = [row[target_names.index(f"{pv.name}_{c}")]
                    for c in ("p_out_mw", "q_out_mvar")]
            pv_rows.append([sid, ts, sense, pv.name, pv.bus] + [f"{v:.6f}" for v in vals])
    write_component("output_pvs.csv", pv_headers, pv_rows)

    # ---- output_loads.csv (仅 ev/ac 可调度负荷) ----
    ld_rows, ld_headers = [], ["sample_id", "time_slot", "sense", "name", "bus", "type",
                               "p_out_mw"]
    for ts, sid, sense, row, _, _ in records:
        for ld in net.loads.values():
            if not ld.dispatchable:
                continue
            val = row[target_names.index(f"{ld.name}_p_out_mw")]
            ld_rows.append([sid, ts, sense, ld.name, ld.bus, ld.type, f"{val:.6f}"])
    write_component("output_loads.csv", ld_headers, ld_rows)


# =====================================================================
# 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="用已训练 KNN 模型对场景做概率化预测")
    parser.add_argument("--config", default=None,
                        help="场景配置 output_mc_config.csv (output/{场景}/output_mc_config.csv, 含 model 索引 + 抽样配置)")
    parser.add_argument("--scenario", default=None, help="场景名 (scenario/{scenario}/), 覆盖配置")
    parser.add_argument("--model", default=None,
                        help="模型名 = 训练集文件夹名 (knn_lib/{model}/), 覆盖配置")
    parser.add_argument("--model-dir", default=None, help="模型目录 (覆盖配置, 一般无需传)")
    parser.add_argument("--n", type=int, default=None, help="每断面抽样次数 (覆盖配置)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖配置)")
    parser.add_argument("--start-time", default=None, help="起始断面 (覆盖配置)")
    parser.add_argument("--end-time", default=None, help="结束断面 (覆盖配置)")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    # 0. 场景配置 (output_mc_config.csv): CLI --config 显式 > 自动查找 > --scenario 目录内查找
    config_path = args.config
    if config_path is None:
        found = [os.path.join("output", d, "output_mc_config.csv")
                 for d in sorted(os.listdir("output"))
                 if os.path.isfile(os.path.join("output", d, "output_mc_config.csv"))]
        if len(found) == 1:
            config_path = found[0]
        elif len(found) > 1:
            raise ValueError(f"output/ 下存在多个 output_mc_config.csv ({found}), 请用 --config 指定")
    if config_path is None and args.scenario:
        dflt = os.path.join("output", args.scenario, "output_mc_config.csv")
        if os.path.exists(dflt):
            config_path = dflt
    cfg_global, comp_cfg = {}, {}
    if config_path and os.path.exists(config_path):
        cfg_global, comp_cfg = load_mc_config(config_path)
        print(f"读取场景配置: {config_path} "
              f"(全局 {len(cfg_global)} 项, 组件分布 {len(comp_cfg)} 个)")
    elif args.config:
        print(f"  警告: 未找到配置文件 {args.config}")

    # 全局键: model 索引模型库; scenario 场景名; n_samples/seed/start/end 抽样运行参数
    scenario = args.scenario or cfg_global.get("scenario")
    if not scenario:
        raise ValueError("未指定场景: 请用 --scenario 或 output_mc_config.csv 的 scenario 键")
    model = args.model or cfg_global.get("model")
    if not model:
        raise ValueError("未指定模型: 请用 --model 或 output_mc_config.csv 的 model 键 (训练集文件夹名)")
    model_dir = args.model_dir or os.path.join("knn_lib", model)
    n_samples = args.n if args.n is not None else int(cfg_global.get("n_samples", 200))
    seed = args.seed if args.seed is not None else int(cfg_global.get("seed", 42))
    start_time = args.start_time or cfg_global.get("start_time", "0:00")
    end_time = args.end_time or cfg_global.get("end_time", "23:45")

    # 1. 加载模型与列名
    fnames = json.load(open(os.path.join(model_dir, "knn_feature_names.json"), encoding="utf-8"))
    tnames = json.load(open(os.path.join(model_dir, "knn_target_names.json"), encoding="utf-8"))
    models = {s: joblib.load(os.path.join(model_dir, f"knn_model_{s}.joblib"))
              for s in ("min", "max")}
    scalers = {s: joblib.load(os.path.join(model_dir, f"knn_scaler_{s}.joblib"))
               for s in ("min", "max")}
    print(f"模型已加载: {model_dir} (特征 {len(fnames)}, 目标 {len(tnames)})")

    # 2. 加载场景网络
    net = load_network(resolve_model_path(scenario))
    shapes = {name: sh for name, sh in net.shapes.items() if sh.mult}
    print(f"场景已加载: {scenario} (曲线 {len(shapes)} 条)")

    # 3. 逐断面抽样 + 预测
    slots = _slots(start_time, end_time)
    rng = np.random.default_rng(seed)
    records = []
    print(f"断面: {_slot_label(*slots[0][:2])} ~ {_slot_label(*slots[-1][:2])} "
          f"共 {len(slots)} 个, N={n_samples}/断面")
    for h, m, t in slots:
        ts = _slot_label(h, m)
        # EV/AC 当前挂载 cur (预测时不抽样, 固定曲线值, 仅用于 output_sample 记录)
        cur_vals = {}
        for base in ("EV_Bus19", "EV_Bus7", "AC_Bus2"):
            f = f"{base}_cur"
            cur_vals[base] = shapes[f].mult[t] if f in shapes else 1.0
        X = build_slot_features(fnames, net, shapes, t, rng, comp_cfg, n_samples)
        for sense in ("min", "max"):
            pred = models[sense].predict(scalers[sense].transform(X))
            for i in range(n_samples):
                records.append((ts, i + 1, sense, pred[i], X[i], cur_vals))

    # 4. 输出 (output/{场景名}/); 模型由 mc_config 的 model 键索引到总库, 场景下不保存副本
    out_dir = os.path.join("output", scenario)
    save_outputs(out_dir, records, tnames, fnames, net)
    print(f"\n预测完成: {len(records)} 行 (断面×样本×sense), 已保存至 {out_dir}")


if __name__ == "__main__":
    main()

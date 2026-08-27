# -*- coding: utf-8 -*-
"""
train_knn.py
基于 MC 抽样 OPF 结果训练 KNN 回归模型 (根节点注入预测)

数据来源 (训练集 csv 目录由 knn_config.csv 的 dataset_csv 指定, 如 training_dataset/training_dataset_default/csv/):
  - training_dataset_sample.csv  输入特征之一: 各负荷功率 (Load1~Load32) 等抽样值
  - training_dataset_system.csv  输出目标: 根节点有功/无功 (p_sub_mw, q_sub_mvar), 按 sense 区分
  - training_dataset_storage.csv 输出目标: 各储能净有功 (p_net_mw) / 无功 (q_mvar) / 能量 (se_mwh)
  - training_dataset_pvs.csv     输出目标: 各光伏有功 (p_out_mw) / 无功 (q_out_mvar)
  - training_dataset_loads.csv   输出目标: type 为 ev/ac 的可调度负荷实际有功 (p_out_mw)

训练两个 KNN 模型 (同一套输入特征, 不同输出 sense):
  - min: 最小化根节点注入场景
  - max: 最大化根节点注入场景

特征 (每样本一行, 按 sample_id+time_slot+sense 对齐; 默认算例共 45 维):
  固定负荷功率 Load1~32 (32) + 储能初始状态 p_init/se_init (2) + PV 辐照度 (5)
  + EV/AC 可调上下限 lb/ub (6)

用法:
  python train_knn.py                                     # 训练集位置与参数均读取 knn_config.csv
  python train_knn.py --config knn_lib/{训练集文件夹名}/knn_config.csv
  python train_knn.py --k 7 --weights uniform             # CLI 显式覆盖配置

训练集位置与 KNN 参数统一在 knn_config.csv 中配置 (name,value 两列):
  dataset_csv,training_dataset/training_dataset_storage_bus18_sample/csv   ← 训练集 csv 目录 (相对工作目录)
  knn_k,5
  knn_weights,distance
  knn_test_size,0.2
  knn_seed,42
优先级: CLI 显式参数 > knn_config.csv 非空值 > 内置默认值。

结果输出 (knn_lib/{训练集文件夹名}/, 训练模型总库, 与使用场景解耦):
  knn_params.csv      模型参数 + 数据规模 (每 sense 一行)
  knn_metrics.csv     评估指标 R²/MAE/RMSE (每 sense×目标 一行)
  knn_predictions.csv 测试集逐样本预测对比 (每 sense×样本×目标 一行)
  knn_model_{min,max}.joblib / knn_scaler_{min,max}.joblib  模型与标准化器
  knn_feature_names.json / knn_target_names.json            特征/目标列名
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from save_output import save_knn_results, save_knn_model  # KNN 结果/模型导出 (output/)

# 可调度负荷类型 (training_dataset_loads.csv 的 type 列)
DISPATCHABLE_TYPES = {"ev", "ac"}
# 训练集数据目录键 (knn_config.csv 的 name 列; 值 = 相对工作目录的 csv 目录)
DATASET_KEY = "dataset_csv"
# 默认训练集 csv 目录 (--csv 与 knn_config.csv 的 dataset_csv 均未给时兜底)
DEFAULT_CSV = os.path.join("training_dataset", "training_dataset_storage_bus18_sample", "csv")
# KNN 全局参数键 (knn_config.csv 的 name 列)
KNN_KEYS = {"knn_k", "knn_weights", "knn_test_size", "knn_seed",
            "knn_metric", "knn_p", "knn_algorithm", "knn_leaf_size", "knn_n_jobs"}
# 参数默认值 (配置留空或缺失时生效; 与 sklearn KNeighborsRegressor 默认一致)
KNN_DEFAULTS = {"knn_k": 5, "knn_weights": "distance", "knn_test_size": 0.2, "knn_seed": 42,
                "knn_metric": "minkowski", "knn_p": 2, "knn_algorithm": "auto",
                "knn_leaf_size": 30, "knn_n_jobs": None}


# =====================================================================
# 数据加载与拼接 (流式读取, 只提取需要的列/行)
# =====================================================================

def _head(path: str) -> list:
    """读取 CSV 表头"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f))


def load_knn_config(path: str) -> dict:
    """从 knn_config.csv 读取 KNN 训练参数与训练集位置 (name 列) → {键: 字符串值}

    配置文件位于模型总库: knn_lib/{训练集文件夹名}/knn_config.csv。
    CSV 结构与 mc_config 一致 (前两列 name,value):
      name,value
      dataset_csv,training_dataset/training_dataset_default/csv   ← 训练集 csv 目录 (相对工作目录)
      knn_k,5
      knn_weights,distance
      knn_test_size,0.2
      knn_seed,42
      knn_metric,          ← 留空 = 使用默认值 (minkowski)
      knn_p,
      ...
    """
    cfg = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            cells = [c.strip() for c in row]
            if not cells or not cells[0] or cells[0].lower() == "name":
                continue
            # 仅收录已知键且 value 非空 (留空 = 默认值, 不写入 cfg)
            if cells[0] in (KNN_KEYS | {DATASET_KEY}) and len(cells) > 1 and cells[1]:
                cfg[cells[0]] = cells[1]
    return cfg


def resolve_knn_params(args, config_path: str = None, cfg: dict = None) -> dict:
    """合并 KNN 参数 (优先级: CLI 显式 > knn_config.csv 非空值 > 默认值)

    cfg: 已由调用方加载的 knn_config.csv 内容; 为 None 时按 config_path 自行加载。
    """
    if cfg is None:
        cfg = {}
        if config_path and os.path.exists(config_path):
            cfg = load_knn_config(config_path)
            print(f"读取 KNN 配置: {config_path}")
    get = lambda key, cast: (cast(cfg[key]) if cfg.get(key) is not None
                             else KNN_DEFAULTS[key])
    params = {
        "k": args.k if args.k is not None else get("knn_k", int),
        "weights": args.weights if args.weights is not None else get("knn_weights", str),
        "test_size": args.test_size if args.test_size is not None else get("knn_test_size", float),
        "seed": args.seed if args.seed is not None else get("knn_seed", int),
        "metric": get("knn_metric", str),
        "p": get("knn_p", int),
        "algorithm": get("knn_algorithm", str),
        "leaf_size": get("knn_leaf_size", int),
        "n_jobs": get("knn_n_jobs", int),
    }
    return params


def load_knn_dataset(csv_dir: str) -> dict:
    """加载并拼接输入特征与输出目标, 返回 {sense: (X, y, feature_names)}

    - X: numpy (n_samples, n_features), y: numpy (n_samples, 2) [p_sub, q_sub]
    - feature_names: list[str], 与 X 列一一对应
    - 仅保留 training_dataset_system 中 status=OPTIMAL 的样本
    """
    # ---- 1. training_dataset_sample: 负荷功率 + 其余抽样量 (AM~BB: PV 辐照度/储能初始/可调度区间) ----
    p = os.path.join(csv_dir, "training_dataset_sample.csv")
    hdr = _head(p)
    load_feats = [c for c in hdr if c.endswith("_mw") and c != "fixed_total_mw"]  # 排除冗余合计列
    # AM~BB 列: 非负荷、非标识的抽样量 (PV_Bus*_irr / BESS_*_se_init/p_init / 可调度 *_lb|_ub)
    # 取消 EV/AC 的 cur 列, 仅保留可调区间 lb/ub
    extra_feats = [c for c in hdr if c not in ("sample_id", "time_slot", "fixed_total_mw")
                   and not c.endswith("_mw") and not c.endswith("_cur")]
    li = {c: hdr.index(c) for c in load_feats + extra_feats}
    base = {}
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            base[(row[0], row[1])] = ([float(row[li[c]]) for c in load_feats],
                                      [float(row[li[c]]) for c in extra_feats])

    # ---- 2. training_dataset_system: 输出目标 + 状态 (按 sense) ----
    p = os.path.join(csv_dir, "training_dataset_system.csv")
    hdr = _head(p)
    ix = {c: hdr.index(c) for c in ("sample_id", "time_slot", "sense",
                                    "p_sub_mw", "q_sub_mvar", "status")}
    system = {}   # (sid, ts, sense) -> (y 向量, status)
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            key = (row[ix["sample_id"]], row[ix["time_slot"]], row[ix["sense"]])
            system[key] = ([float(row[ix["p_sub_mw"]]), float(row[ix["q_sub_mvar"]])],
                           row[ix["status"]])

    # ---- 3. training_dataset_storage: 每储能净有功/无功输出/能量状态 ----
    p = os.path.join(csv_dir, "training_dataset_storage.csv")
    hdr = _head(p)
    ix = {c: hdr.index(c) for c in ("sample_id", "time_slot", "sense",
                                    "name", "p_net_mw", "q_mvar", "se_mwh")}
    st_map, st_names = {}, []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            key = (row[ix["sample_id"]], row[ix["time_slot"]], row[ix["sense"]])
            name = row[ix["name"]]
            st_map.setdefault(key, {})[name] = (float(row[ix["p_net_mw"]]),
                                                float(row[ix["q_mvar"]]),
                                                float(row[ix["se_mwh"]]))
            if name not in st_names:
                st_names.append(name)

    # ---- 4. training_dataset_pvs: 每光伏有功/无功输出 ----
    p = os.path.join(csv_dir, "training_dataset_pvs.csv")
    hdr = _head(p)
    ix = {c: hdr.index(c) for c in ("sample_id", "time_slot", "sense",
                                    "name", "p_out_mw", "q_out_mvar")}
    pv_map, pv_names = {}, []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            key = (row[ix["sample_id"]], row[ix["time_slot"]], row[ix["sense"]])
            name = row[ix["name"]]
            pv_map.setdefault(key, {})[name] = (float(row[ix["p_out_mw"]]), float(row[ix["q_out_mvar"]]))
            if name not in pv_names:
                pv_names.append(name)

    # ---- 5. training_dataset_loads: 仅 type 为 ev/ac 的负荷实际有功 (跳过其余 ~123 万行) ----
    p = os.path.join(csv_dir, "training_dataset_loads.csv")
    hdr = _head(p)
    ix = {c: hdr.index(c) for c in ("sample_id", "time_slot", "sense",
                                    "name", "type", "p_out_mw")}
    ld_map, ld_names = {}, []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            if row[ix["type"]] not in DISPATCHABLE_TYPES:
                continue
            key = (row[ix["sample_id"]], row[ix["time_slot"]], row[ix["sense"]])
            name = row[ix["name"]]
            ld_map.setdefault(key, {})[name] = float(row[ix["p_out_mw"]])
            if name not in ld_names:
                ld_names.append(name)

    # ---- 输入特征: 仅 training_dataset_sample (负荷 + AM~BB 抽样量) ----
    feature_names = load_feats + extra_feats

    # ---- 输出目标: 根节点注入 + 资源出力 (原 training_dataset_storage/pvs/loads 特征转为目标) ----
    target_names = (["p_sub_mw", "q_sub_mvar"]
                    + [f"{n}_p_net_mw" for n in st_names] + [f"{n}_q_mvar" for n in st_names]
                    + [f"{n}_se_mwh" for n in st_names]
                    + [f"{n}_p_out_mw" for n in pv_names] + [f"{n}_q_out_mvar" for n in pv_names]
                    + [f"{n}_p_out_mw" for n in ld_names])

    # ---- 拼接 (按 sense 拆分) ----
    datasets = {}
    for sense in ("min", "max"):
        X_rows, y_rows, keys = [], [], []
        for (sid, ts), (load_vals, extra_vals) in base.items():
            key = (sid, ts, sense)
            if key not in system:
                continue
            y_sys, status = system[key]
            if status != "OPTIMAL":
                continue
            st_vals = st_map.get(key, {})
            pv_vals = pv_map.get(key, {})
            ld_vals = ld_map.get(key, {})
            X_rows.append(list(load_vals) + list(extra_vals))
            y_row = list(y_sys)
            for n in st_names:
                v = st_vals.get(n, (0.0, 0.0, 0.0))
                y_row += [v[0], v[1], v[2]]
            for n in pv_names:
                v = pv_vals.get(n, (0.0, 0.0))
                y_row += [v[0], v[1]]
            for n in ld_names:
                y_row.append(ld_vals.get(n, 0.0))
            y_rows.append(y_row)
            keys.append((sid, ts))
        datasets[sense] = (np.asarray(X_rows, dtype=float),
                           np.asarray(y_rows, dtype=float),
                           feature_names, target_names, keys)
    return datasets


# =====================================================================
# KNN 训练
# =====================================================================

def train_knn(X: np.ndarray, y: np.ndarray, k: int = 5, weights: str = "distance",
              test_size: float = 0.2, random_state: int = 42,
              metric: str = "minkowski", p: int = 2, algorithm: str = "auto",
              leaf_size: int = 30, n_jobs: int = None, keys: list = None,
              target_names: list = None) -> dict:
    """训练 KNN 多输出回归模型 (根节点注入 + 资源出力联合预测)

    输出目标由 load_knn_dataset 动态生成: p_sub/q_sub + 储能 p_net/q/se
    + PV p_out/q_out + EV/AC p_out。

    metric/algorithm/leaf_size/n_jobs 与 sklearn KNeighborsRegressor 对应,
    由 knn_config.csv 的 knn_* 键控制 (留空取默认值)。

    keys: 与 X 行对应的样本键列表 [(sample_id, time_slot), ...],
          用于回溯测试集对应哪些样本 (输出预测对比 CSV 时使用)。
    target_names: 输出目标列名列表 (与 y 列一一对应)。

    返回 dict:
      model/scaler      : 训练好的 KNN 模型与标准化器
      X_test/y_test/pred: 测试集与预测值 (便于外部评估)
      keys_test         : 测试集对应的样本键列表 (keys 为 None 时亦为 None)
      target_names      : 输出目标列名列表
      metrics           : {r2: list, mae: list, rmse: list} (每输出列一个)
      n_train/n_test    : 样本数
    """
    indices = np.arange(X.shape[0])
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, indices, test_size=test_size, random_state=random_state)
    scaler = StandardScaler().fit(Xtr)
    model = KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric, p=p,
                                algorithm=algorithm, leaf_size=leaf_size, n_jobs=n_jobs)
    model.fit(scaler.transform(Xtr), ytr)
    pred = model.predict(scaler.transform(Xte))

    metrics = {
        "r2": [float(r2_score(yte[:, j], pred[:, j])) for j in range(y.shape[1])],
        "mae": [float(mean_absolute_error(yte[:, j], pred[:, j])) for j in range(y.shape[1])],
        "rmse": [float(root_mean_squared_error(yte[:, j], pred[:, j])) for j in range(y.shape[1])],
    }
    return {"model": model, "scaler": scaler,
            "X_test": Xte, "y_test": yte, "pred": pred, "metrics": metrics,
            "keys_test": [keys[i] for i in ite] if keys is not None else None,
            "target_names": target_names,
            "n_train": Xtr.shape[0], "n_test": Xte.shape[0]}


def print_knn_result(sense: str, feat_names: list, out: dict) -> None:
    """打印单 sense 的训练评估结果 (每个输出目标一行)"""
    m = out["metrics"]
    mdl = out["model"]
    print(f"\n  [{sense}] KNN (k={mdl.n_neighbors}, weights={mdl.weights}, "
          f"metric={mdl.metric}, p={mdl.p}, algorithm={mdl.algorithm})")
    print(f"    样本: 训练 {out['n_train']}, 测试 {out['n_test']}, "
          f"特征 {len(feat_names)}, 输出目标 {len(out['target_names'])}")
    print(f"    {'目标':<24}{'R2':>10}{'MAE':>12}{'RMSE':>12}")
    for j, col in enumerate(out["target_names"]):
        print(f"    {col:<24}{m['r2'][j]:>10.4f}{m['mae'][j]:>12.6f}{m['rmse'][j]:>12.6f}")


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="训练根节点注入 KNN 回归模型 (min/max)")
    parser.add_argument("--csv", default=None,
                        help="结果 CSV 目录 (覆盖 knn_config.csv 的 dataset_csv; 缺省读配置)")
    parser.add_argument("--config", default=None,
                        help="knn_config.csv 路径 (缺省 knn_lib/{训练集文件夹名}/knn_config.csv)")
    parser.add_argument("--k", type=int, default=None, help="KNN 邻居数 (覆盖配置)")
    parser.add_argument("--weights", default=None, choices=["uniform", "distance"],
                        help="权重方式 (覆盖配置)")
    parser.add_argument("--test-size", type=float, default=None,
                        help="测试集比例 (覆盖配置)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖配置)")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)  # 逐行刷新, 便于观察进度

    # 1) 定位配置文件: --config 显式; 否则按 --csv/内置默认推导的训练集名找 knn_lib/{训练集名}/
    config_path = args.config
    if config_path is None:
        probe = args.csv or DEFAULT_CSV
        default_cfg = os.path.join("knn_lib",
                                   os.path.basename(os.path.dirname(probe)), "knn_config.csv")
        if os.path.exists(default_cfg):
            config_path = default_cfg
    elif not os.path.exists(config_path):
        sys.exit(f"错误: 指定的配置文件不存在: {config_path}")
    # 2) 读取配置 (含训练集位置 dataset_csv 与 KNN 参数)
    cfg = {}
    if config_path and os.path.exists(config_path):
        cfg = load_knn_config(config_path)
        print(f"读取 KNN 配置: {config_path}")
    # 3) 最终训练集 csv 目录: CLI --csv > knn_config.csv 的 dataset_csv > 内置默认
    csv_dir = args.csv or cfg.get(DATASET_KEY) or DEFAULT_CSV
    train_set = os.path.basename(os.path.dirname(csv_dir))
    params = resolve_knn_params(args, config_path, cfg)

    datasets = load_knn_dataset(csv_dir)
    print(f"加载数据集: {csv_dir}")
    for sense in ("min", "max"):
        X, y, feat_names, target_names, keys = datasets[sense]
        print(f"  [{sense}] 有效样本 {X.shape[0]} 个, 特征 {X.shape[1]} 个, 输出目标 {y.shape[1]} 个")

    sense_data = {}
    for sense in ("min", "max"):
        X, y, feat_names, target_names, keys = datasets[sense]
        out = train_knn(X, y, k=params["k"], weights=params["weights"],
                        test_size=params["test_size"], random_state=params["seed"],
                        metric=params["metric"], p=params["p"], algorithm=params["algorithm"],
                        leaf_size=params["leaf_size"], n_jobs=params["n_jobs"],
                        keys=keys, target_names=target_names)
        print_knn_result(sense, feat_names, out)
        sense_data[sense] = out

    # 训练产物输出到 knn_lib/{训练集文件夹名}/ (模型总库, 与使用场景解耦)
    out_dir = os.path.join("knn_lib", train_set)
    save_knn_results(out_dir, sense_data, params, X.shape[1], csv_dir)
    save_knn_model(out_dir, sense_data, datasets["min"][2])  # 保存模型供预测复用
    print(f"\nKNN 训练结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()

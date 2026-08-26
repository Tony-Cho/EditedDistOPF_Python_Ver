# -*- coding: utf-8 -*-
"""
save_output.py
KNN 代理模型结果导出模块 (输出到根目录 knn_lib/{训练集文件夹名}/)

提供入口:
    save_knn_results(out_dir, sense_data, params, n_features, data_csv)

与 save_results.py 的职责区分:
  - save_results.py : OPF 优化结果导出 → training_dataset/{scenario}/ 与 training_dataset/{scenario}_sample/csv/
  - save_output.py  : KNN 训练结果导出 → knn_lib/{训练集文件夹名}/
"""

from __future__ import annotations

import csv
import json
import os

import joblib


def save_knn_model(out_dir: str, sense_data: dict, feature_names: list) -> None:
    """保存 KNN 模型 + 标准化器 + 特征/目标列名 (供 scenario_dataset_mc.py 复用)

    - knn_model_{sense}.joblib       KNN 回归模型 (KNeighborsRegressor)
    - knn_scaler_{sense}.joblib      StandardScaler (训练集拟合)
    - knn_feature_names.json         输入特征列名 (48, 顺序与模型一致)
    - knn_target_names.json          输出目标列名 (18, 与 y 列一一对应)
    """
    first = next(iter(sense_data.values()))
    with open(os.path.join(out_dir, "knn_feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "knn_target_names.json"), "w", encoding="utf-8") as f:
        json.dump(first["target_names"], f, ensure_ascii=False, indent=1)
    for sense, out in sorted(sense_data.items()):
        joblib.dump(out["model"], os.path.join(out_dir, f"knn_model_{sense}.joblib"))
        joblib.dump(out["scaler"], os.path.join(out_dir, f"knn_scaler_{sense}.joblib"))
    print(f"  模型已保存: {os.path.join(out_dir, 'knn_model_{min,max}.joblib')} "
          f"(特征 {len(feature_names)} 个, 目标 {len(first['target_names'])} 个)")


def save_knn_results(out_dir: str, sense_data: dict, params: dict,
                     n_features: int, data_csv: str) -> None:
    """将 KNN 训练结果写入 out_dir (3 张 CSV):

    - knn_params.csv      模型参数 + 数据规模 (每 sense 一行)
    - knn_metrics.csv     评估指标 R²/MAE/RMSE (每 sense×目标 一行)
    - knn_predictions.csv 测试集逐样本预测对比 (每 sense×样本×目标 一行)

    sense_data: {sense: train_knn 返回的 dict}
    目标列名取自 out["target_names"] (与 y 列一一对应, 由 load_knn_dataset 动态生成)。
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- knn_params.csv ----
    with open(os.path.join(out_dir, "knn_params.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sense", "k", "weights", "metric", "p", "algorithm",
                    "leaf_size", "n_jobs", "test_size", "seed",
                    "n_train", "n_test", "n_features", "n_targets", "data_csv"])
        for sense, out in sorted(sense_data.items()):
            w.writerow([sense, out["model"].n_neighbors, out["model"].weights,
                        out["model"].metric, out["model"].p, out["model"].algorithm,
                        out["model"].leaf_size, out["model"].n_jobs if out["model"].n_jobs else "",
                        params["test_size"], params["seed"],
                        out["n_train"], out["n_test"], n_features,
                        len(out["target_names"]), data_csv])
    print(f"  已保存: {os.path.join(out_dir, 'knn_params.csv')}")

    # ---- knn_metrics.csv ----
    with open(os.path.join(out_dir, "knn_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sense", "target", "r2", "mae", "rmse"])
        for sense, out in sorted(sense_data.items()):
            m = out["metrics"]
            for j, col in enumerate(out["target_names"]):
                w.writerow([sense, col, f"{m['r2'][j]:.6f}",
                            f"{m['mae'][j]:.6f}", f"{m['rmse'][j]:.6f}"])
    print(f"  已保存: {os.path.join(out_dir, 'knn_metrics.csv')}")

    # ---- knn_predictions.csv ----
    with open(os.path.join(out_dir, "knn_predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sense", "sample_id", "time_slot", "target",
                    "y_true", "y_pred", "abs_err"])
        for sense, out in sorted(sense_data.items()):
            yte, pred, keys_test = out["y_test"], out["pred"], out["keys_test"]
            for i, (sid, ts) in enumerate(keys_test):
                for j, col in enumerate(out["target_names"]):
                    yt = yte[i, j]
                    yp = pred[i, j]
                    w.writerow([sense, sid, ts, col, f"{yt:.6f}", f"{yp:.6f}",
                                f"{abs(yt - yp):.6f}"])
    print(f"  已保存: {os.path.join(out_dir, 'knn_predictions.csv')}")

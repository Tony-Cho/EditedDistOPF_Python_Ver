# -*- coding: utf-8 -*-
"""
eval_knn.py — 用合成匹配数据集正确评估 KNN 泛化能力

要点：KNN 在【整个 train 集上天】完整训练（不自切），
然后在【同分布的不同天】的 val / test 上评估 → 这才是 match 的验证（train/val/test 同源不同随机日）。

特征/目标构建复用 train_knn.load_knn_dataset（同一套列约定，保证与训练时完全一致）。

用法：
  python eval_knn.py --train dataset/model_default/train --val dataset/model_default/val \
                     --test dataset/model_default/test --k 5
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from train_knn import load_knn_dataset


def fit_full(X, y, k, weights="distance"):
    scaler = StandardScaler().fit(X)
    model = KNeighborsRegressor(n_neighbors=k, weights=weights)
    model.fit(scaler.transform(X), y)
    return scaler, model


def report(name, y_true, pred, target_names):
    print(f"\n=== {name} ===")
    print(f"  样本 {y_true.shape[0]} 个, 输出目标 {target_names.shape[0] if hasattr(target_names,'shape') else len(target_names)} 个")
    print(f"  {'目标':<24}{'R2':>10}{'MAE':>12}{'RMSE':>12}")
    r2s, maes, rms = [], [], []
    for j, col in enumerate(target_names):
        r2 = r2_score(y_true[:, j], pred[:, j])
        mae = mean_absolute_error(y_true[:, j], pred[:, j])
        rmse = root_mean_squared_error(y_true[:, j], pred[:, j])
        r2s.append(r2); maes.append(mae); rms.append(rmse)
        print(f"  {col:<24}{r2:>10.4f}{mae:>12.6f}{rmse:>12.6f}")
    print(f"  {'~平均~':<24}{np.mean(r2s):>10.4f}{np.mean(maes):>12.6f}{np.mean(rms):>12.6f}")
    return r2s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    train_ds = load_knn_dataset(args.train)
    print(f"加载训练集: {args.train}", [d[0].shape for d in train_ds.values()])

    for sense in ("min", "max"):
        Xtr, ytr, feats, targets, _ = train_ds[sense]
        scaler, model = fit_full(Xtr, ytr, args.k)
        print(f"\n[{sense}] 模型: KNN k={args.k}, 特征 {feats.shape[0] if hasattr(feats,'shape') else len(feats)} 个")
        tr_pred = model.predict(scaler.transform(Xtr))
        report(f"train (fit) [{sense}]", ytr, tr_pred, targets)
        for tag in ("val", "test"):
            if getattr(args, tag):
                ds = load_knn_dataset(getattr(args, tag))[sense]
                Xe, ye, _, tnames, _ = ds
                pred = model.predict(scaler.transform(Xe))
                report(f"{tag} [{sense}]", ye, pred, tnames)


if __name__ == "__main__":
    main()
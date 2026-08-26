# -*- coding: utf-8 -*-
"""
sampling.py
抽样函数库 (独立模块, 供 training_dataset_mc 等复用)

内置分布 (后续可补充):
- truncated_normal : 截断正态 (拒绝采样, 无需 scipy), 参数 mu/sigma/lo/hi
- uniform          : 均匀分布, 参数 lo/hi

统一入口:
    from sampling import sample, truncated_normal, truncated_normal_vec, uniform
    x  = sample("truncated_normal", {"mu": 0.5, "sigma": 0.05, "lo": 0.0, "hi": 1.0}, rng)
    xs = truncated_normal_vec(mu, sigma, lo, hi, rng)   # 批量 (输入为数组)
"""

from __future__ import annotations

import math

import numpy as np


def truncated_normal(mu, sigma, lo, hi, rng):
    """标量截断正态采样 (拒绝采样): X ∈ [lo, hi], E[X] ≈ mu"""
    while True:
        x = rng.normal(mu, sigma)
        if lo <= x <= hi:
            return x


def truncated_normal_vec(mu, sigma, lo, hi, rng):
    """向量化截断正态采样 (拒绝采样): 输入均为数组, 输出与 mu 同形状"""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = mu.size
    out = np.empty(n)
    mask = np.ones(n, dtype=bool)
    while mask.any():
        x = rng.normal(mu, sigma)
        ok = (x >= lo) & (x <= hi)
        take = mask & ok
        out[take] = x[take]
        mask &= ~ok
    return out


def uniform(lo, hi, rng):
    """标量均匀分布采样: X ∈ [lo, hi]"""
    return rng.uniform(lo, hi)


# 已注册分布 (便于扩展与提示)
AVAILABLE = {
    "truncated_normal": "mu, sigma, lo(可选), hi(可选) 或由调用方由 cv×mu 折算",
    "uniform": "lo, hi",
}


def sample(dist: str, params: dict, rng):
    """按分布名 + 参数字典采样单个值。

    params 须含分布要求的参数 (见 AVAILABLE); lo/hi 缺失时默认 ±inf。
    """
    name = (dist or "").strip().lower()
    if name == "truncated_normal":
        return truncated_normal(
            float(params["mu"]),
            float(params["sigma"]),
            float(params.get("lo", -math.inf)),
            float(params.get("hi", math.inf)),
            rng,
        )
    if name == "uniform":
        return rng.uniform(float(params["lo"]), float(params["hi"]))
    raise ValueError(f"未知分布: {dist!r} (可用: {', '.join(AVAILABLE)})")

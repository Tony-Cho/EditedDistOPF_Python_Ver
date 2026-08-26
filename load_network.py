# -*- coding: utf-8 -*-
"""
load_network.py
配电网模型加载模块 (独立, 供 main.py / training_dataset_mc.py 等复用)

自动识别输入并加载网络:
- .dss 文件          → parse_dss (DSS 解析器)
- 目录含 circuit.csv → parse_csv (CSV 解析器)
- 目录含 Master.dss  → parse_dss(目录/Master.dss)

用法:
    from load_network import load_network, resolve_model_path

    net = load_network("data/csv_case33/model_x")       # CSV 模型目录 (前缀 model_)
    net = load_network("data/opendss_case33/scenario_x")   # DSS 场景目录
    net = load_network("xxx.dss")                          # DSS 文件
    net = load_network(resolve_model_path("scenario_x"))   # 场景名 → 路径 (CSV 优先)
"""

from __future__ import annotations

import os

from parse_dss import parse_dss
from parse_csv import parse_csv


# 场景数据根目录 (可按需修改): {CSV_DATA_DIR}/{model}/, {SCENARIO_DATA_DIR}/{scenario}/ 与 {DSS_DATA_DIR}/{scenario}/
CSV_DATA_DIR = "data/csv_case33"        # 旧场景目录 (文件夹与文件前缀均 model_*)
SCENARIO_DATA_DIR = "scenario"          # 新场景目录 (根目录下, 文件夹前缀 scenario_*)
DSS_DATA_DIR = "data/opendss_case33"


def resolve_model_path(scenario: str) -> str:
    """场景名 → 模型路径: CSV 场景目录优先 (data/csv_case33 再 scenario/), 退回 DSS 场景目录。

    data/csv_case33 下的模型目录现以 model_ 为前缀 (如 model_storage_bus18);
    传入的 scenario 名兼容 scenario_/model_ 两种前缀, 自动映射。

    直接传入 .dss 文件 / 已存在的目录时原样返回。
    """
    if scenario.lower().endswith(".dss") or os.path.isdir(scenario):
        return scenario
    # 兼容前缀: 如 scenario_storage_bus18 ↔ model_storage_bus18
    names = [scenario]
    alt = {"scenario_": "model_", "model_": "scenario_"}
    for prefix in alt:
        if scenario.startswith(prefix):
            names.append(alt[prefix] + scenario[len(prefix):])
    # output_ 前缀: 去除 output_ 后使用原场景名 (如 output_scenario_trail_1 → scenario_trail_1)
    if scenario.startswith("output_"):
        names.append(scenario[7:])
    for root in (CSV_DATA_DIR, SCENARIO_DATA_DIR):
        for name in names:
            d = os.path.join(root, name)
            if os.path.isdir(d):
                return d                   # CSV 场景目录优先
    return os.path.join(DSS_DATA_DIR, scenario)   # DSS 场景目录 (load_network 内找 Master.dss)


def _preprocess_dispatchable(net):
    """可调度负荷预处理 (统一层, 覆盖 parse_csv / parse_dss 两路):
    - 可调下限 lb 高于当前挂载 cur 时收紧为 cur (负荷不允许再往下削减),
      避免 z_lb = lb/cur > 1 使 OPF 不可行或可调区间异常;
    - 可调上限 ub 低于当前挂载 cur 时放松为 cur (当前时段负荷高于上报 ub 均值时,
      ub 均值放松至当前负荷, 保证 cur ≤ ub), 避免 z_ub < 1 压缩当前负荷。
    """
    for ld in net.loads.values():
        if not ld.dispatchable:
            continue
        if ld.mult_lb > ld.base_ratio:
            print(f"  提示: 负荷 {ld.name} 可调下限 {ld.mult_lb:.4f} 高于当前挂载 "
                  f"{ld.base_ratio:.4f}, 已收紧为当前挂载 (不允许继续削减)")
            ld.mult_lb = ld.base_ratio
        if ld.mult_ub < ld.base_ratio:
            print(f"  提示: 负荷 {ld.name} 可调上限 {ld.mult_ub:.4f} 低于当前挂载 "
                  f"{ld.base_ratio:.4f}, 已放松为当前挂载 (保证当前负荷可达)")
            ld.mult_ub = ld.base_ratio
        ld.z_lb = ld.mult_lb / ld.base_ratio if ld.base_ratio > 0 else 0.0
        ld.z_ub = ld.mult_ub / ld.base_ratio if ld.base_ratio > 0 else 1e6
    return net


def load_network(model_path: str):
    """自动识别输入并加载网络:
    - .dss 文件          → parse_dss
    - 目录含 circuit.csv → parse_csv
    - 目录含 Master.dss  → parse_dss(目录/Master.dss)
    """
    if model_path.lower().endswith(".dss"):
        return _preprocess_dispatchable(parse_dss(model_path))
    if os.path.isdir(model_path):
        if (os.path.exists(os.path.join(model_path, "model_circuit.csv"))
                or os.path.exists(os.path.join(model_path, "scenario_circuit.csv"))):
            return _preprocess_dispatchable(parse_csv(model_path))
        master = os.path.join(model_path, "Master.dss")
        if os.path.exists(master):
            return _preprocess_dispatchable(parse_dss(master))
        raise FileNotFoundError(f"目录中未找到 circuit.csv 或 Master.dss: {model_path}")
    raise FileNotFoundError(f"无法识别输入: {model_path}")

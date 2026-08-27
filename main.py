# -*- coding: utf-8 -*-
"""
main.py
IEEE 33 配电网 VPP OPF 优化主程序 (Python + Gurobi 重构版)

对应原 Julia 项目 main.jl 的功能：
1. 加载 OpenDSS 模型 (IEEE 33 + VPP 资源)
2. 运行 VPP OPF 优化 (最小化/最大化根节点注入功率)
3. 保存结果到 JSON

差异说明：
- 求解器: Ipopt (NLP) → Gurobi (LP)
- 模型:   AC OPF (ACPUPowerModel) → LinDistFlow (线性化支路潮流)
- 系统:   三相非平衡显式建模 → 三相平衡单相等值
"""

from __future__ import annotations # 启用类型提示

import csv
import io
import os
import shutil
import sys

from parse_dss import Network # 网络数据结构类型标注 (运行时鸭子类型)
from opf_model import build_and_solve_opf, print_result_summary # OPF 模型函数
from save_results import save_case_results # 结果导出模块
from load_network import load_network, resolve_model_path # 模型加载 (独立模块)


# =====================================================================
# 路径与场景配置 (集中管理, 可按需修改)
# =====================================================================
DEFAULT_SCENARIO = "model_default"   # 未指定命令行参数时的默认场景 (默认算例)
RESULT_DIR = "demo_result"          # 结果输出根目录: {RESULT_DIR}/{场景名}/{场景名}.txt + 组件 CSV


def demo_result_name(scenario_name: str) -> str:
    """将 model_/scenario_ 前缀替换为 demo_result_, 如 model_storage_bus18 → demo_result_storage_bus18"""
    prefix = next((p for p in ("model_", "scenario_") if scenario_name.startswith(p)), None)
    return "demo_result_" + scenario_name[len(prefix):] if prefix else scenario_name


# =====================================================================
# 后处理工具
# =====================================================================

def print_summary(result: dict):
    """打印简要摘要 (兼容 Julia print_summary)"""
    print(f"Termination status: {result['termination_status']}")
    if "objective" in result:
        print(f"Objective value: {result['objective']}")

def get_scenario_name(dss_path: str) -> str:
    """从模型路径提取场景名称 (如 'model_storage_bus18' 或 'scenario_max_a')"""
    parts = dss_path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("scenario_") or p.startswith("model_"):
            return p
    return "default"


# =====================================================================
# VPP 资源展示 (对应 Julia add_vpp_resources!)
# =====================================================================

def show_vpp_resources(network: Network):
    """展示已加载的 VPP 资源 (PV 和储能) 信息"""
    print("检查 VPP 资源...")
    print(f"  - 光伏 (solar): {len(network.pvs)} 个")
    print(f"  - 储能 (storage): {len(network.storages)} 个")

    if network.pvs:
        print("  光伏列表:")
        for pv in network.pvs.values():
            print(f"    - {pv.name}: bus={pv.bus}, P_max={pv.pmpp_kw} kW, "
                  f"irradiance={pv.irradiance} (可用 {pv.p_avail_pu * network.base_mva:.4f} MW)")
        # 设置 PV 无功上下限 (按各 PV 数据字段 pf 计算)
        print(f"  设置光伏无功上下限 (按各 PV 的 pf)...")
        from parse_dss import pf_to_q_factor
        for pv in network.pvs.values():
            q_max_kw = pv.pmpp_kw * pf_to_q_factor(pv.pf)
            print(f"    {pv.name}: pf={pv.pf}, qg ∈ [{-q_max_kw:.4f}, {q_max_kw:.4f}] kvar (总)")
    print("  注意: 当前光伏出力在优化中是可调的 (0~100%)")

    if network.storages:
        print("  储能列表:")
        for st in network.storages.values():
            print(f"    - {st.name}: bus={st.bus}, kW={st.kw}, kWh={st.kwh}")

        print("  储能参数确认:")
        for st in network.storages.values():
            print(f"    {st.name}: energy_ub={st.energy_ub_pu:.4f} pu·h, "
                  f"charge_ub={st.charge_ub_pu:.4f} pu, "
                  f"discharge_ub={st.discharge_ub_pu:.4f} pu")

    # 可调度负荷识别
    print("  自动识别可调度负荷...")
    dispatchable_count = sum(1 for ld in network.loads.values() if ld.dispatchable)
    for ld in network.loads.values():
        if ld.dispatchable:
            print(f"    设置 {ld.name} → dispatchable")
    print(f"    共识别 {dispatchable_count} 个可调度负荷")


# =====================================================================
# 主流程 (对应 Julia run_vpp_demo)
# =====================================================================

def run_vpp_demo(model_path: str) -> tuple:
    """运行 VPP 优化演示 (两个场景: min 和 max)"""
    print("=" * 50)
    print("VPP 虚拟电厂优化演示 (Python + Gurobi)")
    print("=" * 50)
    print()

    # 1. 加载配电网模型 (自动识别 DSS 文件 / CSV 目录)
    print("1. 加载配电网模型...")
    network = load_network(model_path)

    # 2. 检查 VPP 资源
    print()
    print("3. 检查 VPP 资源...")
    show_vpp_resources(network)

    # 3. 运行 VPP OPF 优化
    print()
    print("4. 运行 VPP OPF 优化...")

    # 场景1: 最小化根节点注入
    print("\n" + "-" * 40)
    print("【场景1】最小化根节点注入功率")
    print("-" * 40)
    result_min, var_min = build_and_solve_opf(network, "min", verbose=False)
    print()
    print("=" * 50)
    print("场景1 结果: 最小化根节点注入功率")
    print("=" * 50)
    print_result_summary(result_min, network, var_min)

    # 场景2: 最大化根节点注入
    print("\n" + "-" * 40)
    print("【场景2】最大化根节点注入功率")
    print("-" * 40)
    result_max, var_max = build_and_solve_opf(network, "max", verbose=False)
    print()
    print("=" * 50)
    print("场景2 结果: 最大化根节点注入功率")
    print("=" * 50)
    print_result_summary(result_max, network, var_max)

    return result_min, result_max, network, var_min, var_max # 返回结果、网络、变量值


# =====================================================================
# 主入口
# =====================================================================

def main():
    # 支持命令行指定场景名 / 路径: python main.py scenario_storage_bus18
    #   - 场景名: 优先 data/csv_case33/{model} (CSV 目录, 前缀 model_), 退回 data/opendss_case33/{scenario} (DSS 目录)
    #   - 也支持直接传 .dss 文件、CSV 目录或 DSS 场景目录, 保持向后兼容
    scenario = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCENARIO
    # 场景名 → 路径 (CSV 目录优先, DSS 退回); 直接传 .dss 文件/目录时原样返回
    model_path = resolve_model_path(scenario)

    print("Loading case...")
    print()
    print("-" * 40)
    print("Running VPP OPF...")

    # 自动从 model_path 提取场景名称, 输出统一到 demo_result/{场景名}/
    scenario_name = get_scenario_name(model_path)
    out_name = demo_result_name(scenario_name)
    case_dir = os.path.join(RESULT_DIR, out_name)
    csv_dir = os.path.join(case_dir, "csv")          # 组件 CSV 统一放 csv/ 子目录
    shutil.rmtree(case_dir, ignore_errors=True)      # 清空旧输出, 避免新旧混淆
    os.makedirs(csv_dir, exist_ok=True)
    output_txt = os.path.join(case_dir, f"{out_name}.txt")

    # 捕获 print 输出到字符串缓冲区
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()

    try:
        result_min, result_max, network, var_min, var_max = run_vpp_demo(model_path)

        print()
        print_summary(result_min)
        print_summary(result_max)

        save_case_results(network, result_min, var_min, result_max, var_max,
                          csv_dir)

        print("Done.")
    finally:
        # 恢复 stdout 并写入文件
        sys.stdout = old_stdout
        output_text = buf.getvalue()

    # 打印到控制台
    print(output_text, end="")

    # 保存概览到 txt 文件
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"   概览已保存: {output_txt}")
    print(f"   组件 CSV 已保存至: {csv_dir}")


if __name__ == "__main__":
    main()

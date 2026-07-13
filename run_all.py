# -*- coding: utf-8 -*-
"""
==============================================================================
一键运行入口 — 区域健康风险预测与智能诊疗决策
功能: 按顺序执行完整数据处理流水线
用法: python run_all.py [--skip-download]
==============================================================================
"""

import os, sys, argparse, time
from pathlib import Path
from datetime import datetime


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║    区域健康风险预测与智能诊疗决策                              ║
║    Health Risk Prediction & Intelligent Clinical Decision     ║
║    完整数据处理流水线                                         ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def run_module(module_name, description):
    """运行单个模块并计时"""
    print(f"\n{'#'*60}")
    print(f"# {description}")
    print(f"# 模块: {module_name}")
    print(f"{'#'*60}")

    start = time.time()
    try:
        exec(f"import {module_name}; {module_name}.main()")
        elapsed = time.time() - start
        print(f"\n✅ {description} — 完成 ({elapsed:.0f}s)")
        return True, elapsed
    except ImportError as e:
        elapsed = time.time() - start
        print(f"\n⚠️  {description} — 跳过 (模块未安装: {e})")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ {description} — 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description="区域健康风险预测与智能诊疗决策 — 一键运行")
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过数据下载 (数据已存在时使用)")
    parser.add_argument("--task", type=int, choices=[1, 2, 3],
                        help="仅运行指定任务 (1=数据, 2=模型, 3=可视化)")
    args = parser.parse_args()

    print_banner()
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {os.getcwd()}")
    print()

    total_start = time.time()
    results = {}

    # ============================================================
    # 任务一: 多模态医疗数据获取与隐私安全融合
    # ============================================================
    if args.task is None or args.task == 1:
        print("\n" + "=" * 60)
        print("  📦 任务一: 多模态医疗数据获取与隐私安全融合")
        print("=" * 60)

        # 3.1 数据获取
        if not args.skip_download:
            ok, t = run_module("data_acquisition", "3.1 多源公开医疗数据集采集")
            results["data_acquisition"] = {"ok": ok, "time": t}
        else:
            print("\n⏭️  跳过数据下载 (--skip-download)")

        # 3.2 数据清洗
        ok, t = run_module("data_cleaning", "3.2 数据清洗与标准化")
        results["data_cleaning"] = {"ok": ok, "time": t}

        # 3.3 数据融合
        ok, t = run_module("data_fusion", "3.3 多模态数据融合与特征工程")
        results["data_fusion"] = {"ok": ok, "time": t}

        # 3.4 隐私保护
        ok, t = run_module("privacy_protection", "3.4 隐私安全方案设计")
        results["privacy_protection"] = {"ok": ok, "time": t}

    # ============================================================
    # 任务二: 健康风险评估与趋势预测建模 (后续阶段)
    # ============================================================
    if args.task is None or args.task == 2:
        print("\n" + "=" * 60)
        print("  🤖 任务二: 健康风险评估与趋势预测建模")
        print("=" * 60)

        print("""
⚠️  任务二需要创建以下模块:
    - chronic_disease_model.py  (4.1 慢性病风险评估)
    - health_trend_forecast.py  (4.2 人群健康趋势分析)
    - medical_resource_forecast.py (4.3 医疗资源需求预测)

请确保上述文件已存在于 src/ 目录下。
        """)

    # ============================================================
    # 任务三: 数据可视化与公共卫生决策支持 (后续阶段)
    # ============================================================
    if args.task is None or args.task == 3:
        print("\n" + "=" * 60)
        print("  📊 任务三: 数据可视化与公共卫生决策支持")
        print("=" * 60)

        print("""
⚠️  任务三需要创建以下模块:
    - dashboard_app.py         (5.1 Web 可交互式可视化面板)
    - report_generator.py      (5.2 报告自动导出)

请确保上述文件已存在于 src/ 目录下。
        """)

    # 总结
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  执行总结")
    print("=" * 60)

    for name, result in results.items():
        status = "✅" if result["ok"] else "❌"
        print(f"  {status} {name}: {result['time']:.0f}s")

    print(f"\n总耗时: {total_elapsed:.0f}s")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 输出成果清单
    print("\n📋 任务一输出成果检查:")
    check_outputs([
        ("清洗后数据", "data/processed/", "*.parquet"),
        ("融合特征矩阵", "data/fused/", "*.parquet"),
        ("隐私保护数据", "data/privacy_protected/", "*.parquet"),
        ("数据获取报告", "data/", "data_acquisition_report.md"),
        ("数据清洗报告", "data/", "data_cleaning_report.md"),
        ("数据融合报告", "data/", "data_fusion_report.md"),
        ("隐私保护报告", "data/", "privacy_protection_report.md"),
    ])

    print("\n✨ 流水线执行完毕!")


def check_outputs(items):
    """检查输出文件是否存在"""
    for name, directory, pattern in items:
        p = Path(directory)
        if p.exists():
            matches = list(p.glob(pattern))
            if matches:
                print(f"  ✅ {name}: {len(matches)} 个文件")
            else:
                print(f"  ⚠️  {name}: 无匹配文件 ({directory}/{pattern})")
        else:
            print(f"  ⚠️  {name}: 目录不存在 ({directory})")


if __name__ == "__main__":
    # 确保在项目根目录运行
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    sys.path.insert(0, str(Path(__file__).parent))
    main()

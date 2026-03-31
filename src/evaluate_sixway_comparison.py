import json
from evaluation_core import run_unified_evaluation


def run_sixway_comparison(dataset_path="data/mini_dev.json", test_limit=None):
    """
    论文风格六组对比（按你当前项目能力映射）：
    1) official_no_hints
    2) enhanced_no_hints
    3) fused_no_hints
    4) official_hints
    5) enhanced_hints
    6) fused_hints
    """
    experiments = [
        ("official_no_hints", "official", False),
        ("enhanced_no_hints", "enhanced", False),
        ("fused_no_hints", "fused", False),
        ("official_hints", "official", True),
        ("enhanced_hints", "enhanced", True),
        ("fused_hints", "fused", True),
    ]

    summary_rows = []
    for name, mode, use_hints in experiments:
        result_path = f"data/evaluation_results_{name}.json"
        summary = run_unified_evaluation(
            mode=mode,
            dataset_path=dataset_path,
            test_limit=test_limit,
            include_dataset_evidence=use_hints,
            save_path=result_path,
        )
        row = {
            "name": name,
            "mode": mode,
            "hints": use_hints,
            "accuracy": round(summary["accuracy"], 2),
            "correct": summary["correct"],
            "total": summary["total"],
            "result_file": result_path,
        }
        summary_rows.append(row)
        print(
            f"✅ {name}: {row['accuracy']:.2f}% "
            f"({row['correct']}/{row['total']}) -> {row['result_file']}"
        )

    summary_path = "data/evaluation_results_sixway_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📊 六组准确率总览")
    for row in summary_rows:
        print(f"{row['name']:<22} {row['accuracy']:>6.2f}%   ({row['correct']}/{row['total']})")
    print(f"💾 汇总文件: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_sixway_comparison()

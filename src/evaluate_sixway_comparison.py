import json
import os
from datetime import datetime
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

    dataset_tag = os.path.splitext(os.path.basename(dataset_path))[0] or "dataset"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("data", "evaluation_runs", f"sixway_{dataset_tag}_{run_tag}")
    os.makedirs(run_dir, exist_ok=True)

    summary_rows = []
    for name, mode, use_hints in experiments:
        result_path = os.path.join(run_dir, f"evaluation_results_{name}.json")
        summary = run_unified_evaluation(
            mode=mode,
            dataset_path=dataset_path,
            test_limit=test_limit,
            include_dataset_evidence=use_hints,
            save_path=result_path,
            run_tag=run_tag,
        )
        if not isinstance(summary, dict):
            summary = {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "error": "run_unified_evaluation did not return summary",
            }
        row = {
            "name": name,
            "mode": mode,
            "hints": use_hints,
            "accuracy": round(summary["accuracy"], 2),
            "correct": summary["correct"],
            "total": summary["total"],
            "execution_success_rate": round(summary.get("execution_success_rate", 0.0), 2),
            "error_count": summary.get("error_count", 0),
            "result_file": summary.get("save_path", result_path),
            "summary_file": summary.get("summary_path"),
        }
        if "error" in summary:
            row["error"] = summary["error"]
        summary_rows.append(row)
        print(
            f"✅ {name}: {row['accuracy']:.2f}% "
            f"({row['correct']}/{row['total']}) -> {row['result_file']}"
        )

    summary_path = os.path.join(run_dir, f"evaluation_results_sixway_summary_{run_tag}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    latest_pointer_path = os.path.join("data", "evaluation_results_sixway_latest.json")
    latest_payload = {
        "run_tag": run_tag,
        "dataset_path": dataset_path,
        "test_limit": test_limit,
        "run_dir": run_dir,
        "summary_path": summary_path,
        "rows": summary_rows,
        "best_row_by_accuracy": max(summary_rows, key=lambda r: r.get("accuracy", 0.0)) if summary_rows else None,
    }
    with open(latest_pointer_path, "w", encoding="utf-8") as f:
        json.dump(latest_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📊 六组准确率总览")
    for row in summary_rows:
        print(f"{row['name']:<22} {row['accuracy']:>6.2f}%   ({row['correct']}/{row['total']})")
    print(f"💾 汇总文件: {summary_path}")
    print(f"🧭 latest 指针: {latest_pointer_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_sixway_comparison()

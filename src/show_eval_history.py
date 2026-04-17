import argparse
import json
from pathlib import Path


def load_history(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda x: (x.get('ended_at') or '', x.get('run_tag') or ''))
    return rows


def print_latest(rows, limit):
    print(f"\n最近 {limit} 次评测记录")
    print('-' * 110)
    print(f"{'结束时间':19} {'mode':10} {'acc':>8} {'exec':>8} {'ok/total':>12} {'耗时(s)':>9}  result")
    print('-' * 110)
    for row in rows[-limit:]:
        ended_at = (row.get('ended_at') or '')[:19]
        mode = row.get('mode', '-')
        acc = f"{float(row.get('accuracy', 0.0)):.2f}%"
        exec_rate = f"{float(row.get('execution_success_rate', 0.0)):.2f}%"
        correct = int(row.get('correct', 0))
        total = int(row.get('total', 0))
        duration = float(row.get('duration_seconds', 0.0))
        result_file = Path(row.get('save_path', '-')).name
        print(f"{ended_at:19} {mode:10} {acc:>8} {exec_rate:>8} {f'{correct}/{total}':>12} {duration:>9.2f}  {result_file}")


def print_best_by_mode(rows):
    best = {}
    for row in rows:
        mode = row.get('mode', 'unknown')
        score = float(row.get('accuracy', 0.0))
        if mode not in best or score > float(best[mode].get('accuracy', 0.0)):
            best[mode] = row

    print("\n各 mode 历史最佳")
    print('-' * 90)
    print(f"{'mode':10} {'best_acc':>10} {'exec':>8} {'ok/total':>12} {'结束时间':19}")
    print('-' * 90)
    for mode in sorted(best.keys()):
        row = best[mode]
        acc = f"{float(row.get('accuracy', 0.0)):.2f}%"
        exec_rate = f"{float(row.get('execution_success_rate', 0.0)):.2f}%"
        correct = int(row.get('correct', 0))
        total = int(row.get('total', 0))
        ended_at = (row.get('ended_at') or '')[:19]
        print(f"{mode:10} {acc:>10} {exec_rate:>8} {f'{correct}/{total}':>12} {ended_at:19}")


def main():
    parser = argparse.ArgumentParser(description='查看评测历史准确率与关键指标')
    parser.add_argument('--history', default='data/evaluation_runs/history.jsonl', help='历史记录 jsonl 文件路径')
    parser.add_argument('--limit', type=int, default=10, help='显示最近多少条记录')
    args = parser.parse_args()

    rows = load_history(args.history)
    if not rows:
        print(f"未找到评测历史记录: {args.history}")
        return

    limit = max(1, args.limit)
    print_latest(rows, limit)
    print_best_by_mode(rows)


if __name__ == '__main__':
    main()

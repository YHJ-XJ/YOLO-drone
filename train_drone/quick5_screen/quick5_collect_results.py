from pathlib import Path
import csv


# 跑完 quick5_screen_train.py 后运行这个脚本，自动汇总每个模型前5轮的最好指标。
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def read_best(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def get(row, key):
        # Ultralytics 的表头有时带空格，这里做一次兼容。
        for k, v in row.items():
            if k.strip() == key:
                return float(v)
        return 0.0

    best = max(rows, key=lambda r: get(r, "metrics/mAP50(B)"))
    return {
        "epoch": int(get(best, "epoch")) + 1,
        "precision": get(best, "metrics/precision(B)"),
        "recall": get(best, "metrics/recall(B)"),
        "map50": get(best, "metrics/mAP50(B)"),
        "map5095": get(best, "metrics/mAP50-95(B)"),
    }


if __name__ == "__main__":
    print("quick5结果汇总：")
    print("name, best_epoch, P, R, mAP50, mAP50-95")
    for csv_path in sorted(RUNS_DIR.glob("quick5_*/results.csv")):
        best = read_best(csv_path)
        if best is None:
            continue
        print(
            f"{csv_path.parent.name}, {best['epoch']}, "
            f"{best['precision']:.4f}, {best['recall']:.4f}, "
            f"{best['map50']:.4f}, {best['map5095']:.4f}"
        )

from pathlib import Path

from ultralytics import YOLO


# 训练完成后，用 best.pt 在 test 集上做最终测试。
# 脚本会自动寻找最近一次 yolov26n_baseline_pcb 开头的训练结果目录。
THIS_DIR = Path(__file__).resolve().parent
YOLO_ROOT = THIS_DIR.parents[1]
DATA_YAML = THIS_DIR / "yolov26n_baseline_pcb_data.yaml"


def find_latest_best_weight() -> Path:
    runs = [
        p
        for p in (YOLO_ROOT / "train_pcb").glob("yolov26n_baseline_pcb*")
        if p.is_dir() and (p / "weights" / "best.pt").exists()
    ]
    if not runs:
        raise FileNotFoundError("没有找到 yolov26n_baseline_pcb 的 best.pt，请先完成训练。")
    latest_run = max(runs, key=lambda p: p.stat().st_mtime)
    return latest_run / "weights" / "best.pt"


if __name__ == "__main__":
    best_weight = find_latest_best_weight()
    print(f"Using weight: {best_weight}")

    model = YOLO(str(best_weight))
    model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        workers=8,
        device=0,
        project=str(YOLO_ROOT / "train_pcb"),
        name="yolov26n_baseline_pcb_test",
    )

from pathlib import Path

from ultralytics import YOLO

# CARPK 正式划分版 baseline：
# train/val 来自官方训练集拆分，test 保留官方测试集。
# 这一组结果将作为后续所有 CARPK 改进模型的正式对照基线。
THIS_DIR = Path(__file__).resolve().parent
YOLO_ROOT = THIS_DIR.parents[1]

DATA_YAML = THIS_DIR / "yolov26n_baseline_carpk_formal_data.yaml"
WEIGHTS = YOLO_ROOT / "yolo26n.pt"

model = YOLO(str(WEIGHTS))

model.train(
    data=str(DATA_YAML),
    imgsz=640,
    epochs=100,
    batch=16,
    workers=8,
    device=0,
    project=str(YOLO_ROOT / "train_carpk"),
    name="yolov26n_baseline_carpk_formal",
    patience=20,
    close_mosaic=10,
    amp=True,
)

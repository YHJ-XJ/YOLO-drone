from pathlib import Path

from ultralytics import YOLO

# PCB 缺陷检测 baseline。
# 这一组不加入任何自研模块，用来判断 PCB 数据集是否适合作为“高基线 + 再涨点”的论文主数据集。
THIS_DIR = Path(__file__).resolve().parent
YOLO_ROOT = THIS_DIR.parents[1]

DATA_YAML = THIS_DIR / "yolov26n_baseline_pcb_data.yaml"
WEIGHTS = YOLO_ROOT / "yolo26n.pt"

model = YOLO(str(WEIGHTS))

model.train(
    data=str(DATA_YAML),
    imgsz=640,
    epochs=100,
    batch=16,
    workers=8,
    device=0,
    project=str(YOLO_ROOT / "train_pcb"),
    name="yolov26n_baseline_pcb",
    patience=25,
    close_mosaic=10,
    amp=True,
)

import os
from pathlib import Path

# 低成本微调实验：关闭 CS-SlideLoss，避免引入新的不稳定变量。
os.environ["YOLO_CS_SLIDE"] = "0"

from ultralytics import YOLO

# 使用 Tile960 训练得到的 best.pt 作为起点，再用原始 VisDrone 训练集短程微调。
# 这样比继续跑切片训练便宜很多，也能减少“切片训练分布”和原始验证集之间的偏差。
root = Path("/root/autodl-tmp/yolo26_drone_package/yolo26")
tile_best = root / "train_drone" / "yolov26_P2_FGDDown_DySample_TGPFuse3_TCA_P3_Tile960" / "weights" / "best.pt"
local_tile_best = root / "train_drone" / "yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960" / "weights" / "best.pt"

if tile_best.exists():
    weight_path = tile_best
elif local_tile_best.exists():
    weight_path = local_tile_best
else:
    raise FileNotFoundError(
        "没有找到 Tile960 的 best.pt。请先确认 Tile960 结果文件夹已经上传，并且 weights/best.pt 存在。"
    )

model = YOLO(str(weight_path))

model.train(
    data=(
        "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30/"
        "yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30_data.yaml"
    ),
    imgsz=960,
    epochs=30,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_TCA_P3_Tile960_FT30",
    patience=15,
    close_mosaic=0,
    mosaic=0.2,
    mixup=0.0,
    copy_paste=0.0,
    amp=True,
)

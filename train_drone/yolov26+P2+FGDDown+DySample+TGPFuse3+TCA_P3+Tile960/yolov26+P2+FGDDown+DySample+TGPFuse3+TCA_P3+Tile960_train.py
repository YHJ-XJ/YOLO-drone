import os


# 下一阶段冲精度实验：使用切片训练增强小目标，不启用已经验证不稳定的 CS-SlideLoss。
os.environ["YOLO_CS_SLIDE"] = "0"

from ultralytics import YOLO


# 结构沿用当前最有希望冲精度的四检测头模型：
# YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + TCA_P3。
# 主要变化不是继续堆网络模块，而是训练数据改为 960 重叠切片，让小目标在训练时占比更大。
model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/"
    "yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960.yaml"
)
model.load("yolo26n.pt")

model.train(
    data=(
        "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/"
        "yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_data.yaml"
    ),
    imgsz=960,
    epochs=200,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_TCA_P3_Tile960",
    patience=80,
    close_mosaic=20,
    amp=True,
)

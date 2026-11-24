import os

# 结构实验：关闭 CS-SlideLoss，避免损失函数和结构改进混在一起。
os.environ["YOLO_CS_SLIDE"] = "0"

from ultralytics import YOLO

# TCA_P3P4：
# 在 TCA_P3 的基础上，额外对 P5SPF 后的 P4 语义特征加入轻量卷积注意力。
# 目的：验证 P4 语义层增强是否能进一步提升 VisDrone 小目标和复杂背景场景下的 mAP。
model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4/"
    "yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4.yaml"
)
model.load("yolo26n.pt")

model.train(
    data=(
        "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4/"
        "yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4_data.yaml"
    ),
    imgsz=960,
    epochs=200,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_P5SPFPrune_TCA_P3P4",
    patience=100,
    close_mosaic=10,
    amp=True,
)

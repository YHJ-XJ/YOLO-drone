import os

# 本实验是结构实验，不启用 CS-SlideLoss，避免和 TCA 的效果混在一起。
os.environ["YOLO_CS_SLIDE"] = "0"

from ultralytics import YOLO

# TCA_P3：Tiny Convolutional Attention at P3
# 只在 P4->P3 融合后的 P3 特征处加入轻量卷积注意力。
# 目标是在几乎不大幅增加参数的前提下，加强小目标局部纹理与上下文响应。
model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3.yaml"
)
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3_data.yaml",
    imgsz=960,
    epochs=200,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_P5SPFPrune_TCA_P3",
    patience=100,
    close_mosaic=10,
    amp=True,
)

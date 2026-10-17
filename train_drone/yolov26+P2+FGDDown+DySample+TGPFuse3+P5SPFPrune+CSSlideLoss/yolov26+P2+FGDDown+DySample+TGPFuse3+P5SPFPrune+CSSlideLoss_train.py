import os


# CS-SlideLoss：类别-尺度感知滑动损失
# 该损失只在训练阶段启用，不改变模型结构，不增加推理参数量和 FLOPs。
# 设计目的：
# 1. 用滑动难样本权重强化低置信度/难分类样本；
# 2. 用类别频次权重缓解 VisDrone 类别不均衡；
# 3. 用尺度权重提高小目标正样本的训练贡献。
os.environ["YOLO_CS_SLIDE"] = "1"
os.environ["YOLO_CS_SLIDE_GAIN"] = "0.6"
os.environ["YOLO_CS_SLIDE_GAMMA"] = "1.0"
os.environ["YOLO_CS_SCALE_GAIN"] = "0.5"
os.environ["YOLO_CS_SCALE_MAX"] = "1.8"

from ultralytics import YOLO


model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss.yaml"
)
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss_data.yaml",
    imgsz=960,
    epochs=200,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_P5SPFPrune_CSSlideLoss",
    patience=100,
    close_mosaic=10,
    amp=True,
)

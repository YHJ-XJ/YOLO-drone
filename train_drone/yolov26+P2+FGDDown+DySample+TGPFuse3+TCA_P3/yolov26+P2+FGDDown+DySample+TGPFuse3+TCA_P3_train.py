import os


# 冲精度结构实验：关闭 CS-SlideLoss，只验证 TCA_P3 对完整四检测头结构的作用。
os.environ["YOLO_CS_SLIDE"] = "0"

from ultralytics import YOLO


# YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + TCA_P3
# 保留 P2/P3/P4/P5 四检测头，不做 P5SPFPrune。
# 目的：在当前 TGPFuse3 强精度路线基础上，只增强 P3 小目标敏感特征层。
model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3/"
    "yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3.yaml"
)
model.load("yolo26n.pt")

model.train(
    data=(
        "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3/"
        "yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3_data.yaml"
    ),
    imgsz=960,
    epochs=200,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_TCA_P3",
    patience=100,
    close_mosaic=10,
    amp=True,
)

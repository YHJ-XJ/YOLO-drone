import os


# 消融实验：关闭 CS-SlideLoss，只观察删 P5 且不使用 P5SPF 时的结构影响。
os.environ["YOLO_CS_SLIDE"] = "0"

from ultralytics import YOLO


# P5Prune_NoSPF：
# 直接删除 P5 检测头，不引入 P5SPF 语义保留模块。
# 目的：与 P5SPFPrune 对比，证明 P5SPF 的必要性。
model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF/"
    "yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF.yaml"
)
model.load("yolo26n.pt")

model.train(
    data=(
        "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF/"
        "yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF_data.yaml"
    ),
    imgsz=960,
    epochs=150,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_P5Prune_NoSPF",
    patience=80,
    close_mosaic=10,
    amp=True,
)

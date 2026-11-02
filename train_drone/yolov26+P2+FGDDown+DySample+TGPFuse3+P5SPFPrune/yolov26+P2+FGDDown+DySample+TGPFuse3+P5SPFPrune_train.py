from ultralytics import YOLO

# P5SPFPrune：P5 Semantic Preserving Fusion Pruning
# 核心思想：
# 1. 不再直接保留 P5 检测头，减少大步长检测分支对边缘部署的负担；
# 2. 删除 P5 前，先用 P5SPF 将 P5 深层语义注入 P4；
# 3. 最终保持 P2/P3/P4 三个检测层，更偏向无人机小目标检测。

model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune.yaml"
)
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune_data.yaml",
    imgsz=960,
    epochs=200,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_P5SPFPrune",
    patience=100,
    close_mosaic=10,
    amp=True,
)

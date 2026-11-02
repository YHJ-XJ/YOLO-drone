from ultralytics import YOLO

# TPKDStudent-v2：Tiny-Prior Knowledge Distillation Student 第二版
# v1 的 width=0.20 压缩过猛，精度基本只接近 baseline。
# v2 将 width 提到 0.225，并把蒸馏权重 dis 从 0.5 降到 0.3：
# 1. 给 Student 更多通道容量，避免小目标细节特征过早丢失；
# 2. 降低 Teacher 约束强度，让 Student 先学好检测任务本身；
# 3. 保持和 Teacher 相同拓扑，确保现有蒸馏代码能按层索引稳定对齐。

model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent-v2/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent-v2.yaml"
)
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent-v2/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent-v2_data.yaml",
    imgsz=960,
    epochs=150,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_TPKDStudent_v2",
    patience=80,
    close_mosaic=10,
    amp=True,
    distill_model="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3/weights/best.pt",
    dis=0.3,
)

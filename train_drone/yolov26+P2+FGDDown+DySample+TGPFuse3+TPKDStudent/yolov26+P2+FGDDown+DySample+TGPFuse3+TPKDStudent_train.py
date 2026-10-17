from ultralytics import YOLO


# TPKDStudent：Tiny-Prior Knowledge Distillation Student
# 目的：用当前最强 TGPFuse3 作为 Teacher，训练一个更窄的同拓扑 Student。
# 这样既保留 P2/TGPFuse 小目标先验融合路径，又降低通道宽度，适合作为轻量化第一步。

model = YOLO(
    "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent.yaml"
)
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent_data.yaml",
    imgsz=960,
    epochs=150,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3_TPKDStudent",
    patience=80,
    close_mosaic=10,
    amp=True,
    # Teacher 使用当前最强 TGPFuse3 的 best.pt。上传云端时要确保这个权重路径存在。
    distill_model="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3/weights/best.pt",
    # dis 是蒸馏损失权重。第一版不宜过大，避免 Teacher 过强导致 Student 学不动。
    dis=0.5,
)

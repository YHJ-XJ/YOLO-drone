from ultralytics import YOLO


# 本实验：YOLO26 + P2 + FGDDown + DySample
# 注意：训练前请确认 ultralytics/utils/loss.py 中 NWD_ENABLED=False，保证这次只验证结构改进。
model = YOLO("train_drone/yolov26+P2+FGDDown+DySample/yolov26+P2+FGDDown+DySample.yaml")
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown+DySample/yolov26+P2+FGDDown+DySample_data.yaml",
    imgsz=960,
    epochs=200,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample",
    patience=100,
    close_mosaic=10,
    amp=True,
)

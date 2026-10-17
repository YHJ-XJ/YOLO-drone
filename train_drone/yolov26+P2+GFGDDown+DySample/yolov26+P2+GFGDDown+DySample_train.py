from ultralytics import YOLO

# 本实验：YOLO26 + P2 + GFGDDown + DySample
# 单变量对照：相对 E12，只把 FGDDown 替换为门控版 GFGDDown。
# 训练前确认：ultralytics/utils/loss.py 中 NWD_ENABLED=False。
model = YOLO("train_drone/yolov26+P2+GFGDDown+DySample/yolov26+P2+GFGDDown+DySample.yaml")
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+GFGDDown+DySample/yolov26+P2+GFGDDown+DySample_data.yaml",
    imgsz=960,
    epochs=200,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_GFGDDown_DySample",
    patience=100,
    close_mosaic=10,
    amp=True,
)

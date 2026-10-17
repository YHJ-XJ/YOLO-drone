from ultralytics import YOLO


# CARPK 无人机车辆检测 baseline。
# 先用官方 YOLO26n 预训练权重，确认该数据集是否能形成高 mAP50 基线。
model = YOLO("yolo26n.pt")

model.train(
    data="train_carpk/yolov26n_baseline_carpk/yolov26n_baseline_carpk_data.yaml",
    imgsz=640,
    epochs=50,
    batch=16,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_carpk",
    name="yolov26n_baseline_carpk",
    patience=20,
    close_mosaic=10,
    amp=True,
)

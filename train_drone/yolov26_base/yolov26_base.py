from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="train_drone/yolov26_base/yolov26_base_data.yaml",
    imgsz=960,
    epochs=100,
    batch=4,
    workers=0,
    device="0",
    project="F:/图像深度学习科研/面向边缘设备部署的无人机小目标轻量化检测算法研究/yolo26/train_drone",
    name="yolov26_base",
    close_mosaic=10,
    amp=False,
)

from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-p2.yaml")
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26_p2_autodl/visdrone_autodl.yaml",
    imgsz=960,
    epochs=100,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_p2_autodl",
    close_mosaic=10,
    amp=True,
)

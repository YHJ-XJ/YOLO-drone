from ultralytics import YOLO

model = YOLO("train_drone/yolov26+P2+FGDDown/yolov26+P2+FGDDown.yaml")
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown/yolov26+P2+FGDDown_data.yaml",
    imgsz=960,
    epochs=150,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown",
    close_mosaic=10,
    amp=True,
)

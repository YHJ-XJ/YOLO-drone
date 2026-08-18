from ultralytics import YOLO


model = YOLO("train_drone/yolov26+DCLDown/yolov26+DCLDown.yaml")
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+DCLDown/yolov26+DCLDown_data.yaml",
    imgsz=960,
    epochs=150,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_DCLDown",
    close_mosaic=10,
    amp=True,
    patience=50,
)

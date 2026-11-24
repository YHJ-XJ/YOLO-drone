from ultralytics import YOLO

# YOLO26 + P2 + FGDDown + DySample + TGPFuse3
# 5轮预验证：判断普通 TGPFuse 替换3处关键Concat后，早期趋势是否优于当前E12底座。
model = YOLO("train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3/yolov26+P2+FGDDown+DySample+TGPFuse3.yaml")
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3/yolov26+P2+FGDDown+DySample+TGPFuse3_data.yaml",
    imgsz=960,
    epochs=5,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TGPFuse3",
    patience=5,
    close_mosaic=0,
    amp=True,
    plots=False,
    save=False,
    deterministic=False,
)

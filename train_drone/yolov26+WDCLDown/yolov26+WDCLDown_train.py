from ultralytics import YOLO

# 本实验：YOLO26 baseline + WDCLDown（小波细节对比轻量下采样）。
# 目的：验证 Haar 高频支路是否比 DCLDown 的 MaxPool-AvgPool 对比支路更能保留小目标细节。
# 对比对象：
#   YOLO26 baseline        mAP50-95 = 0.24407
#   YOLO26 + DCLDown       mAP50-95 = 0.24469
#   YOLO26 + DySample      mAP50-95 = 0.24843
# 判断：若 mAP50-95 明显高于 DCLDown 的 0.24469，说明小波高频分支有效，下一步组合 WDCLDown + DySample。
model = YOLO("train_drone/yolov26+WDCLDown/yolov26+WDCLDown.yaml")

# 加载 YOLO26n 预训练权重（除 WDCLDown 外结构一致，权重可部分迁移）。
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+WDCLDown/yolov26+WDCLDown_data.yaml",
    imgsz=960,
    epochs=200,
    patience=100,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_WDCLDown",
    close_mosaic=10,
    amp=True,
)

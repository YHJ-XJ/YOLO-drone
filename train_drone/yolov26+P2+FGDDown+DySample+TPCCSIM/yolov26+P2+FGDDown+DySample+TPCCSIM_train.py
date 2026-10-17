from ultralytics import YOLO

# YOLO26 + P2 + FGDDown + DySample + TPCCSIM
# 5轮预验证：判断Detect前小目标先验一致性校准是否值得正式训练。
model = YOLO("train_drone/yolov26+P2+FGDDown+DySample+TPCCSIM/yolov26+P2+FGDDown+DySample+TPCCSIM.yaml")
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+P2+FGDDown+DySample+TPCCSIM/yolov26+P2+FGDDown+DySample+TPCCSIM_data.yaml",
    imgsz=960,
    epochs=5,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_P2_FGDDown_DySample_TPCCSIM",
    patience=5,
    close_mosaic=0,
    amp=True,
    plots=False,
    save=False,
    deterministic=False,
)

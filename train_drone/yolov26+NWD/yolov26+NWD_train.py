from ultralytics import YOLO


# 本实验：YOLO26 baseline + NWD（Normalized Wasserstein Distance）
# 目的：模型结构完全等于 baseline，只在 box loss 中用 NWD 替换一半 CIoU，
#       单独衡量 NWD 对 tiny object 定位的净收益。
# 对比基准：YOLO26 baseline mAP50-95 = 0.24407（batch=4/amp=False/100ep 的旧结果）。
# 注意：为控制变量，本实验用 batch=8/amp=True/200ep，与 DySample 等改进实验一致；
#       若要得到"同条件无 NWD"的对照，把 ultralytics/utils/loss.py 里的 NWD_ENABLED 改成 False 再跑一遍即可。
model = YOLO("train_drone/yolov26+NWD/yolov26+NWD.yaml")

# 加载 YOLO26n 预训练权重（结构完全一致，全部权重可直接迁移）。
model.load("yolo26n.pt")

model.train(
    data="train_drone/yolov26+NWD/yolov26+NWD_data.yaml",
    imgsz=960,
    epochs=200,
    patience=100,
    batch=8,
    workers=8,
    device=0,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_NWD",
    close_mosaic=10,
    amp=True,
)

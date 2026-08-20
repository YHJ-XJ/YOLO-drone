from ultralytics import YOLO


# 本实验：YOLO26 + DCLDown + DySample
# 目标：在不增加 P2 检测头的情况下，同时改进下采样细节保留和上采样特征恢复。
# 控制变量：数据集、输入尺寸、batch、AMP、预训练权重均与前面 AutoDL 实验保持一致。
model = YOLO("train_drone/yolov26+DCLDown+DySample/yolov26+DCLDown+DySample.yaml")

# 加载 YOLO26n 预训练权重。
# DCLDown 和 DySample 是新增/替换模块，能加载上的层会自动迁移，加载不上的新层会随机初始化。
model.load("yolo26n.pt")

model.train(
    # 数据集路径写在 data yaml 中，AutoDL 上对应 /root/autodl-tmp/yolo26_drone_package/datasets/VisDrone-YOLO。
    data="train_drone/yolov26+DCLDown+DySample/yolov26+DCLDown+DySample_data.yaml",
    # VisDrone 小目标密集，沿用 960 输入尺寸。
    imgsz=960,
    # DySample 单独实验使用 200 轮、patience=100，这里保持一致，给组合模块充分收敛机会。
    epochs=200,
    patience=100,
    # RTX 4090 上 batch=8 已经稳定跑通，继续保持公平对比。
    batch=8,
    workers=8,
    device=0,
    # 训练结果统一保存到 train_drone 下，便于和 baseline / DCLDown / DySample 横向对比。
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_DCLDown_DySample",
    # 最后 10 轮关闭 mosaic，利于检测框定位收敛。
    close_mosaic=10,
    # 4090 建议开启 AMP，可降低显存并加快训练。
    amp=True,
)

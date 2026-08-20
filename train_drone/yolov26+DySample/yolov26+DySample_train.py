from ultralytics import YOLO


# 本实验：YOLO26 baseline + DySample
# 目的：只替换颈部 neck 的两处 nearest 上采样，验证动态采样上采样对 VisDrone 小目标检测的作用。
# 注意：这一步不加 P2、不加 DCLDown，方便后续做清晰的消融实验。
model = YOLO("train_drone/yolov26+DySample/yolov26+DySample.yaml")

# 加载 YOLO26n 预训练权重，让实验和前面的 baseline / DCLDown 保持一致。
model.load("yolo26n.pt")

model.train(
    # AutoDL 服务器中的数据集路径写在这个 data yaml 里。
    data="train_drone/yolov26+DySample/yolov26+DySample_data.yaml",
    # VisDrone 小目标较多，沿用之前正式实验的 960 输入尺寸。
    imgsz=960,
    # 先设 200 轮，并用 patience=50 早停；如果 50 轮无提升会自动停止。
    epochs=200,
    patience=50,
    # RTX 4090 上 batch=8 已经在前面实验验证可用，继续保持控制变量一致。
    batch=8,
    workers=8,
    device=0,
    # 所有实验输出统一放在 train_drone 下，便于横向对比。
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_DySample",
    # 最后 10 轮关闭 mosaic，通常有利于定位框收敛。
    close_mosaic=10,
    # 4090 上开启 AMP 可以加速并降低显存占用；和前面 AutoDL 实验保持一致。
    amp=True,
)

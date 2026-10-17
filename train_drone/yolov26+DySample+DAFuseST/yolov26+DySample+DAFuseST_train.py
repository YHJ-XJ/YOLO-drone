from ultralytics import YOLO

# 本实验：YOLO26 + DySample + DAFuseST
# 目的：保留已经证明有效的 DySample，再用 DAFuseST 替换前两处跨尺度 Concat。
# 核心假设：小目标依赖浅层细节，DySample 负责更细致地上采样，DAFuseST 负责自适应选择融合权重。
model = YOLO("train_drone/yolov26+DySample+DAFuseST/yolov26+DySample+DAFuseST.yaml")

# 使用 YOLO26n 预训练权重。
# DySample 和 DAFuseST 中新增的小模块会随机初始化，其余能匹配的权重会自动迁移。
model.load("yolo26n.pt")

model.train(
    # AutoDL 上的数据集路径由 data yaml 指定。
    data="train_drone/yolov26+DySample+DAFuseST/yolov26+DySample+DAFuseST_data.yaml",
    # 沿用 VisDrone 正式实验输入尺寸，保证与前面实验可比。
    imgsz=960,
    # 与 DySample 单独实验保持一致，给动态采样和动态融合足够收敛时间。
    epochs=200,
    patience=100,
    # RTX 4090 已验证 batch=8 稳定；继续保持控制变量一致。
    batch=8,
    workers=8,
    device=0,
    # 所有结果统一保存到 train_drone 下。
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="yolov26_DySample_DAFuseST",
    # 最后 10 轮关闭 mosaic，有利于定位框精修。
    close_mosaic=10,
    # 4090 上开启 AMP，速度和显存更友好。
    amp=True,
)

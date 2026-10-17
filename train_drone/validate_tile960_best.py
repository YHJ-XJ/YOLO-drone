from pathlib import Path

from ultralytics import YOLO


# 低成本复验脚本：只验证 Tile960 的 best.pt，不训练。
# 用于确认当前 best.pt 的真实指标，并生成 PR/F1 曲线和混淆矩阵。
root = Path("/root/autodl-tmp/yolo26_drone_package/yolo26")
weight_candidates = [
    root / "train_drone" / "yolov26_P2_FGDDown_DySample_TGPFuse3_TCA_P3_Tile960" / "weights" / "best.pt",
    root / "train_drone" / "yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960" / "weights" / "best.pt",
]

for path in weight_candidates:
    if path.exists():
        weight_path = path
        break
else:
    raise FileNotFoundError("没有找到 Tile960 的 best.pt，请检查结果文件夹是否已上传。")

model = YOLO(str(weight_path))
model.val(
    data="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30_data.yaml",
    imgsz=960,
    batch=8,
    workers=8,
    device=0,
    plots=True,
    project="/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone",
    name="val_Tile960_best",
)

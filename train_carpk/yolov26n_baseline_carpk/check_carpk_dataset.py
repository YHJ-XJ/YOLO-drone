from pathlib import Path


# 简单检查 CARPK YOLO 数据集是否放对位置。
root = Path("/root/autodl-tmp/yolo26_drone_package/datasets/CARPK-YOLO")

for split in ["train", "val", "test"]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = []
    labels = []
    if image_dir.exists():
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if label_dir.exists():
        labels = list(label_dir.glob("*.txt"))
    print(f"{split}: images={len(images)}, labels={len(labels)}, image_dir={image_dir.exists()}, label_dir={label_dir.exists()}")

print("CARPK dataset check finished.")

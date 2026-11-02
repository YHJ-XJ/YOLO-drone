from pathlib import Path

# 检查 AutoDL 上传后的正式 CARPK 数据集结构是否完整。
# 运行位置建议为：/root/autodl-tmp/yolo26_drone_package/yolo26
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PACKAGE_ROOT / "datasets" / "CARPK-YOLO-Formal"


def count_boxes(label_dir: Path) -> int:
    box_count = 0
    for label_path in label_dir.glob("*.txt"):
        text = label_path.read_text(encoding="utf-8").strip()
        if text:
            box_count += len(text.splitlines())
    return box_count


def check_split(split: str) -> None:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    images = sorted(image_dir.glob("*.png"))
    labels = sorted(label_dir.glob("*.txt"))
    print(
        f"{split}: images={len(images)}, labels={len(labels)}, boxes={count_boxes(label_dir)}, "
        f"image_dir={image_dir.exists()}, label_dir={label_dir.exists()}"
    )

    missing = []
    label_names = {p.stem for p in labels}
    for image_path in images:
        if image_path.stem not in label_names:
            missing.append(image_path.name)
    if missing:
        raise FileNotFoundError(f"{split} 缺少标签文件，示例：{missing[:5]}")


if __name__ == "__main__":
    print(f"dataset root: {DATASET_ROOT}")
    for split_name in ("train", "val", "test"):
        check_split(split_name)
    print("CARPK formal dataset check finished.")

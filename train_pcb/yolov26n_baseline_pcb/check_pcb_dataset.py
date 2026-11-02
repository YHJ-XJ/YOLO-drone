from __future__ import annotations

from pathlib import Path

# 检查 AutoDL 上传后的 PCB 数据集结构是否完整。
# 运行位置建议为：/root/autodl-tmp/yolo26_drone_package/yolo26
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PACKAGE_ROOT / "datasets" / "PCB-YOLO"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def count_boxes(label_dir: Path) -> tuple[int, int]:
    """统计目标框数量和空标签文件数量。."""
    box_count = 0
    empty_count = 0
    for label_path in label_dir.glob("*.txt"):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_count += 1
            continue
        box_count += len(text.splitlines())
    return box_count, empty_count


def check_split(split: str) -> None:
    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
    labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
    box_count, empty_count = count_boxes(label_dir) if label_dir.exists() else (0, 0)

    print(
        f"{split}: images={len(images)}, labels={len(labels)}, boxes={box_count}, "
        f"empty_labels={empty_count}, image_dir={image_dir.exists()}, label_dir={label_dir.exists()}"
    )

    label_names = {p.stem for p in labels}
    missing = [p.name for p in images if p.stem not in label_names]
    if missing:
        raise FileNotFoundError(f"{split} 缺少标签文件，示例：{missing[:5]}")


if __name__ == "__main__":
    print(f"dataset root: {DATASET_ROOT}")
    for split_name in ("train", "val", "test"):
        check_split(split_name)
    print("PCB dataset check finished.")

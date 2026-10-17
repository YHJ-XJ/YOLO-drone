from __future__ import annotations

import random
import shutil
from pathlib import Path


# CARPK 正式论文划分脚本：
# 1. 原官方 train 按 80% / 20% 拆成 train / val
# 2. 原官方 test 只作为最终 test
# 3. 固定随机种子，保证之后论文实验可以复现
PROJECT_ROOT = Path(r"F:\图像深度学习科研\面向边缘设备部署的无人机小目标轻量化检测算法研究")
SRC = PROJECT_ROOT / "YOLO格式数据集" / "CARPK-YOLO"
DST = PROJECT_ROOT / "YOLO格式数据集" / "CARPK-YOLO-Formal"

SEED = 20260830
VAL_RATIO = 0.2


def clear_dir(path: Path) -> None:
    """清空并重建目标目录，避免重复运行时残留旧文件。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_pair(image_path: Path, label_path: Path, image_dst: Path, label_dst: Path) -> None:
    """复制一张图片和对应 YOLO 标签。"""
    shutil.copy2(image_path, image_dst / image_path.name)
    shutil.copy2(label_path, label_dst / label_path.name)


def count_boxes(label_dir: Path) -> int:
    """统计一个 split 中所有目标框数量。"""
    total = 0
    for label_path in label_dir.glob("*.txt"):
        text = label_path.read_text(encoding="utf-8").strip()
        if text:
            total += len(text.splitlines())
    return total


def main() -> None:
    for split in ("train", "val", "test"):
        clear_dir(DST / "images" / split)
        clear_dir(DST / "labels" / split)

    train_images = sorted((SRC / "images" / "train").glob("*.png"))
    random.seed(SEED)
    random.shuffle(train_images)

    val_count = int(round(len(train_images) * VAL_RATIO))
    val_names = {p.name for p in train_images[:val_count]}

    for image_path in train_images:
        label_path = SRC / "labels" / "train" / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")

        split = "val" if image_path.name in val_names else "train"
        copy_pair(
            image_path,
            label_path,
            DST / "images" / split,
            DST / "labels" / split,
        )

    for image_path in sorted((SRC / "images" / "test").glob("*.png")):
        label_path = SRC / "labels" / "test" / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
        copy_pair(
            image_path,
            label_path,
            DST / "images" / "test",
            DST / "labels" / "test",
        )

    summary_lines = [
        "CARPK-YOLO-Formal",
        f"seed: {SEED}",
        f"val_ratio_from_official_train: {VAL_RATIO}",
        "split rule: official train -> train/val, official test -> test",
        "",
    ]
    for split in ("train", "val", "test"):
        image_count = len(list((DST / "images" / split).glob("*.png")))
        label_count = len(list((DST / "labels" / split).glob("*.txt")))
        box_count = count_boxes(DST / "labels" / split)
        summary_lines.append(f"{split}: images={image_count}, labels={label_count}, boxes={box_count}")

    (DST / "dataset_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

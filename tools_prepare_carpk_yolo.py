from pathlib import Path
import shutil

import cv2
import numpy as np


PROJECT_ROOT = Path(r"F:\图像深度学习科研\面向边缘设备部署的无人机小目标轻量化检测算法研究")
SRC_ROOT = PROJECT_ROOT / "数据集汇总" / "drone_car数据集" / "datasets" / "CARPK_devkit" / "data"
LOCAL_OUT = PROJECT_ROOT / "YOLO格式数据集" / "CARPK-YOLO"
PACKAGE_OUT = PROJECT_ROOT / "AutoDL上传包" / "yolo26_drone_package" / "datasets" / "CARPK-YOLO"


def read_split(split_file: Path):
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert_one(stem: str, out_img_dir: Path, out_label_dir: Path):
    img_path = SRC_ROOT / "Images" / f"{stem}.png"
    ann_path = SRC_ROOT / "Annotations" / f"{stem}.txt"
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    # Windows 下 cv2.imread 对中文路径不稳定，使用 imdecode 兼容中文路径。
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    h, w = img.shape[:2]

    labels = []
    if ann_path.exists():
        for line in ann_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            x1, y1, x2, y2 = map(float, parts[:4])
            x1 = max(0.0, min(x1, w - 1))
            x2 = max(0.0, min(x2, w - 1))
            y1 = max(0.0, min(y1, h - 1))
            y2 = max(0.0, min(y2, h - 1))
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1 or bh <= 1:
                continue
            xc = x1 + bw / 2
            yc = y1 + bh / 2
            labels.append(f"0 {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}")

    shutil.copy2(img_path, out_img_dir / img_path.name)
    (out_label_dir / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return len(labels)


def prepare_root(out_root: Path):
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_ids = read_split(SRC_ROOT / "ImageSets" / "train.txt")
    test_ids = read_split(SRC_ROOT / "ImageSets" / "test.txt")

    stats = {}
    for split, ids in [("train", train_ids), ("val", test_ids), ("test", test_ids)]:
        boxes = 0
        for stem in ids:
            boxes += convert_one(stem, out_root / "images" / split, out_root / "labels" / split)
        stats[split] = {"images": len(ids), "boxes": boxes}

    (out_root / "dataset_summary.txt").write_text(
        "\n".join(f"{k}: images={v['images']}, boxes={v['boxes']}" for k, v in stats.items()) + "\n",
        encoding="utf-8",
    )
    return stats


def main():
    if not (SRC_ROOT / "Images").exists() or not (SRC_ROOT / "Annotations").exists():
        raise FileNotFoundError(f"CARPK source not found: {SRC_ROOT}")

    for out in [LOCAL_OUT, PACKAGE_OUT]:
        stats = prepare_root(out)
        print(f"Prepared {out}")
        for split, values in stats.items():
            print(f"  {split}: images={values['images']}, boxes={values['boxes']}")


if __name__ == "__main__":
    main()

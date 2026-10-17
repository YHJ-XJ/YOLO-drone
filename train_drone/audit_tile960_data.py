# P1-3 Tile960 数据统计脚本：切片数、重复次数、空背景比例、每 epoch 步数
# 在 AutoDL 运行：python train_drone/audit_tile960_data.py
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def count_tiles(tile_dir):
    """统计切片训练集的图片数和标签数."""
    img_dir = tile_dir / "images" / "train"
    lbl_dir = tile_dir / "labels" / "train"

    imgs = list(img_dir.glob("*.jpg")) if img_dir.exists() else []
    lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    n_imgs = len(imgs)
    n_lbls = len(lbls)

    # 统计空背景（标签文件为空）
    n_empty = 0
    total_objs = 0
    for lbl in lbls:
        lines = lbl.read_text().strip().split("\n")
        if len(lines) == 1 and not lines[0]:
            n_empty += 1
        else:
            total_objs += len(lines)

    return {
        "n_images": n_imgs,
        "n_labels": n_lbls,
        "n_empty": n_empty,
        "empty_ratio": n_empty / n_lbls if n_lbls > 0 else 0,
        "total_objects": total_objs,
        "avg_objs_per_tile": total_objs / n_lbls if n_lbls > 0 else 0,
    }


def count_original(data_yaml_path):
    """统计原始 VisDrone 训练集的图片数."""
    # 假设原始数据在 datasets/VisDrone-YOLO
    orig_dir = ROOT / "datasets" / "VisDrone-YOLO" / "images" / "train"
    imgs = list(orig_dir.glob("*.jpg")) if orig_dir.exists() else []
    return len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-dir", default="datasets/VisDrone-YOLO-Tile960/train")
    ap.add_argument("--orig-data", default="ultralytics/cfg/datasets/VisDrone.yaml")
    ap.add_argument("--out", default="train_drone/audit_tile960_data")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tile_path = ROOT / args.tile_dir
    n_orig = count_original(ROOT / args.orig_data)
    tile_stats = count_tiles(tile_path)

    # 计算切片倍率
    tile_ratio = tile_stats["n_images"] / n_orig if n_orig > 0 else 0

    # 估算每 epoch 步数（batch_size=8）
    steps_orig = n_orig / 8
    steps_tile = tile_stats["n_images"] / 8

    result = {
        "original_images": n_orig,
        "tile_images": tile_stats["n_images"],
        "tile_ratio": tile_ratio,
        "empty_tiles": tile_stats["n_empty"],
        "empty_ratio": tile_stats["empty_ratio"],
        "total_objects": tile_stats["total_objects"],
        "avg_objs_per_tile": tile_stats["avg_objs_per_tile"],
        "steps_per_epoch_orig": steps_orig,
        "steps_per_epoch_tile": steps_tile,
        "steps_ratio": steps_tile / steps_orig if steps_orig > 0 else 0,
    }

    csv_path = out / "tile960_stats.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader()
        w.writerow(result)

    print(f"[audit_tile960_data] done -> {csv_path}")
    print(f"  原图数: {n_orig}")
    print(f"  切片数: {tile_stats['n_images']} ({tile_ratio:.2f}x)")
    print(f"  空背景: {tile_stats['n_empty']} ({tile_stats['empty_ratio']:.1%})")
    print(f"  每 epoch 步数: 原图 {steps_orig:.0f} vs 切片 {steps_tile:.0f} ({steps_tile / steps_orig:.2f}x)")


if __name__ == "__main__":
    main()

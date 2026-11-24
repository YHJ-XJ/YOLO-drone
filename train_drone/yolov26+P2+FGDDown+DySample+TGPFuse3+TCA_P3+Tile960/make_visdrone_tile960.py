import argparse
import shutil
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo_labels(label_path: Path):
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        labels.append((int(float(cls)), float(x), float(y), float(w), float(h)))
    return labels


def yolo_to_xyxy(label, img_w, img_h):
    cls, x, y, w, h = label
    cx, cy = x * img_w, y * img_h
    bw, bh = w * img_w, h * img_h
    return cls, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def xyxy_to_yolo(cls, x1, y1, x2, y2, tile_w, tile_h):
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    return cls, cx / tile_w, cy / tile_h, bw / tile_w, bh / tile_h


def build_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def clip_box_to_tile(box, x0, y0, tile_w, tile_h):
    cls, x1, y1, x2, y2 = box
    nx1 = max(x1, x0) - x0
    ny1 = max(y1, y0) - y0
    nx2 = min(x2, x0 + tile_w) - x0
    ny2 = min(y2, y0 + tile_h) - y0
    return cls, nx1, ny1, nx2, ny2


def box_center_in_tile(box, x0, y0, tile_w, tile_h):
    _, x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return x0 <= cx < x0 + tile_w and y0 <= cy < y0 + tile_h


def process_split(src_root: Path, out_root: Path, split: str, tile: int, overlap: float, min_keep: float):
    img_dir = src_root / "images" / split
    label_dir = src_root / "labels" / split
    out_img_dir = out_root / "images" / split
    out_label_dir = out_root / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    stride = int(tile * (1 - overlap))
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    saved = 0
    skipped_empty = 0

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        labels = read_yolo_labels(label_dir / f"{img_path.stem}.txt")
        boxes = [yolo_to_xyxy(label, w, h) for label in labels]

        for y0 in build_starts(h, tile, stride):
            for x0 in build_starts(w, tile, stride):
                tile_img = img[y0 : y0 + tile, x0 : x0 + tile]
                th, tw = tile_img.shape[:2]
                new_labels = []

                for box in boxes:
                    if not box_center_in_tile(box, x0, y0, tw, th):
                        continue
                    _cls, x1, y1, x2, y2 = box
                    old_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    clipped = clip_box_to_tile(box, x0, y0, tw, th)
                    _, cx1, cy1, cx2, cy2 = clipped
                    new_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
                    if old_area <= 0 or new_area / old_area < min_keep:
                        continue
                    yolo = xyxy_to_yolo(*clipped, tw, th)
                    if yolo[3] <= 0 or yolo[4] <= 0:
                        continue
                    new_labels.append(yolo)

                # 训练集只保留含目标图块，避免背景块过多拖慢训练。
                if split == "train" and not new_labels:
                    skipped_empty += 1
                    continue

                name = f"{img_path.stem}_x{x0}_y{y0}_s{tile}{img_path.suffix.lower()}"
                cv2.imwrite(str(out_img_dir / name), tile_img)
                label_text = "\n".join(f"{cls} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}" for cls, x, y, bw, bh in new_labels)
                (out_label_dir / f"{Path(name).stem}.txt").write_text(
                    label_text + ("\n" if label_text else ""), encoding="utf-8"
                )
                saved += 1

        if idx % 200 == 0:
            print(f"{split}: {idx}/{len(images)} images processed, {saved} tiles saved")

    print(f"{split}: done, {saved} tiles saved, {skipped_empty} empty train tiles skipped")


def main():
    parser = argparse.ArgumentParser(description="Generate overlapped 960x960 VisDrone YOLO training tiles.")
    parser.add_argument("--src", default="/root/autodl-tmp/yolo26_drone_package/datasets/VisDrone-YOLO")
    parser.add_argument("--out", default="/root/autodl-tmp/yolo26_drone_package/datasets/VisDrone-YOLO-Tile960")
    parser.add_argument("--tile", type=int, default=960)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min-keep", type=float, default=0.30)
    parser.add_argument("--clean", action="store_true", help="Remove old output folder before generating.")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)

    process_split(src_root, out_root, "train", args.tile, args.overlap, args.min_keep)
    print("Tile dataset prepared.")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()

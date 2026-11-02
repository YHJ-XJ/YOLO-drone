# P1-2 o2o/o2m 对照脚本：对比不同 max_det 与 NMS 分支的 recall
# 在 AutoDL 运行：python train_drone/compare_o2o_o2m.py --weight xxx/best.pt
import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.utils.metrics import box_iou

ROOT = Path(__file__).resolve().parents[1]


def validate_with_mode(model, dataloader, end2end, max_det, device):
    """验证 o2o 或 o2m 模式."""
    model.model[-1].end2end = end2end
    model.model[-1].max_det = max_det

    stats = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{'o2o' if end2end else 'o2m'} max_det={max_det}"):
            imgs = batch["img"].to(device)
            preds = model.predict(imgs, verbose=False)

            for i, pred in enumerate(preds):
                gt_boxes = batch["bboxes"][i].to(device)  # (N, 4) xyxy
                gt_cls = batch["cls"][i].to(device).long()  # (N,)

                if len(pred.boxes) == 0:
                    stats.append({"n_gt": len(gt_boxes), "n_pred": 0, "tp": 0, "fp": 0, "fn": len(gt_boxes)})
                    continue

                pred_boxes = pred.boxes.xyxy
                pred_cls = pred.boxes.cls.long()

                iou = box_iou(gt_boxes, pred_boxes)
                tp = ((iou > 0.5).any(dim=1) & (gt_cls.unsqueeze(1) == pred_cls.unsqueeze(0)).any(dim=1)).sum().item()
                fn = len(gt_boxes) - tp
                fp = len(pred_boxes) - tp

                stats.append(
                    {"n_gt": len(gt_boxes), "n_pred": len(pred_boxes), "tp": int(tp), "fp": int(fp), "fn": int(fn)}
                )

    total_tp = sum(s["tp"] for s in stats)
    total_gt = sum(s["n_gt"] for s in stats)
    recall = total_tp / total_gt if total_gt > 0 else 0
    return recall, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-dets", type=int, nargs="+", default=[300, 500, 1000])
    ap.add_argument("--out", default="train_drone/audit_o2o_o2m")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ROOT / args.weight))
    model.model.float().eval().to(device)

    dataloader = build_dataloader(
        str(ROOT / args.data), batch=args.batch, imgsz=640, augment=False, split="val", workers=4, shuffle=False
    )[0]

    results = []
    # o2m+NMS
    print("[o2m] NMS 分支")
    recall_o2m, _ = validate_with_mode(model, dataloader, end2end=False, max_det=300, device=device)
    results.append({"mode": "o2m", "max_det": "NMS", "recall": recall_o2m})

    # o2o with different max_det
    for md in args.max_dets:
        print(f"[o2o] max_det={md}")
        recall_o2o, _ = validate_with_mode(model, dataloader, end2end=True, max_det=md, device=device)
        results.append({"mode": "o2o", "max_det": md, "recall": recall_o2o})

    csv_path = out / f"{Path(args.weight).parent.parent.name}_o2o_o2m.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "max_det", "recall"])
        w.writeheader()
        w.writerows(results)

    print(f"[compare_o2o_o2m] done -> {csv_path}")
    for r in results:
        print(f"  {r['mode']} max_det={r['max_det']}: recall={r['recall']:.4f}")


if __name__ == "__main__":
    main()

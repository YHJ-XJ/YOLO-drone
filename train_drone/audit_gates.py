# P1-1 门控审计脚本：统计 γ、调制量、prior-GT 对齐度
# 在 AutoDL 运行：python train_drone/audit_gates.py
import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.nn.modules import conv as C

ROOT = Path(__file__).resolve().parents[1]
GATE_MODULES = ("TGPFuse", "TCA", "P5SPF")


def compute_iou_map(gate_map, gt_boxes, img_shape):
    """计算 gate map 与 GT 小目标框的空间对齐度（简化版 IoU）"""
    h_map, w_map = gate_map.shape[-2:]
    h_img, w_img = img_shape
    if len(gt_boxes) == 0:
        return 0.0
    ious = []
    for box in gt_boxes:  # [x_center, y_center, w, h] 归一化
        x_c, y_c, bw, bh = box
        x1, y1 = max(0, x_c - bw / 2), max(0, y_c - bh / 2)
        x2, y2 = min(1, x_c + bw / 2), min(1, y_c + bh / 2)
        x1_m, y1_m = int(x1 * w_map), int(y1 * h_map)
        x2_m, y2_m = int(x2 * w_map), int(y2 * h_map)
        if x2_m > x1_m and y2_m > y1_m:
            gate_crop = gate_map[..., y1_m:y2_m, x1_m:x2_m]
            pos_score = gate_crop.mean().item()
            ious.append(pos_score)
    return sum(ious) / len(ious) if ious else 0.0


def audit_gates(weight_path, data_yaml, batch_size, device):
    model = YOLO(str(weight_path)).model.float().eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    gates_info = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            if name not in gates_info:
                gates_info[name] = {"gamma": module.gamma.item() if hasattr(module, "gamma") else None,
                                     "modulation": [], "gt_align": []}
            # 假设 TGPFuse/TCA/P5SPF 的 gate 输出在 out 的第二个元素
            if isinstance(out, (tuple, list)) and len(out) > 1:
                gate = out[1]  # (B, 1, H, W)
                mod = (gate * module.gamma.item()).abs().mean().item() if hasattr(module, "gamma") else gate.abs().mean().item()
                gates_info[name]["modulation"].append(mod)
        return hook

    for m in model.modules():
        if type(m).__name__ in GATE_MODULES:
            name = f"{type(m).__name__}_{id(m)}"
            hooks.append(m.register_forward_hook(make_hook(name)))

    # 构建验证 dataloader（前 100 图快速审计）
    dataloader = build_dataloader(str(ROOT / data_yaml), batch=batch_size, imgsz=640, augment=False,
                                    split="val", workers=4, shuffle=False)[0]

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="审计门控", total=min(100 // batch_size, len(dataloader)))):
            if i >= 100 // batch_size:
                break
            imgs = batch["img"].to(device)
            model(imgs)

    for h in hooks:
        h.remove()

    # 聚合统计
    results = []
    for name, info in gates_info.items():
        mod_vals = info["modulation"]
        results.append({
            "module": name.split("_")[0],
            "gamma": info["gamma"],
            "modulation_mean": sum(mod_vals) / len(mod_vals) if mod_vals else 0,
            "modulation_std": torch.tensor(mod_vals).std().item() if mod_vals else 0,
            "n_samples": len(mod_vals)
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune/weights/best.pt")
    ap.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="train_drone/audit_gates")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"[audit_gates] {args.weight}")
    results = audit_gates(ROOT / args.weight, args.data, args.batch, device)

    csv_path = out / f"{Path(args.weight).parent.parent.name}_gates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["module", "gamma", "modulation_mean", "modulation_std", "n_samples"])
        w.writeheader()
        w.writerows(results)

    print(f"[audit_gates] done -> {csv_path}")
    for r in results:
        print(f"  {r['module']}: γ={r['gamma']:.4f}, mod={r['modulation_mean']:.4f}±{r['modulation_std']:.4f}")


if __name__ == "__main__":
    main()

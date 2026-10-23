import argparse
import gc
import os
from pathlib import Path

# 本地 8GB 显存容易出现显存碎片，提前设置 PyTorch CUDA 分配策略。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from ultralytics import YOLO

# quick5 是“快速预筛选”，不是正式论文结果。
# 目的：用很少轮次观察 early mAP、显存和速度，把明显不值得正式训练的模块先淘汰。
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent


CANDIDATES = [
    {
        "name": "quick5_E12_P2_FGDDown_DySample",
        "model": ROOT / "train_drone/yolov26+P2+FGDDown+DySample/yolov26+P2+FGDDown+DySample.yaml",
    },
    {
        "name": "quick5_TGPFuse3",
        "model": ROOT / "train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3/yolov26+P2+FGDDown+DySample+TGPFuse3.yaml",
    },
    {
        "name": "quick5_TGPFuseSDFM3",
        "model": EXP_DIR / "yolov26_P2_FGDDown_DySample_TGPFuseSDFM3.yaml",
    },
    {
        "name": "quick5_TPCCSIM",
        "model": EXP_DIR / "yolov26_P2_FGDDown_DySample_TPCCSIM.yaml",
    },
    {
        "name": "quick5_TGPFuseSDFM3_TPCCSIM",
        "model": EXP_DIR / "yolov26_P2_FGDDown_DySample_TGPFuseSDFM3_TPCCSIM.yaml",
    },
]


def train_one(candidate, args):
    print(f"\n========== Start {candidate['name']} ==========")
    model = YOLO(str(candidate["model"]))
    model.load(str(ROOT / "yolo26n.pt"))
    model.train(
        data=str(EXP_DIR / "visdrone_quick5_local.yaml"),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(EXP_DIR / "runs"),
        name=f"{args.tag}_{candidate['name']}",
        exist_ok=True,
        patience=args.epochs,
        close_mosaic=0,
        amp=args.amp,
        fraction=args.fraction,
        plots=False,
        save=False,
        deterministic=False,
        verbose=False,
    )
    # 连续跑多个模型时主动释放显存，减少本地 8GB 显卡的显存碎片问题。
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26无人机小目标改进模块quick5预筛选")
    parser.add_argument("--epochs", type=int, default=5, help="每个模型快速训练轮数，默认5")
    parser.add_argument("--batch", type=int, default=1, help="本地显存有限，默认batch=1；AutoDL可改大")
    parser.add_argument("--imgsz", type=int, default=960, help="保持和正式训练一致，默认960")
    parser.add_argument("--workers", type=int, default=0, help="Windows本地建议0，避免多进程读数据问题")
    parser.add_argument("--device", default=0, help="GPU编号，默认0；CPU可写cpu")
    parser.add_argument("--amp", action="store_true", help="开启混合精度，显存更省；不加此参数则关闭AMP")
    parser.add_argument("--fraction", type=float, default=0.05, help="本地快速筛选只用部分训练集，默认0.05")
    parser.add_argument("--tag", default="micro960_b1_f005", help="结果文件夹前缀，避免覆盖正式quick5")
    parser.add_argument("--only", default="", help="只跑名字包含该关键词的模型，例如 TGPFuseSDFM")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected = [c for c in CANDIDATES if not args.only or args.only.lower() in c["name"].lower()]
    print(f"将运行 {len(selected)} 个quick5实验，batch={args.batch}, imgsz={args.imgsz}, epochs={args.epochs}")
    for candidate in selected:
        train_one(candidate, args)

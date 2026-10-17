# P0-1 复杂度审计：从已训练 best.pt 统一重算参数量与 FLOPs，不训练。
# 在 AutoDL 的 yolo26 目录运行：python train_drone/audit_complexity.py  （输出到 train_drone/audit_complexity/）
import argparse
import copy
import csv
import math
import subprocess
from pathlib import Path

import thop
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules import conv as C
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NC = 10
# 这些模块内含 THOP 不计数的函数式运算（grid_sample、逐元素门控乘法、Haar/池化差分等）
UNCOUNTED_NAMES = ("DySample", "TGPFuse", "TGPFuseSDFM", "DAFuseST", "P5SPF", "TCA", "TPCCSIM", "WDCLDown", "DCLDown")
UNCOUNTED = tuple(getattr(C, n) for n in UNCOUNTED_NAMES if hasattr(C, n))
GRID_SAMPLE_FLOPS_PER_ELEM = 7  # 双线性插值约定：4 乘 3 加，不含坐标与权重计算


def gflops(module, inp):
    return thop.profile(copy.deepcopy(module), inputs=[inp], verbose=False)[0] / 1e9 * 2


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def level_name(stride):
    return f"P{int(round(math.log2(float(stride))))}"


def git_version():
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "ultralytics"], cwd=ROOT, text=True).strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def profile_layers(model, imgsz, device, tag):
    n_backbone = len(model.yaml["backbone"])
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    y, rows = [], []
    for m in model.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        if isinstance(m, Detect):
            heads = [("o2m", m.cv2, m.cv3)] if getattr(m, "cv2", None) is not None else []
            if hasattr(m, "one2one_cv2"):
                heads.append(("o2o", m.one2one_cv2, m.one2one_cv3))
            for branch, box, cls in heads:
                for i, f in enumerate(x):
                    rows.append(dict(tag=tag, imgsz=imgsz, idx=m.i, type=f"Detect.{branch}",
                                     stage=f"detect_{level_name(m.stride[i])}", out_stride=int(m.stride[i]),
                                     params=n_params(box[i]) + n_params(cls[i]), gflops=gflops(box[i], f) + gflops(cls[i], f)))
            break
        flops = gflops(m, x.copy() if isinstance(x, list) else x)
        x = m(x)
        stride = imgsz // x.shape[-1]
        stage = "backbone" if m.i < n_backbone else f"neck_{level_name(stride)}"
        rows.append(dict(tag=tag, imgsz=imgsz, idx=m.i, type=m.type, stage=stage, out_stride=stride,
                         params=n_params(m), gflops=flops))
        y.append(x if m.i in model.save else None)
    return rows


def uncounted_stats(model, imgsz, device):
    stats, hooks = {}, []
    for m in model.modules():
        if isinstance(m, UNCOUNTED):
            def hook(mod, inp, out, name=type(m).__name__):
                o = out if torch.is_tensor(out) else out[0]
                s = stats.setdefault(name, [0, 0])
                s[0] += 1
                s[1] += o.numel()
            hooks.append(m.register_forward_hook(hook))
    model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    for h in hooks:
        h.remove()
    return stats


def audit(weight, imgszs, device):
    model = YOLO(str(weight)).model.float().eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    det = model.model[-1]
    rec = dict(name=weight.parent.parent.name, weight=str(weight.relative_to(ROOT)), code=git_version(),
               nc_yaml=model.yaml.get("nc"), nc_head=det.nc, n_names=len(model.names),
               levels="/".join(level_name(s) for s in det.stride), end2end=bool(getattr(det, "end2end", False)),
               params_unfused=n_params(model))
    fused = copy.deepcopy(model).fuse(verbose=False)
    rec["o2m_removed"] = getattr(fused.model[-1], "cv2", None) is None
    rec["params_fused"] = n_params(fused)
    layer_rows = []
    for s in imgszs:
        ur = profile_layers(model, s, device, "unfused")
        fr = profile_layers(fused, s, device, "fused")
        layer_rows += ur + fr
        rec[f"gflops_unfused@{s}"] = sum(r["gflops"] for r in ur)
        rec[f"gflops_fused@{s}"] = sum(r["gflops"] for r in fr)
        rec[f"gflops_fused_ultra@{s}"] = get_flops(copy.deepcopy(fused), s)
        rec[f"detect_o2m_unfused@{s}"] = sum(r["gflops"] for r in ur if r["type"] == "Detect.o2m")
        rec[f"detect_o2o_fused@{s}"] = sum(r["gflops"] for r in fr if r["type"] == "Detect.o2o")
        for stage in ("backbone", "neck_P2", "detect_P2"):
            rec[f"{stage}_fused@{s}"] = sum(r["gflops"] for r in fr if r["stage"] == stage)
        un = uncounted_stats(fused, s, device)
        rec[f"gridsample_gflops_est@{s}"] = un.get("DySample", [0, 0])[1] * GRID_SAMPLE_FLOPS_PER_ELEM / 1e9
        rec[f"uncounted@{s}"] = "; ".join(f"{k}x{v[0]}:{v[1] / 1e6:.2f}M elems" for k, v in sorted(un.items()))
    for r in layer_rows:
        r["name"] = rec["name"]
    return rec, layer_rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_report(path, recs, imgszs):
    s = imgszs[-1]
    nan = float("nan")
    lines = [
        f"# 复杂度审计（代码版本 {recs[0].get('code', '?') if recs else '?'}）",
        "",
        "口径：THOP MACs×2，batch=1，全尺寸输入实测（非 stride 代理）。fused = Conv+BN 融合并移除 o2m 头。"
        f"THOP 不计 grid_sample、门控逐元素乘等函数式运算；grid_sample 另按每输出元素 {GRID_SAMPLE_FLOPS_PER_ELEM} FLOPs 估算，未并入总数。"
        "P2 Neck 指输出 stride=4 的 Neck 层。部署结论以导出后端实际图和 Pi 5 延迟为准。",
        "",
        f"| 实验 | nc head/yaml | 检测层 | Params fused/M | GFLOPs fused@640 | GFLOPs fused@{s} | P2 Neck+Detect 占比@{s} | grid_sample 估算@{s} | o2m 已移除 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in recs:
        if "error" in r:
            lines.append(f"| {r['name']} | 出错：{r['error']} | | | | | | | |")
            continue
        tot = r.get(f"gflops_fused@{s}", 0)
        p2 = r.get(f"neck_P2_fused@{s}", 0) + r.get(f"detect_P2_fused@{s}", 0)
        flag = "" if r["nc_head"] == EXPECTED_NC == r["n_names"] else "（类别数异常）"
        lines.append(
            f"| {r['name']} | {r['nc_head']}/{r['nc_yaml']}{flag} | {r['levels']} | {r['params_fused'] / 1e6:.3f} | "
            f"{r.get('gflops_fused@640', nan):.3f} | {tot:.3f} | {(p2 / tot * 100) if tot else 0:.1f}% | "
            f"{r.get(f'gridsample_gflops_est@{s}', 0):.3f} | {r['o2m_removed']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="train_drone/**/weights/best.pt")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960])
    ap.add_argument("--out", default="train_drone/audit_complexity")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    recs, layers = [], []
    with torch.no_grad():
        for w in sorted(ROOT.glob(args.weights)):
            print(f"[audit] {w.relative_to(ROOT)}")
            try:
                rec, rows = audit(w, args.imgsz, device)
            except Exception as e:
                rec, rows = dict(name=w.parent.parent.name, weight=str(w.relative_to(ROOT)), error=repr(e)), []
            recs.append(rec)
            layers += rows
    write_csv(out / "summary.csv", recs)
    write_csv(out / "per_layer.csv", layers)
    write_report(out / "report.md", recs, args.imgsz)
    print(f"[audit] done -> {out}")


if __name__ == "__main__":
    main()

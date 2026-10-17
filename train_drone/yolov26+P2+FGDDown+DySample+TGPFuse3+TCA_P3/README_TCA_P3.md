# TGPFuse3 + TCA_P3 冲精度实验

## 实验定位

本实验用于冲精度，而不是轻量化剪枝。

结构为：

```text
YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + TCA_P3
```

它相对 `TGPFuse3` 只新增 P3 位置的 TCA；相对 `P5SPFPrune+TCA_P3` 则保留 P5 检测头。

## 改动位置

```text
P4 -> P3 融合
-> C3k2
-> TCA
-> 后续 P3->P2、P2->P3、Detect
```

## 为什么这么做

已有实验说明：

```text
TCA 放在 P3 有轻微正收益；
TCA 扩展到 P4 后退步；
P5SPFPrune 有轻量化优势，但可能牺牲一部分精度上限。
```

因此本实验保留完整 P5 检测头，只在 P3 小目标敏感层增强，目标是优先冲 mAP。

## AutoDL 运行命令

在 `yolo26` 目录下运行：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3_train.py
```

## 目标

```text
mAP50    >= 0.445
mAP50-95 >= 0.268
```

如果 130 轮后仍明显低于 `TCA_P3` 的 0.43982 / 0.26568，则可以提前停止。

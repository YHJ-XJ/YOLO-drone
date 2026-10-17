# Tile960_FT30 低成本微调实验

## 当前判断

`Tile960` 虽然中途停止，但结果已经是当前最好：

```text
best mAP50    = 0.44490
best mAP50-95 = 0.27655
```

从第 103 到 117 轮看，mAP50 基本在 0.444 左右横盘，继续恢复训练到 200 轮的收益可能很小，但成本很高。

## 这个实验做什么

不继续用切片数据长时间训练，而是：

```text
Tile960 best.pt -> 原始 VisDrone 训练集 -> 30 轮短程微调
```

这样做的目的：

```text
1. 成本低，预计远低于继续 Tile960 完整训练；
2. 保留切片训练带来的小目标尺度增强能力；
3. 用原始图像微调，使模型重新适应原始验证集分布；
4. 观察能否把 mAP50 从 0.44490 推到 0.45 附近。
```

## AutoDL 运行命令

在 `yolo26` 目录下运行：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30_train.py
```

## 注意

运行前需要保证 `Tile960` 的 `weights/best.pt` 已经在 AutoDL 上。

脚本会优先寻找：

```text
/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone/yolov26_P2_FGDDown_DySample_TGPFuse3_TCA_P3_Tile960/weights/best.pt
```

如果没有，再寻找：

```text
/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/weights/best.pt
```

## 预期

```text
mAP50    >= 0.445
mAP50-95 尽量不低于 0.276
```

如果 15-20 轮后没有超过 Tile960 的 best，不建议继续烧钱。

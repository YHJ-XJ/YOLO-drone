# TCA_P3P4 实验说明

## 实验目的

本实验是在当前较优轻量化模型 `TCA_P3` 基础上继续扩展：

```text
YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune + TCA_P3P4
```

相对 `TCA_P3`，本实验只新增一个 P4 位置的 TCA 模块。

## 改动位置

```text
P5SPF 输出后的 P4 特征 -> TCA -> Detect
```

也就是对融合了 P5 深层语义的 P4 特征进行轻量卷积注意力增强。

## 运行命令

在 AutoDL 的 `yolo26` 目录下运行：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4_train.py
```

## 判断标准

如果结果达到或接近：

```text
mAP50 >= 0.442
mAP50-95 >= 0.267
```

说明 P4 语义注意力增强有效。否则不建议继续堆叠 TCA。

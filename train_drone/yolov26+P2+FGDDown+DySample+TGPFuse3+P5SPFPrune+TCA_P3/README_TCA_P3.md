# YOLO26n + P5SPFPrune + TCA_P3

## 实验目的

在当前最稳的最终结构 `P5SPFPrune` 基础上，只在 P4->P3 融合后的 P3 特征处加入轻量卷积注意力 `TCA`。

该实验用于验证：

```text
Neck 中 P3 融合特征的局部纹理/上下文注意力是否能进一步提升无人机小目标检测精度。
```

## 模块说明

TCA 表示 Tiny Convolutional Attention。

模块结构：

```text
输入特征
-> 1x1 Conv 通道对齐
-> DWConv 3x3 提取局部纹理
-> DWConv 5x5 提取邻域上下文
-> 1x1 Conv + Sigmoid 得到空间注意力
-> 残差增强输出
```

它不是直接照搬 CBAM/SE/CA，而是针对无人机小目标 Neck 融合特征设计的轻量卷积注意力。

## AutoDL 训练命令

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
export PYTHONPATH=$PWD

python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3_train.py
```

## 对比目标

当前最终模型：

```text
P5SPFPrune best mAP50 = 0.43770
P5SPFPrune best mAP50-95 = 0.26391
Params = 2.029M
```

若 TCA_P3 达到：

```text
mAP50 >= 0.442
mAP50-95 >= 0.266
```

说明该卷积注意力方向值得继续扩展到 P4 或 P2->P3 回传融合后。

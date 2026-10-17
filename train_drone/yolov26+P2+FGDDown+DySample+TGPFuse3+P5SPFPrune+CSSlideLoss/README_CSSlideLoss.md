# YOLO26n + P5SPFPrune + CS-SlideLoss

## 实验目的

老师认为当前模型的提升幅度还不够大，因此本实验在不改变推理结构的前提下，针对 VisDrone 的类别不均衡和小目标难样本问题，引入训练阶段损失函数改进。

## 模型结构

模型结构与当前最终模型保持一致：

```text
YOLO26n + P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune
Detect(P2, P3, P4)
```

本实验只启用 `CS-SlideLoss`，不改变网络层、不增加参数量、不增加推理 FLOPs。

## CS-SlideLoss 组成

CS-SlideLoss 表示 Class-Scale Aware Slide Loss，包含三部分：

1. 难样本滑动权重：对低置信度、难分类样本提高 loss 权重。
2. 类别频次权重：根据 VisDrone 训练集类别频次设置温和的逆频率权重。
3. 小目标尺度权重：对分配到小尺度目标的正样本提高训练贡献。

## AutoDL 训练命令

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
export PYTHONPATH=$PWD

python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss_train.py
```

## 判断标准

对比 P5SPFPrune：

```text
P5SPFPrune best mAP50 = 0.43770
P5SPFPrune best mAP50-95 = 0.26391
```

若本实验达到：

```text
mAP50 >= 0.445
mAP50-95 >= 0.268
```

则说明 CS-SlideLoss 对提升精度有明显帮助，可作为训练策略创新点写入论文。

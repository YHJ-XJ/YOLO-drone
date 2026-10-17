# YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune

## 实验目的

本实验不再优先做 Teacher 蒸馏，而是尝试结构级轻量化创新：

先把 P5 的高层语义注入 P4，再删除 P5 检测头，使最终模型仍保持三个检测层 P2/P3/P4。

## 创新点

P5SPF：P5 Semantic Preserving Fusion。

不同于简单删除 P5 检测头，本实验在删除前加入一个轻量融合模块：

1. P5 上采样到 P4 尺度；
2. 用 1x1 Conv 将 P5 通道压到 P4 通道；
3. 用 P4 生成细节门控，保留小目标定位信息；
4. 用 P5 生成语义门控，补偿删除 P5 后的高层语义损失；
5. 输出仍为 P4 通道，不做 Concat 通道膨胀。

## AutoDL 训练命令

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
export PYTHONPATH=$PWD

python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune_train.py
```

## 判断标准

对比当前最强模型：

```text
YOLO26 + P2 + FGDDown + DySample + TGPFuse3
best mAP50 = 0.43843
best mAP50-95 = 0.26402
```

如果 P5SPFPrune 能达到：

```text
mAP50 >= 0.430
mAP50-95 >= 0.258
并且参数量/计算量低于四检测头版本
```

则说明“P5语义保留式剪枝”有继续写成论文创新点的价值。

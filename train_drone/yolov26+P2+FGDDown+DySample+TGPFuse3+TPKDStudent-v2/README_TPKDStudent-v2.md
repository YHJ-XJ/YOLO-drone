# YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + TPKDStudent-v2

## 实验目的

v1 学生模型将 width 从 0.25 降到 0.20，参数量明显降低，但精度基本回到 baseline 附近，说明压缩过猛或蒸馏约束过强。

v2 做两个调整：

1. width 从 0.20 提升到 0.225，给小目标特征保留更多通道容量。
2. distillation loss 权重 dis 从 0.5 降到 0.3，避免 Teacher 约束过强导致 Student 学不稳。

## 创新点定位

该实验服务于论文中的轻量化创新点：

Tiny-Prior Guided Knowledge Distillation。

即用高精度的 TGPFuse3 小目标先验融合模型作为 Teacher，训练更窄的同拓扑 Student，在尽量保留小目标检测精度的同时降低参数量。

## AutoDL 训练命令

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
export PYTHONPATH=$PWD

ls train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3/weights/best.pt

python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent-v2/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent-v2_train.py
```

如果显存不足，优先把训练脚本中的 batch 从 8 改成 4，不建议优先降低 imgsz。

## 判断标准

若达到以下范围，就说明 v2 有继续优化价值：

```text
mAP50 >= 0.420
mAP50-95 >= 0.252
参数量明显低于 TGPFuse3
```

若仍明显低于 baseline，则说明同拓扑窄学生路线不合适，下一步应改为 P2-free student + 自定义特征蒸馏匹配。

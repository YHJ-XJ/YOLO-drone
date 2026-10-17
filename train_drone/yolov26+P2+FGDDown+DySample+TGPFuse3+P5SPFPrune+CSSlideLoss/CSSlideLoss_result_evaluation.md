# CS-SlideLoss 实验结果评估

## 实验配置

模型结构：

```text
YOLO26n + P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune
```

训练策略：

```text
启用 CS-SlideLoss
epochs = 200
imgsz = 960
batch = 8
```

CS-SlideLoss 只改变训练损失，不改变模型结构、参数量和推理计算量。

## 最优结果

```text
best mAP50 = 0.42793，epoch 126
best mAP50-95 = 0.26075，epoch 142
best Precision = 0.54953，epoch 141
best Recall = 0.43943，epoch 157
last mAP50 = 0.42330，epoch 200
last mAP50-95 = 0.25801，epoch 200
```

## 与关键模型对比

| 模型 | best mAP50 | best mAP50-95 |
|---|---:|---:|
| YOLO26n baseline | 0.41202 | 0.24407 |
| TGPFuse3 四检测头 | 0.43843 | 0.26402 |
| P5SPFPrune 三检测头 | 0.43770 | 0.26391 |
| P5SPFPrune + CS-SlideLoss | 0.42793 | 0.26075 |

## 结论

CS-SlideLoss 当前版本没有带来预期提升，反而使 mAP50 相比 P5SPFPrune 下降约 0.98 个百分点，mAP50-95 下降约 0.32 个百分点。

可能原因：

1. 类别频次权重与滑动难样本权重叠加后，对分类分支扰动较强。
2. VisDrone 中多数类如 car、pedestrian 对总体 mAP 贡献较大，过度提高少数类权重可能损害整体排序。
3. 当前 end-to-end YOLO26 的分类分支较敏感，直接加权 BCE 可能改变正负样本学习平衡。

当前建议：

```text
不把 CS-SlideLoss 纳入最终模型。
最终模型仍采用 P5SPFPrune。
若后续继续优化 loss，应降低权重强度，或只在训练后期微调阶段启用。
```

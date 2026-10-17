# Tile960 小目标增强训练

## 为什么做这个

前面的结构实验已经接近平台期，继续堆注意力模块收益很小。VisDrone 的核心难点是目标太小、密集、遮挡多，因此下一阶段优先做切片训练：把大图切成重叠的 960x960 图块，让小目标在训练视野里变大。

## 实验结构

```text
YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + TCA_P3 + Tile960
```

模型结构沿用当前冲精度路线，主要变化是训练数据。

## 第一步：生成切片训练集

在 AutoDL 的 `yolo26` 目录下运行：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/make_visdrone_tile960.py --clean
```

默认输入：

```text
/root/autodl-tmp/yolo26_drone_package/datasets/VisDrone-YOLO
```

默认输出：

```text
/root/autodl-tmp/yolo26_drone_package/datasets/VisDrone-YOLO-Tile960
```

## 第二步：正式训练

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_train.py
```

## 注意

这里的训练集是切片图，验证集仍然用原始 VisDrone val，因此结果仍然可以和之前实验大致对比。

如果训练速度明显变慢，是正常的，因为切片会增加训练图片数量。这个实验的目的就是冲精度，不是省时间。

## 目标

```text
mAP50    >= 0.45
mAP50-95 >= 0.27
```

如果这个实验有效，再考虑做最终部署版；如果无效，就说明瓶颈不在单纯小目标尺度，需要转向标签清洗、类别重采样或推理阶段切片。

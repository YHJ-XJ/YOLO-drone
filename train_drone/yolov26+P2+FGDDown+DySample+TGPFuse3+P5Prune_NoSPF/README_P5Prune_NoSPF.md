# P5Prune_NoSPF 消融实验说明

## 实验目的

这个实验不是为了冲最高精度，而是为了论文消融。

它验证一个问题：

```text
如果直接删掉 P5 检测头，但不使用 P5SPF 语义保留融合，结果会不会下降？
```

如果该模型明显低于 `P5SPFPrune`，就能证明 P5SPF 的作用不是简单删头，而是有效保留了深层语义。

## 改动位置

```text
保留 Detect(P2, P3, P4)
删除 P5 检测分支
不加入 P5SPF
```

## 运行命令

在 AutoDL 的 `yolo26` 目录下运行：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF/yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF_train.py
```

## 判断标准

对比对象：

```text
P5SPFPrune best mAP50    = 0.43770
P5SPFPrune best mAP50-95 = 0.26391
```

如果 `P5Prune_NoSPF` 低于它，就说明 P5SPF 是有效模块。

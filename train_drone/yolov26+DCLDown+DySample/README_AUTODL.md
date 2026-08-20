# YOLO26 + DCLDown + DySample AutoDL 训练说明

本实验用于验证组合改进：

```text
YOLO26 baseline
+ DCLDown  替换 P2/4 -> P3/8 下采样 Conv
+ DySample 替换 neck/head 中两处 nearest 上采样
```

## 训练命令

在 AutoDL 终端中进入 yolo26 根目录：

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
```

建议用 tmux 防止断开连接导致训练中断：

```bash
tmux new -s yolov26_dcl_dysample
```

在 tmux 中运行：

```bash
python train_drone/yolov26+DCLDown+DySample/yolov26+DCLDown+DySample_train.py
```

退出但保持训练运行：

```text
Ctrl+B，然后按 D
```

重新进入：

```bash
tmux attach -t yolov26_dcl_dysample
```

## 结果保存位置

训练结果会保存到：

```text
/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone/yolov26_DCLDown_DySample
```

## 对比目标

重点对比已有实验：

```text
YOLO26 baseline: mAP50-95 = 0.24407
YOLO26 + DCLDown: mAP50-95 = 0.24469
YOLO26 + DySample: mAP50-95 = 0.24843
YOLO26 + P2 + FGDDown: mAP50-95 = 0.25567
```

如果本实验 mAP50-95 明显高于 0.24843，说明 DCLDown 和 DySample 有组合收益。
如果精度持平但 GFLOPs 明显低于 P2 类模型，仍然适合保留为轻量化路线。

# YOLO26 + DySample + DAFuseST AutoDL 训练说明

本实验是当前推荐主线：

```text
YOLO26 baseline
+ DySample  替换两处 nearest 上采样
+ DAFuseST  替换前两处跨尺度 Concat
```

## 设计目的

```text
DySample 负责更自适应地恢复高分辨率特征。
DAFuseST 负责在融合浅层细节和深层语义时学习动态权重。
不加入 P2 检测头，避免明显增加边缘部署开销。
```

## 训练命令

在 AutoDL 终端中进入 yolo26 根目录：

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
```

建议使用 tmux：

```bash
tmux new -s yolov26_dysample_dafusest
```

在 tmux 中运行：

```bash
python train_drone/yolov26+DySample+DAFuseST/yolov26+DySample+DAFuseST_train.py
```

退出但保持训练：

```text
Ctrl+B，然后按 D
```

重新进入：

```bash
tmux attach -t yolov26_dysample_dafusest
```

## 结果保存位置

```text
/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone/yolov26_DySample_DAFuseST
```

## 判断标准

重点对比：

```text
YOLO26 baseline: mAP50-95 = 0.24407
YOLO26 + DySample: mAP50-95 = 0.24843
YOLO26 + P2 + FGDDown: mAP50-95 = 0.25567
```

如果本实验超过 0.24843，说明 DAFuseST 与 DySample 有组合收益。
如果接近或超过 0.252，则可以作为当前论文主模型候选。

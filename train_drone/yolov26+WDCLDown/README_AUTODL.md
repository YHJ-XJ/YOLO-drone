# YOLO26 + WDCLDown AutoDL 训练说明

本实验验证 WDCLDown（小波细节对比轻量下采样）对小目标细节保留的净收益。

## 设计目的

```text
YOLO26 baseline（结构不变）
+ WDCLDown  只替换 backbone P2->P3 的 stride=2 Conv
```

WDCLDown 三条支路：
1. PixelUnshuffle(2) -> 1x1 Conv —— 空间-通道重排（SPDConv/reorg 思想）
2. DWConv(3,s=2) -> 1x1 Conv —— 可学习局部结构下采样
3. Haar 高频(LH/HL/HH) -> 1x1 Conv —— 方向性边缘/纹理（小波池化思想）

相比 DCLDown 的 MaxPool-AvgPool 对比支路，Haar 高频能同时保留水平/垂直/对角三个方向的细节。

## 训练命令

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
```

```bash
tmux new -s yolov26_wdcldown
python train_drone/yolov26+WDCLDown/yolov26+WDCLDown_train.py
```

退出保持训练：`Ctrl+B` 然后按 `D`；重进：`tmux attach -t yolov26_wdcldown`

## 结果保存位置

```text
/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone/yolov26_WDCLDown
```

## 判断标准

```text
YOLO26 baseline:      mAP50-95 = 0.24407
YOLO26 + DCLDown:     mAP50-95 = 0.24469
YOLO26 + DySample:    mAP50-95 = 0.24843
```

- 若 `mAP50-95 > 0.24469`（超过 DCLDown），说明小波高频分支更有效，下一步做 `WDCLDown + DySample`。
- 若 `mAP50-95` 接近 DCLDown 但参数量/GFLOPs 更低，也值得保留。
- 若 `mAP50-95 < 0.244`，则 WDCLDown 不理想，回退 DySample 单独路线。

## 说明：本次代码同步需包含的改动

```text
ultralytics/nn/modules/conv.py       新增 WDCLDown 类 + DAFuseST v2 重写
ultralytics/nn/modules/__init__.py   注册 WDCLDown
ultralytics/nn/tasks.py              注册 WDCLDown（base_modules）
train_drone/yolov26+WDCLDown/*       本实验文件
```

同步到 AutoDL 后无需 `pip install -e .`，直接运行训练脚本即可。

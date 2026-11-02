# YOLO26 + NWD AutoDL 训练说明

本实验验证 NWD（Normalized Wasserstein Distance）小目标定位损失的净收益。

## 设计目的

```text
YOLO26 baseline（结构不变）
+ NWD  替换 box loss 中一半 CIoU
```

NWD 是"损失级"改进：只影响训练，推理阶段零额外计算，与轻量部署路线完全兼容。

## 训练命令

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
```

```bash
tmux new -s yolov26_nwd
python train_drone/yolov26+NWD/yolov26+NWD_train.py
```

退出保持训练：`Ctrl+B` 然后按 `D`；重进：`tmux attach -t yolov26_nwd`

## 结果保存位置

```text
/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone/yolov26_NWD
```

## 判断标准

```text
YOLO26 baseline: mAP50-95 = 0.24407
```

如果本实验明显超过 0.24407，说明 NWD 对 tiny object 有效，下一步组合 `DySample + NWD`。

## 关键开关（ultralytics/utils/loss.py 顶部）

```python
NWD_ENABLED = True  # 总开关
NWD_ALPHA = 0.5  # NWD 在 box loss 中的占比
NWD_CONSTANT = 16.0  # 像素归一化常数，越小对 tiny box 越敏感
```

若要得到"同条件无 NWD"的对照，把 `NWD_ENABLED` 改为 `False` 重跑即可。

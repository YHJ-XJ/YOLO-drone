# YOLO26n Baseline CARPK

## 运行前检查数据集

在 AutoDL 的 `yolo26` 目录下运行：

```bash
python train_carpk/yolov26n_baseline_carpk/check_carpk_dataset.py
```

确认 `train` 和 `val` 都有图片和标签后，再训练。

## 开始训练

```bash
python train_carpk/yolov26n_baseline_carpk/yolov26n_baseline_carpk_train.py
```

## 省钱判断

先跑 50 轮。如果 mAP50 能快速接近或超过 90%，CARPK 很适合继续做无人机车辆小目标方向。

如果 50 轮后 mAP50 仍不高，先不要继续加 epoch，优先检查数据集转换和标签是否正确。

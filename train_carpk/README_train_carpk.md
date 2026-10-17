# train_carpk 说明

这个目录用于 CARPK 无人机车辆检测方向实验，和 `train_drone` 并列。

当前先准备 baseline：

```text
YOLO26n baseline CARPK
```

CARPK 仍然属于无人机视角，但类别通常只有 car 一类，比 VisDrone 简单很多，更适合验证“无人机小目标 + 高基线 + 轻量化改进”的论文路线。

## 数据集建议格式

请把 CARPK 数据集整理为 YOLO 标准格式：

```text
/root/autodl-tmp/yolo26_drone_package/datasets/CARPK-YOLO/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

如果没有 test，可以先不放 test，但 `val` 必须有。

## 类别

当前按单类车辆检测写：

```text
car
```

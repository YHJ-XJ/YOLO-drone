# 新方向 baseline 运行说明

## 已建立目录

与 `train_drone` 并列新增：

```text
train_pcb/
train_carpk/
```

## 当前目标

先不改模型，直接跑 YOLO26n baseline，确认数据集本身是否适合“高基线 + 提升 2 个点”的论文路线。

## 推荐先跑顺序

第一优先级：

```text
train_carpk/yolov26n_baseline_carpk/
```

原因：它仍然是无人机视角，和原来的论文方向衔接更自然。

第二优先级：

```text
train_pcb/yolov26n_baseline_pcb/
```

原因：它更容易获得 90% 以上 mAP50，但方向会从无人机小目标转向工业缺陷检测。

## 省钱策略

每个数据集先跑 50 轮：

```text
imgsz=640
epochs=50
batch=16
patience=20
```

如果 30-50 轮内 mAP50 已经达到 90% 左右，再考虑：

```text
1. 加到 100 轮；
2. 引入前面有效的 FGDDown / DySample / TGPFuse / TCA；
3. 做完整消融实验。
```

如果 baseline 低于预期，先不要继续烧钱，优先检查：

```text
1. 数据集是否转 YOLO 格式正确；
2. 类别 names 是否和 label 中 class id 对应；
3. train/val 是否划分合理；
4. 是否有大量空标签、错标签或重复标签。
```

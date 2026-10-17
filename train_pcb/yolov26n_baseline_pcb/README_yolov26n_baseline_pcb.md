# YOLO26n PCB Baseline

这一组用于测试 PCB 缺陷检测方向是否适合作为新的论文主数据集。

## 数据集

```text
datasets/PCB-YOLO
```

类别：

```text
0: Missing_Hole
1: MouseBite
2: Open_Circuit
3: Short_Circuit
4: Spur
5: Spurious_Copper
```

## AutoDL 运行

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
python train_pcb/yolov26n_baseline_pcb/check_pcb_dataset.py
python train_pcb/yolov26n_baseline_pcb/yolov26n_baseline_pcb_train.py
python train_pcb/yolov26n_baseline_pcb/validate_yolov26n_baseline_pcb_test.py
```

## 训练参数

```text
imgsz=640
epochs=100
batch=16
workers=8
patience=25
amp=True
```

先看 baseline 的 `mAP50` 是否处于 `90%-96%` 区间。如果太低，方向不稳；如果高到 `99%` 左右，提升空间又太小。比较理想的是 `mAP50` 高于 90%，但 `mAP50-95` 还有明显提升空间。

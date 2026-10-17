# train_pcb 说明

这个目录用于新的 PCB 缺陷检测方向实验，和 `train_drone` 并列。

当前先准备 baseline：

```text
YOLO26n baseline PCB
```

目的不是立刻改模型，而是先确认 PCB 数据集本身能不能达到较高基线，例如 mAP50 90% 以上。若 baseline 已经很高，再把前面无人机方向中有效的模块迁移过来，争取提升 2 个百分点以上。

## 数据集建议格式

请把 PCB 数据集整理为 YOLO 标准格式：

```text
/root/autodl-tmp/yolo26_drone_package/datasets/PCB-YOLO/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

若没有 test，可以先不放 test，但 `val` 必须有。

## 默认类别

当前按常见 PCB 缺陷 6 类写：

```text
missing_hole
mouse_bite
open_circuit
short
spur
spurious_copper
```

如果你下载的数据集类别名不同，只需要改 `yolov26n_baseline_pcb_data.yaml` 里的 `names`。

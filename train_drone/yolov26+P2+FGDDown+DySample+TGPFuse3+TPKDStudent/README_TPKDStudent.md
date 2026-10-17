# TPKDStudent 实验说明

## 目的

当前最强模型是 `YOLO26 + P2 + FGDDown + DySample + TGPFuse3`，精度最高但包含 P2 高分辨率检测分支，计算量偏大。

本实验先不激进删除 P2，而是做更稳的轻量化第一步：

- 保留 TGPFuse3 的层级结构和小目标先验融合路径；
- 将宽度系数从 `0.25` 降到 `0.20`；
- 使用 TGPFuse3 的 `best.pt` 作为 Teacher 做知识蒸馏；
- 观察 Student 是否能以更小模型接近 Teacher 精度。

## 创新表述

可暂命名为：

`Tiny-Prior Knowledge Distillation Student, TPKD-Student`

重点不是普通蒸馏，而是让已经验证有效的 TGPFuse 小目标先验融合结构进入轻量 Student，并通过 Teacher 的小目标响应进行约束。

## AutoDL 运行

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
export PYTHONPATH=$PWD
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent_train.py
```

## 注意

请确认 Teacher 权重存在：

```bash
ls train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3/weights/best.pt
```

如果路径不存在，需要先把 TGPFuse3 的 `weights/best.pt` 上传到该位置。

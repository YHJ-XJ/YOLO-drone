# YOLO26n CARPK Formal Baseline

这一组是 CARPK 正式划分版 baseline，用来作为论文主实验的基准模型。

## 数据划分

- `train`：官方 train 的 80%
- `val`：官方 train 的 20%
- `test`：官方 test，保留为最终测试集
- 随机种子：`20260830`
- 类别数：1 类，`car`

## AutoDL 运行命令

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
python train_carpk/yolov26n_baseline_carpk_formal/check_carpk_formal_dataset.py
python train_carpk/yolov26n_baseline_carpk_formal/yolov26n_baseline_carpk_formal_train.py
```

## 训练参数

- `imgsz=640`
- `epochs=100`
- `batch=16`
- `workers=8`
- `patience=20`
- `amp=True`

如果这组正式 baseline 的 `mAP50` 仍然保持在 90% 以上，就可以开始跑后续改进模型。

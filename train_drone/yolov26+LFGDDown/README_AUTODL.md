# YOLO26 + LFGDDown AutoDL Run

This experiment keeps the original YOLO26 P3-P5 detection heads and replaces only the backbone P2 -> P3 downsampling Conv with LFGDDown.

## Expected Package Layout

```text
/root/autodl-tmp/yolo26_drone_package
  datasets/VisDrone-YOLO
  yolo26
    ultralytics
    yolo26n.pt
    train_drone/yolov26+LFGDDown
```

## Setup

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
pip install -e .
python -c "from ultralytics import YOLO; model=YOLO('train_drone/yolov26+LFGDDown/yolov26+LFGDDown.yaml'); model.load('yolo26n.pt'); model.info(imgsz=960); print('ok')"
```

## Train

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
python train_drone/yolov26+LFGDDown/yolov26+LFGDDown_train.py
```

## Watch GPU

```bash
watch -n 1 nvidia-smi
```

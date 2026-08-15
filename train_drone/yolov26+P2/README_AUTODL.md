# YOLO26-P2 AutoDL Training

Recommended upload location:

```bash
/root/autodl-tmp/yolo26_drone_package
```

After extracting the package, the expected structure is:

```text
/root/autodl-tmp/yolo26_drone_package
├── yolo26
│   ├── ultralytics
│   ├── pyproject.toml
│   ├── yolo26n.pt
│   └── train_drone/yolov26_p2_autodl
│       ├── train_yolov26_p2_autodl.py
│       └── visdrone_autodl.yaml
└── datasets/VisDrone-YOLO
    ├── images
    ├── labels
    └── visdrone.yaml
```

Run:

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
conda create -n yolo26 python=3.10 -y
conda activate yolo26
pip install -e .
python train_drone/yolov26_p2_autodl/train_yolov26_p2_autodl.py
```

Recommended tmux run:

```bash
tmux new -s yolov26_p2
cd /root/autodl-tmp/yolo26_drone_package/yolo26
conda activate yolo26
python train_drone/yolov26_p2_autodl/train_yolov26_p2_autodl.py
```

Detach tmux:

```text
Ctrl+B, then D
```

Reattach:

```bash
tmux attach -t yolov26_p2
```

Results will be saved to:

```text
/root/autodl-tmp/yolo26_drone_package/yolo26/train_drone/yolov26_p2_autodl
```

If CUDA memory is not enough, change `batch=8` to `batch=6` or `batch=4`.

# VisDrone 无人机小目标检测实验目录

本目录包含基于 YOLOv26n 在 VisDrone 数据集上的小目标检测改进实验的所有训练脚本、配置、结果和文档。

## 目录结构

```
train_drone/
├── README.md                           # 本文件
├── 当前创新优先级_2026-10-16.md        # 当前实验路线与优先级（执行入口）
├── SCI投稿与后续研究计划.md             # 论文投稿计划与研究规划
├── audit_complexity.py                  # 复杂度审计脚本（P0 诊断）
├── docs/                               # 技术文档与分析
│   └── YOLO小目标改进点记忆文档.md      # 历史改进点与技术记忆
├── ablation_table_paper_style.md       # 消融实验表（论文格式）
├── ablation_summary_p5spf.md           # P5SPFPrune 消融汇总
├── 下一步实验计划_TCA后.md             # TCA 实验后续计划（历史）
├── yolov26_base/                       # Baseline 实验
├── yolov26+P2/                         # 添加 P2 检测层
├── yolov26+P2+FGDDown/                 # +FGDDown 下采样
├── yolov26+P2+FGDDown+DySample/        # +DySample 上采样
├── yolov26+P2+FGDDown+DySample+TGPFuse3/               # +TGPFuse 融合
├── yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune/    # +P5 语义融合与剪枝
├── yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3/  # +TCA 注意力
├── yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/     # Tile960 切片训练
└── validate_tile960_best.py            # Tile960 验证脚本
```

## 快速开始

### 1. 优先阅读文档（按顺序）

1. **[当前创新优先级_2026-10-16.md](当前创新优先级_2026-10-16.md)** — 当前执行路线与 STP-SCA 设计
2. **[SCI投稿与后续研究计划.md](SCI投稿与后续研究计划.md)** — 论文投稿计划与期刊选择
3. **[docs/YOLO小目标改进点记忆文档.md](docs/YOLO小目标改进点记忆文档.md)** — 历史改进点与技术细节

### 2. 运行诊断脚本（P0/P1，在 AutoDL 上）

**P0: 复杂度审计**（修正 nc 口径）

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
python train_drone/audit_complexity.py
```

输出到 `train_drone/audit_complexity/`：`summary.csv` + `per_layer.csv` + `report.md`

详细说明见 [audit_complexity_README.md](audit_complexity_README.md)

**P1-1: 门控审计**（γ、调制量、prior-GT 对齐度）

```bash
python train_drone/audit_gates.py --weight train_drone/xxx/weights/best.pt
```

输出到 `train_drone/audit_gates/`

**P1-2: o2o/o2m 对照**（max_det 300/500/1000）

```bash
python train_drone/compare_o2o_o2m.py --weight train_drone/xxx/weights/best.pt
```

输出到 `train_drone/audit_o2o_o2m/`

**P1-3: Tile960 数据统计**（切片数、重复次数、步数）

```bash
python train_drone/audit_tile960_data.py
```

输出到 `train_drone/audit_tile960_data/`

### 3. 查看已有实验结果

每个实验目录下包含：
- `*_train.py` — 训练脚本
- `*.yaml` — 模型配置
- `*_data.yaml` — 数据集配置
- `args.yaml` — 训练超参数
- `results.csv` — 训练日志
- `weights/best.pt` — 最优权重

查看某个实验的最优结果：

```bash
# 查看最后一行（最优 epoch）
tail -1 yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune/results.csv
```

## 当前状态（2026-10-16）

### 已完成实验

| 实验 | mAP50 | mAP50-95 | Params | 说明 |
|---|---:|---:|---:|---|
| baseline | 0.41202 | 0.24407 | 2.572M | YOLOv26n 基线 |
| +P2 | 0.41804 | 0.24977 | ? | 添加 P2 检测层 |
| +FGDDown | 0.42714 | 0.25567 | ? | 双路径下采样 |
| +DySample | 0.43210 | 0.25982 | ? | 学习式上采样 |
| +TGPFuse3 | 0.43843 | 0.26402 | ? | 小目标先验融合 |
| +P5SPFPrune | 0.43777 | 0.26391 | 2.029M | P5 语义融合与剪枝（-21%） |
| +TCA_P3 | 0.43982 | 0.26568 | 2.037M | P3 注意力 |
| Tile960 | 0.44490 | 0.27655 | ? | 切片训练（ep117 中断） |

**注意**：上表参数量/FLOPs 因 nc 口径不一致（P2/P2+FGDDown 是 nc:80，后续 nc:10）需重算。

### 待做诊断（P0/P1）

- [ ] **P0**: 从 23 个 best.pt 统一重算参数量与 FLOPs（`audit_complexity.py`）
- [ ] **P1-1**: 门控审计（γ、调制量、prior-GT 对齐度）
- [ ] **P1-2**: o2o/o2m 对照（max_det 300/500/1000）
- [ ] **P1-3**: Tile960 数据统计（切片数、重复次数、步数对比）
- [ ] **P1-4**: 导出冒烟（DySample/grid_sample 支持、Pi 5 延迟）

### 待做训练（P2/P3）

- [ ] **P2**: 补 Tile960 同结构对照（TGPFuse3+TCA_P3 非切片、等步数）
- [ ] **P3**: STP-SCA 快速筛选（A0/A1/A4/A5，各 100ep）

## 代码模块

新增模块位于 [`../ultralytics/nn/modules/conv.py`](../ultralytics/nn/modules/conv.py)：

- `FGDDown` — 双路径下采样（PixelUnshuffle + Pool）
- `DySample` — 学习式上采样（grid_sample）
- `TGPFuse` — 小目标先验引导融合（残差门控 γ=0.1）
- `P5SPF` — P5 语义保持融合
- `TCA` — 微小卷积注意力
- `LFGDDown`、`GFGDDown`、`DCLDown`、`WDCLDown` — 其他下采样变体

## 数据集

**VisDrone-YOLO** 格式（10 类，忽略 `ignored regions` 与 `others`）：
- 训练集：6471 图
- 验证集：548 图
- 测试集：1610 图（dev）

类别：`pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`

## 训练环境

- **本地开发**：Windows 11，F 盘 yolo26 目录，Git 管理
- **训练平台**：AutoDL RTX 4090，Ubuntu，`/root/autodl-tmp/yolo26_drone_package/`
- **训练参数**：batch=8, imgsz=960, epochs=150-200, amp=True, patience=50-100

## 引用

如果使用本实验代码或改进模块，请引用：

```bibtex
@misc{yolo26-uav-small-object,
  author = {Your Name},
  title = {Towards More Effective Small Object Detection for UAV Images},
  year = {2026},
  note = {Under review}
}
```

## 联系

- 项目负责人：[Your Name]
- 导师：[Advisor Name]
- 机构：[Your University]

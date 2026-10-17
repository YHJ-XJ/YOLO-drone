# YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune + TCA_P3 结果评估

## 训练状态

本次训练提前停止在第 164 轮。虽然没有自然跑到完整 epoch，但最后 10 轮已经基本进入平台期，因此提前停止是合理的。

## CSV 峰值结果

| 指标 | 最优值 | 对应 epoch |
|---|---:|---:|
| Precision | 0.56070 | 148 |
| Recall | 0.45111 | 128 |
| mAP50 | 0.43982 | 124 |
| mAP50-95 | 0.26568 | 116 |

最后一轮结果：

| epoch | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 164 | 0.55187 | 0.43971 | 0.43692 | 0.26429 |

## best.pt 本地复验结果

使用 `weights/best.pt` 在本地 VisDrone-YOLO 验证集重新验证：

| 指标 | 数值 |
|---|---:|
| Precision | 0.54648 |
| Recall | 0.44095 |
| mAP50 | 0.43927 |
| mAP50-95 | 0.26568 |

逐类 mAP50 / mAP50-95：

| 类别 | mAP50 | mAP50-95 |
|---|---:|---:|
| pedestrian | 0.519 | 0.250 |
| people | 0.402 | 0.164 |
| bicycle | 0.195 | 0.0928 |
| car | 0.833 | 0.588 |
| van | 0.470 | 0.335 |
| truck | 0.383 | 0.264 |
| tricycle | 0.302 | 0.176 |
| awning-tricycle | 0.176 | 0.111 |
| bus | 0.589 | 0.429 |
| motor | 0.524 | 0.248 |

## 与前序实验对比

| 模型 | best mAP50 | best mAP50-95 | Params/M |
|---|---:|---:|---:|
| YOLO26 baseline | 0.41202 | 0.24407 | 2.572 |
| P2 + FGDDown + DySample + TGPFuse3 | 0.43843 | 0.26402 | 2.542 |
| P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune | 0.43770 | 0.26391 | 2.029 |
| P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune + TCA_P3 | 0.43982 | 0.26568 | 2.037 |

## 结论

TCA_P3 是正向改进，但提升幅度偏小。它的价值主要在于：几乎不增加参数量的情况下，把 P5SPFPrune 的精度补回来并略微超过 TGPFuse3，同时保持约 2.04M 参数量。

这条路线可以作为当前最优轻量化候选，但还不足以单独支撑“明显大幅提升”的主创新。后续应优先做 Neck 结构级创新，而不是继续堆叠普通注意力。

## 建议

1. 保留本实验作为有效消融项。
2. 下一步优先尝试 `TCA_P3P4`：在 P3 和 P5SPF 后的 P4 两个关键融合特征处加入轻量卷积注意力。
3. 必须补一个 `P5Prune_NoSPF` 消融：直接删除 P5 检测头但不加 P5SPF，用来证明 P5SPF 不是简单删头。
4. 暂时不要继续使用 CS-SlideLoss，该路线已经证明不稳定且会降低整体 mAP。

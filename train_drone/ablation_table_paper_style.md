# 表：提出的算法模块在 VisDrone 验证集上的消融实验结果

> 注：当前 Ultralytics 标准验证结果没有直接输出 `AP_s`、`AP_xs`，因此本表先采用 `mAP50` 和 `mAP50-95` 作为主要精度指标。`FLOPs/B` 采用 `imgsz=640` 下的 THOP 统一统计口径，单位为 GFLOPs。

> **2026-10-16 审计警告：下表 Params/FLOPs 暂不可作为论文正式消融结论。** 前两项 P2 YAML 为 `nc:80`，DySample 版本为 `nc:10`，两者差值混入检测头类别数变化；baseline 亦源于 80 类预训练配置。同为 10 类的后半段行可保留为探索性相对比较，但仍须从已训练 `best.pt` 按同一未融合/融合/最终后端口径重算，确认 THOP 对 DySample `grid_sample` 的计数范围。表中精度历史记录不因此自动失效，但训练预算也未完全对齐。统一类别数后，P2 相对基线的百分比不能预判会增大或减小。

## Markdown 论文表

| Baseline | P2 | FGDDown | DySample | TGPFuse | P5SPF | mAP50 / % | mAP50-95 / % | Params / M | FLOPs / B |
|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| √ |  |  |  |  |  | 41.20 | 24.41 | 2.572 | 6.119 |
| √ | √ |  |  |  |  | 41.80 | 24.98 | 2.662 | 9.530 |
| √ | √ | √ |  |  |  | 42.71 | 25.57 | 2.665 | 9.565 |
| √ | √ | √ | √ |  |  | 43.21 | 25.98 | 2.535 | 7.615 |
| √ | √ | √ | √ | √ |  | **43.84** | **26.40** | 2.542 | 7.752 |
| √ | √ | √ | √ | √ | √ | 43.77 | 26.39 | **2.029** | 7.777 |

## LaTeX 三线表版本

```latex
\begin{table}[htbp]
\centering
\caption{提出的算法模块在 VisDrone 验证集上的消融实验结果}
\label{tab:ablation_modules}
\begin{tabular}{ccccccrrrr}
\toprule
Baseline & P2 & FGDDown & DySample & TGPFuse & P5SPF
& mAP$_{50}$/\% & mAP$_{50:95}$/\% & Params/M & FLOPs/B \\
\midrule
\checkmark &  &  &  &  &  & 41.20 & 24.41 & 2.572 & 6.119 \\
\checkmark & \checkmark &  &  &  &  & 41.80 & 24.98 & 2.662 & 9.530 \\
\checkmark & \checkmark & \checkmark &  &  &  & 42.71 & 25.57 & 2.665 & 9.565 \\
\checkmark & \checkmark & \checkmark & \checkmark &  &  & 43.21 & 25.98 & 2.535 & 7.615 \\
\checkmark & \checkmark & \checkmark & \checkmark & \checkmark &  & \textbf{43.84} & \textbf{26.40} & 2.542 & 7.752 \\
\checkmark & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & 43.77 & 26.39 & \textbf{2.029} & 7.777 \\
\bottomrule
\end{tabular}
\end{table}
```

## 结果解读

1. 引入 P2 检测层后，mAP50-95 从 24.41% 提升到 24.98%，说明高分辨率检测层对无人机小目标有效，但计算量明显增加。
2. 加入 FGDDown 后，mAP50-95 提升到 25.57%，说明细粒度下采样能缓解小目标信息在下采样过程中的损失。
3. 加入 DySample 后，历史记录的 mAP50-95 为 25.98%；当前参数量和 FLOPs 差值混入 `nc` 配置变化，不能归因于 DySample，需重算同口径复杂度。
4. 加入 TGPFuse 后，模型达到最高精度，mAP50 为 43.84%，mAP50-95 为 26.40%。
5. 加入 P5SPF 后，模型由四检测头改为三检测头；历史同预算未对齐的精度差值很小。参数量下降幅度及实际部署收益须经 10 类权重和最终后端重算后确定。

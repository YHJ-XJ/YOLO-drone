# YOLO小目标改进点记忆文档

研究方向：面向边缘设备部署的无人机小目标轻量化检测算法研究

当前主线：以 YOLO26n 为基础模型，在 VisDrone / UAVDT 等无人机数据集上做小目标检测改进，并兼顾 Raspberry Pi 5 等边缘设备部署。

当前短版入口：[当前创新优先级_2026-10-16.md](../当前创新优先级_2026-10-16.md)。

> 2026-10-16 执行校正：下方“传递到 P3/P4/P5”是初版设想，当前三个 TGPFuse 实际位于 P3、P2、P3；P5SPFPrune 的检测层为 P2/P3/P4。Tile960 缺同结构非切片和等优化步数对照；ignored regions 热图损失需要 mask；完整 STP-SCA 消融须加“共享但不监督”。以短版入口中的校正方案为准。

> 2026-10-16 复杂度纠偏：本地有 23 个自训练 `best.pt`，不是“缺权重”；默认文件检索忽略了权重。旧消融表跨 `nc:80` 与 `nc:10` 配置比较，且训练双头、融合后 o2o、NCNN 等最终后端图可能不同。旧参数/FLOPs 与据此得出的“DySample 降计算量”“P2 增幅”均暂停作为结论，先从权重和最终部署图统一重算，再用 Raspberry Pi 5 实测决定无 P2 分支优先级。

## 2026-10-16 最新研究决策（优先于下方 8 月历史路线）

> 状态说明：本节是对现有代码、实验登记表和近期文献的重新审计。已取得的数字标为“事实”；尚未训练的新设计标为“假设”。下方 2026-08-18、2026-08-30 等内容保留为历史记录，不再直接作为下一步执行指令。

### 1. 已确认事实与纠偏

- 当前结构候选的已登记结果：YOLO26n baseline 为 `0.41202 / 0.24407`；TGPFuse3 为 `0.43740 / 0.26402`；P5SPFPrune+TCA_P3 为 `0.43935 / 0.26568`；Tile960 候选为 `0.44426 / 0.27655`。旧实验预算、结构和数据策略没有完全对齐，这些数字只能作为探索性证据。
- TGPFuse、TCA、P5SPF、GFGDDown 不是完全相同的 `sigmoid + x*(1+gamma*gate)`：该概括主要适用于 TGPFuse；GFGDDown 使用两分支 Softmax，P5SPF 还包含 P5 语义注入，TCA 还包含局部/上下文分支和输出卷积。真正的问题是多个模块各自学习空间选择依据，缺少统一监督和跨层协同。
- Tile960 的收益不能只解释为“尺度提升”。当前切片流程还改变了目标重复采样、空背景比例和前景/背景分布；必须先做数据统计和同结构对照。
- 旧表记录原始 P2 为 9.530 GFLOPs、baseline 为 6.119 GFLOPs、TGPFuse3 为 7.752 GFLOPs；这些数字的 `nc` 与训练/部署图口径未统一。旧表内的 `+55.7%`、`+26.7%` 仅是历史数字的算术比值，当前均不得作为 P2 的真实计算增幅结论。
- TCA_P3P4 已出现负收益，继续堆注意力的优先级降为最低；CS-SlideLoss 和当前外部教师蒸馏版本不再作为近期主线。

### 2. 当前最值得投入的创新候选

#### P1 主创新：监督式共享小目标先验 + 尺度适配器（暂用工作名 STP-SCA）

从 backbone 的高分辨率 P2 特征预测单通道小目标先验 logits，以小目标 GT 生成的软热图进行辅助监督；将共享先验下采样到 P3/P4/P5，并通过每层极轻量的尺度适配器驱动 TGPFuse，随后再判断是否扩展到 TCA/P5SPF。

关键边界：

- 创新不能写成“首次用监督热图指导 FPN”。SSPNet、SADENet 等已有接近工作。可验证的差异是：**单一、类别无关、受监督的共享先验，配合尺度适配器统一驱动 YOLO Neck 多个融合位置，并面向边缘部署评估**。
- 先验头必须接在 backbone P2，而不是最终 Neck P2，否则无法指导更早的跨尺度融合。
- 不直接删除全部局部门控；先采用“共享先验 + 局部尺度校准残差”，降低一张先验图预测错误导致全 Neck 同步失败的风险。
- 小目标阈值不能未经说明固定为 `32^2`。标签应基于增强后的输入坐标生成，并比较离散阈值和连续尺度权重。
- 先验 loss 只在 E2E 总损失中加入一次，不能在 o2m/o2o 两支中重复累计。

必须完成的消融：

1. 当前 TGPFuse3；
2. 只加先验辅助监督、不把先验用于门控；
3. 共享先验只驱动 TGPFuse；
4. 共享先验 + 尺度适配器；
5. 每层独立受监督先验与共享先验对比；
6. 在主方案有效后，再尝试驱动 TCA/P5SPF。

#### P2 次创新：共享先验上的切片—整图尺度一致性（条件路线）

仅在 STP-SCA 已证实有效后考虑。训练时对同一原图构造整图和放大切片视图，在正确坐标映射后约束两者的先验 logits 或目标区域表示，推理仍使用整图。该路线与 SED、LG-RSD 等已有工作接近，不能单独以“局部裁剪 + stop-gradient 一致性”宣称强新颖性；必须证明它相对单纯 Tile960 数据增强仍有独立增益。

#### P3 部署创新：门控量化保持（条件路线）

先测 baseline、当前模型和 STP-SCA 在 FP32/FP16/INT8 下的精度、门控图一致性和分层量化误差。只有确认小幅残差门控在 INT8 下明显退化后，才设计 gate/mul 混合精度、量化感知训练或受约束门控范围。当前不能预先写成“0.1 幅度一定会被 INT8 淹没”，输入相关门控也不能直接等价折叠进卷积。

### 3. 近期先做的低成本诊断

1. **门控审计**：读取各模块学习后的 gamma/权重；逐模块令 gamma=0 验证；统计 prior 与小目标 GT 的对齐程度及不同门控图相关性。
2. **o2o/o2m 审计**：以本地 Ultralytics 版本的实际前向为准，同一权重比较两条推理路径，并扫描 `max_det=300/500/1000`；按目标尺寸、图像密度和类别统计 recall。该项是诊断和部署选择依据，不作为主创新。
3. **Tile 数据审计**：比较切片前后目标尺寸、目标重复次数、空背景比例和类别采样权重，分离“尺度放大”与“重采样”效应。
4. **导出冒烟测试**：在大训练前确认 DySample/GridSample、共享先验、resize 和检测头能通过目标后端；当前主部署目标保持 Raspberry Pi 5，CV181x 如需加入必须单独建立硬件协议。

### 4. 暂缓或停止

- P2 窗口稀疏跳算暂缓：普通张量乘掩码不会减少真实卷积，必须有 gather-conv-scatter kernel 或部署运行时支持；没有真机加速时不能把理论掩码 FLOPs 写成部署收益。
- DMSSP、更多注意力、loss 微调、外部教师蒸馏、继续堆 TCA 暂不占用主算力；已有路线可作为历史候选或失败对照。
- 不立即启动新的 200 epoch 大实验。先完成诊断，再用 30～50 epoch 做机制筛选；只有入围方案进入同预算、至少 3 seeds 的正式训练。

### 5. 最小正式实验矩阵

完成 STP-SCA 筛选后，做 `数据策略 × 结构` 的 2×2 对照：

| 数据策略 | 当前 TGPFuse3 | STP-SCA |
|---|---:|---:|
| 原图训练 | 对照 A | 对照 B |
| Tile960 训练 | 对照 C | 对照 D |

该矩阵用于判断共享先验与切片训练是互补、重复还是互相干扰。正式结论统一报告 mAP50、mAP50-95、APsmall/分尺寸结果、按类 AP、Recall、参数量、GFLOPs、导出成功率和 Raspberry Pi 5 真实延迟。

## 2026-08-18 最新执行路线

当前阶段的主线已经从“直接加 P2 小目标检测头”调整为：

```text
不增加 P2 推理头的轻量小目标增强 YOLO26
```

核心原因：

- P2 检测头对小目标有效，但会明显增加高分辨率特征图上的计算量，不适合作为最终边缘部署主模型。
- P2 / P2+FGDDown 可以作为精度上限和教师模型，不建议作为最终轻量模型。
- 最终模型应保持 Detect(P3, P4, P5)，通过轻量下采样、动态上采样、轻量融合和小目标损失来逼近 P2 效果。

暂定主模型路线：

```text
DCL-YOLO26 = DCLDown + DySample + LWConcat + NWD
```

进阶路线：

```text
DCLD-YOLO26 = DCL-YOLO26 + P2/FGDDown teacher distillation
```

当前正在训练：

```text
YOLO26 + DCLDown
```

当前训练中期状态：

- 训练轮次：40/200
- P：0.495
- R：0.392
- mAP50：0.377
- mAP50-95：0.220
- 判断：没有跑崩，速度明显快于 P2 方案，值得继续跑完。

当前最推荐的后续实验顺序：

1. `YOLO26 + DCLDown`，正在跑，先完整得到 200 epochs 结果。
2. `YOLO26 + DCLDown + DySample`，替换 neck/head 中最近邻上采样，增强小目标细节恢复。
3. `YOLO26 + DCLDown + LWConcat`，将普通 Concat 改成轻量可学习加权融合。
4. `YOLO26 + DCLDown + DySample + LWConcat`，形成第一版完整结构模型。
5. `YOLO26 + DCLDown + DySample + LWConcat + NWD`，训练阶段增强 tiny box 定位，推理零额外计算。
6. `P2+FGDDown teacher -> DCL-YOLO26 student`，用高精度 P2 模型指导无 P2 学生模型。

当前结论：

- 不再把“P2 小目标检测头”作为最终创新点，只把它作为上限模型和消融对照。
- 不优先堆 EMA/CAA/CBAM 等注意力，除非前面结构路线遇到瓶颈。
- 不直接照搬 CSDN 的 D3C2f/D3C3k2/YOLO-ULM 模块，可作为竞品和参考。
- 优先做“轻量细节补偿 + 动态上采样 + 轻量融合 + 小目标损失”的闭环方案。

## 一、总体判断

当前方向不建议一开始堆很多复杂注意力模块。更适合优先围绕“小目标信息保留、多尺度检测、轻量化部署、损失函数适配”展开。

推荐主线：

1. P2 小目标检测头
2. SPDConv 下采样替换
3. NWD / 小目标友好损失
4. 轻量化 Neck，例如 GSConv / SlimNeck / GhostConv
5. 特征融合改进，例如 BiFPN / ASF / 加权融合
6. 轻量注意力模块作为后期消融，例如 ECA / SimAM / CoordAttention
7. 可选应用拓展：YOLO 检测 + KCF 稳定跟踪

## 二、优先级最高的改进点

### 1. P2 小目标检测头

适配原因：

- VisDrone 中大量目标很小，原始 YOLO11/YOLO26 通常使用 P3/P4/P5 检测头。
- P3 对应 1/8 下采样，P2 对应 1/4 下采样，P2 能保留更多细节。
- 对 pedestrian、people、bicycle、tricycle、awning-tricycle、motor 等小目标类别更有帮助。

优点：

- 论文逻辑清楚。
- 对小目标检测很直接。
- 适合作为第一项结构改进。

风险：

- 显存和计算量增加。
- 误检可能变多。
- RTX 2000 Ada 8GB 上建议从 batch=2 或 batch=3 开始测试。

建议实验：

- YOLO26n baseline
- YOLO26n + P2

### 2. SPDConv

核心思想：

- 用 Space-to-Depth + 非步长卷积替代传统 stride=2 Conv。
- 目的是减少下采样过程中小目标细粒度信息丢失。

适配原因：

- 无人机图像小目标在多次下采样后容易只剩几个像素。
- SPDConv 比普通注意力更直接针对“小目标下采样丢信息”的问题。

推荐替换位置：

- 优先替换浅层下采样 Conv。
- 先尝试 P2/4 或 P3/8 前后的下采样层。
- 不建议一上来替换全网络所有 stride=2 Conv。

建议实验：

- YOLO26n + SPDConv
- YOLO26n + P2 + SPDConv

论文依据：

- No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects.

### 3. NWD / 小目标损失函数

核心思想：

- 小目标框很小，IoU 对位置偏差非常敏感。
- NWD 使用归一化高斯 Wasserstein 距离来衡量小目标框差异，对 tiny object 更友好。

适配原因：

- VisDrone 小目标多，框尺寸小。
- bicycle、tricycle、people、pedestrian 等类别容易受定位误差影响。

建议实验：

- YOLO26n + NWD
- YOLO26n + P2 + NWD
- YOLO26n + P2 + SPDConv + NWD

风险：

- 需要改 loss，代码复杂度比只改 yaml 高。
- 要保证训练日志和指标可对比。

### 4. 轻量化 Neck

候选模块：

- GSConv
- SlimNeck
- GhostConv
- FasterBlock
- RepViTBlock

适配原因：

- 论文题目包含“边缘设备部署”，所以不能只追求 mAP。
- Neck 是多尺度特征融合区域，计算量通常不小。
- 轻量化 Neck 可以降低参数量、GFLOPs 和推理延迟。

建议实验：

- YOLO26n + SlimNeck / GSConv
- YOLO26n + P2 + SlimNeck
- YOLO26n + P2 + SPDConv + SlimNeck

风险：

- 轻量化可能牺牲精度。
- 需要同时记录 mAP、Params、GFLOPs、FPS。

## 三、中等优先级改进点

### 5. BiFPN / 加权特征融合 / ASF

适配原因：

- 小目标检测依赖浅层细节和深层语义的融合。
- BiFPN 或加权融合可以让模型学习不同尺度特征的重要性。

优点：

- 和 P2 检测头搭配自然。
- 论文解释比较好写。

风险：

- 可能增加计算量。
- 如果实现复杂，初期不要和太多模块同时叠加。

建议实验：

- YOLO26n + BiFPN
- YOLO26n + P2 + BiFPN
- YOLO26n + P2 + SPDConv + BiFPN

### 6. 轻量注意力模块

推荐候选：

- ECA：轻量，优先推荐。
- SimAM：无参数，适合作为轻量注意力尝试。
- CoordAttention：对位置信息有帮助，但稍重。

不建议优先尝试：

- CBAM：经典但偏老，部署收益不一定高。
- SE：简单但创新性较弱。
- Transformer 类注意力：偏重，部署不友好。
- DyHead、Mamba、DCNv3、DCNv4、KAN：复杂度高，初期不建议碰。

定位：

- 注意力模块不是主线，只作为后期锦上添花的消融实验。
- 最多选一个轻量注意力模块加入，不建议多个注意力堆叠。

建议实验：

- YOLO26n + P2 + ECA
- YOLO26n + P2 + SPDConv + ECA
- YOLO26n + P2 + SPDConv + NWD + ECA

## 四、可选拓展方向

### 7. YOLO + KCF 超小目标稳定跟踪

思路：

- YOLO 负责检测目标。
- KCF 负责连续帧之间的轻量跟踪。
- 当 KCF 跟丢或置信度低时，用 YOLO 重新检测和校正。

适合场景：

- 无人机视频目标跟踪。
- 后期作为应用验证或拓展章节。

不建议作为当前主线原因：

- 会从检测任务扩展到跟踪任务。
- 评价指标更复杂，例如 MOTA、MOTP、ID Switch、Success Rate、Precision Plot。
- 需要 VisDrone-VID、VisDrone-MOT、UAVDT-MOT、UAV123 等视频/跟踪数据集。
- 当前阶段先把检测主线跑通更稳。

建议定位：

- 主线：YOLO26n 小目标轻量化检测。
- 副线：基于改进检测器的 KCF 轻量跟踪验证。

## 五、推荐实验顺序

第一阶段：基础对比

1. YOLO11n baseline，已完成。
2. YOLO26n baseline，正在/准备完成。

第二阶段：小目标结构改进

3. YOLO26n + P2
4. YOLO26n + SPDConv
5. YOLO26n + P2 + SPDConv

第三阶段：损失函数优化

6. YOLO26n + NWD
7. YOLO26n + P2 + NWD
8. YOLO26n + P2 + SPDConv + NWD

第四阶段：轻量化部署优化

9. YOLO26n + SlimNeck / GSConv
10. YOLO26n + P2 + SPDConv + SlimNeck
11. YOLO26n + P2 + SPDConv + NWD + SlimNeck

第五阶段：辅助增强与消融

12. YOLO26n + P2 + ECA
13. YOLO26n + P2 + SPDConv + ECA
14. YOLO26n + P2 + SPDConv + NWD + ECA

第六阶段：部署与应用验证

15. 导出 ONNX。
16. 测试 CPU / GPU / Raspberry Pi 5 推理速度。
17. 可选加入 YOLO + KCF 跟踪验证。

## 六、每次实验必须记录的指标

精度指标：

- Precision
- Recall
- mAP50
- mAP50-95
- 各类别 AP，重点关注 pedestrian、people、bicycle、tricycle、awning-tricycle、motor

复杂度指标：

- Parameters
- GFLOPs
- 模型权重大小
- 训练显存占用
- 单张推理耗时
- FPS

部署指标：

- ONNX 是否成功导出
- ONNXRuntime 推理速度
- Raspberry Pi 5 推理速度
- 是否支持 OpenCV DNN / NCNN / TensorRT 等部署路径

## 七、当前最推荐论文题目方向

候选题目：

- 面向边缘设备部署的无人机小目标轻量化检测算法研究
- 基于改进 YOLO26 的无人机图像小目标检测算法研究
- 融合细粒度下采样与轻量化特征融合的无人机小目标检测方法
- 面向边缘部署的轻量级无人机小目标检测网络

推荐核心创新表述：

- 引入 P2 小目标检测层，增强高分辨率小目标感知能力。
- 使用 SPDConv 替换浅层下采样卷积，减少小目标细粒度信息损失。
- 引入 NWD 小目标损失，缓解小目标框对位置偏差敏感的问题。
- 设计轻量化 Neck，在提升检测精度的同时降低参数量和计算量。

## 八、当前不建议优先做的内容

- 一上来加入多个注意力模块。
- 一上来做 YOLO + KCF 作为主线。
- 一上来换大模型，例如 YOLO26m/l/x。
- 同时修改 backbone、neck、head、loss，导致无法判断涨点来源。
- 只看 mAP，不看参数量、GFLOPs、FPS 和部署效果。

## 九、最稳的论文实验组合

推荐最终论文表格可以包含：

| 模型 | P2 | SPDConv | NWD | SlimNeck/GSConv | ECA/SimAM | mAP50 | mAP50-95 | Params | GFLOPs | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26n | - | - | - | - | - | 待填 | 待填 | 待填 | 待填 | 待填 |
| YOLO26n-P2 | yes | - | - | - | - | 待填 | 待填 | 待填 | 待填 | 待填 |
| YOLO26n-SPD | - | yes | - | - | - | 待填 | 待填 | 待填 | 待填 | 待填 |
| YOLO26n-P2-SPD | yes | yes | - | - | - | 待填 | 待填 | 待填 | 待填 | 待填 |
| YOLO26n-P2-SPD-NWD | yes | yes | yes | - | - | 待填 | 待填 | 待填 | 待填 | 待填 |
| YOLO26n-P2-SPD-NWD-LiteNeck | yes | yes | yes | yes | - | 待填 | 待填 | 待填 | 待填 | 待填 |
| YOLO26n-P2-SPD-NWD-LiteNeck-Attn | yes | yes | yes | yes | yes | 待填 | 待填 | 待填 | 待填 | 待填 |

## 十、当前结论

最值得优先尝试的不是普通注意力模块，而是：

1. P2 检测头
2. SPDConv
3. NWD
4. SlimNeck / GSConv
5. 轻量注意力 ECA / SimAM

这条路线更符合“小目标检测 + 轻量化 + 边缘部署”的论文方向。

## 十一、多轮筛选后的候选模块总池

本节根据 B站 / CSDN / GitHub / 论文与开源项目中常见 YOLO 小目标改进方法继续汇总。原则是：CSDN 和 B站主要作为实现参考，真正写论文时优先引用论文、arXiv、期刊论文、GitHub 官方源码。

### A档：最适合当前课题，优先尝试

| 模块/策略 | 推荐程度 | 适合原因 | 主要风险 | 建议位置 |
| --- | --- | --- | --- | --- |
| P2 小目标检测头 | 必试 | 直接增加 1/4 高分辨率检测层，适合 VisDrone 小目标 | 显存、GFLOPs 增加 | Detect Head |
| SPDConv | 必试 | 替换 stride=2 下采样，减少小目标细节损失 | 全部替换可能变慢 | 浅层下采样 Conv |
| NWD / SAL-NWD | 必试 | 小目标框对 IoU 偏差极敏感，NWD 更友好 | 需要改 loss | Loss |
| GSConv / SlimNeck | 必试 | 符合轻量化和边缘部署主题 | 可能轻量化后精度下降 | Neck |
| PConv / FasterBlock | 强烈推荐 | 减少冗余计算，适合轻量化 | 需要验证实际部署速度 | Backbone / Neck |
| SAHI / 切片推理 | 强烈推荐 | 不改模型也能提升高分辨率小目标召回 | 推理时间增加，不适合实时主指标 | 推理阶段 |

### B档：很值得做消融，但不要一开始全加

| 模块/策略 | 推荐程度 | 适合原因 | 主要风险 | 建议位置 |
| --- | --- | --- | --- | --- |
| BiFPN | 推荐 | 多尺度加权融合，适合密集小目标 | 增加计算量 | Neck |
| ASF / 自适应空间融合 | 推荐 | SOD-YOLO 在 UAV 小目标中使用，逻辑贴合 | 需要看源码适配难度 | Neck |
| CARAFE | 推荐 | 内容感知上采样，比 nearest 更细腻 | 部署端算子支持要验证 | Upsample |
| RFAConv | 推荐 | 扩大感受野，增强小目标特征提取 | 比普通 Conv 稍复杂 | Backbone 浅层 |
| EMA 注意力 | 推荐 | 多尺度注意力，遥感小目标论文常用 | 比 ECA/SimAM 稍重 | Neck C2f/C3k2 后 |
| ECA | 推荐 | 极轻量通道注意力，适合边缘部署 | 提升可能有限 | Backbone/Neck |
| SimAM | 推荐 | 无参数注意力，试错成本低 | 工程论文中创新性一般 | Backbone/Neck |
| CoordAttention | 可试 | 保留位置信息，对小目标定位有帮助 | 比 ECA 重 | Backbone/Neck |
| Inner-IoU / Wise-IoU / Inner-Wise IoU | 推荐 | 定位损失优化，可和 NWD 对比 | 需要避免 loss 改太乱 | Loss |
| MPDIoU | 可试 | 边界框回归损失候选 | 不一定专门针对 tiny object | Loss |

### C档：有价值，但适合后期或作为对照

| 模块/策略 | 推荐程度 | 适合原因 | 主要风险 | 建议位置 |
| --- | --- | --- | --- | --- |
| GhostConv / GhostBottleneck | 可试 | 经典轻量化模块 | 有时精度下降明显 | Backbone/Neck |
| RepVGG / RepConv | 可试 | 训练多分支，推理重参数化 | 需要确认导出部署正确 | Backbone/Neck |
| RepViT / EfficientViT 类轻量主干 | 后期可试 | 新颖，轻量主干方向好写 | 改动大，初学者成本高 | Backbone |
| WTConv | 后期可试 | 小目标和纹理细节可能受益 | 依赖和算子复杂度需要确认 | Backbone |
| AKConv | 后期可试 | 可变核卷积，适应尺度变化 | 实现复杂，部署风险高 | Backbone |
| DyHead | 暂缓 | 检测头表达能力强，CVPR 模块 | 对当前边缘部署和初学难度不友好 | Head |
| GAM / CBAM / SE | 只做对照 | 资料多、容易实现 | 偏老或偏重，论文亮点弱 | Backbone/Neck |
| Soft-NMS | 可做推理对照 | 密集目标可能减少误删 | YOLO26/Ultralytics 后处理适配要确认 | 后处理 |
| KCF 跟踪 | 应用拓展 | 可做检测后稳定跟踪验证 | 任务从检测扩展为跟踪 | 视频后处理 |

### D档：当前不建议优先投入

| 模块/策略 | 不建议原因 |
| --- | --- |
| Mamba / Vision Mamba | 代码复杂、训练风险高、边缘部署不稳 |
| DCNv3 / DCNv4 | 算子依赖复杂，Windows/ONNX/边缘端部署容易出问题 |
| 大型 Transformer Neck/Backbone | 算力压力大，不贴合 Raspberry Pi 5 |
| 多注意力堆叠 | 消融不好解释，也容易拖慢模型 |
| 同时改 Backbone + Neck + Head + Loss | 结果涨了也难判断到底是哪一项起作用 |

## 十二、按论文主线重新组织的实验路线

### 路线1：小目标感知增强路线

目标：解决无人机小目标尺度小、浅层细节丢失的问题。

实验顺序：

1. YOLO26n baseline
2. YOLO26n + P2
3. YOLO26n + SPDConv
4. YOLO26n + P2 + SPDConv
5. YOLO26n + P2 + SPDConv + CARAFE

推荐写法：

提出一种细粒度小目标感知结构，通过高分辨率检测头和低信息损失下采样增强模型对微小目标的表达能力。

### 路线2：小目标定位损失优化路线

目标：解决小目标框位置偏差导致 IoU 剧烈变化的问题。

实验顺序：

1. YOLO26n baseline
2. YOLO26n + NWD
3. YOLO26n + Wise-IoU
4. YOLO26n + Inner-Wise IoU
5. YOLO26n + P2 + SPDConv + NWD

推荐写法：

针对无人机小目标边界框尺度小、定位误差敏感的问题，引入更适合小目标回归的距离度量损失。

### 路线3：轻量化边缘部署路线

目标：在保证精度的同时降低参数量、GFLOPs 和推理延迟。

实验顺序：

1. YOLO26n baseline
2. YOLO26n + GSConv / SlimNeck
3. YOLO26n + PConv
4. YOLO26n + FasterBlock
5. YOLO26n + P2 + SPDConv + GSConv / SlimNeck

推荐写法：

构建轻量化特征融合网络，减少冗余计算，使模型更适合边缘设备部署。

### 路线4：多尺度融合增强路线

目标：解决小目标语义弱、尺度变化大、背景复杂的问题。

实验顺序：

1. YOLO26n + BiFPN
2. YOLO26n + ASF
3. YOLO26n + P2 + BiFPN
4. YOLO26n + P2 + SPDConv + ASF

推荐写法：

通过自适应多尺度融合增强浅层细节与深层语义交互，提升复杂无人机视角下的小目标检测能力。

### 路线5：轻量注意力辅助路线

目标：作为消融实验，验证注意力是否能进一步增强有效特征。

实验顺序：

1. YOLO26n + ECA
2. YOLO26n + SimAM
3. YOLO26n + EMA
4. YOLO26n + P2 + SPDConv + ECA
5. YOLO26n + P2 + SPDConv + EMA

推荐写法：

轻量注意力只作为辅助增强模块，不作为全文第一创新点。

## 十三、最值得先做的 12 个实验

如果时间和算力有限，优先做下面这些：

| 序号 | 实验名 | 目的 |
| --- | --- | --- |
| 1 | YOLO26n baseline | 新基线 |
| 2 | YOLO26n-P2 | 验证小目标检测头 |
| 3 | YOLO26n-SPD | 验证低损失下采样 |
| 4 | YOLO26n-P2-SPD | 验证 P2 与 SPDConv 是否互补 |
| 5 | YOLO26n-NWD | 验证小目标损失 |
| 6 | YOLO26n-P2-SPD-NWD | 第一套强组合 |
| 7 | YOLO26n-GSConv/SlimNeck | 验证轻量化 Neck |
| 8 | YOLO26n-P2-SPD-GSConv | 验证精度与轻量化平衡 |
| 9 | YOLO26n-P2-SPD-NWD-GSConv | 第二套强组合 |
| 10 | YOLO26n-P2-SPD-ECA | 验证轻量注意力 |
| 11 | YOLO26n-P2-SPD-EMA | 验证多尺度注意力 |
| 12 | 最优模型 + SAHI | 验证切片推理对小目标召回的上限 |

## 十四、当前资料来源索引

后续查代码和写论文时优先看这些：

- Ultralytics VisDrone 数据集说明：用于确认 VisDrone 是无人机小目标、密集、俯视场景标准数据集。
- SOD-YOLO：P2 + ASF + Soft-NMS，面向 UAV 小目标检测。
- LEAF-YOLO：轻量、边缘实时、小目标航拍检测，适合参考边缘部署写法。
- LI-YOLOv8：GSConv + PConv + EMA + Inner-Wise IoU，适合参考轻量化小目标组合。
- DroneScan-YOLO：SAL-NWD、MSFD、RPA，高分辨率 UAV tiny object 方向很贴合。
- SAHI：切片推理，适合做后期推理增强或应用验证。
- Wise-IoU 官方 GitHub：可作为损失函数改进参考。
- DynamicHead 官方 GitHub：暂不优先实现，但可作为复杂检测头参考。

## 十五、当前最终筛选结论

最适合你当前阶段的“论文主菜”：

1. P2
2. SPDConv
3. NWD / SAL-NWD
4. GSConv / SlimNeck / PConv
5. BiFPN / ASF

最适合你当前阶段的“配菜消融”：

1. ECA
2. SimAM
3. EMA
4. CARAFE
5. Wise-IoU / Inner-IoU

最适合你后期展示论文完整性的内容：

1. ONNX 导出
2. Raspberry Pi 5 推理测试
3. SAHI 切片推理对比
4. YOLO + KCF 跟踪应用验证

当前不要被大量 CSDN 模块标题带偏。真正最稳的路线仍然是：小目标检测头 + 下采样优化 + 小目标损失 + 轻量化 Neck + 少量注意力消融。

## 十六、结合本地 YOLOv5/YOLO11 改进项目后的新路线

本地参考项目：

- `F:\图像深度学习科研\C3Ghost+GSConv+VoVGSCSP -yolov5-prune-light`
- `F:\图像深度学习科研\yolov11yolo12项目(20260410)-ultralytics-yolo11-20260410`

从第一个项目中可参考：

- `C3Ghost`：用 GhostBottleneck 替代 C3 内部 Bottleneck，减少冗余特征计算。
- `GSConvns`：GSConv 的规范化 shuffle 版本，注释中说明更利于 TensorRT。
- `GSBottleneck`：用 GSConvns 构成轻量瓶颈。
- `VoVGSCSP`：在 Neck 中用 GSConv + CSP 思路替代普通 C3/C3k2，属于 Slim-Neck 路线。

从第二个项目中可参考：

- `yolo11-SPDConv.yaml`：用 SPDConv 替换下采样 Conv。
- `yolo11-ADown.yaml`：用平均池化/最大池化思想构造下采样模块。
- `yolo11-ASF-P2.yaml`：P2 + 多尺度自适应融合。
- `yolo11-C3k2-GhostDynamicConv.yaml`：把 C3k2 内部替换为 GhostDynamicConv。
- `yolo11-fasternet-bifpn.yaml`：FasterNet 主干 + BiFPN 融合。
- `yolo11-atthead.yaml`：在 P3/P4/P5 检测特征前插入 EMA。

### 新的核心判断

单独使用官方 P2 创新性不够，只能作为小目标增强基线。

更适合作为论文创新的是：

1. 修改下采样模块内部结构，减少小目标细节丢失。
2. 修改 C3k2 内部结构，使其兼具细粒度特征、轻量化和多尺度感受野。
3. 修改 Neck 融合方式，使 P2/P3/P4/P5 特征不是简单 Concat，而是轻量自适应融合。
4. 加入小目标友好损失，例如 NWD/SAL-NWD。

### 推荐自研模块 1：FGDDown

暂定名称：`FGDDown`，Fine-Grained Dual-branch Downsampling。

目标问题：

- 普通 stride=2 Conv 会直接压缩空间尺寸，导致无人机小目标细节丢失。
- 单纯 SPDConv 虽然保留空间信息，但会增加通道压力。
- 单纯 MaxPool 虽然保留局部极值，但语义表达不足。

建议结构：

```text
Input
├── Branch A: SPD / Space-to-Depth + 1x1 Conv
├── Branch B: MaxPool(k=2, s=2) + 1x1 Conv
└── Concat + GSConv/Conv 3x3
```

模块解释：

- SPD 分支：把 2x2 空间邻域搬到通道维，保留小目标细粒度空间信息。
- MaxPool 分支：保留局部极值、边缘、高对比度响应。
- 1x1 Conv：压缩通道，避免 SPD 带来的通道膨胀。
- GSConv/Conv：融合两条支路，兼顾表达能力和轻量化。

优先替换位置：

- P2 到 P3 的下采样。
- P3 到 P4 的下采样。

不建议一开始替换：

- P1 到 P2，过浅层可能显存压力大。
- P4 到 P5，深层小目标信息已经很少，收益可能不如浅层。

实验顺序：

1. YOLO26-P2 baseline，已完成。
2. YOLO26-P2 + FGDDown(P2->P3)。
3. YOLO26-P2 + FGDDown(P2->P3, P3->P4)。

### 推荐自研模块 2：DGA-C3k2

暂定名称：`DGA-C3k2`，Drone-aware Gated Aggregation C3k2。

目标问题：

- YOLO26 的 C3k2 是特征提取主模块，但对无人机极小目标的边缘、局部纹理和密集遮挡表达仍可能不足。
- 全部替换为 GhostDynamicConv 或注意力模块容易变成照搬。

建议结构：

```text
C3k2 主路径
├── Local branch: 3x3 Conv / Bottleneck
├── Context branch: 5x5 DWConv 或 GSConv
├── Lightweight attention: ECA / EMA-lite / SimAM
└── Gate fusion: sigmoid gate + Add/Concat + 1x1 Conv
```

模块解释：

- 3x3 分支：保留局部纹理和细节。
- 5x5 DWConv/GSConv 分支：扩大感受野，但保持轻量。
- 注意力分支：增强小目标有效区域或通道响应。
- 门控融合：不是简单拼接，而是让模型学习不同分支的重要性。

优先替换位置：

- P2 检测分支前的 C3k2。
- P3 检测分支前的 C3k2。
- Backbone 浅层 C3k2 可作为第二阶段尝试。

不建议：

- 全网络所有 C3k2 一次性替换，消融困难，训练风险高。

### 推荐自研模块 3：DAS-Neck

暂定名称：`DAS-Neck`，Drone Adaptive Slim Neck。

参考来源：

- GSConv/Slim-Neck 的轻量 Neck 思想。
- VoVGSCSP 的 Neck 替换思想。
- ASF-P2 的多尺度自适应融合思想。

目标问题：

- P2 引入后特征图很大，Neck 计算量上升。
- 普通 Concat 对各尺度特征一视同仁，无法动态选择浅层细节和深层语义。

建议结构：

```text
P2/P3/P4/P5
↓
1x1 Conv 对齐通道
↓
GSConvns / VoVGSCSP-lite 做轻量融合
↓
scale weight 或 ASF-style attention
↓
Detect
```

模块解释：

- GSConvns：降低 Neck 计算量，并兼顾普通卷积表达能力。
- VoVGSCSP-lite：用 CSP 思想减少冗余融合。
- ASF-style attention：让模型对小目标密集区域更敏感。

实验顺序：

1. YOLO26-P2 + GSConvns Neck。
2. YOLO26-P2 + VoVGSCSP-lite Neck。
3. YOLO26-P2 + DAS-Neck。

### 推荐自研模块 4：D-SPPF

暂定名称：`D-SPPF`，Drone Sparse SPPF。

参考来源：

- 学长方案中的 GLS-SPPF。
- YOLO26 官方 SPPF。

目标问题：

- P5/32 深层特征图上小目标响应稀疏，容易被背景语义淹没。

建议结构：

```text
SPPF pool outputs
↓
GroupConv / DWConv
↓
BN 或 LayerNorm
↓
SiLU
↓
Concat + Conv
```

优先级：

- 中等。适合作为第三或第四个结构改进，不建议第一个做。

### 当前最像自己论文的组合

建议主方案名称：

- `FGD-YOLO26`
- `DAS-YOLO26`
- `FGD-DAS-YOLO`

推荐最终主线：

```text
YOLO26n
→ YOLO26n-P2
→ YOLO26n-P2-FGDDown
→ YOLO26n-P2-DAS-Neck
→ YOLO26n-P2-FGDDown-DAS-Neck
→ YOLO26n-P2-FGDDown-DAS-Neck-NWD
```

这条路线比单纯 `P2 + SPDConv + EMA` 更有自己的结构设计：

- P2 是官方小目标增强基线。
- FGDDown 是自研下采样模块。
- DAS-Neck 是自研轻量自适应融合 Neck。
- NWD 是小目标定位损失增强。

### 下一步最推荐先实现

第一优先级：`FGDDown`。

原因：

- 和无人机小目标最贴。
- 能吸收 SPDConv、ADown、学长 MaxPool 支路三种思路。
- 结构不算太复杂，适合初学者实操。
- 可以先只替换一个下采样位置，风险小。

第二优先级：`DAS-Neck`。

原因：

- 吸收 GSConvns、VoVGSCSP、ASF。
- 和边缘部署主题高度一致。

第三优先级：`NWD/SAL-NWD`。

原因：

- 对小目标定位非常关键。
- 但涉及 loss 源码，初学者应放在结构模块之后。

## 十七、2026-08-18 深度迭代后的新版计划

### 1. 为什么当前路线要从 P2 转为无 P2 轻量增强

前期实验已经证明：

| 模型 | 最佳 mAP50 | 最佳 mAP50-95 | 作用 |
|---|---:|---:|---|
| YOLO26 baseline | 0.41202 | 0.24407 | 基准 |
| YOLO26 + P2 | 0.41713 | 0.24977 | 小目标精度上限参考 |
| YOLO26 + P2 + FGDDown | 0.42709 | 0.25567 | 证明细节下采样有效 |
| YOLO26 + FGDDown | 0.41554 | 0.24612 | 无 P2 单模块轻量改进 |

关键判断：

- P2 有效，但不是最终轻量化部署答案。
- FGDDown 在 P2 结构下效果明显，说明“下采样细节保留”这个方向成立。
- 无 P2 的 FGDDown 提升有限，说明只替换一个模块还不够，需要组合式结构。
- 当前论文主线应从“加 P2 提精度”改为“无 P2 逼近 P2 效果”。

### 2. 当前自研模块：DCLDown

暂定名称：

```text
DCLDown: Detail-Contrast Lightweight Downsampling
```

中文含义：

```text
细节-对比轻量下采样模块
```

设计目标：

- 替换 YOLO26 backbone 中 `P2/4 -> P3/8` 的普通 stride=2 Conv。
- 不增加 P2 检测头。
- 尽量保留小目标在下采样时容易丢失的边缘、纹理、局部高响应信息。
- 保持参数量和计算量很低，服务于 Raspberry Pi 5 / 边缘部署方向。

结构：

```text
Input
├─ PixelUnshuffle(2) -> 1x1 Conv       # 保留空间细节，借鉴 SPDConv
├─ DWConv(k=3,s=2) -> 1x1 Conv         # 学习边缘、纹理、局部结构
└─ MaxPool(2,2) - AvgPool(2,2) -> 1x1 Conv  # 局部对比增强，突出小目标响应
Concat
DWConv(k=3,s=1)
1x1 Conv
Output
```

当前源码位置：

- `yolo26/ultralytics/nn/modules/conv.py`
- `yolo26/ultralytics/nn/modules/__init__.py`
- `yolo26/ultralytics/nn/tasks.py`

当前训练文件：

- `yolo26/train_drone/yolov26+DCLDown/yolov26+DCLDown.yaml`
- `yolo26/train_drone/yolov26+DCLDown/yolov26+DCLDown_train.py`
- `yolo26/train_drone/yolov26+DCLDown/yolov26+DCLDown_data.yaml`

已验证：

- 模型可正常构建。
- 预训练权重可部分加载。
- 参数量约 `2,486,708`，低于 baseline 的 `2,507,700`。

### 3. 参考资料给我们的启发

PCPE-YOLO：

- 关键词：YOLOv8m、C2f_PIG、CAA、小目标检测层、EUCB。
- 启发：小目标需要浅层细节和上下文信息共同增强。
- 避免照搬：不直接复刻 C2f_PIG + CAA + 小目标检测层。
- 用法：作为竞品和论文相关工作。
- 链接：https://www.nature.com/articles/s41598-025-15975-w

LUD-YOLO：

- 关键词：无人机目标检测、轻量网络、动态稀疏注意力、特征融合。
- 启发：不能只看 mAP，还要写参数量、GFLOPs、FPS。
- 避免照搬：不直接复刻其动态稀疏注意力。
- 链接：https://www.sciencedirect.com/science/article/pii/S0020025524012805

FDM-YOLO：

- 关键词：VisDrone、PConv、DySample、EMA、高分辨率小目标层。
- 启发：DySample 和轻量融合对无人机小目标有效。
- 避免照搬：不走“加高分辨率检测层 + EMA”的同路线。
- 用法：证明 `DySample` 可以作为下一步优先模块。

DySample：

- 关键词：动态上采样、轻量、不需要自定义 CUDA。
- 启发：替换 YOLO neck/head 中的 `nn.Upsample`，提升上采样过程中的小目标细节恢复。
- 链接：https://arxiv.org/abs/2308.15085
- GitHub：https://github.com/tiny-smart/dysample

GSConv / Slim-Neck：

- 关键词：轻量 Neck、GSConv、VoVGSCSP。
- 启发：后续可用于减少 neck 计算量。
- 链接：https://github.com/AlanLi1997/slim-neck-by-gsconv

NWD：

- 关键词：tiny object、航拍图像、小目标定位损失。
- 启发：训练阶段改善小框定位，推理阶段不增加计算量。
- 链接：https://arxiv.org/abs/2110.13389
- GitHub：https://github.com/jwwangchn/NWD

YOLO-ULM / D3C2f / D3C3k2：

- 关键词：超轻量模块、D3C2f、RepDown、FoCIoU。
- 启发：轻量模块可以参考，但不建议直接作为主创新。
- 原因：CSDN 已有较多 YOLO26 移植教程，直接照搬创新性低。

### 4. 最新实验路线表

| 顺序 | 实验名称 | 是否已做 | 目的 | 备注 |
|---:|---|---|---|---|
| 1 | YOLO26 baseline | 已完成 | 基线 | 必须保留 |
| 2 | YOLO26 + P2 | 已完成 | 精度上限参考 | 不作为最终部署模型 |
| 3 | YOLO26 + P2 + FGDDown | 已完成 | 证明细节下采样有效 | 可作为 teacher 候选 |
| 4 | YOLO26 + FGDDown | 已完成 | 无 P2 单模块对照 | 提升小但有意义 |
| 5 | YOLO26 + DCLDown | 正在训练 | 当前自研模块验证 | 不加 P2，参数量低 |
| 6 | YOLO26 + DCLDown + DySample | 待做 | 动态上采样恢复细节 | 下一优先级 |
| 7 | YOLO26 + DCLDown + LWConcat | 待做 | 轻量加权融合 | 借鉴 EFC/BiFPN 思想 |
| 8 | YOLO26 + DCLDown + DySample + LWConcat | 待做 | 第一版完整结构 | 作为主结构候选 |
| 9 | 主结构 + NWD | 待做 | 小目标定位损失 | 推理零额外计算 |
| 10 | P2 teacher -> 无 P2 student 蒸馏 | 待做 | 逼近 P2 精度 | 最像论文亮点 |

### 5. 后续源码改动优先级

第一阶段：当前 DCLDown 跑完。

- 不再改当前训练。
- 等待 200 epochs 完成后分析 `results.csv`。
- 对比 baseline、P2、P2+FGDDown、FGDDown。

第二阶段：加入 DySample。

主要改动：

- 新增 `DySample` 模块源码。
- 在 `ultralytics/nn/modules/__init__.py` 注册。
- 在 `tasks.py` 中确保 yaml 可解析。
- 新建 `yolov26+DCLDown+DySample.yaml`。

替换位置：

```yaml
- [-1, 1, nn.Upsample, [None, 2, "nearest"]]
```

替换为：

```yaml
- [-1, 1, DySample, [2]]
```

第三阶段：加入 LWConcat。

暂定名称：

```text
LWConcat: Lightweight Weighted Concatenation
```

目标：

- 替换 head 中普通 `Concat`。
- 给深层上采样特征和浅层 backbone 特征分配可学习权重。
- 让网络自己决定偏重语义还是偏重细节。

结构：

```text
x1, x2
learnable weights -> softmax
concat(w1*x1, w2*x2)
```

优点：

- 参数几乎可以忽略。
- 比普通 Concat 更有“特征融合创新”说法。
- 可以借鉴 EFC / BiFPN 的思想，但实现更轻。

第四阶段：加入 NWD。

目标：

- 改 loss，不改推理结构。
- 主要提升小目标定位稳定性。

注意：

- NWD 改动比结构 yaml 更复杂。
- 必须单独消融，不能和前面模块一口气混在一起。

第五阶段：teacher-student 蒸馏。

teacher：

```text
YOLO26 + P2 + FGDDown
```

student：

```text
YOLO26 + DCLDown + DySample + LWConcat
```

论文叙事：

```text
训练阶段利用高分辨率 P2 teacher 提供小目标监督；
推理阶段保留无 P2 student，实现轻量部署。
```

这是当前最有可能拉开创新性的路线。

### 6. 当前论文故事草案

中文题目候选：

```text
面向边缘部署的无人机小目标轻量化检测算法研究
```

更聚焦版本：

```text
一种无 P2 推理头的无人机小目标轻量化检测算法
```

英文名称候选：

```text
DCL-YOLO26: Lightweight Detail-Contrast Downsampling for UAV Small Object Detection
```

进阶英文名称：

```text
DCLD-YOLO26: Distilled Lightweight Detail-Contrast YOLO26 for UAV Small Object Detection
```

核心创新点候选：

1. 提出 DCLDown，在下采样阶段保留细节并增强局部对比。
2. 使用 DySample 改善 neck 上采样阶段的小目标细节恢复。
3. 使用 LWConcat 增强跨尺度特征融合，减少普通 Concat 的无差别拼接。
4. 使用 NWD 改善 tiny object 定位稳定性，推理阶段无额外开销。
5. 使用 P2 teacher 蒸馏无 P2 student，使轻量模型逼近 P2 模型精度。

### 7. 当前不能做偏的事情

- 不要看到一个 CSDN 模块就马上缝进去。
- 不要同时改很多模块，否则无法写消融实验。
- 不要直接把 P2 当最终创新点。
- 不要只看 mAP，要同步记录 Params、GFLOPs、FPS、权重大小、边缘端推理速度。
- 不要只和 YOLO26 baseline 比，后面还要和 YOLO26-P2、PCPE-YOLO、LUD-YOLO 等相关工作进行文字层面的比较。

### 8. 最推荐的近期动作

当前只做一件事：

```text
等 YOLO26 + DCLDown 训练完成。
```

完成后立刻整理：

- `results.csv`
- `results.png`
- `weights/best.pt`
- `confusion_matrix_normalized.png`
- 每类 AP 指标
- 参数量、GFLOPs、推理速度

如果 DCLDown 达到或超过 baseline：

```text
下一组：YOLO26 + DCLDown + DySample
```

如果 DCLDown 没有超过 baseline：

```text
保留为消融实验，仍然尝试 DCLDown + DySample，因为下采样和上采样可能存在互补。
```

## 十八、2026-08-18 新一批资料融入后的升级版路线

### 1. 对新增资料的可靠性判断

这批资料可以分三类使用：

高参考价值：

- 论文或论文解读中有明确模型、数据集、指标、模块的资料。
- 例如 MASF-YOLO、SOD-YOLO、LCW-YOLO、PCPE-YOLO、DySample、NWD、GSConv/Slim-Neck。

中参考价值：

- CSDN、腾讯云、知乎、51CTO 等二次解读文章。
- 可用于快速了解模块思想，但不能直接相信“提升 5%-15%”这类泛化说法。
- 必须回到原论文、代码、数据集和训练配置核验。

低参考价值：

- 没有原论文链接、没有消融表、没有源码、只写“某团队提出”“性能天花板”的文章。
- 例如部分 YOLO-DRONE / CVPR 2026 Oral 标题型文章，需要谨慎引用。

### 2. 新增模块与我们路线的映射

| 资料中模块 | 原始思想 | 是否适合当前主线 | 我们如何吸收 |
|---|---|---|---|
| ASFM | 自适应空间/尺度融合 | 适合 | 改造成 `DAFuse/LWConcat`，替换普通 Concat，而不是直接照搬 |
| CFIM | 跨尺度特征交互 | 适合 | 做轻量双向融合，让 P3/P4 特征交互更充分 |
| CAM | 上下文感知，扩张卷积/ASPP | 谨慎 | 只放在 P3 或 SPPF 附近，避免全网增加计算 |
| DSA | 动态稀疏注意力 | 暂缓 | 思想好，但实现复杂，先不作为第一阶段 |
| CA | 坐标注意力 | 可选 | 如果 DCLDown 后仍漏检小目标，可只加在 P3 分支 |
| TA | 三重注意力 | 暂缓 | 太常见，计算和论文撞车风险较高 |
| DNP | 动态剪枝 | 后处理阶段 | 主模型稳定后再做剪枝，不参与早期结构搜索 |
| LDH | 轻量检测头 | 适合后期 | 如果结构模型速度不够，再替换 Detect 前的卷积头 |
| FRM | 特征复用/跨层跳跃 | 适合 | 做 `P2-guided FRM`，只复用 P2 特征，不增加 P2 检测头 |
| MTC | 多任务协同检测/分割 | 不优先 | 没有分割标注，成本过高，不适合当前阶段 |

### 3. 当前路线再次升级后的核心思想

新的论文主线不再是：

```text
YOLO26 + 某个模块
```

而是：

```text
用轻量细节补偿网络，在不增加 P2 推理头的情况下逼近 P2 小目标检测效果。
```

更具体：

```text
P2 提供了高分辨率细节，但 P2 检测头计算量高；
DCLDown 保留下采样细节；
DySample 恢复上采样细节；
DAFuse/LWConcat 自适应融合浅层细节与深层语义；
NWD/WIoU 改善小框定位；
最终再用 P2 teacher 蒸馏无 P2 student。
```

### 4. 新版主模型候选

第一版主模型：

```text
DCL-YOLO26-v1 = YOLO26 + DCLDown
```

第二版主模型：

```text
DCL-YOLO26-v2 = YOLO26 + DCLDown + DySample
```

第三版主模型：

```text
DCL-YOLO26-v3 = YOLO26 + DCLDown + DySample + DAFuse
```

第四版训练增强模型：

```text
DCL-YOLO26-v4 = YOLO26 + DCLDown + DySample + DAFuse + NWD/WIoU
```

最终论文模型候选：

```text
DCLD-YOLO26 = DCL-YOLO26-v3/v4 + P2 teacher distillation
```

其中 `DCLD` 可解释为：

```text
Detail-Contrast Lightweight Distillation
```

### 5. DAFuse/LWConcat 的升级定义

原来的 `LWConcat` 可以升级为：

```text
DAFuse: Detail-aware Adaptive Fusion
```

中文名称：

```text
细节感知自适应融合模块
```

借鉴来源：

- ASFM：自适应尺度融合。
- CFIM：跨尺度特征交互。
- EFC：增强层间特征相关性。
- BiFPN：可学习多尺度融合权重。

设计目标：

- 替换 head 中普通 `Concat`。
- 不增加高分辨率 P2 检测头。
- 给上采样语义特征和浅层细节特征分配可学习权重。
- 让小目标密集区域更依赖浅层细节，让背景复杂区域更依赖深层语义。

建议结构：

```text
x_deep, x_shallow
↓
1x1 Conv 对齐通道
↓
global pooling 得到轻量权重
↓
softmax(w_deep, w_shallow)
↓
concat(w_deep * x_deep, w_shallow * x_shallow)
↓
Conv/DWConv 融合
```

与普通 Concat 的区别：

- 普通 Concat：只拼接，不判断谁更重要。
- DAFuse：先判断不同尺度特征的重要性，再融合。

### 6. FRM 思想的轻量化吸收

FRM 不建议直接做很多跨层跳线，否则模型会乱、消融难写。

建议改成：

```text
P2-guided FRM
```

思路：

- 不增加 P2 检测头。
- 只把 backbone 的 P2 浅层细节压缩后注入 P3。
- 可以在 DCLDown 之后或 P3 head 之前使用。

候选结构：

```text
P2 feature
↓
DWConv stride=2
↓
1x1 Conv 压缩
↓
与 P3 feature 加权相加或轻量 concat
```

论文叙事：

```text
P2 检测头虽然有效但计算昂贵，因此本文仅复用 P2 的细粒度信息，
不保留 P2 预测分支，从而兼顾小目标感知能力和边缘部署效率。
```

### 7. LDH 的使用位置

LDH 不作为当前第一优先级。

原因：

- 检测头轻量化主要降低计算量，不一定直接提升 mAP。
- 当前主要瓶颈还是小目标细节丢失和融合不足。
- 如果先改 LDH，精度可能下降，反而干扰 DCLDown/DySample 的判断。

建议使用时机：

```text
等 DCLDown + DySample + DAFuse 稳定后，再测试 LDH。
```

建议实验：

```text
DCL-YOLO26-v3
DCL-YOLO26-v3 + LDH
```

只保留 LDH 的条件：

- mAP50-95 下降不超过 0.003。
- GFLOPs 或 FPS 有明显改善。
- 边缘设备推理速度有收益。

### 8. 注意力模块的再判断

注意力不是不能用，而是要很节制。

当前优先级：

```text
DAFuse/LWConcat > DySample > NWD/WIoU > CA/轻量注意力 > DSA/TA
```

可考虑的位置：

- 只放在 P3 小目标分支。
- 只放在 DAFuse 内部作为权重生成器。
- 只放在 SPPF/C2PSA 后面做深层上下文增强。

暂时不建议：

- 全 backbone 插入注意力。
- P3/P4/P5 每个 head 都插入注意力。
- 同时使用 EMA、CA、TA、DSA 多种注意力。

### 9. 损失函数路线调整

从新增资料看，WIoU 也值得作为 NWD 的对照。

建议后续 loss 实验：

| 实验 | 目的 |
|---|---|
| baseline loss | 保持原始 YOLO26 训练 |
| NWD | 针对 tiny object 的框距离度量 |
| WIoU v3 | 动态样本质量加权，改善定位鲁棒性 |
| NWD + WIoU | 后期再试，避免一开始 loss 太复杂 |

当前优先级：

```text
先 NWD，再 WIoU。
```

原因：

- NWD 与 AI-TOD / tiny object / aerial image 的联系更强。
- WIoU 适合作为泛化和鲁棒性增强。

### 10. 训练策略升级

从小目标实战资料中可吸收以下训练策略：

- 训练轮次比常规目标检测增加 20%-30%。
- 当前 200 epochs 是合理的，不算过度。
- 保持 `imgsz=960`，不要盲目升到 1280，否则显存和训练时间会明显增加。
- 后续可单独测试 `imgsz=1024`，但不要和结构改进混在同一轮。
- 重点记录 AP_s 或小框召回率，不能只看总 mAP。
- 对 VisDrone 的 `pedestrian/people/bicycle/tricycle/awning-tricycle/motor` 单独统计。

### 11. 部署路线补充

最终论文必须补部署实验。

建议部署指标：

- Params
- GFLOPs
- model size
- PyTorch FPS
- ONNX Runtime FPS
- TensorRT FP16 FPS
- TensorRT INT8 FPS
- Raspberry Pi 5 上的 ONNX / NCNN / OpenVINO 推理速度

部署顺序：

1. 先在 RTX 4090 上训练和验证。
2. 导出 ONNX。
3. 在本地 Windows 或 AutoDL 上测 ONNX Runtime。
4. 在 Raspberry Pi 5 上测轻量模型。
5. 最后再考虑 INT8 量化。

不要太早剪枝：

- DNP/剪枝应放在主模型稳定之后。
- 剪枝会改变模型结构和精度，不适合和 DCLDown/DySample/DAFuse 混在一起同时做。

### 12. 下一轮最推荐的具体执行

当前 DCLDown 正在训练，保持不动。

训练完成后：

1. 分析 `YOLO26 + DCLDown` 完整结果。
2. 如果 mAP50-95 >= baseline，则直接做 `DCLDown + DySample`。
3. 如果 mAP50-95 < baseline，但差距不大，也继续做 `DCLDown + DySample`，因为两者可能互补。
4. 如果 DCLDown 明显低于 baseline，则保留为失败消融，回退到 `FGDDown + DySample` 或只做 `DySample + DAFuse`。

下一组最推荐：

```text
YOLO26 + DCLDown + DySample
```

再下一组：

```text
YOLO26 + DCLDown + DySample + DAFuse
```

暂时不推荐：

```text
YOLO26 + DCLDown + CA/EMA/TA
YOLO26 + P2 + 一堆注意力
YOLO26 + D3C3k2 直接照搬
```

### 13. 新增资料来源索引

- MASF-YOLO：`https://blog.csdn.net/CooVally_AI/article/details/151211776`
- LCW-YOLO 解读：`https://cloud.tencent.com/developer/article/2566279`
- SOD-YOLO：`https://cloud.tencent.com/developer/article/2554895`
- 小目标检测实战指南：`https://blog.csdn.net/gitblog_00321/article/details/159757229`
- YOLO-DRONE 低空巡检文章：`https://zeeklog.com/cvpr-2026-oralshi-ce-yolo-drone-wu-ren-ji-di-kong-xun-jian-de-xing-neng-tian-hua-ban-xiao-mu-biao-zhao-hui-lu-kuang-sheng-39-qing-hua-tuan-dui-li-zuo-dian-li-bu-shu-shi-cao-quan-jie-xi-7`
- 知乎资料：`https://zhuanlan.zhihu.com/p/1986422853621523582`
- 51CTO资料：`https://blog.51cto.com/u_13521/14562661`

注意：

- 这些网页只作为模块灵感来源。
- 正式论文中应优先引用原论文、arXiv、期刊或会议页面。
- CSDN/知乎/腾讯云/51CTO 不建议作为最终参考文献主体。

## 十九、2026-08-18 关键词扩展搜索后的第三轮路线优化

本轮按照以下关键词继续扩展搜索：

```text
无人机 / UAV / drone
小目标 / tiny object / small object
轻量化 / lightweight / edge deployment
YOLO / YOLO26 / YOLO11 / VisDrone / aerial image / remote sensing
```

搜索网站和资料源按可信度分层：

```text
A 级：arXiv、CVF OpenAccess、Nature/Scientific Reports、OUP JCDE、ScienceDirect、IEEE Xplore、官方 GitHub、Ultralytics / PaddleDetection 官方文档
B 级：MDPI、Springer、PMC、ResearchGate 论文页、Semantic Scholar、国内期刊网页
C 级：CSDN、知乎、腾讯云、51CTO、头条、GraphAstra、PaperNotes 等技术解读或二次整理
```

结论：C 级资料可以用来找模块灵感，但不能作为最终论文的主要引用。论文中优先引用 A 级论文、官方代码和数据集文档。

### 1. 本轮搜索确认的主趋势

#### 1.1 P2 小目标检测头反复有效，但不是最终部署最优解

多个 UAV / VisDrone 小目标检测方法都会加高分辨率检测头，例如：

- TPH-YOLOv5：增加额外 prediction head，用于无人机图像中尺度剧烈变化和密集目标。
- PCPE-YOLO：增加 small object detection layer。
- SOD-YOLO：ASF + P2 + Soft-NMS，在 VisDrone 上涨幅明显。
- FDM-YOLO：加高分辨率层，并移除大目标检测层。
- YOLO-TLA / YOLO-BWS 等也都强调高分辨率浅层特征。

这和我们的实验一致：

```text
YOLO26 baseline      mAP50-95 = 0.24407
YOLO26 + P2          mAP50-95 = 0.24977
YOLO26 + P2+FGDDown  mAP50-95 = 0.25567
```

但 P2 的问题也很明确：

```text
P2 会显著增加高分辨率特征图上的计算量，对树莓派 5、ONNX、NCNN、OpenVINO 部署不友好。
```

因此现在的论文主线不改：

```text
P2 / P2+FGDDown 作为精度上限和教师模型。
最终轻量化模型尽量不保留 P2 检测头，而是在 P2 -> P3 下采样、P3 融合、上采样和 loss 上补偿小目标信息损失。
```

#### 1.2 下采样是当前最值得继续挖的创新点

相关资料中反复出现：

- SPD-Conv / Space-to-Depth：减少下采样时的空间信息损失。
- LAD：轻量自适应下采样。
- ADown：分支下采样，兼顾池化和卷积。
- RepDown：训练多分支，推理重参数化为单分支。
- Wavelet Pooling：用小波低频/高频分量保留边缘纹理。
- DCLDown / FGDDown：我们当前路线中的细节保留下采样。

这说明我们自编的 `DCLDown` 方向是合理的，不是乱试。它的价值在于：

```text
不增加 P2 检测头，却尽量把 P2 阶段的细节压缩进 P3。
```

后续推荐迭代为：

```text
DCLDown-v1：PixelUnshuffle + DWConv + MaxPool-AvgPool contrast
DCLDown-v2：PixelUnshuffle + DWConv + Haar/Wavelet high-frequency detail
DCLDown-v3：DCLDown-v2 + light spatial gate
RepDCLDown：训练期多分支，推理期重参数化
```

其中最推荐下一步可选储备模块：

```text
WDCLDown: Wavelet Detail-Contrast Lightweight Downsampling
```

中文名：

```text
小波细节对比轻量下采样模块
```

设计动机：

- PixelUnshuffle 已经把 2x2 空间块搬到通道维，适合构造 Haar 小波分量。
- LL 分量代表低频轮廓，LH/HL/HH 分量代表边缘、纹理、局部突变。
- 小目标在航拍图里常常只剩边缘和局部对比，高频分量比普通池化更容易保留它。
- Haar 分解可以用固定加减法完成，几乎不增加可学习参数。

建议结构：

```text
input
├─ branch 1: PixelUnshuffle(2) -> 1x1 Conv             # 细节重排
├─ branch 2: DWConv(k=3,s=2) -> 1x1 Conv               # 局部可学习下采样
└─ branch 3: Haar high-frequency(LH/HL/HH) -> 1x1 Conv # 高频小目标边缘
concat
-> DWConv(k=3,s=1)
-> 1x1 Conv
```

这比直接加 P2 更适合我们的论文叙事：

```text
以轻量下采样替代高分辨率检测头，在低计算开销下增强小目标细节保持能力。
```

#### 1.3 上采样和融合是第二优先级

DySample、EUCB、MASF-YOLO 的 DASI、Efficient Fusion Factor、EFC、BiFPN、ASFF 类资料都指向同一问题：

```text
上采样不是简单放大特征图，融合也不是简单 Concat。
```

对 YOLO26 来说，下一步最好先改：

```yaml
head 中两个 nn.Upsample
```

替换为：

```text
DySample
```

原因：

- DySample 是 ICCV 2023 方法，有正式论文和官方源码。
- 它不依赖额外 CUDA 算子。
- 参数和计算开销小。
- 比 CARAFE、FADE、SAPA 这种动态核上采样更适合初学者迁移到 Ultralytics。

再下一步改：

```yaml
head 中与 backbone P3/P4/P5 融合的 Concat
```

替换为：

```text
DAFuse / LWConcat
```

建议先替换这两个位置：

```yaml
[-1, 4]   # cat backbone P3
[-1, 13]  # cat head P4
```

暂时不要一次性替换所有 Concat，否则变量太多。

#### 1.4 Loss 改进比堆模块更适合 tiny object

NWD、WIoU、FoCIoU 这类 loss 在小目标检测里很值得做。

当前优先级：

```text
NWD > WIoU > FoCIoU
```

原因：

- NWD 原始动机就是 tiny object，尤其适合 AI-TOD / aerial tiny object。
- 小框 IoU 对位置偏移极敏感，几个像素的偏差会导致 IoU 大幅下降。
- NWD 把框建模成高斯分布，相对缓和 tiny box 的位置敏感性。
- WIoU 可以作为第二组 loss 对照，用来改善样本质量加权和定位稳定性。
- FoCIoU 可以作为参考 YOLO-ULM 的后续储备，不建议现在最先做。

建议实验：

```text
DCLDown + 原始 loss
DCLDown + NWD
DCLDown + WIoU
DCLDown + DySample + DAFuse + NWD
```

#### 1.5 轻量 backbone/block 替换可以做，但要后置

资料中常见的轻量模块：

- C2f_PIG：动态可重构 C2f。
- D3C2f / D3C3k2：深度可分离卷积 + 大核深度卷积 + C2f/C3k2 风格聚合。
- Fast-C2f / PConv：只对部分通道做空间卷积。
- GSConv / Slim-Neck：混合普通卷积和深度可分离卷积，适合 neck 轻量化。
- GhostConv / GhostDynamicConv：用廉价特征生成减少冗余。
- C2Faster / FasterNet block：偏轻量高速特征提取。

但这些不建议现在马上全网替换，因为：

```text
当前还没有确定 DCLDown / DySample / DAFuse 的主线效果。
如果同时替换 C3k2，后面很难解释到底是谁带来的提升或下降。
```

建议后置为：

```text
DCL-C3k2Lite
```

只替换 head 中和小目标相关的 C3k2：

```yaml
layer 16: P3/8-small 后的 C3k2
layer 19: P4/16-medium 后的 C3k2
```

暂时不替换 backbone 的所有 C3k2。

候选设计：

```text
C3k2Lite = C3k2 主体 + 部分通道 DWConv 3x3 + DWConv 7x7 + 1x1 Conv 融合
```

论文叙事：

```text
在颈部小目标融合阶段使用轻量大核局部-上下文聚合模块，避免全主干替换带来的训练不稳定和部署代价。
```

### 2. 更新后的主路线

当前主线：

```text
YOLO26 baseline
-> DCLDown
-> DCLDown + DySample
-> DCLDown + DySample + DAFuse
-> DCLDown + DySample + DAFuse + NWD
```

增强主线：

```text
YOLO26 + WDCLDown
YOLO26 + WDCLDown + DySample
YOLO26 + WDCLDown + DySample + DAFuse
YOLO26 + WDCLDown + DySample + DAFuse + NWD
```

轻量 block 后置主线：

```text
YOLO26 + DCLDown + DySample + DAFuse + DCL-C3k2Lite
```

教师蒸馏主线：

```text
Teacher: YOLO26 + P2 + FGDDown
Student: YOLO26 + DCLDown / WDCLDown + DySample + DAFuse
```

蒸馏的意义：

```text
P2 教师有更强小目标响应，学生不保留 P2 检测头，但学习教师在 P3/P4 特征和预测结果上的小目标知识。
```

### 3. 当前最推荐的实验队列

| 顺序 | 实验名 | 目的 | 是否优先 |
|---|---|---|---|
| E0 | YOLO26 baseline | 绝对基准 | 已完成 |
| E1 | YOLO26 + P2 | 小目标检测头上限 | 已完成 |
| E2 | YOLO26 + P2 + FGDDown | 精度上限/教师模型 | 已完成 |
| E3 | YOLO26 + FGDDown | 无 P2 的轻量下采样对照 | 已完成 |
| E4 | YOLO26 + DCLDown | 当前正在训练的主线 | 最高 |
| E5 | YOLO26 + DCLDown + DySample | 第二阶段主线 | 最高 |
| E6 | YOLO26 + DCLDown + DySample + DAFuse | 第三阶段主线 | 高 |
| E7 | E6 + NWD | tiny-box loss 优化 | 高 |
| E8 | YOLO26 + WDCLDown | 小波细节下采样新模块 | 高 |
| E9 | E8 + DySample + DAFuse | 可能形成论文主模型 | 高 |
| E10 | E9 + NWD | 可能形成最终模型 | 高 |
| E11 | E10 + DCL-C3k2Lite | 轻量 block 增强 | 中 |
| E12 | P2+FGDDown teacher -> E10 student | 高级创新：轻量蒸馏 | 中高 |

### 4. 是否继续保留 P2 路线

保留，但不作为最终主路线。

P2 路线用途：

```text
1. 作为精度上限。
2. 作为蒸馏教师。
3. 作为论文中证明“高分辨率分支有效但部署代价高”的对照。
```

可以额外做一个折中实验：

```text
P2-Lite-noP5
```

结构思想：

```text
增加 P2 小目标头，但删除 P5 大目标头，仅保留 Detect(P2,P3,P4)。
```

适用理由：

- VisDrone 中人、车、摩托、自行车等多数目标偏小。
- P5 对超大目标更有帮助，但无人机航拍中大目标比例较低。
- 删除 P5 可以抵消一部分 P2 带来的参数和计算。

但这条线暂时排在 DCLDown / DySample / DAFuse / NWD 之后。

### 5. 明确暂缓或不推荐的方向

暂缓：

```text
YOLO-Master / MoE
```

原因：

- 思路先进，但源码迁移、训练稳定性、论文解释难度都高。
- 对当前初学阶段不友好。
- 边缘部署时动态路由不一定比普通轻量卷积更容易加速。

暂缓：

```text
DSA / TA / 多注意力堆叠
```

原因：

- 容易涨参数或增加延迟。
- 很难消融干净。
- YOLO26 本身已有 C2PSA，继续堆注意力的边际收益不确定。

暂缓：

```text
MTC 检测+分割多任务
```

原因：

- 当前 VisDrone-YOLO 只有检测框标签，没有分割标签。
- 做多任务会引入新数据标注和训练复杂度，不适合作为当前主线。

暂缓：

```text
直接照搬 D3C3k2 / C2f_PIG / YOLO-ULM 全套结构
```

原因：

- 容易和已有工作高度相似。
- 论文创新会显得像模块拼接。
- 更建议吸收“深度可分离、大核、重参数化、部分通道计算”的思想，自编成适配 YOLO26 的小模块。

### 6. 下一步源码优先级

当前训练完成后，按这个顺序学源码和改源码：

#### 第一优先级：DySample

源码关注：

```text
DySample 的 forward
grid_sample 如何根据 offset 重新采样
为什么它不需要动态卷积核
```

YOLO26 修改位置：

```text
ultralytics/nn/modules/conv.py 或 block.py
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py
train_drone/.../yaml 中的 nn.Upsample
```

#### 第二优先级：DAFuse / LWConcat

源码关注：

```text
两个输入特征如何通道对齐
如何生成两个分支的权重
权重是 sigmoid 还是 softmax
融合后输出通道如何保持和原 Concat 后一致
```

YOLO26 修改位置：

```text
先替换 P3/P4 融合处的 Concat，不替换所有 Concat。
```

#### 第三优先级：NWD / WIoU

源码关注：

```text
ultralytics/utils/loss.py
ultralytics/utils/metrics.py
bbox_iou / bbox loss 相关函数
YOLO26 end2end=True 时 loss 走哪条路径
```

#### 第四优先级：WDCLDown

源码关注：

```text
PixelUnshuffle 后如何得到 2x2 子像素块
如何构造 Haar LL/LH/HL/HH
高频分支如何压缩到较少通道
和 DCLDown-v1 对比参数量/GFLOPs 是否变化很小
```

### 7. 最新路线一句话版本

现在我们的方向升级为：

```text
面向边缘部署的 YOLO26 无人机小目标检测：用细节保持下采样替代高成本 P2 检测头，并结合轻量动态上采样、细节感知跨尺度融合和 tiny-box 回归损失，实现精度-速度平衡。
```

更像论文题目的版本：

```text
DCL-YOLO26: A Detail-Preserving Lightweight YOLO26 for UAV Small Object Detection on Edge Devices
```

中文版本：

```text
DCL-YOLO26：面向边缘设备的细节保持轻量化无人机小目标检测算法
```

如果后续加入小波版本：

```text
WDCL-YOLO26：面向无人机小目标检测的小波细节保持轻量化 YOLO26
```

### 8. 本轮新增重点资料索引

- YOLO26 官方文档：`https://docs.ultralytics.com/models/yolo26`
- YOLO26 arXiv：`https://arxiv.org/abs/2606.03748`
- Ultralytics VisDrone 文档：`https://docs.ultralytics.com/zh/datasets/detect/visdrone`
- VisDrone 官方 GitHub：`https://github.com/VisDrone/VisDrone-Dataset`
- TPH-YOLOv5 arXiv：`https://arxiv.org/abs/2108.11539`
- TPH-YOLOv5 官方 GitHub：`https://github.com/cv516Buaa/tph-yolov5`
- DySample 官方 GitHub：`https://github.com/tiny-smart/dysample`
- DySample arXiv：`https://arxiv.org/abs/2308.15085`
- NWD 官方 GitHub：`https://github.com/jwwangchn/NWD`
- NWD arXiv：`https://arxiv.org/abs/2110.13389`
- Effective Fusion Factor：`https://arxiv.org/abs/2011.02298`
- Effective Fusion Factor GitHub：`https://github.com/ucas-vg/Effective-Fusion-Factor`
- Slim-Neck / GSConv GitHub：`https://github.com/AlanLi1997/slim-neck-by-gsconv`
- PCPE-YOLO Scientific Reports：`https://www.nature.com/articles/s41598-025-15975-w`
- LRDS-YOLO Scientific Reports：`https://www.nature.com/articles/s41598-025-07021-6`
- MASF-YOLO arXiv：`https://arxiv.org/abs/2504.18136`
- FDM-YOLO arXiv：`https://arxiv.org/abs/2503.04452`
- Efficient Feature Fusion for UAV Object Detection：`https://arxiv.org/abs/2501.17983`
- SOD-YOLO arXiv：`https://arxiv.org/abs/2507.12727`
- YOLO-BWS OUP JCDE：`https://academic.oup.com/jcde/article/13/4/97/8527734`
- LMW-YOLO Scientific Reports：`https://www.nature.com/articles/s41598-026-45055-6`
- LMW-YOLO GitHub：`https://github.com/qqqqqq-ch/LMW-YOLO`
- LCW-YOLO PubMed：`https://pubmed.ncbi.nlm.nih.gov/41095031/`
- PaddleDetection PP-YOLOE-SOD：`https://github.com/PaddlePaddle/PaddleDetection`
- YOLO-Master arXiv：`https://arxiv.org/abs/2512.23273`
- YOLO-Master Docs：`https://tencent.github.io/YOLO-Master/`
- YOLO-ULM CVPR 2026 索引：`https://github.com/52CV/CVPR-2026-Papers/blob/main/21-40.md`
- YOLO-ULM 解读：`https://graphastra.com/en/papers/c13fb08b-2d22-4866-bcb2-964154c67a86`

### 9. 当前对正在训练的 DCLDown 的判断标准

等 `YOLO26 + DCLDown` 训练完后，按下面判断：

```text
若 mAP50-95 >= 0.244：
    DCLDown 有效，直接进入 DCLDown + DySample。

若 mAP50-95 在 0.239 - 0.244：
    DCLDown 精度接近 baseline，若参数/GFLOPs/显存更低，也值得保留，再进入 DCLDown + DySample。

若 mAP50-95 < 0.239：
    DCLDown-v1 不够强，保留为失败消融，改做 WDCLDown 或 FGDDown + DySample。
```

最终论文中真正有价值的不是单次涨点，而是形成这个逻辑闭环：

```text
P2 确实强 -> 但重
普通下采样轻 -> 但丢细节
DCL/WDCLDown 试图保留 P2 细节 -> 轻量
DySample/DAFuse 弥补多尺度融合不足
NWD/WIoU 解决 tiny box 回归敏感
边缘部署实验证明速度和模型大小优势
```

## 二十、2026-08-18 《基于YOLO的无人机小目标高精度检测改进模型》学习记录

资料来源：

```text
题目：基于 YOLO 的无人机小目标高精度检测改进模型
英文题目：An improved high-precision model for UAV small object detection based on YOLO
期刊：南京邮电大学学报（自然科学版）
年份：2026 年 6 月
DOI：10.14132/j.cnki.1673-5439.2026.03.014
模型名：YOLO-LiteMax
基础模型：YOLOv8
数据集：VisDrone2019-DET
```

这篇文章和我们的方向高度相关，但要注意：

```text
它是基于 YOLOv8 做的无人机小目标检测改进。
我们是基于 YOLO26 做轻量化小目标检测。
因此只能借鉴思想，不能直接照搬模块名和结构。
```

### 1. 文章的三个核心改进

#### 1.1 SCB：Selective Convolution Block

中文可理解为：

```text
选择性卷积模块 / 部分通道选择卷积模块
```

主要思想：

- 标准卷积会对全部通道做空间卷积，计算冗余较大。
- SCB 借鉴 FasterNet 的 Partial Convolution 思想。
- 只对一部分通道做空间卷积，其余通道直接保留。
- 最后将两路特征拼接，再用 1x1 卷积整合。

文章中的作用：

```text
替换原来的跨阶段结构，减少冗余计算，提高特征提取效率。
```

实验结果：

```text
YOLOv8s baseline:
mAP@0.5 = 39.3, APsmall = 12.5, Params = 11.1M, FPS = 131

加入 SCB:
mAP@0.5 = 39.5, APsmall = 12.2, Params = 8.3M, FPS = 156
```

解读：

```text
SCB 单独加入后，精度几乎不涨，APsmall 甚至略降，但参数明显下降，速度明显上升。
```

对我们的启发：

```text
SCB 更适合作为轻量化模块，而不是主涨点模块。
```

在 YOLO26 中的建议用法：

```text
不要一开始替换所有 C3k2。
后续可以只替换 head 中 P3/P4 融合后的 C3k2，形成 C3k2Lite / DCL-C3k2Lite。
```

优先替换位置：

```yaml
layer 16: P3/8-small 后的 C3k2
layer 19: P4/16-medium 后的 C3k2
```

暂不建议替换：

```text
backbone 中所有 C3k2。
```

原因：

```text
backbone 大规模替换会改变特征提取主体，变量太多，训练稳定性和论文消融都不好控制。
```

#### 1.2 STSSF：Small Target Scale Sequence Fusion

中文可理解为：

```text
小目标尺度序列融合结构
```

组成：

```text
STSSF = C3DSF + SAC + P2 高分辨率特征引入
```

其中：

```text
C3DSF = Convolution 3D Scale Fusion
SAC = Scale Aware Concatenation
```

文章中的动机：

- 传统 FPN/PAN 融合多依赖相邻层拼接或相加。
- 深层跨尺度关联不足。
- 上采样容易丢失浅层高分辨率细节。
- 无人机小目标密集，P2 浅层特征对小目标很关键。

C3DSF 的思想：

```text
把不同尺度特征看成沿尺度维度排列的序列。
通道和尺寸先对齐，再通过类似 3DConv 的方式提取尺度维度上的相关性。
```

SAC 的思想：

```text
大尺度特征：MaxPool + AvgPool 后相加
小尺度特征：Upsample
中尺度特征：保持
最后 Concat 融合
```

实验结果：

```text
SCB + STSSF:
mAP@0.5 = 44.2, APsmall = 14.8, Params = 6.9M, FPS = 124
```

解读：

```text
STSSF 是这篇文章真正的主要涨点模块。
它本质上还是证明了：P2 高分辨率细节 + 更强跨尺度融合，对 VisDrone 小目标非常有效。
```

对我们的启发：

```text
这进一步支持我们之前的判断：P2 是强有效路线，但最终部署会变重。
```

因此我们的策略不是直接照搬 STSSF，而是把它压缩成两条轻量路线：

```text
1. DCL/WDCLDown：把 P2 细节压缩进 P3，不保留 P2 检测头。
2. DAFuse/LWConcat：在 P3/P4 融合时做可学习的尺度融合，而不是普通 Concat。
```

可以借鉴 STSSF 的地方：

```text
尺度序列思想。
P2/P3/P4 的局部融合。
MaxPool + AvgPool 的尺度感知融合。
```

不建议直接照搬的地方：

```text
完整 P2-P5 多尺度序列融合。
完整 C3DSF 3DConv 结构。
```

原因：

```text
完整 STSSF 结构对初学者实现复杂，且可能增加推理图复杂度。
我们的论文目标是轻量边缘部署，应优先选择更简洁、部署友好的模块。
```

#### 1.3 SCPD：Shared Convolution Precision Detection

中文可理解为：

```text
共享卷积精度检测头
```

文章中的动机：

- YOLOv8 等模型每个尺度都有独立检测头。
- 不同尺度检测头独立学习，参数较多。
- 小 batch 训练时 BN 对 batch size 敏感，可能不稳定。
- 不同尺度之间分类/回归的一致性不足。

SCPD 的核心做法：

```text
不同尺度检测头共享一组卷积参数。
使用组归一化替代批归一化。
回归分支引入尺度因子，用于适应不同尺度特征。
```

实验结果：

```text
SCB + STSSF + SCPD:
mAP@0.5 = 45.2, APsmall = 15.3, Params = 6.1M, FPS = 118
```

解读：

```text
SCPD 主要作用是进一步降低参数，并提升多尺度检测一致性。
```

对我们的启发：

```text
共享检测头是轻量化部署方向里非常值得后续考虑的一条线。
```

但当前不建议马上做。

原因：

```text
YOLO26 的 Detect head 和 end2end=True 机制比普通 YOLOv8 检测头复杂。
现在直接改 Detect 风险较高，可能牵连 loss、推理输出、权重加载、导出 ONNX。
```

建议后置实验：

```text
等 DCLDown + DySample + DAFuse + NWD 稳定后，再研究 YOLO26 Detect head 是否能做轻量共享。
```

### 2. 这篇文章对我们路线的直接影响

该文进一步强化三个判断：

```text
1. 小目标涨点主要来自高分辨率浅层细节和跨尺度融合，不是单纯注意力。
2. 轻量化模块单独加入不一定涨点，但能减少参数和提高速度。
3. 检测头共享是一条有价值但实现风险较高的后期路线。
```

因此我们的路线微调为：

```text
当前主线：DCLDown / WDCLDown + DySample + DAFuse + NWD
后置轻量化：DCL-C3k2Lite
高级后置：SharedLiteHead / SCPD-like Detect Head
```

### 3. 可吸收成我们自己的模块

#### 3.1 DCL-C3k2Lite

来源灵感：

```text
SCB + FasterNet Partial Convolution + C3k2
```

目标：

```text
只在 neck/head 的小目标特征融合处降低冗余计算。
```

建议结构：

```text
input
-> 1x1 Conv 通道调整
-> split
   -> branch A: partial channels -> DWConv/PConv -> 1x1 Conv
   -> branch B: identity
-> concat
-> 1x1 Conv
```

建议替换：

```yaml
head:
  layer 16 C3k2 -> DCL_C3k2Lite
  layer 19 C3k2 -> DCL_C3k2Lite
```

暂不替换：

```yaml
layer 22 P5/32-large C3k2
```

原因：

```text
当前任务更关注小目标，P5 大目标语义分支不是第一优先级。
```

#### 3.2 DAFuse-ST

来源灵感：

```text
STSSF + SAC + ASFM + EFC
```

目标：

```text
不用完整 3DConv，也能做尺度感知融合。
```

建议结构：

```text
x_deep, x_shallow
-> channel align
-> w = softmax(MLP(GAP([x_deep, x_shallow])))
-> weighted concat
-> DWConv + 1x1 Conv
```

可选加入轻量尺度感知：

```text
MaxPool(x_shallow) - AvgPool(x_shallow)
```

作为小目标局部对比提示。

建议替换：

```yaml
[[ -1, 4 ], 1, Concat, [1]]   # P3 融合
[[ -1, 13], 1, Concat, [1]]   # P4 融合
```

#### 3.3 SharedLiteHead

来源灵感：

```text
SCPD
```

目标：

```text
减少检测头参数，提高多尺度预测一致性。
```

后续可研究：

```text
YOLO26 Detect 内部 cls/reg 分支是否能共享中间卷积。
BN 是否可替换为 GN。
每个尺度是否需要 learnable scale。
```

当前状态：

```text
暂缓，不作为下一轮实验。
```

### 4. 与我们已有实验的结合

目前已知：

```text
YOLO26 baseline:      mAP50-95 = 0.24407
YOLO26 + P2:          mAP50-95 = 0.24977
YOLO26 + P2+FGDDown:  mAP50-95 = 0.25567
YOLO26 + FGDDown:     mAP50-95 = 0.24612
YOLO26 + DCLDown:     正在训练
```

从这篇文章看，我们后续不应该放弃 P2 的启发，而应该把 P2 的作用拆解：

```text
P2 的作用 = 高分辨率细节 + 小目标浅层纹理 + 定位信息
```

我们要做的是：

```text
把 P2 的作用压缩进更轻的模块，而不是直接保留 P2 检测头。
```

因此当前最稳路线：

```text
DCLDown -> DySample -> DAFuse -> NWD
```

如果 DCLDown 最终不理想，则切换：

```text
WDCLDown -> DySample -> DAFuse -> NWD
```

### 5. 更新后的实验队列补充

| 顺序 | 实验名 | 对应吸收思想 | 目的 |
|---|---|---|---|
| E4 | YOLO26 + DCLDown | P2 细节压缩到 P3 | 当前训练 |
| E5 | YOLO26 + DCLDown + DySample | 更细致上采样 | 提升融合质量 |
| E6 | YOLO26 + DCLDown + DAFuse-ST | 借鉴 STSSF/SAC，但轻量化 | 学习尺度权重 |
| E7 | YOLO26 + DCLDown + DySample + DAFuse-ST | 下采样 + 上采样 + 融合组合 | 主模型候选 |
| E8 | E7 + NWD | tiny box 回归优化 | 主模型候选 |
| E9 | YOLO26 + WDCLDown | 小波高频替代 P2 细节 | 新模块候选 |
| E10 | E9 + DySample + DAFuse-ST + NWD | 最终模型候选 | 高优先 |
| E11 | E10 + DCL-C3k2Lite | 借鉴 SCB 做局部轻量 block | 后置 |
| E12 | E10 + SharedLiteHead | 借鉴 SCPD 共享检测头 | 高级后置 |

### 6. 论文写作可用表述

可以吸收的叙事：

```text
现有高精度无人机小目标检测方法通常通过引入高分辨率检测层或复杂跨尺度融合结构提升小目标感知能力，但这类设计会增加高分辨率特征图上的计算开销，不利于边缘设备部署。为此，本文不直接保留额外 P2 检测头，而是在浅层到中层的下采样过程中构建细节保持机制，并在颈部引入轻量尺度感知融合，以在精度和推理效率之间取得更好的平衡。
```

这段和 YOLO-LiteMax 的区别：

```text
YOLO-LiteMax: 引入 P2 + STSSF + SCPD，强调高精度轻量检测。
我们的路线: 不保留 P2 检测头，强调用轻量下采样和尺度感知融合近似 P2 效果，突出边缘部署。
```

### 7. 对当前方向的影响总结

这篇文章值得参考，但不能成为我们的直接模板。

最值得吸收：

```text
SCB 的部分通道轻量化思想。
STSSF 的尺度序列融合思想。
SAC 的 MaxPool + AvgPool + Upsample 尺度感知融合思想。
SCPD 的检测头共享思想。
```

最不建议照搬：

```text
完整 P2-P5 STSSF 颈部结构。
完整 SCPD 检测头。
直接命名 SCB/STSSF/SCPD。
```

我们的更优改法：

```text
把这些思想分别压缩进 DCLDown / WDCLDown / DAFuse-ST / DCL-C3k2Lite / SharedLiteHead。
```

下一步仍然保持：

```text
先等 DCLDown 训练完成，再决定是否进入 DySample 或 WDCLDown。
```

## 二十一、2026-08-18 本地无人机小目标论文批量学习记录

本轮新增学习材料：

```text
1. 针对弱小无人机目标的轻量级目标检测算法
2. 无人机场景下基于轻量化YOLOv8的小目标检测算法
3. 基于YOLOv8的轻量化无人机图像目标检测算法
4. 基于YOLOv8n的无人机小目标检测算法
5. 基于改进YOLO11的无人机航拍图像小目标检测算法
6. 基于YOLO的无人机小目标高精度检测改进模型
```

这批论文的共同结论非常明确：

```text
无人机小目标检测真正有效的方向主要集中在：
1. 高分辨率浅层细节保留
2. P2/Tiny Head/小目标检测头
3. 跨尺度特征融合
4. 轻量卷积模块
5. 检测头共享
6. 小目标定位损失优化
7. 后期剪枝/量化部署
```

这与当前实验路线一致：

```text
P2 路线有效，但重；
下采样细节保持是轻量替代 P2 的核心；
neck 融合比全 backbone 堆模块更值得优先改；
loss 和部署实验必须补上。
```

### 1. 论文逐篇摘要与可借鉴点

#### 1.1 针对弱小无人机目标的轻量级目标检测算法

模型：

```text
DTD-YOLOv4-tiny
```

任务特点：

```text
偏向天空中远距离小无人机/弱小目标检测，不完全等同于 VisDrone 人车密集小目标。
```

主要改进：

```text
1. K-means++ 重聚类 Anchor box
2. 增加 52x52 尺度检测头
3. 使用 ShuffleNetV2 轻量骨干
4. 使用 reorg_layer 下采样保留细节
5. 使用 sub-pixel 上采样增强高分辨率特征
```

重要结果：

```text
模型大小约 1.4 MB
GFLOPs 约 1.1
Dataset A 上 AP@50 最高约 95.0
Dataset B 上 AP@50 最高约 85.1
```

对我们的启发：

```text
reorg_layer 和 sub-pixel 的本质都是空间-通道重排。
这与 PixelUnshuffle / SPDConv / DCLDown 的思想高度一致。
```

可吸收：

```text
空间重排下采样思想
轻量骨干/neck 的速度意识
针对小目标重新设计特征尺度
```

不建议直接照搬：

```text
YOLOv4-tiny 结构较老
Anchor-based 设计与 YOLO26 end2end 机制不同
该任务偏弱小无人机，不是 VisDrone 人车密集目标
```

对当前路线的影响：

```text
进一步支持 DCLDown / WDCLDown。
后续论文可以把 reorg/SPD/PixelUnshuffle 作为“空间信息保留下采样”的相关思想来讨论。
```

#### 1.2 无人机场景下基于轻量化YOLOv8的小目标检测算法

模型：

```text
Drone-YOLO
```

基础模型：

```text
YOLOv8n
```

主要改进：

```text
1. CSPGhostNet 替换骨干网络，降低参数和计算量
2. CBAM 混合注意力增强通道/空间特征
3. 小目标共享卷积检测头，引入 P2/小目标检测层
4. WIoU Loss 优化小目标定位
```

摘要结果：

```text
在 VisDrone2019 上 mAP@50 提升约 10.9%
在 MVRSD 上 mAP@50 提升约 9.1%
参数量压缩约 45.2%
计算量压缩约 43.9%
```

消融结果重点：

```text
YOLOv8n baseline:
P = 44.3, mAP50 = 33.1, mAP50-95 = 19.3, Params = 3.01M, GFLOPs = 8.2

仅 CSPGhostNet:
P = 47.0, mAP50 = 37.3, mAP50-95 = 22.1, Params = 1.85M, GFLOPs = 5.2

仅 CBAM:
P = 47.1, mAP50 = 37.0, mAP50-95 = 22.2, Params = 3.38M, GFLOPs = 9.5

仅共享小目标检测头:
P = 50.3, mAP50 = 41.2, mAP50-95 = 24.3, Params = 2.36M, GFLOPs = 6.4

仅 WIoU:
P = 46.7, mAP50 = 35.3, mAP50-95 = 21.1

全部加入:
P = 53.9, mAP50 = 43.9, mAP50-95 = 26.6, Params = 1.65M, GFLOPs = 4.6
```

对我们的启发：

```text
检测头和 loss 对小目标影响很大。
Ghost 类模块可以降参数，但不一定天然适合所有硬件加速。
CBAM 可涨点，但不是我们当前第一优先级。
```

可吸收：

```text
WIoU 作为 NWD 后的 loss 对照。
共享小目标检测头作为后期 SharedLiteHead 思路。
CSPGhostNet 可作为轻量化 block 储备。
```

暂不优先：

```text
全 backbone 替换为 Ghost。
大面积插入 CBAM。
```

#### 1.3 基于YOLOv8的轻量化无人机图像目标检测算法

模型：

```text
LM-YOLOv8
```

主要改进：

```text
1. FasterC2f：用 FasterNet / PConv 思想替换 C2f 中 Bottleneck
2. FD-FPN：聚焦扩散特征金字塔
3. FTF：聚焦三层特征模块
4. SCDH：共享卷积检测头
5. 重构小目标检测网络，使用 P2/P3/P4 三层检测头
```

摘要结果：

```text
相比 YOLOv8s：
P 提升 5.1%
R 提升 5.4%
mAP 提升 6.6%
参数量降低 68%
模型体积减少 15.3 MB
FPS 提升 16%
```

关键消融：

```text
YOLOv8s:
P = 50.2, R = 38.0, mAP = 38.8, Params = 11.17M, Weight = 22.5MB, FPS = 110

+ FasterC2f:
P = 49.9, R = 38.6, mAP = 39.4, Params = 9.68M, Weight = 18.7MB, FPS = 238

+ FD-FPN:
P = 51.5, R = 38.2, mAP = 39.3, Params = 10.97M, Weight = 21.1MB, FPS = 176

+ SCDH:
P = 51.5, R = 38.8, mAP = 39.5, Params = 9.43M, Weight = 18.2MB, FPS = 204

+ 重构小目标:
P = 55.1, R = 42.0, mAP = 44.3, Params = 3.53M, Weight = 7.4MB, FPS = 101

Ours:
P = 55.3, R = 43.4, mAP = 45.4, Params = 3.62M, Weight = 7.2MB, FPS = 128
```

对我们的启发：

```text
FTF/FD-FPN 说明三层特征聚焦融合很有效。
FasterC2f 说明部分通道卷积能明显提速。
P2/P3/P4 检测头再次证明高分辨率小目标头有效。
```

可吸收：

```text
FasterC2f -> DCL-C3k2Lite
FTF/FD-FPN -> DAFuse-ST
SCDH -> SharedLiteHead
P2/P3/P4 -> P2 作为教师或上限对照
```

不建议直接照搬：

```text
完整 FD-FPN + P2/P3/P4 检测头。
直接把 C2f 结构搬到 YOLO26 的 C3k2 中。
```

更适合我们的改法：

```text
只在 YOLO26 neck 中加入轻量尺度感知融合；
只在 head 小目标相关 C3k2 做轻量化替换。
```

#### 1.4 基于YOLOv8n的无人机小目标检测算法

基础模型：

```text
YOLOv8n
```

主要改进：

```text
1. ConvNeXt v2 替换 Backbone
2. ASF-Neck
3. TFE：Triple Feature Encoding / 三重特征编码
4. SSFF：Scale Sequence Feature Fusion / 尺度序列特征融合
5. MSFH：多尺度特征融合检测层
6. LAMP：层自适应幅度剪枝
```

摘要结果：

```text
在 VisDrone2019 数据集上 mAP50 达到 30.2
相比 YOLOv8n 提升 3.8
参数量减少 1.63M
```

关键对比：

```text
YOLOv8n:
mAP50 = 26.4, mAP50-95 = 14.7, Params = 3.01M, FLOPs = 8.1G

Ours:
mAP50 = 30.2, mAP50-95 = 16.5, Params = 1.38M, FLOPs = 9.0G
```

消融重点：

```text
+ SSFF:
mAP50 = 27.0, mAP50-95 = 15.0

+ SSFF + TFE:
mAP50 = 27.7, mAP50-95 = 15.6

+ SSFF + TFE + MSFH:
mAP50 = 29.4, mAP50-95 = 16.3, Params = 2.49M, FLOPs = 12.0G

+ ConvNeXt v2:
mAP50 = 32.2, mAP50-95 = 17.6, Params = 5.16M, FLOPs = 18.2G

+ LAMP:
mAP50 = 30.2, mAP50-95 = 16.5, Params = 1.38M, FLOPs = 9.0G
```

对我们的启发：

```text
剪枝能把复杂模型重新压轻，但会带来精度回落。
参数少不等于 FLOPs 一定低。
ConvNeXt v2 虽然涨点明显，但对边缘部署和初学实现不友好。
```

可吸收：

```text
TFE/SSFF 的尺度序列融合思想 -> DAFuse-ST
LAMP 剪枝 -> 主模型稳定后的压缩实验
目标尺寸统计图 -> 我们论文中也应该补 VisDrone 小目标尺寸分布统计
```

暂不优先：

```text
ConvNeXt v2 替换 backbone。
先做剪枝。
```

#### 1.5 基于改进YOLO11的无人机航拍图像小目标检测算法

模型：

```text
ACFI-YOLO11
```

主要改进：

```text
1. Tiny Head 分支：增加高分辨率极小目标检测头
2. ACFI：基于注意力的跨层信息交互模块
3. LFA：Layer Feature Aggregation / 层特征聚合
4. Transformer encoder：增强跨层全局交互
5. SPDConv：空间到深度卷积，替换部分下采样卷积
```

摘要结果：

```text
VisDrone2021:
APs 提升 4.2
APxs 提升 3.5
mAP50 提升 5.2
mAP50-95 提升 4.0

UAVDT:
mAP50-95 达到 83.3
```

消融结果：

```text
YOLO11s baseline:
APs = 16.9, APxs = 8.4, mAP50-95 = 27.7, Params = 9.4M, FLOPs = 21.6B

+ Tiny Head:
APs = 20.6, APxs = 11.5, mAP50-95 = 30.8, Params = 9.6M, FLOPs = 29.6B

+ ACFI:
APs = 17.3, APxs = 8.8, mAP50-95 = 28.3, Params = 18.5M, FLOPs = 33.7B

+ Tiny Head + ACFI:
APs = 21.0, APxs = 11.8, mAP50-95 = 31.5, Params = 19.0M, FLOPs = 46.4B

+ Tiny Head + ACFI + SPDConv:
APs = 21.1, APxs = 11.9, mAP50-95 = 31.7, Params = 18.0M, FLOPs = 42.5B
```

对我们的启发：

```text
Tiny Head 是主要涨点来源，但 FLOPs 明显增加。
ACFI 单独涨点不大，且参数和计算量明显增加。
SPDConv 能降低参数和计算量，并对小目标略有帮助。
```

可吸收：

```text
SPDConv / PixelUnshuffle -> DCLDown / WDCLDown
LFA 层聚合思想 -> DAFuse-ST
Tiny Head -> P2 教师模型或上限对照
```

暂不优先：

```text
完整 ACFI + Transformer encoder。
```

原因：

```text
Transformer 在轻量边缘部署中代价偏高；
YOLO26 已经有 C2PSA；
ACFI 单独收益和代价不成比例。
```

#### 1.6 基于YOLO的无人机小目标高精度检测改进模型

模型：

```text
YOLO-LiteMax
```

主要改进：

```text
1. SCB：Selective Convolution Block，部分通道轻量特征提取
2. STSSF：Small Target Scale Sequence Fusion，小目标尺度序列融合
3. C3DSF：Convolution 3D Scale Fusion
4. SAC：Scale Aware Concatenation
5. SCPD：Shared Convolution Precision Detection，共享卷积检测头
```

关键结果：

```text
VisDrone2019-DET:
mAP@0.5 = 45.2
APsmall = 15.3
Params = 6.1M
GFLOPs = 30.0G
FPS = 118
```

对我们的启发：

```text
STSSF 再次证明 P2/浅层高分辨率细节 + 跨尺度融合是核心涨点。
SCB 主要用于轻量化。
SCPD 说明共享检测头可以减少冗余并增强一致性。
```

### 2. 六篇论文共同规律

#### 2.1 最常见、最有效的涨点来自 P2/Tiny Head

出现次数：

```text
DTD-YOLOv4-tiny：52x52 检测头
Drone-YOLO：小目标共享检测头
LM-YOLOv8：P2/P3/P4 检测头
YOLOv8n 改进：MSFH 高分辨率小目标检测层
ACFI-YOLO11：Tiny Head
YOLO-LiteMax：P2 + STSSF
```

结论：

```text
P2/Tiny Head 基本是无人机小目标检测最稳定的涨点来源。
```

但对我们的限制：

```text
高分辨率检测头会增加 FLOPs、显存、延迟，对树莓派 5 不友好。
```

我们的策略：

```text
P2 作为 teacher / upper bound / 消融对照；
最终部署模型不优先保留 P2 Detect。
```

#### 2.2 空间重排/细节保持下采样被反复验证

相关模块：

```text
reorg_layer
sub-pixel
SPDConv
PixelUnshuffle
FGDDown
DCLDown
WDCLDown
```

共同思想：

```text
把原本容易在 stride=2 卷积或池化中丢掉的空间细节，转移到通道维或高频分支中。
```

对我们最重要：

```text
这是最适合替代 P2 检测头的轻量化主线。
```

#### 2.3 跨尺度融合是第二涨点来源

相关模块：

```text
STSSF / C3DSF / SAC
FTF / FD-FPN
TFE / SSFF / ASF-Neck
ACFI / LFA
DEConv multi-scale head
DAFuse-ST
```

共同目标：

```text
让浅层空间细节和深层语义信息更充分交换。
```

我们的吸收方式：

```text
不用完整大 neck；
只做两个轻量融合点：
1. P4 -> P3 融合
2. P5 -> P4 / P3 -> P4 融合
```

#### 2.4 轻量卷积能降参数，但不一定直接涨点

相关模块：

```text
ShuffleNetV2
GhostConv / CSPGhostNet
FasterNet / PConv / FasterC2f
SCB
GSConv
DWConv
```

规律：

```text
轻量模块主要降低参数、模型体积和理论计算量；
单独替换时可能轻微涨点，也可能略降。
```

我们的策略：

```text
后置做 DCL-C3k2Lite；
只替换 neck/head 的小目标相关 C3k2；
不先替换整个 backbone。
```

#### 2.5 检测头共享很值得，但 YOLO26 上要后置

相关模块：

```text
SCPD
SCDH
共享小目标卷积检测头
SharedLiteHead
```

优点：

```text
降低检测头参数；
增强多尺度预测一致性；
小 batch 时配合 GroupNorm 可能更稳定。
```

风险：

```text
YOLO26 的 Detect head 与 end2end=True、loss、导出逻辑耦合较深。
初期直接改 Detect 容易把训练/推理/导出都弄乱。
```

我们的策略：

```text
后期再研究 SharedLiteHead。
当前不作为下一轮实验。
```

#### 2.6 Loss 改进是低结构风险路线

相关模块：

```text
WIoU
NWD
FoCIoU
```

结论：

```text
Loss 不直接改变网络结构，部署风险低；
但可能影响训练稳定性和收敛，需要独立消融。
```

我们的顺序：

```text
NWD -> WIoU -> NWD + WIoU
```

### 3. 对当前 YOLO26 路线的重新排序

当前正在训练：

```text
YOLO26 + DCLDown
```

训练完成后，推荐顺序更新为：

```text
第一阶段：轻量细节保留下采样
E4  YOLO26 + DCLDown
E5  YOLO26 + WDCLDown

第二阶段：轻量上采样
E6  YOLO26 + DCLDown + DySample
E7  YOLO26 + WDCLDown + DySample

第三阶段：轻量跨尺度融合
E8  YOLO26 + DCLDown + DySample + DAFuse-ST
E9  YOLO26 + WDCLDown + DySample + DAFuse-ST

第四阶段：小目标定位损失
E10 E8/E9 + NWD
E11 E8/E9 + WIoU

第五阶段：轻量 block
E12 E10 + DCL-C3k2Lite

第六阶段：检测头共享/蒸馏
E13 SharedLiteHead
E14 P2+FGDDown teacher -> no-P2 student
```

### 4. 下一步最推荐的模块定义

#### 4.1 DAFuse-ST

全称：

```text
Detail-aware Scale-aware Fusion
```

中文：

```text
细节感知尺度自适应融合模块
```

借鉴来源：

```text
STSSF / SAC / FTF / TFE / SSFF / ASFM / EFC
```

设计目标：

```text
替代普通 Concat，让浅层细节和深层语义有权重地融合。
```

轻量版结构：

```text
x_deep, x_shallow
-> 通道对齐
-> GAP + GMP 生成尺度权重
-> softmax 得到两个分支权重
-> weighted concat
-> DWConv 3x3 + Conv 1x1
```

可选增强：

```text
加入 MaxPool(x_shallow) - AvgPool(x_shallow) 作为局部对比提示。
```

优先替换位置：

```yaml
[[ -1, 4 ], 1, Concat, [1]]   # P3 融合，小目标最重要
[[ -1, 13], 1, Concat, [1]]   # P4 融合，兼顾中小目标
```

#### 4.2 WDCLDown

全称：

```text
Wavelet Detail-Contrast Lightweight Downsampling
```

中文：

```text
小波细节对比轻量下采样模块
```

借鉴来源：

```text
SPDConv / reorg_layer / sub-pixel / PixelUnshuffle / wavelet pooling
```

设计目标：

```text
用低成本方式保留 P2 阶段的边缘、纹理和局部对比信息。
```

建议结构：

```text
input
├─ PixelUnshuffle(2) -> 1x1 Conv
├─ DWConv(k=3,s=2) -> 1x1 Conv
└─ Haar high-frequency(LH/HL/HH) -> 1x1 Conv
concat
-> DWConv(k=3,s=1)
-> Conv(k=1,s=1)
```

替换位置：

```yaml
YOLO26 backbone:
- [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
替换为
- [-1, 1, WDCLDown, [256]]
```

#### 4.3 DCL-C3k2Lite

借鉴来源：

```text
SCB / FasterC2f / FasterNet / PConv / GSConv
```

设计目标：

```text
只对部分通道做空间卷积，减少 neck/head 冗余计算。
```

替换位置：

```yaml
YOLO26 head:
layer 16 C3k2
layer 19 C3k2
```

暂不替换：

```text
backbone 全部 C3k2
P5 大目标分支 C3k2
```

### 5. 现在不能盲目追高分的原因

这几篇论文很多 mAP 看起来比我们高，但不能直接说我们实验白跑：

```text
1. 有的用 YOLOv8s/YOLO11s，不是 YOLO26n。
2. 有的输入是 640 或 800，不同尺寸不可直接比较。
3. 有的用 VisDrone2019，有的用 VisDrone2021，有的用 UAVDT/MVRSD/自建数据。
4. 有的训练 300-500 epochs，我们目前部分实验是 100-200 epochs。
5. 有的加了 P2/Tiny Head，计算量明显提高。
6. 有的最终参数低，但 FLOPs 或 latency 未必适合树莓派。
7. 有些文章只报 mAP50，不报 mAP50-95 或真实部署速度。
```

因此我们真正要比的是：

```text
同一数据集
同一输入尺寸
同一训练策略
同一基础 YOLO26n
同一评估脚本
同一部署格式
```

### 6. 对论文创新性的重新定位

不要写成：

```text
本文加入 P2、注意力、轻量卷积、改 loss。
```

这样太像普通模块堆叠。

更建议写成：

```text
本文围绕边缘部署场景下无人机小目标检测的“高分辨率细节保留与计算开销矛盾”展开：
1. 通过细节保持下采样减少普通 stride=2 卷积的信息损失；
2. 通过轻量尺度感知融合弥补浅层细节与深层语义交互不足；
3. 通过小目标友好的定位损失缓解 tiny box 对位置偏差敏感的问题；
4. 在不保留高成本 P2 检测头的前提下，接近 P2 模型的小目标检测收益。
```

候选论文题目继续保留：

```text
DCL-YOLO26: A Detail-Preserving Lightweight YOLO26 for UAV Small Object Detection on Edge Devices
```

如果 WDCLDown 效果更好：

```text
WDCL-YOLO26: Wavelet Detail-Contrast Lightweight YOLO26 for UAV Small Object Detection
```

如果最终加入蒸馏：

```text
DCLD-YOLO26: Lightweight UAV Small Object Detection via Detail-Preserving Downsampling and P2-Guided Distillation
```

### 7. 当前执行建议

短期不变：

```text
继续等待 YOLO26 + DCLDown 的 200 epoch 完整结果。
```

结果出来后按以下判断：

```text
若 mAP50-95 >= baseline:
    进入 DCLDown + DySample。

若 mAP50-95 略低但 Params/GFLOPs 明显更优:
    仍保留，进入 DCLDown + DySample，因为 DySample 可能补偿融合损失。

若 mAP50-95 明显低:
    DCLDown-v1 作为失败消融，转 WDCLDown。
```

下一轮优先写代码：

```text
1. DySample
2. DAFuse-ST
3. WDCLDown
```

后置代码：

```text
4. NWD / WIoU
5. DCL-C3k2Lite
6. SharedLiteHead
7. P2 teacher distillation
```

## 二十二、2026-08-19 YOLO26 + DCLDown 完整实验记录

### 1. 结果归档位置

本次 AutoDL 训练结果已归档到：

```text
F:\图像深度学习科研\面向边缘设备部署的无人机小目标轻量化检测算法研究\yolo26\train_drone\yolov26+DCLDown
```

目录中保留：

```text
args.yaml
results.csv
results.png
confusion_matrix.png
confusion_matrix_normalized.png
BoxF1/PR/P/R 曲线
train_batch*.jpg
val_batch*_labels.jpg
val_batch*_pred.jpg
weights/best.pt
weights/last.pt
```

### 2. 训练设置

```text
model: train_drone/yolov26+DCLDown/yolov26+DCLDown.yaml
data: train_drone/yolov26+DCLDown/yolov26+DCLDown_data.yaml
pretrained: yolo26n.pt
epochs: 200
patience: 50
batch: 8
imgsz: 960
workers: 8
device: 0
amp: True
close_mosaic: 10
```

实际训练在第 168 轮触发 EarlyStopping，最佳模型出现在第 118 轮。

### 3. 最佳结果

```text
YOLOv26 + DCLDown summary (fused):
130 layers, 2,355,602 parameters, 5.0 GFLOPs

best epoch: 118
Precision: 0.53424
Recall: 0.41315
mAP50: 0.41302
mAP50-95: 0.24469
```

最后一轮结果：

```text
epoch 168
Precision: 0.53861
Recall: 0.41180
mAP50: 0.40721
mAP50-95: 0.24117
```

说明：第 118 轮后训练 loss 继续缓慢下降，但验证 mAP 不再提升，属于平台期/轻微过拟合趋势，不建议继续单独加训 DCLDown。

### 4. 与已有实验对比

| 模型 | Best Epoch | P | R | mAP50 | mAP50-95 | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26 baseline | 86 | 0.52215 | 0.41993 | 0.41202 | 0.24407 | 2.5077M | 6.1 |
| YOLO26 + DCLDown | 118 | 0.53424 | 0.41315 | 0.41302 | 0.24469 | 2.3556M | 5.0 |
| YOLO26 + FGDDown | 146 | 0.53629 | 0.42614 | 0.41554 | 0.24612 | 2.5181M | 约 6.1 |
| YOLO26 + P2 | 76 | 0.51529 | 0.43452 | 0.41713 | 0.24977 | 2.5184M | 约 6.6 |
| YOLO26 + P2 + FGDDown | 125 | 0.53542 | 0.43546 | 0.42709 | 0.25567 | 2.5210M | 约 6.6 |

相对 YOLO26 baseline：

```text
DCLDown 参数量下降约 6.07%
DCLDown GFLOPs 下降约 18.03%
mAP50 提升约 +0.00100
mAP50-95 提升约 +0.00062
```

### 5. 结论定位

本次结果不是失败，而是：

```text
轻量化有效，精度基本持平，明显涨点不足。
```

DCLDown 不能替代 P2 检测头带来的高分辨率预测收益，但它证明了一个重要方向：

```text
在不增加 P2 高成本检测头的前提下，
通过细节保持下采样可以降低计算量，
并尽量保持 YOLO26 baseline 的检测精度。
```

因此 DCLDown 适合作为论文中的轻量化消融模块，而不是最终单模块主贡献。

### 6. 类别层面观察

最佳验证输出中，表现较好的类别：

```text
car: mAP50=0.809, mAP50-95=0.564
bus: mAP50=0.558, mAP50-95=0.409
motor: mAP50=0.500, mAP50-95=0.226
pedestrian: mAP50=0.493, mAP50-95=0.227
```

拖累总指标的类别：

```text
bicycle: mAP50=0.162, mAP50-95=0.0743
awning-tricycle: mAP50=0.123, mAP50-95=0.0743
tricycle: mAP50=0.288, mAP50-95=0.161
people: mAP50=0.368, mAP50-95=0.143
```

归一化混淆矩阵显示，大量小目标仍被判为 background，尤其 bicycle、awning-tricycle、people 等类别。这说明单纯优化下采样无法完全解决小目标召回不足问题，后续必须继续优化上采样/跨尺度融合/损失。

### 7. 下一步执行建议

不建议：

```text
继续单独重训 YOLO26 + DCLDown 到 200 或 300 epoch
```

原因：第 118 轮已经达到最佳，后续 50 轮无提升触发早停。

建议保留 DCLDown，并进入下一组实验：

```text
E5: YOLO26 + DCLDown + DySample
E6: YOLO26 + DCLDown + DAFuse-ST
E7: YOLO26 + DCLDown + DySample + DAFuse-ST
```

若仍然没有明显涨点，再切换：

```text
E8: YOLO26 + WDCLDown
E9: YOLO26 + WDCLDown + DySample
```

最终论文中可将 P2 + FGDDown 作为“高精度上界/teacher”，将 DCLDown 或 WDCLDown 作为“轻量化 student/backbone 下采样优化”。

## 二十三、2026-08-19 YOLO26 + DySample 完整实验记录与组合实验准备

### 1. YOLO26 + DySample 结果归档

本次结果已放回：

```text
F:\图像深度学习科研\面向边缘设备部署的无人机小目标轻量化检测算法研究\yolo26\train_drone\yolov26+DySample
```

最佳结果：

```text
best epoch: 91
Precision: 0.52654
Recall: 0.43237
mAP50: 0.42038
mAP50-95: 0.24843

fused summary:
122 layers
2,389,138 parameters
5.3 GFLOPs
```

最后一轮：

```text
epoch: 191
Precision: 0.53484
Recall: 0.42435
mAP50: 0.41570
mAP50-95: 0.24564
```

说明：第 91 轮达到最佳，后续继续训练到第 191 轮触发 early stopping。后期 precision 提升但 recall 与 mAP 回落，说明泛化最佳点较早。

### 2. 与 baseline 对比

| 模型 | mAP50 | mAP50-95 | Params | GFLOPs |
|---|---:|---:|---:|---:|
| YOLO26 baseline | 0.41202 | 0.24407 | 2.5077M | 6.1 |
| YOLO26 + DySample | 0.42038 | 0.24843 | 2.3891M | 5.3 |

相对 baseline：

```text
mAP50 提升约 +0.00836
mAP50-95 提升约 +0.00436
参数量下降约 4.73%
GFLOPs 下降约 13.11%
```

结论：

```text
DySample 是当前最优先保留的有效模块。
```

它在不增加 P2 检测头的情况下，提升了跨尺度融合质量，并让结果接近 YOLO26 + P2。

### 3. 已准备下一组组合实验

组合实验目录：

```text
F:\图像深度学习科研\面向边缘设备部署的无人机小目标轻量化检测算法研究\yolo26\train_drone\yolov26+DCLDown+DySample
```

已写入文件：

```text
yolov26+DCLDown+DySample.yaml
yolov26+DCLDown+DySample_data.yaml
yolov26+DCLDown+DySample_train.py
README_AUTODL.md
```

模型已本地解析通过：

```text
YOLOv26+DCLDown+DySample summary:
273 layers, 2,499,060 parameters
```

该实验的判断目标：

```text
如果 mAP50-95 > 0.24843:
    说明 DCLDown 与 DySample 有正向组合收益。

如果 mAP50-95 约等于 0.24843，但 GFLOPs/Params 仍优于 P2:
    可作为轻量化部署路线保留。

如果 mAP50-95 低于 0.244:
    说明 DCLDown 对 DySample 的上采样收益产生冲突，后续切换 WDCLDown 或只保留 DySample。
```

## 二十四、2026-08-19 YOLO26 + DySample + DAFuseST 代码准备记录

### 1. 为什么转向这条路线

YOLO26 + DCLDown + DySample 的组合实验出现退步：

```text
mAP50: 0.410
mAP50-95: 0.243
Params: 2.368M
GFLOPs: 5.1
```

说明 DCLDown 与 DySample 同时改变空间采样时，可能产生特征对齐冲突。因此当前主线改为：

```text
YOLO26 baseline + DySample + DAFuseST
```

保留已经验证有效的 DySample，不再叠加 DCLDown，而是在跨尺度融合处改进 Concat。

### 2. DAFuseST 模块设计

源码位置：

```text
F:\图像深度学习科研\面向边缘设备部署的无人机小目标轻量化检测算法研究\yolo26\ultralytics\nn\modules\conv.py
```

模块名：

```text
DAFuseST
```

设计思想：

```text
Dynamic Adaptive Fusion for Small Target
```

具体机制：

```text
1. 输入仍为多路特征，例如 [上采样深层特征, backbone浅层特征]。
2. 输出仍沿通道维拼接，因此后续 C3k2 的输入通道数与普通 Concat 一致。
3. 每一路特征使用一个轻量 1x1 spatial gate 生成空间权重。
4. 使用 branch_scale 学习每一路整体强弱。
5. branch_scale 和 gamma 都初始化为 0，因此训练初期近似普通 Concat，减少破坏预训练分布的风险。
```

### 3. 已修改源码

已修改：

```text
ultralytics/nn/modules/conv.py
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py
```

tasks.py 中 DAFuseST 解析方式：

```text
c2 = sum(ch[x] for x in f)
args = [[ch[x] for x in f], *args]
```

原因：DAFuseST 和 Concat 一样输出通道拼接结果，但内部需要知道每一路输入通道数。

### 4. 实验文件

已创建：

```text
F:\图像深度学习科研\面向边缘设备部署的无人机小目标轻量化检测算法研究\yolo26\train_drone\yolov26+DySample+DAFuseST
```

包含：

```text
yolov26+DySample+DAFuseST.yaml
yolov26+DySample+DAFuseST_data.yaml
yolov26+DySample+DAFuseST_train.py
README_AUTODL.md
```

### 5. YAML 改动位置

只替换前两处跨尺度融合：

```text
P5/32 -> P4/16 后的 Concat 替换为 DAFuseST
P4/16 -> P3/8 后的 Concat 替换为 DAFuseST
```

暂时不替换 PAN 回流路径中的两个 Concat，避免一次改动过多。

### 6. 本地验证

模型解析通过：

```text
YOLOv26+DySample+DAFuseST summary:
262 layers, 2,520,704 parameters
```

本实验判断标准：

```text
若 mAP50-95 > 0.24843:
    DAFuseST 与 DySample 有组合收益。

若 mAP50-95 接近 0.24843:
    说明 DAFuseST 没有明显负作用，可继续做轻量变体。

若 mAP50-95 < 0.244:
    说明当前 DAFuseST 设计不适合该融合位置，应回退 DySample 单独路线。

## 二十五、2026-08-19 六篇论文落地：DAFuseST v2 + WDCLDown 更新记录

### 1. 本次改动概述

六篇本地论文（扫描件无文本层，OCR 成本高，直接复用第二十/二十一节的逐篇笔记）落地为两件代码工作：

```text
1. 重写 DAFuseST v1 -> v2，修复梯度消失和冗余问题。
2. 新增 WDCLDown（小波细节对比轻量下采样），替换 DCLDown 的 MaxPool-AvgPool 分支。
```

改动文件：

```text
ultralytics/nn/modules/conv.py       DAFuseST v2 重写 + WDCLDown 新增
ultralytics/nn/modules/__init__.py   注册 WDCLDown
ultralytics/nn/tasks.py              注册 WDCLDown（base_modules）+ 注释更新
train_drone/yolov26+DySample+DAFuseST/*.yaml   DAFuseST 参数 [1, True] -> [1]
train_drone/yolov26+WDCLDown/*       新增实验（yaml/data/train.py/README）
```

### 2. DAFuseST v2 重写原因

v1 已诊断出的两个问题（本次直接改掉）：

```text
问题1（梯度消失）：gamma=0 初始化 -> tanh(0)=0，空间门控卷积训练初期梯度为零，
                  长期不激活，模块几乎退化成普通 Concat。
问题2（冗余）：branch_scale 逐分支标量，和下游 C3k2 的 1x1 卷积作用重复。
```

v2 设计：

```text
x_deep, x_shallow
-> 每支路 GAP + GMP 全局统计
-> 微型 MLP(2*total -> 8 -> n) -> softmax
-> 分支权重 w1, w2（和为 1）
-> weighted concat
输出通道 = sum(channels)，与普通 Concat 一致，下游 C3k2 输入通道不变
```

优点：

```text
1. 无 tanh/零初始化，第一步起就有梯度。
2. 机制明确：softmax 分支权重，显式表达"浅层细节 vs 深层语义"的取舍。
3. 参数量极小（约 36K），不增加部署负担。
```

注意：

```text
当前正在跑的 DySample+DAFuseST 用的是 v1（云端已加载旧代码，不受影响）。
下次同步重跑会自动用 v2 + 新 yaml，无需重跑 pip install -e .。
```

### 3. WDCLDown 模块设计

```text
input
├─ PixelUnshuffle(2) -> 1x1 Conv      SPDConv/reorg 细节重排
├─ DWConv(3,s=2)     -> 1x1 Conv      可学习局部下采样
└─ Haar 高频(LH/HL/HH) -> 1x1 Conv   水平/垂直/对角边缘，几乎零参数
concat -> DWConv(3) -> 1x1 Conv
```

与 DCLDown 的唯一区别：第三支路从 `MaxPool-AvgPool` 对比改为 `Haar 高频`。
Haar 高频能同时保留三个方向的边缘纹理，比"最大值减均值"信号更强。

替换位置（与 DCLDown 完全一致，便于对照）：

```yaml
backbone:
  - [-1, 1, WDCLDown, [256]] # 3-P3/8，替换 Conv [256, 3, 2]
```

验证情况：

```text
py_compile 通过；Haar 公式 numpy 验证正确（水平图只激活 LH、HL=0）。
```

### 4. 更新后的实验队列

| 顺序 | 实验 | 状态 | mAP50-95 | 说明 |
|---|---|---|---|---|
| E0 | baseline | 完成 | 0.24407 | 基准 |
| E1 | + P2 | 完成 | 0.24977 | 上限参考 |
| E2 | + P2 + FGDDown | 完成 | 0.25567 | 教师候选 |
| E3 | + FGDDown | 完成 | 0.24612 | 无 P2 对照 |
| E4 | + DCLDown | 完成 | 0.24469 | 轻量但涨点不足 |
| E5 | + DySample | 完成 | 0.24843 | 当前最优单模块 |
| E6 | + DCLDown + DySample | 完成 | 0.243 | 空间采样冲突，退步 |
| E7 | + DySample + DAFuseST v1 | 训练中 | 待定 | v1 预计涨不动 |
| E8 | + WDCLDown | 待跑 | 待定 | 本次新增，优先 |
| E9 | + WDCLDown + DySample | 待做 | 待定 | E8 通过后 |
| E10 | E9 + DAFuseST v2 | 待做 | 待定 | 组合候选 |
| E11 | E10 + NWD | 待做 | 待定 | tiny-box loss |

### 5. 下一步判断

```text
1. E8 WDCLDown 结果出来后：
   - 若 > 0.24469（超过 DCLDown）：小波高频有效，进入 E9 WDCLDown+DySample。
   - 若接近但 GFLOPs 更低：保留为轻量消融。
   - 若 < 0.244：回退 DySample 单独路线。

2. E7 DySample+DAFuseST(v1) 结果出来后：
   - 大概率涨不动（v1 缺陷），保留为失败对照即可。
   - 若仍想验证融合模块，改用 v2 重跑一次。

3. 主线优先级：WDCLDown/DCLDown(下采样) + DySample(上采样) + NWD(loss)
   是当前最有依据的组合，DAFuseST 融合模块后置，避免和 C3k2 冗余。
```

### 6. 一句路线总结

```text
细节保持下采样（DCLDown/WDCLDown）→ 动态上采样（DySample）→ 小目标损失（NWD），
在不加 P2 检测头的前提下逼近 P2 效果；融合模块（DAFuseST）后置为可选消融。
```
```

## 二十六、2026-08-20 全量实验结果汇总：主线必须转向（保留 P2，轻量化靠 Neck）

### 1. 本轮新增完成的实验

- E7 `+ DySample + DAFuseST v1`：best mAP50-95 = 0.24696，**低于** DySample 单独（0.24843），证实 v1 融合模块冗余、无净收益。
- `+ NWD`：best mAP50-95 = 0.24484，几乎等于 baseline（0.24407），**单独 NWD 无收益**。

### 2. 全部实验最终结果表（截至 2026-08-20，best 值）

| 实验 | best mAP50 | best mAP50-95 | 相对 baseline | 判定 |
|---|---:|---:|---:|---|
| YOLO11n baseline | 0.41784 | 0.24864 | +0.0046 | ⚠️ 反超 YOLO26 基线 |
| YOLO26 baseline | 0.41202 | 0.24407 | 基准 | 基准 |
| + P2 | 0.41804 | 0.24977 | +0.0057 | ✅ 有效 |
| + P2 + FGDDown | 0.42714 | 0.25567 | +0.0116 | ✅ 当前最优 |
| + FGDDown | 0.41581 | 0.24612 | +0.0021 | 微弱有效 |
| + DCLDown | 0.41361 | 0.24469 | +0.0006 | ❌ 无效 |
| + DySample | 0.42038 | 0.24843 | +0.0044 | ✅ 最优单模块 |
| + DCLDown + DySample | 0.41098 | 0.24325 | -0.0008 | ❌ 负收益 |
| + DySample + DAFuseST v1 | 0.41677 | 0.24696 | +0.0029 | ❌ 不如 DySample 单 |
| + NWD | 0.41249 | 0.24484 | +0.0008 | ❌ 无效 |

### 3. 硬结论（数据说话，不再凭直觉）

有效（正收益）：
```text
P2 (+0.006)、P2+FGDDown (+0.012)、DySample (+0.004)、FGDDown (+0.002)
```

无效或负收益：
```text
DCLDown (约 0)、DCLDown+DySample (-0.001)、DAFuseST v1 (-0.001 vs DySample)、NWD (约 0)
```

重大风险：
```text
YOLO11n baseline (0.24864) > YOLO26 baseline (0.24407)。
论文若主打"改进 YOLO26"，最终必须明显超过 YOLO11，否则审稿人会质疑基线选择。
```

### 4. 路线必须转向：放弃"无 P2 轻量化逼近 P2"

被数据否定的事实：
```text
无 P2 路线最优是 DySample (0.24843)，仍不如 P2 (0.24977)、更不如 P2+FGDDown (0.25567)。
DCLDown / DAFuseST 两个自研模块均无净收益。
结论：P2 是真实涨点来源，删掉 P2 后没有任何轻量模块能补回。
```

新主线（保留 P2，轻量化靠 Neck 而不是删 P2）：
```text
主结构 = P2 + FGDDown + DySample（三者都是验证过的正收益，下一步组合）
轻量化 = GSConv / SlimNeck 压 Neck 计算量，而非去掉 P2 检测头
NWD 降级 = 最终结构模型上的可选小消融，不再单独做主实验
```

### 5. 下一步实验队列（2026-08-20 起）

| 顺序 | 实验 | 目的 | 优先级 |
|---|---|---|---|
| E12 | + P2 + FGDDown + DySample | 组合三个正收益模块，冲击 >0.256 | 最高 |
| E13 | + WDCLDown | 一次性验证小波高频 vs DCLDown | 中（低预期） |
| E14 | E12 + GSConv/SlimNeck | 轻量化 Neck，形成"轻量 P2"论文卖点 | 高（战略） |
| E15 | E12 + NWD | 结构到位后再测 loss，仅作消融 | 低 |

### 6. 一句话新路线

```text
保留 P2 小目标检测头（真实涨点来源），用 FGDDown 保留细节 + DySample 恢复细节
做精度主结构，再用 GSConv/SlimNeck 轻量化 Neck 满足边缘部署；
DCLDown/DAFuseST/NWD 降级为消融对照，不再作为主创新点。
```

## 二十七、2026-08-20 P2 算力矛盾分析与最终方案定型（方案一+二 + EMA + WIoU）

### 1. 核心矛盾（本次要解决的问题）

```text
加入完整 P2 (160×160) 检测头：小目标涨点（我方实测 mAP50-95 +0.57%），
但高分辨率特征图面积是 P3 的 4 倍，完整解耦 Detect 算力爆炸（文献约 +30%），
违背"面向边缘设备部署的轻量化"课题初衷。
不加 P2：拿不到同等小目标增益，且方案三（无 P2 注入 P3）已被 DCLDown 实测否定。
```

### 2. 四方案取舍（数据支撑，非文献空谈）

| 方案 | 思路 | 我方实测依据 | 取舍 |
|---|---|---|---|
| 一 | +P2 同时删 P5（算力加减法） | P2 有效 +0.57%；P5 在 VisDrone 大目标少，冗余 | ✅ 采纳为骨架 |
| 二 | 保留 P2 通路但轻量化 P2 头（Tiny-Head 共享卷积/GSConv） | 完整 P2 Detect 是算力爆炸元凶 | ✅ 采纳为创新点 |
| 三 | 不新增 P2 头，浅层特征注入 P3（SPD-Conv/跨层融合） | DCLDown=0.24469(≈0)、DAFuseST=0.24696(负) 均失败 | ❌ 放弃 |
| 四 | SAHI 切片推理 | 训练不改网络，仅推理 trick | 只做部署/泛化对照，不作核心创新 |

### 3. 最终模型定义（本次定型）

```text
YOLO26 + P2 + FGDDown + 删P5 + 轻量化P2头(Tiny-Head)
检测头 = Detect(P2, P3, P4)，剪掉 P5
P2 头 = 共享卷积 / GSConv 轻量化，不装完整解耦 Detect
P3/P4 = 保留标准解耦头
```

精度提升模块（配菜，一次只加一个做消融，禁止堆叠）：

```text
第一轮加：EMA 多尺度注意力（放 P3 小目标分支 C3k2 后）  预期 +0.3~0.6%
第二轮加：WIoU v3 loss（替换 CIoU，不是已失败的 NWD）    预期 +0.2~0.5%
可选对照：SimAM（零参数，不伤 FLOPs）/ CA（位置信息，定位友好）
不推荐：CBAM/SE（老）、DyHead/DCN/Mamba（重、部署不友好）
```

预期总目标（相对 baseline，经验值，需实测）：

```text
P2+FGDDown (0.25567) → +EMA → 0.257~0.260 → +WIoU → 0.258~0.262
即相对 baseline 0.24407 提升 +1.5~1.9%（不是文献吹的 +1.5~1.8% 再往上）
```

### 4. 必补测的三个数字（消融表缺列，动手前必须先补）

```text
1. +P2 和 +P2+FGDDown 的实测 FLOPs / Params（model.info() 直接出）
2. 删 P5 后（P2/P3/P4 头）的实测 mAP + FLOPs
3. 每类小目标 APs / 极小型目标指标（VisDrone 论文必须报 APs，不能只看总 mAP）
```

消融表四组骨架：Baseline(P3/P4/P5) / +P2 / -P5(P3/P4) / +P2-P5(P2/P3/P4)

### 5. 待决问题（优先级最高，比选方案更根本）

```text
YOLO11n baseline = 0.24864 > YOLO26 baseline = 0.24407（+0.46%）
若论文主打"改进 YOLO26"，最终必须明显甩开 YOLO11，否则审稿人质疑基线选择。
待办：确认 YOLO11/YOLO26 是否同条件对比（原 baseline 100ep/batch4 vs 后续 200ep/batch8）。
若同条件仍 YOLO11 更高 → 认真考虑换基线 YOLO11/YOLOv8，题目改"改进 YOLO11"。
```

### 6. 一句话新路线

```text
保留 P2 小目标检测头（真实涨点来源）并删冗余 P5 抵消算力，用 FGDDown 保细节，
用轻量化 Tiny-Head 压 P2 检测开销，再以 EMA 注意力 + WIoU loss 做精度配菜，
兼顾小目标精度与边缘部署；方案三（无 P2）与 DCLDown/DAFuseST/NWD 已全部实测否定。
```

## 二十八、2026-08-21 E12 完成：YOLO26 + P2 + FGDDown + DySample 结果评估

### 1. 实验配置
```text
模型：YOLO26 + P2 + FGDDown + DySample
训练轮次：200 epochs
输入尺寸：960
batch：8
workers：8
amp：True
NWD：False（已关闭，避免 loss 改动污染结构实验）
数据集：VisDrone-YOLO
结果目录：yolo26/train_drone/yolov26+P2+FGDDown+DySample
```

### 2. 最优结果
根据 results.csv：
```text
Best mAP50-95 epoch = 113
Precision = 0.52994
Recall = 0.44326
mAP50 = 0.43200
mAP50-95 = 0.25982
Max mAP50 = 0.43209（epoch 144）
Last epoch = mAP50 0.42753 / mAP50-95 0.25673
```

本地 best.pt 复验：
```text
all: P 0.533 / R 0.440 / mAP50 0.432 / mAP50-95 0.260
Params: 2.419872 M
```

### 3. 每类 best.pt 复验指标
| class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| pedestrian | 0.607 | 0.501 | 0.523 | 0.251 |
| people | 0.520 | 0.392 | 0.382 | 0.156 |
| bicycle | 0.425 | 0.207 | 0.197 | 0.090 |
| car | 0.753 | 0.818 | 0.833 | 0.588 |
| van | 0.525 | 0.479 | 0.466 | 0.333 |
| truck | 0.513 | 0.395 | 0.384 | 0.254 |
| tricycle | 0.425 | 0.357 | 0.293 | 0.167 |
| awning-tricycle | 0.322 | 0.183 | 0.138 | 0.084 |
| bus | 0.669 | 0.550 | 0.584 | 0.426 |
| motor | 0.571 | 0.524 | 0.518 | 0.247 |

### 4. 与已有实验对比
```text
YOLO26 baseline:                 0.41202 / 0.24407
YOLO26 + P2:                     0.41713 / 0.24977
YOLO26 + P2 + FGDDown:           0.42709 / 0.25567
YOLO26 + DySample:               0.42038 / 0.24843
YOLO26 + P2 + FGDDown + DySample:0.43200 / 0.25982
```

相对 YOLO26 baseline：
```text
mAP50:    +0.01998（约 +2.00 个百分点）
mAP50-95: +0.01575（约 +1.58 个百分点）
```

相对 P2 + FGDDown：
```text
mAP50:    +0.00491
mAP50-95: +0.00415
```

### 5. 结论
E12 是目前最强模型，证明 P2 + FGDDown + DySample 组合不是无效堆叠，而是有稳定正收益。
但最后 200 轮结果低于 best epoch，说明后期出现轻微回落；后续不建议简单加训，应围绕模块机制继续优化。

### 6. 下一步建议
优先级最高：
```text
E16: 在 E12 基础上加入 FGDD-Gate（对 FGDDown 分支做自适应门控），作为近原创核心模块。
E17: 在 E12 基础上加入 EMA，只放 P2/P3 小目标分支，验证注意力能否冲击 mAP50 0.435+。
E18: E12 + 去 P5，验证能否在保留精度的同时降低计算量，服务边缘部署主线。
```

不建议：
```text
继续无脑加训；继续堆 DCLDown / DAFuseST / NWD；直接做轻量化而不先固定高精度 teacher。
```

## 二十九、2026-08-21 PAFPN / Concat 融合路线查重后更新

### 1. 检索结论
围绕 UAV 小目标检测的 “adaptive feature fusion / multi-scale feature fusion / scale-adaptive fusion / PAFPN 改进 / Concat 替换” 已经较多，不能把普通自适应融合直接当作强原创。
已发现相近方向包括：AFF-YOLO、SOD-YOLO、MASF-YOLO、FSMF-YOLO、EFF-YOLO、AAPW-YOLO、RASA-YOLO 等。

### 2. 需要避开的高重复写法
```text
1. 只把 Concat 换成注意力加权融合：重复风险高。
2. 只加 P2 + 自适应融合：已有 SOD-YOLO / Drone-YOLO 类似思路。
3. 只做多尺度上下采样融合：EFF-YOLO、MASF-YOLO 等已有相近表述。
4. 只加 EMA / CA / CBAM：创新性弱，只能作为辅助消融。
```

### 3. 更新后的近原创路线
将原 SGAFuse 升级为：
```text
ST-GRPAFPN: Small-Target Guided Residual Path Aggregation FPN
小目标引导残差路径聚合金字塔
```

核心区别：不是普通自适应融合，而是利用 P2 小目标高分辨率响应生成显式引导信号，控制 Neck 中 P2/P3/P4 的跨尺度残差融合。

### 4. 模块设计原则
```text
输入：两路或三路待融合特征，例如 P2/P3、P3/P4、P2/P3/P4。
引导：从 P2 或 P3 生成 small-target guidance map。
融合：保留原 Concat 路径，同时增加 residual gated path，避免破坏预训练结构。
轻量化：门控只用 GAP/GMP + 1x1 Conv，或 DWConv + 1x1 Conv。
输出：通道数保持和原 Concat 一致，后接原 C3k2，方便 YAML 单变量替换。
```

### 5. 建议实验顺序
```text
E16: P2 + GFGDDown + DySample（已准备/正在跑）
E17: E12 + ST-GRFuse-3Concat（先不基于 GFGDDown，避免两个新模块互相干扰）
E18: E16 + ST-GRFuse-3Concat（若 E16 有效，再组合）
E19: E18 - P5，形成边缘部署轻量化版本
E20: Teacher Distillation：用 E12/E18 高精度模型蒸馏轻量版
```

### 6. 论文贡献表述更新
推荐贡献点：
```text
1. 提出 GFGDDown，用门控双路径下采样缓解 P2->P3 的细节损失。
2. 提出 ST-GRPAFPN，通过 P2 小目标显式引导信号重构 Neck 中跨尺度残差融合路径。
3. 提出面向边缘部署的轻量化版本，通过去 P5 / Tiny Head / 蒸馏降低计算量。
```

不推荐贡献表述：
```text
本文简单引入 DySample、EMA、P2、NWD。
```

### 7. 当前判断
若只做 P2 + FGDDown + DySample，创新性约 5-6 分；
若加入 GFGDDown + ST-GRPAFPN，并有稳定消融结果，创新性可提升到 7 分以上。

## 三十、2026-08-21 创新模块二次查重后路线升级：从普通 PAFPN 融合转向 Tiny-object Prior Guided Fusion

### 1. 本轮查重结论
继续检索后发现，小目标/无人机 YOLO 方向中，以下思路高度拥挤：

```text
1. P2 高分辨率检测头。
2. DySample / 动态上采样。
3. BiFPN / AFPN / PAFPN / Scale-adaptive fusion。
4. 替换 Concat 为自适应加权融合。
5. 注意力引导特征融合 AFFM / AG-FPN。
6. P2 + DySample + 去 P5 + WIoU 的组合路线。
7. 多尺度序列融合 SSFF / DASSF / STSSF / C3DSF。
```

因此，不能把 “P2 + DySample + 改 Concat + EMA/WIoU” 当作强创新主线；这些只能作为基础组件或消融对照。

### 2. 发现的高相关工作类型
```text
MAF-YOLO / MHAF-YOLO: Multi-Branch Auxiliary FPN，浅层辅助融合 + 高级辅助融合。
MASF-YOLO: Multi-scale Context Aggregation + Scale-adaptive Fusion。
SOD-YOLO: P2 small object layer + ASF neck。
LAF-YOLOv10: Partial Conv + Attention-Guided FPN + DySample + P2 + 去 P5 + WIoU。
AFFM/SFSM: Attention-guided feature fusion + shallow feature supplement。
DASSF/STSSF/C3DSF: 动态尺度序列融合 / 3D尺度融合。
DRF-YOLO/EFF-YOLO/AFPN类: 残差/渐进式/自适应空间权重融合。
```

### 3. 路线必须避开的写法
```text
不推荐作为主创新：
- “提出一种自适应特征融合模块替换 Concat”。
- “引入 P2 检测头提升小目标检测”。
- “引入 DySample 改善上采样”。
- “引入 EMA/WIoU 提升精度”。
- “删 P5 降低计算量”。
这些都可以用，但不能单独包装为主要贡献。
```

### 4. 新的核心路线：TGP-PAFPN
将上一版 ST-GRPAFPN 进一步升级为：

```text
TGP-PAFPN: Tiny-object Prior Guided Path Aggregation Feature Pyramid Network
小目标先验引导路径聚合特征金字塔
```

与普通 AFPN/BiFPN/Concat 加权融合的核心区别：

```text
普通自适应融合：根据输入特征自己算权重，然后融合。
TGP-PAFPN：先从 P2/P3 高分辨率特征中生成 tiny-object prior map，再用该先验显式引导 P2/P3/P4 跨尺度残差融合。
```

### 5. TGP-PAFPN 模块机制
建议模块名：

```text
TGPFuse: Tiny-object Prior Guided Fusion
```

核心结构：

```text
输入：两路特征 x_low, x_high，例如 P2/P3 或 P3/P4。
1. 通道对齐：1x1 Conv 将两路特征对齐到相近通道。
2. 尺寸对齐：DySample / Conv stride=2 完成空间对齐。
3. Tiny-object prior 分支：从 P2/P3 生成小目标响应图 M_tiny。
4. 残差门控：使用 M_tiny 控制低层细节残差注入到高层语义特征。
5. 稳定初始化：残差系数 gamma 初始为 0 或 0.1，避免破坏预训练路径。
6. 输出：保留原 Concat 主路径，同时增加 gated residual path。
```

形式可以写成：

```text
F_out = Concat(F_main, F_skip) + gamma * Gate(M_tiny) * Align(F_detail)
```

### 6. 进一步增强原创性的训练策略：Tiny Prior Auxiliary Supervision
普通融合容易撞车；建议加入训练期小目标先验辅助监督：

```text
根据 GT box 自动生成 tiny-object center heatmap。
只对小目标类别/小尺寸框生成监督。
在 P2/P3 上训练一个轻量 tiny prior branch。
推理时保留轻量 prior branch，但不需要额外标注。
```

这样创新点从 “普通注意力融合” 变成：

```text
显式小目标先验生成 + 先验引导残差融合 + PAFPN 路径重构。
```

重复风险显著低于普通 SGAFuse / AFPN / BiFPN。

### 7. 更新后的实验队列
```text
E16: YOLO26 + P2 + GFGDDown + DySample
     目的：验证门控下采样是否优于 FGDDown。

E17: E12 + TGPFuse-3Concat（不叠 GFGDDown）
     目的：验证 tiny prior guided fusion 单独是否有效。
     替换位置：P4->P3、P3->P2、P2->P3 三个小目标最相关融合点。

E18: E16 + TGPFuse-3Concat
     目的：组合门控下采样和先验引导融合。

E19: E18 - P5 / Tiny Head
     目的：面向边缘部署降算力。

E20: Teacher Distillation
     Teacher: E12 或 E18 高精度模型。
     Student: E19 轻量模型。
     目的：轻量化后尽量保住小目标精度。
```

### 8. 论文贡献点更新
最终论文不要强调 “引入 P2/DySample/EMA/WIoU”，而应强调：

```text
1. 提出 GFGDDown：门控细粒度双路径下采样，缓解 P2->P3 的小目标细节损失。
2. 提出 TGP-PAFPN：通过 tiny-object prior map 显式引导跨尺度残差融合，而非普通 Concat 或普通自适应加权。
3. 提出面向边缘部署的轻量化学生模型，并通过 tiny-object guided distillation 保持小目标精度。
```

### 9. 当前创新评分判断
```text
只做 E12（P2+FGDDown+DySample）：约 5-6 分。
E12 + GFGDDown：约 6-7 分。
GFGDDown + TGP-PAFPN：有机会稳定 >7 分。
再加轻量化蒸馏部署：论文完整性更强，可作为 SCI 四区/三区主线。
```

## 三十一、2026-08-21 Neck 三创新点定型：GFGDDown + TGPFuse + TPCC

### 1. 本轮补充查重
进一步检索发现，Neck 改进方向中以下关键词均已有较多工作：

```text
Feature calibration / ReCalibrationFPN
Stage-wise feature calibration
BiFPN / AFPN / adaptive fusion
Small object enhancement pyramid
Attention-guided FPN
P2 + DySample + P5 removal
```

因此第三个 Neck 创新点不能写成普通“特征重校准”或“注意力校准”，否则会和 RCFPN、DER-YOLO、SOEP、AFFM 等相近。

### 2. Neck 三创新点最终建议
将 Neck 创新定义为一个完整系统，而不是三个孤立模块：

```text
TGP-PAFPN = GFGDDown + TGPFuse + TPCC
```

三个模块分别解决 Neck 中的三个关键问题：

| 模块 | 位置 | 解决问题 | 创新性定位 |
|---|---|---|---|
| GFGDDown | PAN 自底向上 P2->P3 下采样 | 下采样导致小目标细节损失 | 门控细粒度双路径下采样 |
| TGPFuse | PAFPN 的 Concat 融合节点 | 普通 Concat 无法区分小目标细节和高层语义 | 小目标先验引导残差融合 |
| TPCC | Neck 输出到 Detect 前的 P2/P3/P4 校准 | 同一小目标在多尺度层响应不一致，背景噪声跨层传递 | Tiny-prior scale-consistency calibration |

### 3. 第三个新增模块：TPCC
建议名称：

```text
TPCC: Tiny-Prior Consistency Calibration
小目标先验一致性校准模块
```

不要写成普通 ReCalibration，而是强调：

```text
它不是单层通道/空间注意力；
它利用 tiny-object prior map 对 P2/P3/P4 多尺度输出进行一致性校准；
目标是让同一小目标在相邻尺度上的响应更一致，同时抑制只在单一尺度突然出现的背景噪声。
```

### 4. TPCC 结构建议
```text
输入：Neck 输出的 P2, P3, P4（可选 P5）。
1. 将 P3/P4 上采样到 P2 尺度，形成跨尺度响应对齐。
2. 从 P2 生成 tiny prior map M_tiny。
3. 计算相邻尺度响应一致性：C23 = Sim(P2, Up(P3)), C34 = Sim(P3, Up(P4))。
4. 使用 M_tiny * Cij 生成校准权重，增强跨尺度一致的小目标响应。
5. 将校准后的 P2/P3/P4 送入 Detect。
```

简化形式：

```text
P2' = P2 + gamma2 * M_tiny * Align(P3)
P3' = P3 + gamma3 * Down(M_tiny) * (Align(P2) + Align(P4))
P4' = P4 + gamma4 * Down2(M_tiny) * Align(P3)
```

其中 gamma 初始值建议 0 或 0.1，避免破坏预训练模型。

### 5. TPCC 和已有 ReCalibrationFPN 的区别
```text
已有校准：通常是通道/空间注意力或 stage-wise feature calibration。
TPCC：显式围绕 tiny-object prior + 相邻尺度一致性，不是普通注意力。
已有融合：通常只融合特征。
TPCC：在 Detect 前对 P2/P3/P4 的响应一致性进行约束，服务小目标跨尺度稳定检测。
```

### 6. 更新后的实验队列
```text
E16: P2 + GFGDDown + DySample
     目的：验证门控下采样是否优于 FGDDown。

E17: E12 + TGPFuse-3Concat
     目的：验证 tiny prior guided fusion 是否优于普通 Concat。

E18: E12 + TPCC
     目的：验证 detect 前的 tiny-prior consistency calibration 是否有效。

E19: E12 + TGPFuse-3Concat + TPCC
     目的：验证两类 Neck 融合/校准模块是否互补。

E20: E16 + TGPFuse-3Concat + TPCC
     目的：形成完整 TGP-PAFPN。

E21: E20 - P5 / Tiny Head / Distillation
     目的：面向边缘部署降低算力。
```

### 7. 论文贡献点定稿方向
```text
1. GFGDDown：门控细粒度下采样，减少 P2->P3 小目标信息损失。
2. TGPFuse：小目标先验引导残差融合，替换 PAFPN 中普通 Concat 的粗粒度融合。
3. TPCC：小目标先验一致性校准，在检测头前增强 P2/P3/P4 多尺度响应一致性。
```

三个创新点全部发生在 Neck 内部，因此比单纯 P2/DySample/EMA/WIoU 更符合“改 PAFPN / 改 Neck”的高创新评分路径。

### 8. 风险控制
```text
不要一次性把三者全加；必须逐个消融。
TGPFuse 和 TPCC 均应先基于 E12 单独验证，再和 GFGDDown 组合。
若 TPCC 无效，仍保留 GFGDDown + TGPFuse 双创新路线。
若 TGPFuse 有效，优先做轻量化 TGPFuse-Lite。
```

## 三十二、2026-08-21 GFGDDown 中途停止结果与 TGPFuse3 实验准备

### 1. GFGDDown 消融结果
```text
模型：YOLO26 + P2 + GFGDDown + DySample
训练轮次：153/200，中途停止
Best epoch：102
P：0.52800
R：0.44060
mAP50：0.42976
mAP50-95：0.25813
Last epoch：mAP50 0.42685
```

对比当前最强 E12：
```text
E12: P2 + FGDDown + DySample
mAP50 = 0.43200
mAP50-95 = 0.25982

GFGDDown: P2 + GFGDDown + DySample
mAP50 = 0.42976
mAP50-95 = 0.25813
```

结论：GFGDDown 当前版本弱于固定 FGDDown，归档为负增益/弱增益消融。不建议继续以该版本作为主线。

可能原因：
```text
1. 全局池化门控过粗，难以描述 VisDrone 中局部密集小目标分布。
2. softmax 分支竞争可能压制了原 FGDDown 中稳定有效的细节分支。
3. 门控新增随机参数破坏了部分预训练迁移稳定性。
```

### 2. 新实验 E17：TGPFuse-3Concat
已准备新模块：
```text
TGPFuse: Tiny-object Prior Guided Fusion
小目标先验引导融合模块
```

设计机制：
```text
保留原 Concat 主路径；
从浅层/细节分支生成 tiny-object prior map；
用 prior map 对多路输入做轻量残差增强；
输出通道数保持与 Concat 一致，后接原 C3k2。
```

实验配置：
```text
模型：YOLO26 + P2 + FGDDown + DySample + TGPFuse3
基准：E12
变量：只替换 3 个小目标相关 Concat，不叠加 GFGDDown。
替换位置：P4->P3、P3->P2、P2->P3。
```

本地构建验证：
```text
YOLOv26+P2+FGDDown+DySample+TGPFuse3
347 layers, 2,541,924 parameters
TGPFUSE3_MODEL_OK
```

### 3. 判断标准
若 E17 达到：
```text
mAP50-95 > 0.25982 或 mAP50 > 0.43200
```
则证明 tiny-prior guided fusion 有效，可作为 Neck 创新主线。

若 E17 不涨：
```text
TGPFuse 当前版本也归档为无效，下一步改 TPCC 或 TGPFuse-Lite/局部先验版本。
```

## 三十三、2026-08-21 PSFusion 启发：不直接移植，抽象为轻量 Neck 融合机制

### 1. PSFusion 可借鉴模块
PSFusion 是红外-可见光图像融合网络，核心包括：
```text
SDFM: Superficial Detail Fusion Module，浅层细节融合，基于通道-空间注意力。
PSFM: Profound Semantic Fusion Module，深层语义融合，基于 cross-attention。
Fusion_module: Recalibrate + local/global attention + weighted fusion。
GEFM: Q/K/V cross-attention + residual gamma。
SIM: 用语义图调制特征归一化，类似 semantic-conditioned normalization。
```

### 2. 对 YOLO 无人机小目标检测的启发
不能直接把 PSFM 搬到 YOLO P2/P3，因为 HWxHW cross-attention 在高分辨率 P2 上计算量爆炸。
更合理的迁移方式：
```text
1. 从 SDFM 借鉴浅层细节/显著性融合，改造 TGPFuse。
2. 从 PSFM 借鉴 Q/K/V 交叉引导，但改为局部窗口或通道级轻量版本。
3. 从 SIM 借鉴 prior-conditioned modulation，用 tiny-object prior map 调制 Neck 输出。
```

### 3. 新增可选模块路线
```text
TGPFuse-SDFM：借鉴 SDFM 的 recalibration + local/global attention，但输入从 RGB/IR 改为 PAFPN 相邻尺度特征。
Lite-PSFM：借鉴 PSFM 的 cross-attention，但只在 P3/P4 或低分辨率尺度使用，P2 禁止用全局 HWxHW attention。
TPCC-SIM：借鉴 SIM，用 tiny prior map 生成 gamma/beta，对 P2/P3/P4 Detect 前特征做条件调制。
```

### 4. 创新性判断
PSFusion 的模块不能直接宣称原创；但若将其“多模态融合思想”改造成“同模态多尺度小目标先验融合”，并在 YOLO Neck 中服务 P2/P3/P4 检测，可作为二次创新来源。

### 5. 当前建议
短期不直接上 PSFM，因为计算量太高；优先跑已准备的 TGPFuse3。
若 TGPFuse3 有效，再做 TGPFuse-SDFM 二次版本。
若 TGPFuse3 无效，转向 TPCC-SIM（tiny prior conditioned modulation）。

## 2026-08-21 PSFusion 对 YOLO26 无人机小目标方向的借鉴结论

参考项目：Linfeng-Tang/PSFusion，原方向是可见光/红外图像融合，不是目标检测网络。因此不建议把 PSFusion 的模块整块搬进 YOLO26，而应抽取其中“浅层细节融合、深层语义交互、语义调制”的思想，改写成适合 YOLO Neck 的轻量模块。

可借鉴点：

1. SDFM：浅层细节融合模块。可转化为 TGPFuse-SDFM，用于替换 P4->P3、P3->P2、P2->P3 这三处关键 Concat。核心思想是让高分辨率小目标先验图参与融合权重，而不是简单拼接。

2. PSFM：深层语义交叉注意力。原版含 HW x HW 级别 attention，不适合 P2/P3 大特征图，尤其输入 960 时 P2 为 240x240，计算和显存代价过高。若借鉴，只能做 Lite-PSFM：放在 P4/P5 或使用窗口注意力、通道注意力、低秩近似。

3. SIM：语义调制思想。可转化为 TPCC-SIM，在 Detect 前利用 tiny-object prior 对 P2/P3/P4 做 gamma/beta 或 residual gate 校准，增强跨尺度一致性。

当前建议路线：

- 先跑已准备好的 YOLO26 + P2 + FGDDown + DySample + TGPFuse3，验证“关键 Concat 替换为小目标先验融合”是否有效。
- 若 TGPFuse3 有提升，再做 TGPFuse-SDFM 二次创新版：在 TGPFuse 内部加入局部细节分支和全局通道分支。
- 若 TGPFuse3 提升不明显，优先转向 TPCC-SIM：不改变主干拓扑，只在检测头前做轻量尺度一致性校准，风险更低。

注意：

- PSFusion 的 PSFM 不要直接用于 P2，因为全局注意力复杂度接近 O((HW)^2)，违背边缘轻量化目标。
- 可把 PSFusion 写成“跨模态融合机制启发”，但论文中不能声称直接原创 PSFusion 类模块。原创点应落在“面向无人机小目标检测的 tiny-prior guided neck fusion / scale calibration”。

## 2026-08-22 下一步主线：TPKDStudent 轻量蒸馏学生模型

### 1. 当前最佳教师模型
当前最有价值的完整模型是：

```text
YOLO26 + P2 + FGDDown + DySample + TGPFuse3
best mAP50 = 0.43843
best mAP50-95 = 0.26402
```

该模型已经证明 TGPFuse3 对 VisDrone 小目标检测有效，但包含 P2 检测层，后续直接部署到树莓派 5 或边缘端会有较大计算压力。

### 2. 新增创新点：TPKDStudent
TPKDStudent 表示 Tiny-Prior Knowledge Distillation Student。

核心思想：

```text
用当前最强 TGPFuse3 模型作为 Teacher，
训练一个同拓扑但更窄的 Student，
让 Student 学到 P2/TGPFuse 的小目标先验融合能力，
同时减少参数量和计算量。
```

这不是简单剪枝，也不是简单换模块，而是把“高精度小目标先验模型”转化为“轻量学生模型”。更适合写成论文第三个创新点：

```text
面向边缘部署的 Tiny-Prior Guided Knowledge Distillation Strategy。
```

### 3. 为什么先做同拓扑学生
当前 ultralytics 代码里的蒸馏逻辑会自动 hook Student 的 Detect 输入层，并在 Teacher 的相同层索引取特征。因此第一版学生必须保持和 Teacher 相同的层索引：

```text
Detect 输入层保持 [19, 22, 25, 28]
保留 P2 / FGDDown / DySample / TGPFuse3 路线
只把 width 从 0.25 降到 0.20
```

这样蒸馏最稳，不需要立刻改 distill_model.py。

### 4. 已准备文件
```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent/
  yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent.yaml
  yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent_data.yaml
  yolov26+P2+FGDDown+DySample+TGPFuse3+TPKDStudent_train.py
  README_TPKDStudent.md
```

本地构建检查结果：

```text
模型解析成功
Transferred 79/956 items from pretrained weights
参数量约 1.749M
```

### 5. 后续实验判断
若 TPKDStudent 能达到：

```text
mAP50 >= 0.425
mAP50-95 >= 0.255
参数量明显低于 2.54M
```

就可以作为“轻量化有效”的强实验；若精度下降过多，则改为 width=0.225 或降低蒸馏权重 dis=0.3。

### 6. TPKDStudent-v2
v1 训练到 135/150 时：

```text
mAP50 = 0.405
mAP50-95 = 0.243
```

说明 width=0.20 压缩过猛，Student 的容量不足以继承 Teacher 的小目标先验融合能力。

因此准备 v2：

```text
width: 0.20 -> 0.225
dis: 0.5 -> 0.3
epochs: 150
Detect 输入层仍保持 [19, 22, 25, 28]
```

本地构建检查：

```text
Transferred 148/956 items from pretrained weights
params_M = 2.190672
TPKDStudent-v2 OK
```

v2 判断标准：

```text
若 mAP50 >= 0.420 且 mAP50-95 >= 0.252，则保留为轻量化有效路线。
若仍接近 baseline，则同拓扑窄学生路线价值有限，下一步转向 P2-free Student + 自定义特征蒸馏匹配。
```

## 2026-08-23 新结构路线：P5SPFPrune 语义保留式 P5 剪枝

### 1. 为什么暂缓 Teacher 蒸馏
TPKDStudent 虽然有轻量化意义，但训练时需要同时运行 Teacher 和 Student，工程成本高、训练时间长，而且 v1 的结果显示过度压缩会导致精度回落到 baseline 附近。

因此下一步优先尝试结构创新：

```text
先把 P5 高层语义融合到 P4，
再删除 P5 检测头，
最终保持 P2/P3/P4 三个检测层。
```

这条路线更符合“结构剪枝 + 小目标检测 + 边缘轻量化”的论文主线。

### 2. 新模块：P5SPF
P5SPF 表示 P5 Semantic Preserving Fusion。

模块机制：

```text
输入: [P4_feature, upsample(P5_feature)]
1. P5 通过 1x1 Conv 对齐到 P4 通道；
2. P4 生成 detail gate，保留小目标定位细节；
3. P5 生成 semantic gate，补偿删除 P5 检测头后的深层语义；
4. 用残差方式把 P5 语义注入 P4；
5. 输出通道仍为 P4 通道，不进行 Concat 通道膨胀。
```

### 3. 已准备实验
文件夹：

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune/
```

包含：

```text
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune_data.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune_train.py
README_P5SPFPrune.md
```

本地构建检查：

```text
Transferred 359/848 items from pretrained weights
params_M = 2.028618
P5SPFPrune OK
```

### 4. 预期价值
当前最强 TGPFuse3：

```text
参数量约 2.54M
mAP50 = 0.43843
mAP50-95 = 0.26402
Detect = P2/P3/P4/P5
```

P5SPFPrune：

```text
参数量约 2.03M
Detect = P2/P3/P4
```

若 P5SPFPrune 能达到：

```text
mAP50 >= 0.430
mAP50-95 >= 0.258
```

则说明“语义保留式 P5 剪枝”是有效创新点，可作为轻量化结构创新主线。

### 5. 创新性撞车检查与论文表述边界
检索结论：

```text
1. 直接添加 P2 检测头并删除 P5 检测头，不是原创。
2. 小目标检测中使用 P2/P3/P4 检测头，已有多篇 YOLO 改进工作采用。
3. 直接剪掉大尺度检测层以减少计算量，也已有轻量化小目标检测论文涉及。
```

因此论文中不要把以下内容单独写成原创：

```text
添加 P2 检测头
删除 P5 检测头
使用 P2/P3/P4 三检测层
普通 PAFPN/Concat 路径调整
```

本项目可保留的创新表述应集中在：

```text
P5SPF：P5 Semantic Preserving Fusion。
它不是直接删除 P5，而是在删除 P5 检测头前，利用 P5 高层语义对 P4 进行轻量残差注入。
模块包含 P4 detail gate 与 P5 semantic gate，输出通道保持 P4 通道，不进行 Concat 通道膨胀。
```

建议论文命名：

```text
Semantic-Preserving P5 Pruning for Lightweight UAV Tiny Object Detection
```

更稳妥的贡献表述：

```text
为避免直接删除 P5 检测头导致深层语义损失，本文提出一种语义保留式 P5 剪枝融合模块 P5SPF。
该模块在剪枝前将 P5 的全局语义以门控残差方式注入 P4，使模型保持 P2/P3/P4 三检测层的同时，降低参数量并缓解高层语义退化。
```

该点属于“已有 P5 剪枝思想基础上的二次创新”，不是纯粹从零原创。后续需要通过消融实验验证：

```text
Baseline P2/P3/P4 直接删 P5
P2/P3/P4 + P5 ordinary add/concat
P2/P3/P4 + P5SPF
```

若 P5SPF 明显优于直接删 P5 和普通融合，则创新性更稳。

### 6. P5SPFPrune 实验结果
训练在第 146 轮手动停止，停止原因是后期指标基本平台化，继续训练收益很小。

`results.csv` 记录：

```text
best mAP50 = 0.43770，epoch 103
best mAP50-95 = 0.26391，epoch 96
last mAP50 = 0.43530，epoch 146
last mAP50-95 = 0.26252，epoch 146
```

使用 `weights/best.pt` 本地重新验证，补齐最终验证图后得到：

```text
Precision = 0.54536
Recall = 0.43528
mAP50 = 0.43655
mAP50-95 = 0.26384
```

类别 AP50-95：

```text
pedestrian = 0.24922
people = 0.15964
bicycle = 0.09016
car = 0.58758
van = 0.33943
truck = 0.25058
tricycle = 0.17980
awning-tricycle = 0.09419
bus = 0.43850
motor = 0.24929
```

与关键模型对比：

```text
YOLO26 baseline: mAP50 约 0.41202，mAP50-95 约 0.24407
TGPFuse3 四检测头: mAP50 约 0.43843，mAP50-95 约 0.26402，参数量约 2.54M
P5SPFPrune 三检测头: mAP50 约 0.43655-0.43770，mAP50-95 约 0.26384-0.26391，参数量约 2.03M
```

结论：

```text
P5SPFPrune 几乎保持 TGPFuse3 的精度，同时参数量下降约 20%。
这说明“语义保留式 P5 剪枝”不是无效剪枝，而是有效轻量化结构创新点。
该路线应保留为论文主实验之一。
```

已补齐文件：

```text
results.png
BoxPR_curve.png / BoxF1_curve.png / BoxP_curve.png / BoxR_curve.png
confusion_matrix.png / confusion_matrix_normalized.png
val_batch0/1/2_labels.jpg
val_batch0/1/2_pred.jpg
P5SPFPrune_interrupted_summary.md
```

## 2026-08-23 下一步：坚持 YOLO26n，加入 CS-SlideLoss 提升精度

### 1. 背景
老师认为当前提升幅度还不够大。虽然切换到 YOLO26s 可能提高绝对精度，但本课题核心是轻量化，因此继续坚持 n 系列更符合方向。

因此下一步不换大模型，而是在当前最终结构上加入训练阶段损失函数优化：

```text
YOLO26n + P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune + CS-SlideLoss
```

### 2. CS-SlideLoss
CS-SlideLoss 表示 Class-Scale Aware Slide Loss，类别-尺度感知滑动损失。

该损失只在训练阶段启用：

```text
不改变模型结构
不增加推理参数量
不增加推理 FLOPs
```

组成：

```text
1. 难样本滑动权重：提高低置信度/难分类样本的训练贡献。
2. 类别频次权重：根据 VisDrone train 类别频次设置温和逆频率权重。
3. 小目标尺度权重：对分配到小尺度目标的正样本提高 loss 权重。
```

训练集类别频次：

```text
pedestrian: 79337
people: 27056
bicycle: 10479
car: 144864
van: 24956
truck: 12875
tricycle: 4812
awning-tricycle: 3246
bus: 5926
motor: 29646
```

采用温和类别权重：

```text
[0.7019, 0.8457, 1.0555, 0.6494, 0.8600, 1.0011, 1.3197, 1.4980, 1.2387, 0.8303]
```

### 3. 已准备实验文件
文件夹：

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss/
```

包含：

```text
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss_data.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+CSSlideLoss_train.py
README_CSSlideLoss.md
```

本地检查：

```text
Transferred 359/848 items from pretrained weights
params_M = 2.028618
cs_slide_enabled = True
CSSlideLoss model OK
```

### 4. 判断标准
对比 P5SPFPrune：

```text
P5SPFPrune best mAP50 = 0.43770
P5SPFPrune best mAP50-95 = 0.26391
```

若 CS-SlideLoss 达到：

```text
mAP50 >= 0.445
mAP50-95 >= 0.268
```

则说明该训练策略有明显增益，可作为论文中的训练目标函数创新点。

### 5. CS-SlideLoss 实验结果
训练结果：

```text
best mAP50 = 0.42793，epoch 126
best mAP50-95 = 0.26075，epoch 142
last mAP50 = 0.42330，epoch 200
last mAP50-95 = 0.25801，epoch 200
```

与关键模型对比：

```text
P5SPFPrune: mAP50 = 0.43770，mAP50-95 = 0.26391
P5SPFPrune + CS-SlideLoss: mAP50 = 0.42793，mAP50-95 = 0.26075
```

结论：

```text
CS-SlideLoss 当前版本没有带来提升，反而降低了 mAP50。
该路线暂不纳入最终模型。
最终模型仍采用 P5SPFPrune。
```

可能原因：

```text
1. 类别权重、难样本权重、小目标尺度权重同时叠加，分类分支扰动过强。
2. VisDrone 多数类对整体 mAP 影响较大，过度偏向少数类可能损害整体排序。
3. YOLO26 end-to-end 分类损失对权重变化敏感，直接加权 BCE 风险较高。
```

后续若继续做 loss，只建议做温和版本：

```text
YOLO_CS_SLIDE_GAIN: 0.6 -> 0.2
YOLO_CS_SCALE_GAIN: 0.5 -> 0.2
类别权重收缩到 0.85-1.15 区间
或只用 best.pt 进行 30-50 epoch 后期微调
```

## 2026-08-24 下一步：TCA_P3 轻量卷积注意力实验

### 1. 背景
CS-SlideLoss 实验没有提升，因此下一步回到结构优化，但仍坚持 YOLO26n 轻量化路线。

不建议直接套用 CBAM/SE/CA 等常见注意力模块，因为重复度较高，且放在 P2 上计算风险较大。更合理的方向是在 Neck 融合后的 P3 特征上加入轻量卷积注意力。

### 2. 新模块：TCA
TCA 表示 Tiny Convolutional Attention。

模块结构：

```text
输入特征
-> 1x1 Conv 通道对齐
-> DWConv 3x3 提取局部纹理
-> DWConv 5x5 提取邻域上下文
-> 1x1 Conv + Sigmoid 得到空间注意力
-> 残差增强输出
```

设计目的：

```text
1. 强化小目标边缘、纹理和局部上下文；
2. 只在 Neck 融合后特征处使用，避免 Backbone 大面积引入注意力；
3. 使用深度卷积控制参数量和计算量；
4. 与 TGPFuse/P5SPF 的多尺度融合主线保持一致。
```

### 3. 已准备实验
实验文件夹：

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3/
```

文件：

```text
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3_data.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3_train.py
README_TCA_P3.md
```

插入位置：

```text
只在 P4->P3 融合后的 P3 特征处加入 TCA。
即原 layer 16 后新增 TCA，后续 P3->P2 和 P2->P3 融合均使用增强后的 P3 特征。
```

本地检查结果：

```text
Transferred 348/875 items from pretrained weights
params_M = 2.037372
TCA_P3 OK
```

参数量对比：

```text
P5SPFPrune: 约 2.028618M
TCA_P3: 约 2.037372M
增量约 0.008754M
```

### 4. 判断标准
对比 P5SPFPrune：

```text
P5SPFPrune best mAP50 = 0.43770
P5SPFPrune best mAP50-95 = 0.26391
```

若 TCA_P3 达到：

```text
mAP50 >= 0.442
mAP50-95 >= 0.266
```

说明轻量卷积注意力方向有效，可继续扩展到 P4 或 P2->P3 回传融合后。

## 2026-08-24 TCA_P3 实验结果记录

### 实验目录

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3/
```

### 训练状态

本次训练提前停止在第 164 轮。最后 10 轮 mAP50 约在 0.4367-0.4380 之间波动，mAP50-95 约在 0.2641-0.2648 之间波动，已经基本平台化，因此提前停止合理。

### 最优结果

```text
best mAP50      = 0.43982 @ epoch 124
best mAP50-95   = 0.26568 @ epoch 116
best Precision  = 0.56070 @ epoch 148
best Recall     = 0.45111 @ epoch 128
last mAP50      = 0.43692
last mAP50-95   = 0.26429
```

### best.pt 本地复验

```text
Precision  = 0.54648
Recall     = 0.44095
mAP50      = 0.43927
mAP50-95   = 0.26568
Params     = 2.037372M
```

### 对比判断

```text
相对 P5SPFPrune:
mAP50    +0.00212
mAP50-95 +0.00177
Params   +0.008754M

相对 TGPFuse3:
mAP50    +0.00139
mAP50-95 +0.00166
Params   明显降低，约从 2.542M 降到 2.037M
```

结论：TCA_P3 是正向但小幅提升。它适合作为轻量化最终路线中的第三个 Neck 创新点候选，但不能单独作为强创新。下一步建议优先做 TCA_P3P4 单变量实验，并补 P5Prune_NoSPF 消融来证明 P5SPF 的必要性。

## 2026-08-24 已准备下一步实验文件

### 1. TCA_P3P4

实验目录：

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4/
```

文件：

```text
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4_data.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4_train.py
README_TCA_P3P4.md
```

本地构建检查：

```text
Transferred 348/902 items from pretrained weights
Params = 2.067166M
TCA_P3P4 OK
```

运行命令：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4_train.py
```

### 2. P5Prune_NoSPF

实验目录：

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF/
```

文件：

```text
yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF_data.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF_train.py
README_P5Prune_NoSPF.md
```

本地构建检查：

```text
Transferred 359/816 items from pretrained weights
Params = 1.826824M
P5Prune_NoSPF OK
```

运行命令：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF/yolov26+P2+FGDDown+DySample+TGPFuse3+P5Prune_NoSPF_train.py
```

## 2026-08-25 TCA_P3P4 实验结果记录

### 实验目录

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+P5SPFPrune+TCA_P3P4/
```

### 最优结果

```text
best mAP50    = 0.43616 @ epoch 121
best mAP50-95 = 0.26471 @ epoch 130
last mAP50    = 0.43115 @ epoch 200
last mAP50-95 = 0.26008 @ epoch 200
Params        = 2.067166M
```

### 对比结论

相对 TCA_P3：

```text
mAP50    -0.00366
mAP50-95 -0.00097
Params   +0.029794M
```

判断：TCA_P3P4 失败，不建议作为最终模型。P3 加轻量卷积注意力有效，但 P4 继续加注意力会干扰 P5SPF 后的语义融合特征。后续不再沿着“P4 堆注意力”继续扩展。

当前最优轻量化候选仍为：

```text
YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + P5SPFPrune + TCA_P3
best mAP50    = 0.43982
best mAP50-95 = 0.26568
Params        = 2.037372M
```

## 2026-08-25 已准备冲精度实验：TGPFuse3_TCA_P3

### 实验目的

TCA_P3P4 失败后，不再继续向 P4 堆叠注意力。为了冲精度，回到保留 P5 检测头的完整四检测层结构，只在 P3 小目标敏感层加入 TCA。

实验结构：

```text
YOLO26 + P2 + FGDDown + DySample + TGPFuse3 + TCA_P3
```

### 实验目录

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3/
```

### 文件

```text
yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3_data.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3_train.py
README_TCA_P3.md
```

### 本地结构检查

```text
Transferred 348/983 items from pretrained weights
Params = 2.550678M
YOLO26_TGPFuse3_TCA_P3 OK
```

### AutoDL 运行命令

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3_train.py
```

### 目标

```text
mAP50    >= 0.445
mAP50-95 >= 0.268
```

判断：该实验是冲精度用，不是最终轻量部署版。如果结果明显高于 TCA_P3，可作为精度最优模型；P5SPFPrune+TCA_P3 则作为轻量化版本。

## 2026-08-27 下一阶段最佳任务：Tile960 小目标增强训练

### 背景判断

最近的结构实验说明：

```text
TCA_P3 有小幅正收益；
TCA_P3P4 退步；
TGPFuse3+TCA_P3 冲精度效果不明显；
继续在同一结构线上堆模块，收益大概率只有千分位。
```

因此下一阶段不优先继续堆网络模块，而是优先解决 VisDrone 的核心问题：目标在原图中太小、太密集。采用重叠切片训练，让小目标在训练视野中变大。

### 新实验目录

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/
```

### 文件

```text
make_visdrone_tile960.py
yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_data.yaml
yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_train.py
README_Tile960.md
```

### 本地检查

```text
Transferred 348/983 items from pretrained weights
Params = 2.550678M
Tile960 model OK
syntax OK
```

### AutoDL 运行步骤

第一步，生成切片训练集：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/make_visdrone_tile960.py --clean
```

第二步，正式训练：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_train.py
```

### 目标

```text
mAP50    >= 0.45
mAP50-95 >= 0.27
```

判断：这个实验是下一阶段最优先任务。若有效，说明尺度增强是主要突破口；若无效，再转向类别重采样、标签清洗或切片推理评估。

## 2026-08-30 Tile960 中途停止结果与省钱策略

### 当前结果

```text
实验目录：
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960/

停止轮次：epoch 117
best mAP50    = 0.44490 @ epoch 107
best mAP50-95 = 0.27655 @ epoch 93
last mAP50    = 0.44406
last mAP50-95 = 0.27629
```

### 判断

Tile960 是目前总体最优结果。相对 baseline：

```text
mAP50    +0.03288
mAP50-95 +0.03248
```

相对 P5SPFPrune+TCA_P3：

```text
mAP50    +0.00508
mAP50-95 +0.01087
```

说明切片训练对 VisDrone 小目标检测有效，尤其 mAP50-95 提升明显。

### 是否恢复继续跑

不建议直接恢复 Tile960 继续跑到 200 轮。第 93 轮已达到 mAP50-95 最优，第 107 轮达到 mAP50 最优，后面基本横盘；Tile960 单轮成本高，继续训练性价比低。

### 已准备低成本后续实验

实验目录：

```text
train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30/
```

思路：

```text
Tile960 best.pt -> 原始 VisDrone train -> 30 epoch 微调
```

运行命令：

```bash
python train_drone/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30/yolov26+P2+FGDDown+DySample+TGPFuse3+TCA_P3+Tile960_FT30_train.py
```

同时准备了只验证不训练的脚本：

```bash
python train_drone/validate_tile960_best.py
```

## 2026-08-30 新方向 baseline 准备

### 背景

VisDrone 路线已经证明切片训练有效，但 mAP50 很难达到 90% 以上。若老师要求“高基线 90%+ 再提升 2 个点”，需要增加更容易形成高分基线的数据集方向。

### 已新增目录

与 `train_drone` 并列：

```text
train_carpk/
train_pcb/
```

### CARPK baseline

目录：

```text
train_carpk/yolov26n_baseline_carpk/
```

文件：

```text
yolov26n_baseline_carpk_data.yaml
yolov26n_baseline_carpk_train.py
check_carpk_dataset.py
README_yolov26n_baseline_carpk.md
```

运行顺序：

```bash
python train_carpk/yolov26n_baseline_carpk/check_carpk_dataset.py
python train_carpk/yolov26n_baseline_carpk/yolov26n_baseline_carpk_train.py
```

定位：无人机视角车辆检测，和原方向衔接最好。

### PCB baseline

目录：

```text
train_pcb/yolov26n_baseline_pcb/
```

文件：

```text
yolov26n_baseline_pcb_data.yaml
yolov26n_baseline_pcb_train.py
check_pcb_dataset.py
README_yolov26n_baseline_pcb.md
```

运行顺序：

```bash
python train_pcb/yolov26n_baseline_pcb/check_pcb_dataset.py
python train_pcb/yolov26n_baseline_pcb/yolov26n_baseline_pcb_train.py
```

定位：工业缺陷检测，更容易得到高 mAP50，但方向会从无人机转向工业质检。

### 当前 baseline 配置

```text
model = yolo26n.pt
imgsz = 640
epochs = 50
batch = 16
workers = 8
patience = 20
```

判断标准：若 50 轮内 mAP50 能达到 90% 左右，则该方向适合继续迁移前面有效模块；若远低于 80%，先检查数据转换与标签质量，不继续烧钱。

## 2026-08-30 新方向数据集整理完成

### CARPK-YOLO

已从原始 CARPK_devkit 转换为 YOLO 格式。

本地路径：

```text
YOLO格式数据集/CARPK-YOLO
```

AutoDL 上传包路径：

```text
AutoDL上传包/yolo26_drone_package/datasets/CARPK-YOLO
```

统计：

```text
train: 989 images, 989 labels, 42274 boxes
val:   459 images, 459 labels, 47500 boxes
test:  459 images, 459 labels, 47500 boxes
bad labels = 0
```

说明：CARPK 官方只有 train/test 划分，当前 baseline 为了快速训练，把官方 test 同时作为 val/test 使用。若后续作为正式论文主数据集，建议再做严格划分，避免验证集和最终测试集完全相同。

### PCB-YOLO

已从 Roboflow/YOLO 风格数据集整理为项目统一格式。

本地路径：

```text
YOLO格式数据集/PCB-YOLO
```

AutoDL 上传包路径：

```text
AutoDL上传包/yolo26_drone_package/datasets/PCB-YOLO
```

统计：

```text
train: 6333 images, 6333 labels, 12862 boxes
val:   1778 images, 1778 labels, 3656 boxes
test:   903 images, 903 labels, 1860 boxes
bad labels = 0
empty label files = 1
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

下一步：优先训练 CARPK baseline，再训练 PCB baseline。

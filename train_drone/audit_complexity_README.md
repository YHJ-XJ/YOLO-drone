# 复杂度审计脚本使用说明

`audit_complexity.py` 用于从已训练的 `best.pt` 统一重算参数量与 FLOPs，解决消融表中 nc 口径不一致的问题。

## 问题背景

历史消融表中：
- `yolov26+P2` 和 `yolov26+P2+FGDDown` 的 YAML 配置为 `nc: 80`
- 后续实验 YAML 配置为 `nc: 10`

这导致：
- 表中"DySample 降低参数量和计算量"是类别数差异，不是模块作用
- 部分行的 FLOPs 偏大约 1.9 GFLOPs（80 类 vs 10 类检测头差异）
- P2 相对 baseline 的增幅需重算，不能预判方向

## 脚本功能

从每个 `best.pt` 读取并统计：

1. **实际类别数核验**：`nc_yaml` vs `nc_head` vs `len(model.names)`
2. **三种推理口径**：
   - 未融合双头（训练时）
   - 融合后仅 o2o（NMS-free）
   - 最终导出后端实际图（待补充）
3. **逐层分解**：backbone、P2 Neck、其他 Neck、检测头
4. **未统计算子标注**：grid_sample（DySample 使用）、门控逐元素乘等函数式运算

## 运行方式

**在 AutoDL 上运行**（本地 Python 环境没有 torch）：

```bash
cd /root/autodl-tmp/yolo26_drone_package/yolo26
python train_drone/audit_complexity.py
```

默认扫描 `train_drone/**/weights/best.pt`（23 个权重）。

## 输出结果

输出到 `train_drone/audit_complexity/`：

### 1. `summary.csv`

每行一个实验，列包括：
- `name`: 实验目录名
- `nc_yaml`, `nc_head`, `n_names`: 类别数核验
- `levels`: 检测层（如 `P2/P3/P4`）
- `end2end`: 是否有 o2o 头
- `params_unfused`, `params_fused`: 融合前后参数量
- `o2m_removed`: o2m 头是否已移除
- `gflops_unfused@640`, `gflops_fused@640`: 640 输入的 GFLOPs
- `gflops_unfused@960`, `gflops_fused@960`: 960 输入的 GFLOPs
- `gflops_fused_ultra@960`: Ultralytics `get_flops()` 的结果（对照）
- `detect_o2m_unfused@960`, `detect_o2o_fused@960`: 检测头 GFLOPs
- `backbone_fused@960`, `neck_P2_fused@960`, `detect_P2_fused@960`: 分阶段 GFLOPs
- `gridsample_gflops_est@960`: grid_sample 估算（7 FLOPs/elem，未并入总数）
- `uncounted@960`: 未统计算子列表（DySample/TGPFuse/TCA/P5SPF 等）

### 2. `per_layer.csv`

逐层记录，列包括：
- `name`, `tag` (unfused/fused), `imgsz`, `idx`, `type`, `stage`, `out_stride`, `params`, `gflops`

### 3. `report.md`

Markdown 报告，表格格式：

| 实验 | nc head/yaml | 检测层 | Params fused/M | GFLOPs fused@640 | GFLOPs fused@960 | P2 Neck+Detect 占比@960 | grid_sample 估算@960 | o2m 已移除 |
|---|---|---|---:|---:|---:|---:|---:|---|

标注类别数异常、grid_sample 估算值、o2m 是否移除。

## 注意事项

1. **THOP 约定**：MACs×2，batch=1，全尺寸输入（非 stride 代理）
2. **未统计算子**：
   - `grid_sample`（DySample）：按 7 FLOPs/elem 估算，未并入总数
   - 门控逐元素乘（TGPFuse/TCA/P5SPF）：THOP 不计数
3. **融合检查**：脚本检查 `fuse()` 是否移除 o2m 头，若未移除会标注
4. **部署口径**：当前只统计 Python 推理图，最终部署图需导出后端（ONNX/NCNN/TRT）实测

## 后续工作

1. 用此脚本输出修正 `ablation_table_paper_style.md` 和 `ablation_summary_p5spf.md`
2. 对比修正前后的 P2 增幅百分比
3. 补充部署后端导出与 Pi 5 延迟测试

## 依赖

- Python 3.8+
- PyTorch
- ultralytics
- thop（`pip install ultralytics-thop`）

## 代码版本

脚本会记录 Git commit（`git rev-parse --short HEAD`）到结果中，便于追溯代码版本。

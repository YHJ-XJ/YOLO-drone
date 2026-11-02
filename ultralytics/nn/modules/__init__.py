# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics neural network modules.

This module provides access to various neural network components used in Ultralytics models, including convolution
blocks, attention mechanisms, transformer components, and detection/segmentation heads.

Examples:
    Visualize a module with Netron
    >>> from ultralytics.nn.modules import Conv
    >>> import torch
    >>> import subprocess
    >>> x = torch.ones(1, 128, 40, 40)
    >>> m = Conv(128, 128)
    >>> f = f"{m._get_name()}.onnx"
    >>> torch.onnx.export(m, x, f)
    >>> subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)  # pip install onnxslim
"""

from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .conv import (
    CBAM,
    P5SPF,  # P5语义保留式剪枝融合，用于删除P5检测头前把P5语义注入P4
    TCA,  # 小目标轻量卷积注意力，用于Neck融合后特征校准
    TPCCSIM,  # Detect前的小目标先验一致性校准
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DAFuseST,  # 自编模块5-小目标动态自适应融合，用于替换部分Concat
    DCLDown,
    DWConv,
    DWConvTranspose2d,
    DySample,  # 自编模块4-动态采样上采样，用于替换YOLO颈部nearest上采样
    FGDDown,  # 自编模块1-提升P2->P3下采样时小目标信息检测的精度
    Focus,
    GFGDDown,  # 门控细粒度双路径下采样，用于替换 P2->P3 的 FGDDown
    GhostConv,
    Index,
    LFGDDown,  # 自编模块2-提升P2->P3下采样时小目标信息检测的精度，且适合轻量化网络
    LightConv,
    RepConv,
    SpatialAttention,
    TGPFuse,  # 小目标先验引导融合，用于替换 Neck 中关键 Concat
    TGPFuseSDFM,  # PSFusion-SDFM启发的小目标浅层细节融合
    WDCLDown,  # 自编模块-小波细节对比轻量下采样，用于替换P2->P3下采样Conv
)
from .head import (
    OBB,
    OBB26,
    Classify,
    Depth,
    Detect,
    LRPCHead,
    Pose,
    Pose26,
    RTDETRDecoder,
    Segment,
    Segment26,
    SemanticSegment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "AIFI",
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CBAM",
    "CIB",
    "DFL",
    "ELAN1",
    "MLP",
    "OBB",
    "OBB26",
    "P5SPF",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "TCA",
    "TPCCSIM",
    "A2C2f",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ChannelAttention",
    "Classify",
    "Concat",
    "ContrastiveHead",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DAFuseST",
    "DCLDown",
    "DWConv",
    "DWConvTranspose2d",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "Depth",
    "Detect",
    "DySample",
    "FGDDown",  # 自编模块1-提升P2->P3下采样时小目标信息检测的精度
    "Focus",
    "GFGDDown",
    "GhostBottleneck",
    "GhostConv",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Index",
    "LFGDDown",  # 自编模块2-提升P2->P3下采样时小目标信息检测的精度，且适合轻量化网络
    "LRPCHead",
    "LayerNorm2d",
    "LightConv",
    "MLPBlock",
    "MSDeformAttn",
    "MaxSigmoidAttnBlock",
    "Pose",
    "Pose26",
    "Proto",
    "RTDETRDecoder",
    "RepC3",
    "RepConv",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "Segment",
    "Segment26",
    "SemanticSegment",
    "SpatialAttention",
    "TGPFuse",
    "TGPFuseSDFM",
    "TorchVision",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
    "WDCLDown",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "YOLOESegment26",
    "v10Detect",
)

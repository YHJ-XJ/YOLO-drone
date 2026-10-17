# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = (
    "CBAM",
    "TPCCSIM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DAFuseST",
    "DCLDown",
    "DWConv",
    "DWConvTranspose2d",
    "DySample",
    "FGDDown",  # 自编模块1-提升P2->P3下采样时小目标信息检测的精度
    "Focus",
    "GhostConv",
    "Index",
    "LFGDDown",  # 自编模块2-提升P2->P3下采样时小目标信息检测的精度，且适合轻量化网络
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "TGPFuse",
    "TGPFuseSDFM",
    "WDCLDown",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.批归一化层，使得每一层的输入数据分布保持稳定，避免梯度消失或梯度爆炸。同时在轻量化时该模块可以与上述模块进行融合以减少计算量
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.可以在这里指定想要的激活函数
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):  # 此处即为上述提到的融合模块对应的前向传播，用于部署的时候
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=None):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        act = nn.ReLU() if act is None else act
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class FGDDown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        # 第一条分支-保持空间信息量不损失
        self.s2d = nn.PixelUnshuffle(2)  # 下采样2倍
        self.cv_spd = Conv(c1 * 4, c_, 1, 1)

        # 第二条分支-保留突出特征
        self.pool = nn.MaxPool2d(2, 2)
        self.cv_pool = Conv(c1, c_, 1, 1)

        # 融合两个分支
        self.fuse = Conv(c2, c2, 3, 1)

    def forward(self, x):
        x1 = self.cv_spd(self.s2d(x))
        x2 = self.cv_pool(self.pool(x))
        return self.fuse(torch.cat((x1, x2), 1))


# 相比于上述的模块，该模块为适合轻量化的目标检测网络，减少参数量和计算量
class LFGDDown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2

        self.s2d = nn.PixelUnshuffle(2)
        self.cv_spd = Conv(c1 * 4, c_, 1, 1)

        # 与上述的最大池化不同的是这个模块有可学习参数，能学习边缘、纹理、局部结构，较灵活
        self.dw_down = DWConv(c1, c1, 3, 2)
        self.cv_dw = Conv(c1, c2 - c_, 1, 1)

        self.fuse_dw = DWConv(c2, c2, 3, 1)
        self.fuse_pw = Conv(c2, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv_spd(self.s2d(x))
        x2 = self.cv_dw(self.dw_down(x))
        return self.fuse_pw(self.fuse_dw(torch.cat((x1, x2), 1)))


class GFGDDown(nn.Module):
    """Gated Fine-Grained Dual-path Downsampling（门控细粒度双路径下采样）。.

    PixelUnshuffle 分支保留小目标细粒度空间信息，Pool 分支保留显著响应。 门控分支根据输入特征自适应分配两条路径的权重，用于替代固定拼接。
    """

    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2

        # 细节分支：空间到通道重排，尽量减少 P2->P3 下采样时的小目标信息损失。
        self.s2d = nn.PixelUnshuffle(2)
        self.cv_spd = Conv(c1 * 4, c_, 1, 1)

        # 显著分支：池化保留强响应区域，作为目标主体和边缘显著性的补偿。
        self.pool = nn.MaxPool2d(2, 2)
        self.cv_pool = Conv(c1, c2 - c_, 1, 1)

        # 轻量门控：只学习两条分支的权重，额外参数很少。
        hidden = max(c1 // 4, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, 1, bias=True),
            nn.Softmax(dim=1),
        )

        # 与 FGDDown 保持相同融合方式，便于单变量消融比较。
        self.fuse = Conv(c2, c2, 3, 1)

    def forward(self, x):
        w = self.gate(x)
        x1 = self.cv_spd(self.s2d(x)) * w[:, 0:1]
        x2 = self.cv_pool(self.pool(x)) * w[:, 1:2]
        return self.fuse(torch.cat((x1, x2), 1))


class DCLDown(nn.Module):
    """Detail-Contrast Lightweight Downsampling for small object detection."""

    def __init__(self, c1, c2):
        super().__init__()
        c_spd = c2 // 2
        c_local = c2 // 4
        c_contrast = c2 - c_spd - c_local

        self.s2d = nn.PixelUnshuffle(2)
        self.cv_spd = Conv(c1 * 4, c_spd, 1, 1)

        self.dw_down = DWConv(c1, c1, 3, 2)
        self.cv_local = Conv(c1, c_local, 1, 1)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.cv_contrast = Conv(c1, c_contrast, 1, 1)

        self.fuse_dw = DWConv(c2, c2, 3, 1)
        self.fuse_pw = Conv(c2, c2, 1, 1)

    def forward(self, x):
        x_spd = self.cv_spd(self.s2d(x))
        x_local = self.cv_local(self.dw_down(x))
        x_contrast = self.cv_contrast(self.maxpool(x) - self.avgpool(x))
        return self.fuse_pw(self.fuse_dw(torch.cat((x_spd, x_local, x_contrast), 1)))


class WDCLDown(nn.Module):
    """Wavelet Detail-Contrast Lightweight Downsampling（小波细节对比轻量下采样）。.

    三条并行分支在把分辨率减半的同时保留小目标细节：
    1. PixelUnshuffle(2) -> 1x1 Conv   空间-通道重排，借鉴 SPDConv / reorg_layer
    2. DWConv(3, s=2)      -> 1x1 Conv   可学习局部结构下采样
    3. Haar 高频(LH/HL/HH) -> 1x1 Conv   方向性边缘/纹理，借鉴小波池化思想
    其中第 3 条分支是固定的加减法，几乎没有可学习参数，却能把小目标最依赖的边缘与 局部突变信息保留下来。相比 DCLDown 的 MaxPool-AvgPool 对比分支，Haar 高频能同时
    保留水平/垂直/对角三个方向的细节，信号更强。
    """

    def __init__(self, c1, c2):
        super().__init__()
        c_spd = c2 // 2
        c_local = c2 // 4
        c_hf = c2 - c_spd - c_local  # = c2 // 4

        # 分支1：空间重排保留细节
        self.s2d = nn.PixelUnshuffle(2)
        self.cv_spd = Conv(c1 * 4, c_spd, 1, 1)

        # 分支2：可学习局部下采样
        self.dw_down = DWConv(c1, c1, 3, 2)
        self.cv_local = Conv(c1, c_local, 1, 1)

        # 分支3：Haar 高频边缘/纹理
        self.cv_hf = Conv(c1 * 3, c_hf, 1, 1)

        self.fuse_dw = DWConv(c2, c2, 3, 1)
        self.fuse_pw = Conv(c2, c2, 1, 1)

    @staticmethod
    def _haar_hf(x):
        """返回 2x2 Haar 变换的高频子带 LH/HL/HH（无学习参数）。."""
        b, c, h, w = x.shape
        x = F.pixel_unshuffle(x, 2)  # [B, 4C, H/2, W/2]，每个 2x2 块拆成 a,b,c,d
        x = x.view(b, c, 4, h // 2, w // 2)
        a, b_, c_, d = x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3]
        lh = (a - b_ + c_ - d) * 0.5  # 水平边缘
        hl = (a + b_ - c_ - d) * 0.5  # 垂直边缘
        hh = (a - b_ - c_ + d) * 0.5  # 对角边缘
        return torch.cat([lh, hl, hh], dim=1)

    def forward(self, x):
        x_spd = self.cv_spd(self.s2d(x))
        x_local = self.cv_local(self.dw_down(x))
        x_hf = self.cv_hf(self._haar_hf(x))
        return self.fuse_pw(self.fuse_dw(torch.cat((x_spd, x_local, x_hf), 1)))


class DySample(nn.Module):
    """动态采样上采样模块：用可学习偏移替代固定 nearest 上采样。."""

    def __init__(self, c1, scale=2, groups=4, dyscope=False):
        super().__init__()
        # c1 是输入通道数，由 tasks.py 根据上一层自动补进来；YAML 里不用手写 c1。
        # scale=2 表示特征图高宽各放大 2 倍，正好对应 YOLO 颈部 P5->P4、P4->P3 的上采样。
        # groups=4 表示分组采样，能减少偏移学习的开销；c1 必须能被 groups 整除。
        assert c1 % groups == 0, f"DySample: input channels c1={c1} must be divisible by groups={groups}"
        self.scale = scale
        self.groups = groups
        self.dyscope = dyscope

        # 每个采样点需要 x/y 两个方向的偏移；pixel_shuffle 后得到更密的上采样网格。
        offset_channels = 2 * groups * scale * scale
        self.offset = nn.Conv2d(c1, offset_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.constant_(self.offset.bias, 0)

        # dyscope=True 时额外学习一个偏移幅度门控；第一次实验先设 False，变量更干净。
        if self.dyscope:
            self.scope = nn.Conv2d(c1, offset_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0)

        # init_pos 是初始规则采样位置，让模块一开始接近普通双线性/规则上采样，训练更稳。
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        """生成 scale x scale 子像素网格的初始偏移。."""
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        pos = torch.stack(torch.meshgrid(h, h, indexing="ij")).transpose(1, 2)
        return pos.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        """根据学习到的 offset 对原特征图做 grid_sample，得到动态上采样特征。."""
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)

        # 构造原始网格中心点坐标，0.5 表示像素中心。
        coords_h = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="xy")).unsqueeze(0).unsqueeze(2)

        # grid_sample 要求坐标范围归一化到 [-1, 1]。
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        # pixel_shuffle 把低分辨率偏移重排成高分辨率采样网格。
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
        coords = coords.reshape(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        # 分组采样：每组通道共享一套采样偏移，兼顾精度和轻量化。
        x = x.reshape(b * self.groups, -1, h, w)
        out = F.grid_sample(x, coords, mode="bilinear", align_corners=False, padding_mode="border")
        return out.reshape(b, -1, self.scale * h, self.scale * w)

    def forward(self, x):
        # 0.25/0.5 是偏移幅度限制，避免一开始采样点乱跑导致训练不稳定。
        if self.dyscope:
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)


class TGPFuse(nn.Module):
    """Tiny-object Prior Guided Fusion（小目标先验引导融合）。.

    该模块用于替换 Neck 中关键 Concat：
    1. 保留原 Concat 的通道拼接主路径，保证预训练结构尽量稳定；
    2. 从浅层/细节分支生成 tiny-object prior map；
    3. 用 prior map 对各分支做轻量残差增强，提升小目标相关融合响应。
    """

    def __init__(self, channels, dimension=1, prior_index=-1, reduction=4, gamma_init=0.1):
        super().__init__()
        self.channels = channels
        self.d = dimension
        self.n = len(channels)
        self.prior_index = prior_index if prior_index >= 0 else self.n + prior_index
        total = sum(channels)
        prior_c = channels[self.prior_index]
        hidden = max(prior_c // reduction, 8)

        # 小目标先验分支：默认从更浅层/细节分支生成空间先验图。
        self.prior = nn.Sequential(
            Conv(prior_c, hidden, 1, 1),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # 分支残差权重：对每一路融合输入生成轻量空间权重。
        self.branch_gate = nn.Conv2d(total, self.n, 1, bias=True)
        nn.init.constant_(self.branch_gate.weight, 0)
        nn.init.constant_(self.branch_gate.bias, 0)

        # 残差强度系数，较小初始化避免训练初期破坏原 Concat 路径。
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        prior = self.prior(xs[self.prior_index])
        cat = torch.cat(xs, self.d)
        weights = self.branch_gate(cat).softmax(1)
        outs = [x * (1.0 + self.gamma * prior * weights[:, i : i + 1]) for i, x in enumerate(xs)]
        return torch.cat(outs, self.d)


class P5SPF(nn.Module):
    """P5 Semantic Preserving Fusion（P5语义保留式剪枝融合）。.

    用于删除 P5 检测头前，把 P5 深层语义轻量注入 P4：
    1. P5 分支先通过 1x1 Conv 对齐到 P4 通道；
    2. 从 P4 生成细节门控，保留小目标定位信息；
    3. 从 P5 生成语义门控，补偿删除 P5 检测头后的高层语义损失；
    4. 输出通道仍等于 P4，不做通道拼接，避免计算量明显增加。
    """

    def __init__(self, channels, reduction=4, gamma_init=0.1):
        super().__init__()
        assert len(channels) == 2, "P5SPF expects two inputs: [P4_feature, P5_upsampled_feature]"
        c_p4, c_p5 = channels
        hidden = max(c_p4 // reduction, 8)

        # 将上采样后的 P5 语义特征压到 P4 通道，避免 Concat 带来的通道膨胀。
        self.p5_proj = Conv(c_p5, c_p4, 1, 1)

        # P4 细节门控：强调高分辨率下的小目标边缘、纹理与定位信息。
        self.detail_gate = nn.Sequential(
            Conv(c_p4, hidden, 1, 1),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # P5 语义门控：根据深层语义判断哪些区域需要被注入到 P4。
        self.semantic_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c_p4, hidden, 1, 1),
            nn.Conv2d(hidden, c_p4, 1, bias=True),
            nn.Sigmoid(),
        )

        # 残差强度小初始化，使训练初期接近原 P4 特征，避免破坏预训练表示。
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.fuse = Conv(c_p4, c_p4, 3, 1)

    def forward(self, xs):
        p4, p5 = xs
        p5 = self.p5_proj(p5)
        detail = self.detail_gate(p4)
        semantic = self.semantic_gate(p5)
        out = p4 * (1.0 + self.gamma * detail) + self.gamma * semantic * p5
        return self.fuse(out)


class TCA(nn.Module):
    """Tiny Convolutional Attention（小目标轻量卷积注意力）。.

    用于 Neck 融合后的 P3/P4 特征校准：
    1. 1x1 Conv 先做通道对齐；
    2. 深度 3x3 卷积提取局部边缘和纹理；
    3. 深度 5x5 卷积扩大局部上下文感受野；
    4. 生成轻量空间注意力图，以残差方式增强小目标响应。

    该模块不改变特征图尺寸，适合插入 Detect 前或 Neck 融合后的关键层。
    """

    def __init__(self, c1, c2, reduction=4, gamma_init=0.1):
        super().__init__()
        hidden = max(c2 // reduction, 8)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # 局部纹理分支：更关注小目标边缘、角点和密集局部结构。
        self.local = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        # 上下文分支：5x5 深度卷积扩大感受野，但不引入大量参数。
        self.context = nn.Sequential(
            nn.Conv2d(c2, c2, 5, padding=2, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
        )

        self.attn = nn.Sequential(
            Conv(c2 * 2, hidden, 1, 1),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # 小初始化：训练初期接近恒等映射，降低破坏预训练特征的风险。
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x):
        x = self.proj(x)
        local = self.local(x)
        context = self.context(x)
        weight = self.attn(torch.cat((local, context), 1))
        return self.out(x * (1.0 + self.gamma * weight))


class TGPFuseSDFM(nn.Module):
    """Tiny-object Guided Superficial Detail Fusion Module（小目标先验浅层细节融合）。.

    设计动机来自 PSFusion 的 SDFM，但这里不是做红外/可见光融合，而是用于 YOLO Neck 中 相同尺度特征的融合。它保留 Concat 的输出形式，同时增加局部细节和全局通道两类轻量门控。
    """

    def __init__(self, channels, dimension=1, prior_index=-1, reduction=4, gamma_init=0.1):
        super().__init__()
        self.channels = channels
        self.d = dimension
        self.n = len(channels)
        self.prior_index = prior_index if prior_index >= 0 else self.n + prior_index
        total = sum(channels)
        prior_c = channels[self.prior_index]
        hidden = max(prior_c // reduction, 8)

        # 空间细节先验：从浅层/细节分支提取 1 通道小目标响应图，强调边缘、纹理和密集小目标区域。
        self.spatial_prior = nn.Sequential(
            Conv(prior_c, hidden, 1, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # 全局分支权重：用平均池化+最大池化估计每一路特征的重要性，避免普通 Concat 粗暴拼接。
        branch_hidden = max(total // reduction, 8)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(total, branch_hidden, 1, 1),
            nn.Conv2d(branch_hidden, self.n, 1, bias=True),
        )
        self.max_gate = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Conv(total, branch_hidden, 1, 1),
            nn.Conv2d(branch_hidden, self.n, 1, bias=True),
        )

        # 小初始化：训练初期近似普通 Concat，减少破坏预训练权重带来的震荡。
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, xs):
        if not isinstance(xs, (list, tuple)):
            xs = [xs]
        prior = self.spatial_prior(xs[self.prior_index])
        cat = torch.cat(xs, self.d)
        weights = (self.global_gate(cat) + self.max_gate(cat)).softmax(1)
        outs = [x * (1.0 + self.gamma * prior * weights[:, i : i + 1]) for i, x in enumerate(xs)]
        return torch.cat(outs, self.d)


class TPCCSIM(nn.Module):
    """Tiny-Prior Consistency Calibration with SIM-style modulation.

    用在 Detect 前的单个尺度特征上，做轻量的 gamma/beta 调制。它不改变特征尺寸和通道数， 因此适合快速插入 P2/P3/P4/P5 检测层前做消融。
    """

    def __init__(self, c1, c2=None, reduction=8, gamma_init=0.1):
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(c1 // reduction, 8)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

        # tiny prior：局部深度卷积提取小目标/边缘响应。
        self.prior = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # SIM 风格调制：由全局上下文生成通道级 gamma/beta，再被 tiny prior 控制空间位置。
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c2, hidden, 1, 1),
            nn.Conv2d(hidden, 2 * c2, 1, bias=True),
        )
        self.gamma_scale = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        x = self.proj(x)
        gamma, beta = self.context(x).chunk(2, 1)
        prior = self.prior(x)
        return x * (1.0 + self.gamma_scale * prior * gamma.sigmoid()) + self.gamma_scale * prior * beta


class DAFuseST(nn.Module):
    """细节感知尺度自适应融合模块（v2）：用 softmax 分支权重替换普通 Concat。.

    相比 v1 的两点修正：
    1. 去掉 gamma=0 / tanh(0)=0 的门控，避免空间门控卷积在训练初期梯度为零、长期不激活。
    2. 不再用逐分支标量 branch_scale（它和下游 C3k2 的 1x1 卷积作用重复），改为
    GAP+GMP -> 微型 MLP -> softmax 得到"分支权重"，机制更明确且无梯度消失。
    """

    def __init__(self, channels, dimension=1):
        super().__init__()
        # channels 是待融合各路特征的通道数列表，由 tasks.py 根据 from 索引自动传入。
        # dimension=1 表示沿通道维拼接，和 YOLO 原始 Concat 的行为保持一致。
        self.channels = channels
        self.d = dimension
        self.n = len(channels)
        total = sum(channels)

        # 每个分支先做全局平均池化和全局最大池化，得到 2*channels 的统计描述子。
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # 微型 MLP 把 2*total 的统计描述子压成 n 个分支 logits，再 softmax 得到分支权重。
        # 不用 tanh/零初始化，因此从第一步起就有梯度，不会出现 v1 的空间门控"冻住"问题。
        hidden = max(4, self.n * 4)
        self.attn = nn.Sequential(
            nn.Conv2d(2 * total, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, self.n, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, xs):
        # xs 是 YAML 中 [[-1, 6], ...] 这类 from 列表对应的多路特征。
        # 输出仍沿通道维拼接，通道数 = sum(channels)，与普通 Concat 完全一致，
        # 因此后续 C3k2 的输入通道数不变。
        z = torch.cat([torch.cat([self.gap(x), self.gmp(x)], dim=1) for x in xs], dim=1)
        w = self.attn(z)  # [B, n, 1, 1]，n 个分支权重之和为 1
        return torch.cat([w[:, i : i + 1] * x for i, x in enumerate(xs)], dim=self.d)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]

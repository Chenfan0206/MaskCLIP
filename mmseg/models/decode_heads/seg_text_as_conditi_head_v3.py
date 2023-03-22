# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention

from torch.autograd import Variable
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import cv2, random, string, os
import numpy as np
import einops
import math


class CenterPivotConv4d(nn.Module):
    r"""CenterPivot 4D conv"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, bias=True
    ):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size[:2],
            stride=stride[:2],
            bias=bias,
            padding=padding[:2],
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size[2:],
            stride=stride[2:],
            bias=bias,
            padding=padding[2:],
        )

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(
                start=0, end=hb, step=self.stride[2:][0], device=ct.device
            )
            idxw = torch.arange(
                start=0, end=wb, step=self.stride[2:][1], device=ct.device
            )
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (
                idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb
            ).view(-1)
            self.idx_initialized = True
        ct_pruned = (
            ct.view(bsz, ch, ha, wa, -1)
            .index_select(4, self.idx)
            .view(bsz, ch, ha, wa, self.len_h, self.len_w)
        )

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = (
            out1.view(bsz, hb, wb, outch, o_ha, o_wa)
            .permute(0, 3, 4, 5, 1, 2)
            .contiguous()
        )

        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = (
            out2.view(bsz, ha, wa, outch, o_hb, o_wb)
            .permute(0, 3, 1, 2, 4, 5)
            .contiguous()
        )

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y


Conv4d = CenterPivotConv4d


class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(
            in_channel, out_channels, kernel_sizes, spt_strides, group=4
        ):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(
                zip(out_channels, kernel_sizes, spt_strides)
            ):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(
            inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]
        )
        self.encoder_layer3 = make_building_block(
            inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]
        )
        self.encoder_layer2 = make_building_block(
            inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]
        )

        # self.encoder_layer4 = make_building_block(
        #     inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2]
        # )
        # self.encoder_layer3 = make_building_block(
        #     inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2]
        # )
        # self.encoder_layer2 = make_building_block(
        #     inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2]
        # )

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(
            outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]
        )
        self.encoder_layer3to2 = make_building_block(
            outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]
        )

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True),
        )

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = (
            hypercorr.permute(0, 4, 5, 1, 2, 3)
            .contiguous()
            .view(bsz * hb * wb, ch, ha, wa)
        )
        hypercorr = F.interpolate(
            hypercorr, spatial_size, mode='bilinear', align_corners=True
        )
        o_hb, o_wb = spatial_size
        hypercorr = (
            hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb)
            .permute(0, 3, 4, 5, 1, 2)
            .contiguous()
        )
        return hypercorr

    def forward(self, hypercorr_pyramid):

        # torch.Size([2, 3, 25, 25, 25, 25])
        # torch.Size([2, 6, 25, 25, 12, 12])
        # torch.Size([2, 3, 25, 25, 6, 6])

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(
            hypercorr_pyramid[0]
        )  # torch.Size([2, 128, 25, 25, 4, 4])
        hypercorr_sqz3 = self.encoder_layer3(
            hypercorr_pyramid[1]
        )  # torch.Size([2, 128, 25, 25, 2, 2])
        hypercorr_sqz2 = self.encoder_layer2(
            hypercorr_pyramid[2]
        )  # torch.Size([2, 128, 25, 25, 1, 1])

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(
            hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2]
        )
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(
            hypercorr_mix43, hypercorr_sqz2.size()[-4:-2]
        )
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(
        #     hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True
        # )
        logit_mask = self.decoder2(hypercorr_decoded)

        return logit_mask


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(bs, -1)
        # m2 = targets.view(bs, -1)
        m2 = targets.contiguous().view(bs, -1)
        intersection = m1 * m2

        score = 2.0 * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / bs
        return score


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# pre-layernorm


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


# feedforward
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ReduceDim(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# attention
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat(
                (x, context), dim=1
            )  # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat(
                (x, context), dim=1
            )  # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Attention_return_dots(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat(
                (x, context), dim=1
            )  # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return (self.to_out(out), dots)


class CrossTransformer_mine(nn.Module):  #
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention_return_dots(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(
        self, x, context, extract_dots=False
    ):  # x 是用来做query的，context是用来做key和value的
        if not extract_dots:
            for attn, ff in self.layers:
                x = attn(x, context=context) + x
                x = ff(x) + x

            return self.norm(x)
        else:
            _all_x = []
            _all_dots = []
            for attn, ff in self.layers:
                tmp, dots = attn(x, context=context)
                x = tmp + x
                x = ff(x) + x
                _all_x.append(self.norm(x))
                _all_dots.append(dots)
            return _all_x, _all_dots


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        batch_first=True,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            bias=qkv_bias,
        )

        self.norm2_name, norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
        )

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, return_qkv=False):
        q, k, v = None, None, None
        if return_qkv:
            y = self.norm1(x)
            y = F.linear(y, self.attn.attn.in_proj_weight, self.attn.attn.in_proj_bias)
            N, L, C = y.shape
            y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
            y = F.linear(
                y, self.attn.attn.out_proj.weight, self.attn.attn.out_proj.bias
            )
            q, k, v = y.tensor_split(3, dim=0)
            v += x
            v = self.ffn(self.norm2(v), identity=v)

        x = self.attn(self.norm1(x), identity=x)
        x = self.ffn(self.norm2(x), identity=x)
        return x, q, k, v


class ConditionHead(BaseModule):
    def __init__(
        self,
        num_layers=3,
        embed_dims=512,
        num_heads=8,
        mlp_ratio=4,
        num_fcs=2,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        act_cfg=dict(type='GELU'),
    ):
        super(ConditionHead, self).__init__()
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                )
            )

        ## init the weight of the self.layers
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.pre_norm = True
        norm_cfg = dict(type='LN')
        if self.pre_norm:
            self.norm0_name, norm0 = build_norm_layer(norm_cfg, embed_dims, postfix=0)
            self.add_module(self.norm0_name, norm0)

        self.final_norm = True
        if self.final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.return_qkv = [False] * num_layers  # 初始设置为都不返回返回
        # self.out_indices = [num_layers - 1]  # 只返回最后一层的qkv
        self.out_indices = [0, 1, 2]  # 只返回最后一层的qkv

        return_qkv = True
        if isinstance(return_qkv, bool):  # 如果是bool类型
            for out_i in self.out_indices:
                self.return_qkv[out_i] = return_qkv  # 如果是True, 返回在out_indices中的层的qkv
        elif isinstance(return_qkv, list) or isinstance(return_qkv, tuple):
            for i, out_i in enumerate(self.out_indices):  # 如果是list或者tuple，则根据对应位置来返回
                self.return_qkv[out_i] = (return_qkv[i],)
        else:
            raise TypeError('return_qkv must be type of bool, list or tuple')

        self.skip_last_attn = False
        self.with_cls_token = True
        self.output_cls_token = True

    @property
    def norm0(self):
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, inputs, hw_shape):
        x = inputs
        if self.pre_norm:
            x = self.norm0(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x, q, k, v = layer(
                x,
                self.return_qkv[i]
                or (i == len(self.layers) - 1 and self.skip_last_attn),
            )

            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
                    if self.return_qkv[i]:
                        v = self.norm1(v)
                if self.skip_last_attn:
                    if self.with_cls_token:
                        x[:, 1:] = v[:, 1:]
                    else:
                        x = v
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                if self.return_qkv[i]:
                    if self.with_cls_token:
                        q = q[:, 1:]
                        k = k[:, 1:]
                        v = v[:, 1:]
                    v = (
                        v.reshape(B, hw_shape[0], hw_shape[1], C)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                    out = [out, q, k, v]
                outs.append(out)

        return tuple(outs)


@HEADS.register_module()
class SegTextAsConditionHeadV3(BaseDecodeHead):
    def __init__(
        self,
        text_categories,
        text_channels,
        text_embeddings_path,
        visual_projs_path,
        vit=False,
        ks_thresh=0.0,
        pd_thresh=0.0,
        attn_pooling=False,
        num_heads=32,
        decode_mode='text_as_classifier',
        **kwargs,
    ):
        super(SegTextAsConditionHeadV3, self).__init__(**kwargs)

        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.visual_projs_path = visual_projs_path

        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(
                torch.zeros(text_categories, text_channels)
            )
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            print('Loading text embeddings from {}'.format(self.text_embeddings_path))
            self.register_buffer(
                'text_embeddings', torch.randn(text_categories, text_channels)
            )
            self.load_text_embeddings()

        self.vit = vit
        if vit:
            self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.k_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.v_proj = nn.Conv2d(self.in_channels, self.in_channels, 1)
            self.c_proj = nn.Conv2d(self.in_channels, text_channels, 1)
        self.load_visual_projs()

        self.ks_thresh = ks_thresh
        self.pd_thresh = pd_thresh
        self.attn_pooling = attn_pooling
        self.num_heads = num_heads

        self.decode_mode = decode_mode

        if (
            self.decode_mode == 'text_as_cls_token_finetune'
            or self.decode_mode == 'text_as_cls_token_finetune_add'
            or self.decode_mode == 'text_as_cls_token_finetune_add_multiscale'
        ):
            self.decoder_1 = ConditionHead(
                num_layers=3,
                embed_dims=512,
                num_heads=8,
                mlp_ratio=4,
                num_fcs=2,
                attn_drop_rate=0.0,
                drop_rate=0.0,
                drop_path_rate=0.0,
                qkv_bias=True,
                act_cfg=dict(type='GELU'),
            )
            for m in self.decoder_1.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        elif (
            self.decode_mode == 'light_weight_text_as_query'
            or self.decode_mode == 'light_weight_text_as_query_multiscale'
            or self.decode_mode == 'light_weight_text_as_query_RD'
            or self.decode_mode == 'light_weight_text_as_query_RD_support'
            or self.decode_mode == 'light_weight_visual_as_query_support'
            or self.decode_mode == 'light_weight_text_as_query_15_class_training'
            or self.decode_mode
            == 'light_weight_text_as_query_15_class_training_bce_dice'
            or self.decode_mode
            == 'light_weight_text_as_query_15_class_training_bce_dice_out_by_argmax'
            or self.decode_mode == 'light_weight_text_as_query_4d_tensor'
        ):
            self.decoder_1 = CrossTransformer_mine(
                dim=512, depth=3, heads=8, dim_head=512, mlp_dim=2, dropout=0.0
            )

            if (
                self.decode_mode == 'light_weight_text_as_query_RD'
                or self.decode_mode == 'light_weight_text_as_query_RD_support'
                or self.decode_mode == 'light_weight_visual_as_query_support'
            ):
                self.reduction = ReduceDim(
                    in_dim=1024, hidden_dim=256, out_dim=512, dropout=0.0
                )

            if (
                self.decode_mode == 'light_weight_text_as_query_15_class_training'
                or self.decode_mode
                == 'light_weight_text_as_query_15_class_training_bce_dice'
                or self.decode_mode
                == 'light_weight_text_as_query_15_class_training_bce_dice_out_by_argmax'
            ):
                self.bg_cls = nn.Parameter(torch.randn(1, 512))

            if self.decode_mode == 'light_weight_text_as_query_4d_tensor':
                self.decoder_2 = HPNLearner([3, 6, 3])

            for m in self.decoder_1.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        elif self.decode_mode == 'clip_context_affinity_v1' or self.decode_mode == 'clip_context_affinity_v2' or self.decode_mode == 'clip_context_affinity_v3':
            in_channels = [256, 512, 1024, 2048]
            self.number_of_grids = [2304, 576, 144]
            self.pe = nn.ModuleList()
            self.affinity_blocks = nn.ModuleList()

            for in_channel, grid_n in zip(in_channels[1:], [2304, 576, 144]):
                self.affinity_blocks.append(
                    CrossAttention(dim=grid_n, heads=8, dim_head=512, dropout=0.3)
                )
                self.pe.append(PositionalEncoding(d_model=grid_n, dropout=0.5))

            outch1, outch2, outch3 = 16, 64, 128

            stack_ids = [0, 1, 2, 3]
            self.stack_ids = stack_ids

            ## conv是最高层的
            self.conv1 = self.build_conv_block(
                stack_ids[3] - stack_ids[2] + 1,
                [outch1, outch2, outch3],
                [3, 3, 3],
                [1, 1, 1],
            )  # 1/32
            self.conv2 = self.build_conv_block(
                stack_ids[2] - stack_ids[1] + 1,
                [outch1, outch2, outch3],
                [5, 3, 3],
                [1, 1, 1],
            )  # 1/16
            self.conv3 = self.build_conv_block(
                stack_ids[1] - stack_ids[0] + 1,
                [outch1, outch2, outch3],
                [5, 5, 3],
                [1, 1, 1],
            )  # 1/8

            self.conv4 = self.build_conv_block(
                outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]
            )  # 1/32 + 1/16
            self.conv5 = self.build_conv_block(
                outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]
            )  # 1/16 + 1/8

            self.mixer1 = nn.Sequential(
                nn.Conv2d(
                    outch3 + 2 * in_channels[1] + 2 * in_channels[0] + 1,
                    outch3,
                    (3, 3),
                    padding=(1, 1),
                    bias=True,
                ),
                nn.ReLU(),
                nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                nn.ReLU(),
            )

            self.mixer2 = nn.Sequential(
                nn.Conv2d(outch2 + 1, outch2, (3, 3), padding=(1, 1), bias=True),
                nn.ReLU(),
                nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                nn.ReLU(),
            )

            self.mixer3 = nn.Sequential(
                nn.Conv2d(outch1 + 1, outch1, (3, 3), padding=(1, 1), bias=True),
                nn.ReLU(),
                nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True),
            )

    def init_weights(self):
        # super(MaskClipHead, self).init_weights()
        if self.text_embeddings_path is None:
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.load_text_embeddings()
        self.load_visual_projs()

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings[:, :] = loaded[:, :]
        print_log(
            f'Loaded text embeddings from {self.text_embeddings_path}',
            logger=get_root_logger(),
        )

    def load_visual_projs(self):
        loaded = torch.load(self.visual_projs_path, map_location='cuda')
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(
            f'Loaded proj weights from {self.visual_projs_path}',
            logger=get_root_logger(),
        )

    def get_feat_from_inputs(self, inputs):
        hw_shape = inputs[-1][-1].shape[-2:]
        x = inputs[-1]

        q, k, v, cls_token = None, None, None, None
        if self.vit:
            if isinstance(x, list) and len(x) == 4:
                x, q, k, v = x
            if isinstance(x, list) and len(x) == 2:
                x, cls_token = x
            if v is not None:
                feat = self.proj(v)
            else:
                feat = self.proj(x)

            # feat = self.proj(q)
            if cls_token is not None:
                cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
        else:
            if self.attn_pooling:
                N, C, H, W = x.shape
                x = x.view(N, C, -1).permute(2, 0, 1)  # NCHW -> (HW)NC
                x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
                x, _ = F.multi_head_attention_forward(
                    query=x,
                    key=x,
                    value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight[:, :, 0, 0],
                    k_proj_weight=self.k_proj.weight[:, :, 0, 0],
                    v_proj_weight=self.v_proj.weight[:, :, 0, 0],
                    in_proj_weight=None,
                    in_proj_bias=torch.cat(
                        [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
                    ),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight[:, :, 0, 0],
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                )
                feat = x[1:].permute(1, 2, 0).view(N, -1, H, W)
            else:
                q = self.q_proj(x)
                k = self.k_proj(x)
                q = torch.flatten(q, start_dim=2).transpose(-2, -1)
                k = torch.flatten(k, start_dim=2).transpose(-2, -1)
                v = self.v_proj(x)
                feat = self.c_proj(v)

        return (x, q, k, v, cls_token, feat, hw_shape)

    def forward_original_clip_text_as_classifier(self, batch):
        inputs = batch['query_feat']
        # x, q, k, v, cls_token, feat, hw_shape = self.get_features_from_inputs(inputs)
        _, _, _, _, _, feat, hw_shape = self.get_feat_from_inputs(inputs)

        output = self.cls_seg(feat)  # 20 classes
        # if not self.training:
        #     output = self.refine_output(output, k)

        # reshape output to image size
        output = resize(
            input=output,
            size=batch['query_img'].shape[2:],
            mode='bilinear',
            align_corners=False,
        )
        # loss = torch.nn.BCEWithLogitsLoss()(
        #     output.squeeze(1), batch['query_mask'].float()
        # )

        pred_mask_c = output.argmax(dim=1)
        pred_mask_01 = torch.zeros_like(pred_mask_c)
        pred_mask_01[pred_mask_c == batch['class_id'].unsqueeze(-1).unsqueeze(-1)] = 1

        # tmp = torch.sigmoid(output)
        # logits_by_sigmoid = tmp[
        #     torch.arange(tmp.shape[0]), batch['class_id']
        # ].unsqueeze(1)
        # _min = einops.reduce(logits_by_sigmoid, 'n c h w -> n c () ()', 'min')
        # _max = einops.reduce(logits_by_sigmoid, 'n c h w -> n c () ()', 'max')
        # logits_by_sigmoid = (logits_by_sigmoid - _min) / (_max - _min + 1e-8)
        # pred_mask_01 = logits_by_sigmoid >= 0.5
        # pred_mask_01 = pred_mask_01.squeeze(1).float()
        # print()

        ## visualize
        ## by softmax
        tmp = torch.nn.functional.softmax(output, dim=1)

        ## tmp n c h w, batch['class_id'] n , selcet corresponding class from tmp in dim=1
        logits_by_softmax = tmp[
            torch.arange(tmp.shape[0]), batch['class_id']
        ].unsqueeze(1)

        _min = einops.reduce(logits_by_softmax, 'n c h w -> n c () ()', 'min')
        _max = einops.reduce(logits_by_softmax, 'n c h w -> n c () ()', 'max')
        logits_by_softmax = (logits_by_softmax - _min) / (_max - _min + 1e-8)

        # by sigmoid
        tmp = torch.sigmoid(output)
        logits_by_sigmoid = tmp[
            torch.arange(tmp.shape[0]), batch['class_id']
        ].unsqueeze(1)
        _min = einops.reduce(logits_by_sigmoid, 'n c h w -> n c () ()', 'min')
        _max = einops.reduce(logits_by_sigmoid, 'n c h w -> n c () ()', 'max')
        logits_by_sigmoid = (logits_by_sigmoid - _min) / (_max - _min + 1e-8)

        # masks = torch.cat(
        #     [
        #         logits_by_softmax,
        #         logits_by_sigmoid,
        #     ],
        #     dim=2,
        # )

        # masks = einops.rearrange(masks, 'n c h w -> h (n w) c').squeeze(-1)
        # masks = einops.repeat(masks, 'h w -> h w c', c=3)
        # masks = masks.cpu().numpy()
        # masks = (masks * 255).astype(np.uint8)

        # img_paths = batch['query_path']

        # original_images = [cv2.imread(img_path) for img_path in img_paths]
        # original_images = [cv2.resize(img, (224, 224)) for img in original_images]
        # original_images = einops.rearrange(original_images, 'n h w c -> n h w c')
        # original_images = einops.rearrange(original_images, 'n h w c ->h (n w) c')
        # vis = np.concatenate([original_images, masks], axis=0)

        # save_dir = 'log/affinity'
        # if not os.path.exists(save_dir):
        #     os.makedirs(name=save_dir, exist_ok=True)

        # random_str = ''.join(
        #     random.choices(string.ascii_uppercase + string.digits, k=10)
        # )
        # save_path = os.path.join(save_dir, random_str + '.png')
        # cv2.imwrite(save_path, vis)
        # print('save_path', save_path)

        ## by sigmoid

        _all = {
            'pred_logits': output,
            # 'loss': loss,
            'pred_mask_01': pred_mask_01,
            'pred_mask_c': pred_mask_c,
            "logits_by_softmax": logits_by_softmax,
            "logits_by_sigmoid": logits_by_sigmoid,
        }
        return _all

    def forward(self, batch):
        inputs = batch['query_feat']
        hw_shape = batch['query_feat'][-1][-1].shape[-2:]
        x = inputs[-1]
        q, k, v, cls_token = None, None, None, None
        if self.vit:
            if isinstance(x, list) and len(x) == 4:
                x, q, k, v = x
            if isinstance(x, list) and len(x) == 2:
                x, cls_token = x
            if v is not None:
                feat = self.proj(v)
            else:
                feat = self.proj(x)

            # feat = self.proj(q)
            if cls_token is not None:
                cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
        else:
            if self.attn_pooling:
                N, C, H, W = x.shape
                x = x.view(N, C, -1).permute(2, 0, 1)  # NCHW -> (HW)NC
                x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
                x, _ = F.multi_head_attention_forward(
                    query=x,
                    key=x,
                    value=x,
                    embed_dim_to_check=x.shape[-1],
                    num_heads=self.num_heads,
                    q_proj_weight=self.q_proj.weight[:, :, 0, 0],
                    k_proj_weight=self.k_proj.weight[:, :, 0, 0],
                    v_proj_weight=self.v_proj.weight[:, :, 0, 0],
                    in_proj_weight=None,
                    in_proj_bias=torch.cat(
                        [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
                    ),
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0,
                    out_proj_weight=self.c_proj.weight[:, :, 0, 0],
                    out_proj_bias=self.c_proj.bias,
                    use_separate_proj_weight=True,
                    training=self.training,
                    need_weights=False,
                )
                feat = x[1:].permute(1, 2, 0).view(N, -1, H, W)
            else:
                q = self.q_proj(x)
                k = self.k_proj(x)
                q = torch.flatten(q, start_dim=2).transpose(-2, -1)
                k = torch.flatten(k, start_dim=2).transpose(-2, -1)
                v = self.v_proj(x)
                feat = self.c_proj(v)

        if self.decode_mode == 'text_as_classifier':
            output = self.cls_seg(feat)  # 20 classes
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=output,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            # loss = torch.nn.BCEWithLogitsLoss()(
            #     output.squeeze(1), batch['query_mask'].float()
            # )

            pred_mask_c = output.argmax(dim=1)
            pred_mask_01 = torch.zeros_like(pred_mask_c)
            pred_mask_01[pred_mask_c == batch['class_id']] = 1

            # loss = self.loss_decode(output, batch['query_mask'].long())
            ##  text_as_classifier has no loss

            tmp = torch.nn.functional.softmax(output, dim=1)

            ## tmp n c h w, batch['class_id'] n , selcet corresponding class from tmp in dim=1
            logits_by_softmax = tmp[
                torch.arange(tmp.shape[0]), batch['class_id']
            ].unsqueeze(1)

            _min = einops.reduce(logits_by_softmax, 'n c h w -> n c () ()', 'min')
            _max = einops.reduce(logits_by_softmax, 'n c h w -> n c () ()', 'max')
            logits_by_softmax = (logits_by_softmax - _min) / (_max - _min + 1e-8)

            # by sigmoid
            tmp = torch.sigmoid(output)
            logits_by_sigmoid = tmp[
                torch.arange(tmp.shape[0]), batch['class_id']
            ].unsqueeze(1)
            _min = einops.reduce(logits_by_sigmoid, 'n c h w -> n c () ()', 'min')
            _max = einops.reduce(logits_by_sigmoid, 'n c h w -> n c () ()', 'max')
            logits_by_sigmoid = (logits_by_sigmoid - _min) / (_max - _min + 1e-8)

            _all = {
                'pred_logits': output,
                # 'loss': loss,
                'pred_mask_01': pred_mask_01,
                'pred_mask_c': pred_mask_c,
                'logits_by_softmax': logits_by_softmax,
                'logits_by_sigmoid': logits_by_sigmoid,
            }
            return _all

        if self.decode_mode == 'text_as_classifier_trainable_bce':
            output = self.cls_seg(feat)  # 20 classes
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=output,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            # select logits of the class from output in dim 1, output shape: (N, 20, H, W), batch['class_id'] shape: torch.Size([N]), return shape: (N, H, W)
            output = output[torch.arange(output.shape[0]), batch['class_id']].unsqueeze(
                1
            )

            # loss = torch.nn.BCEWithLogitsLoss()(
            #     output.squeeze(1), batch['query_mask'].float()
            # )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            # loss = self.loss_decode(output, batch['query_mask'].long())
            ##  text_as_classifier has no loss

            _all = {
                'pred_logits': output,
                'loss': loss,
                'pred_mask_01': pred_mask_01,
                # 'pred_mask_c': pred_mask_c,
            }
            return _all

        if self.decode_mode == 'text_as_cls_token_finetune':

            ## 构建transformer, 然后特征品拼接。然后取出文本
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            condition = self.text_embeddings[batch['class_id']].unsqueeze(1)
            # TODO, mixup text embeddings and visual features
            x = torch.cat([condition, feat], dim=1)
            tmp = self.decoder_1(x, hw_shape)

            feat = tmp[-1][-1]  # 取最后的v 作为特征

            feat = feat / feat.norm(
                dim=-1, keepdim=True
            )  # normalize for cosine distance

            logits = torch.bmm(
                einops.rearrange(feat, 'b c h w-> b (h w) c'),
                self.text_embeddings[batch['class_id']].unsqueeze(-1),
            )

            logits = einops.rearrange(
                logits, 'b (h w) c -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )

            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'text_as_cls_token_finetune_add':

            init_20_logits = self.cls_seg(feat)  # 16, 20, 32,32
            # selcect this_class_logits [16, 1, 32,32] from init_20_logits, according to the batch['class_id'] (with the shape torch.Size([16])),
            this_class_logits = init_20_logits[
                torch.arange(init_20_logits.shape[0]), batch['class_id'], :, :
            ].unsqueeze(
                1
            )  # 16, 1, 32,32

            ## 构建transformer, 然后特征品拼接。然后取出文本
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            condition = self.text_embeddings[batch['class_id']].unsqueeze(1)
            # TODO, mixup text embeddings and visual features
            x = torch.cat([condition, feat], dim=1)
            tmp = self.decoder_1(x, hw_shape)

            feat = tmp[-1][-1]  # 取最后的v 作为特征

            feat = feat / feat.norm(
                dim=-1, keepdim=True
            )  # normalize for cosine distance

            logits = torch.bmm(
                einops.rearrange(feat, 'b c h w-> b (h w) c'),
                self.text_embeddings[batch['class_id']].unsqueeze(-1),
            )

            logits = einops.rearrange(
                logits, 'b (h w) c -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )

            # add to the init logits
            logits = this_class_logits + 0.5 * logits
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'text_as_cls_token_finetune_add_multiscale':

            init_20_logits = self.cls_seg(feat)  # 16, 20, 32,32
            # selcect this_class_logits [16, 1, 32,32] from init_20_logits, according to the batch['class_id'] (with the shape torch.Size([16])),
            this_class_logits = init_20_logits[
                torch.arange(init_20_logits.shape[0]), batch['class_id'], :, :
            ].unsqueeze(
                1
            )  # 16, 1, 32,32

            ## 构建transformer, 然后特征品拼接。然后取出文本
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            condition = self.text_embeddings[batch['class_id']].unsqueeze(1)
            # TODO, mixup text embeddings and visual features
            x = torch.cat([condition, feat], dim=1)
            tmp = self.decoder_1(x, hw_shape)

            weights = [0.2, 0.3, 0.5]
            for item, weight in zip(tmp, weights):
                feat = item[-1]  # 取最后的v 作为特征
                feat = feat / feat.norm(
                    dim=-1, keepdim=True
                )  # normalize for cosine distance

                logits = torch.bmm(
                    einops.rearrange(feat, 'b c h w-> b (h w) c'),
                    self.text_embeddings[batch['class_id']].unsqueeze(-1),
                )

                logits = einops.rearrange(
                    logits, 'b (h w) c -> b c h w', h=hw_shape[0], w=hw_shape[1]
                )

                # add to the init logits

                this_class_logits = this_class_logits + weight * logits

                # if not self.training:
                #     output = self.refine_output(output, k)

            # reshape output to image size
            logits = this_class_logits
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'light_weight_text_as_query':
            ## 先只用文本作为query
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj
            condition = self.text_embeddings[batch['class_id']].unsqueeze(
                1
            )  # [16, 1, 512]
            # TODO, mixup text embeddings and visual cls_token. and cls_token from support set

            tmp = self.decoder_1(x=condition, context=feat, extract_dots=True)
            _all_x, _all_dots = tmp
            dot = _all_dots[-1]  # 16,8,1,1024
            dot = torch.mean(dot, dim=1, keepdim=False)  # 16,1,1024
            logits = einops.rearrange(
                dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )  # 16,1024
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'light_weight_text_as_query_4d_tensor':
            ## 先只用文本作为query
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj
            condition = self.text_embeddings[batch['class_id']].unsqueeze(
                1
            )  # [16, 1, 512]
            # TODO, mixup text embeddings and visual cls_token. and cls_token from support set

            tmp = self.decoder_1(x=condition, context=feat, extract_dots=True)
            _all_x, _all_dots = tmp
            dot = _all_dots[-1]  # 16,8,1,1024
            dot = torch.mean(dot, dim=1, keepdim=False)  # 16,1,1024
            logits = einops.rearrange(
                dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )  # 16,1024
            # if not self.training:
            #     output = self.refine_output(output, k)

            logits_2 = self.decoder_2(batch['corr'])

            logits = logits + logits_2[:, 1:, :, :]

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'light_weight_text_as_query_15_class_training':
            ## 先只用文本作为query
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            text_embeddings = self.text_embeddings.expand(
                feat.shape[0], -1, -1
            )  # 16 20 512

            text_embeddings = torch.cat(
                [self.bg_cls.expand(feat.shape[0], -1, -1), text_embeddings], dim=1
            )
            tmp = self.decoder_1(x=text_embeddings, context=feat, extract_dots=True)
            _all_x, _all_dots = tmp
            dot = _all_dots[-1]  # 16,8,20,1024
            dot = torch.mean(dot, dim=1, keepdim=False)  # 16,20,1024
            logits = einops.rearrange(
                dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )  # 16 20 32 32
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            output = output[:, 1:, :, :]

            # query_c_labels = torch.unique(
            #     batch['query_mask_c_0_to_21_ignore_novel']
            # )  # 是语义分割的标签，去除了novel的，所以是0-20； 0是bg, 1-20是类别
            # print(
            #     query_c_labels, batch['class_id'] + 1
            # )  # fold1 0 1 2 3 4 5  11 12 13 14 15 16 17 18 19 20 255

            query_c_labels_one_hot = (
                torch.nn.functional.one_hot(
                    batch['query_mask_c_0_to_21_ignore_novel'].long(), num_classes=256
                )
                .permute(0, 3, 1, 2)
                .float()
            )
            query_c_labels_one_hot = query_c_labels_one_hot[
                :, 1:21, :, :
            ]  # 不要bg, 和255的

            ## select the
            # not0_flag = torch.sum(query_c_labels_one_hot, dim=(2, 3)) != 0
            # output_not0 = output[not0_flag]
            # query_c_labels_one_hot_not0 = query_c_labels_one_hot[not0_flag]

            # loss = torch.nn.BCEWithLogitsLoss()(
            #     output_not0, query_c_labels_one_hot_not0
            # )

            # select 0-5, 10-20

            output_15 = torch.cat([output[:, 0:5, :, :], output[:, 10:20, :, :]], dim=1)
            query_c_labels_one_hot_15 = torch.cat(
                [
                    query_c_labels_one_hot[:, 0:5, :, :],
                    query_c_labels_one_hot[:, 10:20, :, :],
                ],
                dim=1,
            )

            loss = torch.nn.BCEWithLogitsLoss()(output_15, query_c_labels_one_hot_15)
            loss_dice = SoftDiceLoss()(output_15, query_c_labels_one_hot_15)
            loss = loss + loss_dice
            ## Nan
            if torch.isnan(loss):
                print('loss is nan', batch['query_path'])
                print('loss is nan', batch['support_path'])
                print()

            # # select logits from the second dim of output (shape: 16 20 32 512) , by batch['class_id'] (shape: 16), return shape: 16 512 512
            output_after_sigmoid = F.sigmoid(
                output[torch.arange(0, output.shape[0]), batch['class_id']]
            )
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            # pred_mask_c = torch.argmax(output, dim=1)
            # pred_mask_01 = torch.zeros_like(pred_mask_c)
            # pred_mask_01[
            #     pred_mask_c == batch['class_id'].unsqueeze(-1).unsqueeze(-1)
            # ] = 1

            # loss = torch.nn.CrossEntropyLoss(ignore_index=255)(
            #     output, batch['query_mask_c_0_to_21_ignore_novel'].long()
            # )

            # output_after_softmax = F.softmax(output, dim=1)
            # pred_mask_c = torch.argmax(output_after_softmax, dim=1)
            # pred_mask_01 = torch.zeros_like(pred_mask_c)
            # pred_mask_01[
            #     pred_mask_c == (batch['class_id'] + 1).unsqueeze(-1).unsqueeze(-1)
            # ] = 1

            _all = {
                # 'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'light_weight_text_as_query_15_class_training_bce_dice':
            ## 先只用文本作为query
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            text_embeddings = self.text_embeddings.expand(
                feat.shape[0], -1, -1
            )  # 16 20 512

            text_embeddings = torch.cat(
                [self.bg_cls.expand(feat.shape[0], -1, -1), text_embeddings], dim=1
            )
            tmp = self.decoder_1(x=text_embeddings, context=feat, extract_dots=True)
            _all_x, _all_dots = tmp
            dot = _all_dots[-1]  # 16,8,20,1024
            dot = torch.mean(dot, dim=1, keepdim=False)  # 16,20,1024
            logits = einops.rearrange(
                dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )  # 16 20 32 32
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            output = output[:, 1:, :, :]

            # query_c_labels = torch.unique(
            #     batch['query_mask_c_0_to_21_ignore_novel']
            # )  # 是语义分割的标签，去除了novel的，所以是0-20； 0是bg, 1-20是类别
            # print(
            #     query_c_labels, batch['class_id'] + 1
            # )  # fold1 0 1 2 3 4 5  11 12 13 14 15 16 17 18 19 20 255

            query_c_labels_one_hot = (
                torch.nn.functional.one_hot(
                    batch['query_mask_c_0_to_21_ignore_novel'].long(), num_classes=256
                )
                .permute(0, 3, 1, 2)
                .float()
            )
            query_c_labels_one_hot = query_c_labels_one_hot[
                :, 1:21, :, :
            ]  # 不要bg, 和255的

            ## ## 一种是筛选出非0的，然后计算loss
            not0_flag = torch.sum(query_c_labels_one_hot, dim=(2, 3)) != 0
            output_not0 = output[not0_flag]
            query_c_labels_one_hot_not0 = query_c_labels_one_hot[not0_flag]

            loss = torch.nn.BCEWithLogitsLoss()(
                output_not0, query_c_labels_one_hot_not0
            )
            loss_dice = SoftDiceLoss()(output_not0, query_c_labels_one_hot_not0)
            loss = loss + loss_dice

            ## 另外一种是直接计算loss，不用筛选
            # loss = torch.nn.BCEWithLogitsLoss()(
            #     output, query_c_labels_one_hot
            # )
            # loss_dice = SoftDiceLoss()(output, query_c_labels_one_hot)
            # loss = loss + loss_dice

            ## Nan
            if torch.isnan(loss):
                print('loss is nan', batch['query_path'])
                print('loss is nan', batch['support_path'])
                print()

            # compute the dice loss
            # loss_dice = torch.nn.dice_loss(

            # # select logits from the second dim of output (shape: 16 20 32 512) , by batch['class_id'] (shape: 16), return shape: 16 512 512
            output_after_sigmoid = F.sigmoid(
                output[torch.arange(0, output.shape[0]), batch['class_id']]
            )
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            # pred_mask_c = torch.argmax(output, dim=1)
            # pred_mask_01 = torch.zeros_like(pred_mask_c)
            # pred_mask_01[
            #     pred_mask_c == batch['class_id'].unsqueeze(-1).unsqueeze(-1)
            # ] = 1

            # loss = torch.nn.CrossEntropyLoss(ignore_index=255)(
            #     output, batch['query_mask_c_0_to_21_ignore_novel'].long()
            # )

            # output_after_softmax = F.softmax(output, dim=1)
            # pred_mask_c = torch.argmax(output_after_softmax, dim=1)
            # pred_mask_01 = torch.zeros_like(pred_mask_c)
            # pred_mask_01[
            #     pred_mask_c == (batch['class_id'] + 1).unsqueeze(-1).unsqueeze(-1)
            # ] = 1

            _all = {
                # 'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if (
            self.decode_mode
            == 'light_weight_text_as_query_15_class_training_bce_dice_out_by_argmax'
        ):
            ## 先只用文本作为query
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            text_embeddings = self.text_embeddings.expand(
                feat.shape[0], -1, -1
            )  # 16 20 512

            text_embeddings = torch.cat(
                [self.bg_cls.expand(feat.shape[0], -1, -1), text_embeddings], dim=1
            )
            tmp = self.decoder_1(x=text_embeddings, context=feat, extract_dots=True)
            _all_x, _all_dots = tmp
            dot = _all_dots[-1]  # 16,8,20,1024
            dot = torch.mean(dot, dim=1, keepdim=False)  # 16,20,1024
            logits = einops.rearrange(
                dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )  # 16 20 32 32
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )
            output = output[:, 1:, :, :]

            # query_c_labels = torch.unique(
            #     batch['query_mask_c_0_to_21_ignore_novel']
            # )  # 是语义分割的标签，去除了novel的，所以是0-20； 0是bg, 1-20是类别
            # print(
            #     query_c_labels, batch['class_id'] + 1
            # )  # fold1 0 1 2 3 4 5  11 12 13 14 15 16 17 18 19 20 255

            query_c_labels_one_hot = (
                torch.nn.functional.one_hot(
                    batch['query_mask_c_0_to_21_ignore_novel'].long(), num_classes=256
                )
                .permute(0, 3, 1, 2)
                .float()
            )
            query_c_labels_one_hot = query_c_labels_one_hot[
                :, 1:21, :, :
            ]  # 不要bg, 和255的

            ## ## 一种是筛选出非0的，然后计算loss
            # not0_flag = torch.sum(query_c_labels_one_hot, dim=(2, 3)) != 0
            # output_not0 = output[not0_flag]
            # query_c_labels_one_hot_not0 = query_c_labels_one_hot[not0_flag]

            # loss = torch.nn.BCEWithLogitsLoss()(
            #     output_not0, query_c_labels_one_hot_not0
            # )
            # loss_dice = SoftDiceLoss()(output_not0, query_c_labels_one_hot_not0)
            # loss = loss + loss_dice

            ## 另外一种是直接计算loss，不用筛选
            loss = torch.nn.BCEWithLogitsLoss()(output, query_c_labels_one_hot)
            loss_dice = SoftDiceLoss()(output, query_c_labels_one_hot)
            loss = loss + loss_dice

            ## Nan
            if torch.isnan(loss):
                print('loss is nan', batch['query_path'])
                print('loss is nan', batch['support_path'])
                print()

            # compute the dice loss
            # loss_dice = torch.nn.dice_loss(

            # # select logits from the second dim of output (shape: 16 20 32 512) , by batch['class_id'] (shape: 16), return shape: 16 512 512
            # output_after_sigmoid = F.sigmoid(
            #     output[torch.arange(0, output.shape[0]), batch['class_id']]
            # )
            # pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            # pred_mask_01[output_after_sigmoid >= 0.5] = 1

            pred_mask_c = torch.argmax(output, dim=1)
            pred_mask_01 = torch.zeros_like(pred_mask_c)
            pred_mask_01[
                pred_mask_c == batch['class_id'].unsqueeze(-1).unsqueeze(-1)
            ] = 1

            # loss = torch.nn.CrossEntropyLoss(ignore_index=255)(
            #     output, batch['query_mask_c_0_to_21_ignore_novel'].long()
            # )

            # output_after_softmax = F.softmax(output, dim=1)
            # pred_mask_c = torch.argmax(output_after_softmax, dim=1)
            # pred_mask_01 = torch.zeros_like(pred_mask_c)
            # pred_mask_01[
            #     pred_mask_c == (batch['class_id'] + 1).unsqueeze(-1).unsqueeze(-1)
            # ] = 1

            _all = {
                # 'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'light_weight_text_as_query_RD':
            ## 先只用文本作为query
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            # TODO, r 是否需要norm
            r = cls_token.unsqueeze(1) * self.text_embeddings[
                batch['class_id']
            ].unsqueeze(1)

            condition = torch.cat(
                [r, self.text_embeddings[batch['class_id']].unsqueeze(1)], dim=-1
            )  # [16, 1, 1024]

            condition = einops.rearrange(condition, 'b n l -> (b n) l')
            condition = self.reduction(condition)
            condition = einops.rearrange(condition, '(b n) l -> b n l', b=feat.shape[0])

            # TODO, mixup text embeddings and visual cls_token. and cls_token from support set

            tmp = self.decoder_1(x=condition, context=feat, extract_dots=True)
            _all_x, _all_dots = tmp
            dot = _all_dots[-1]  # 16,8,1,1024
            dot = torch.mean(dot, dim=1, keepdim=False)  # 16,1,1024
            logits = einops.rearrange(
                dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )  # 16,1024
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'light_weight_text_as_query_RD_support':
            ## 先只用文本作为query
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            # TODO, r 是否需要norm. 文本和text的相似性
            support_cls_token = torch.mean(
                batch['support_cls_token'], dim=1, keepdim=False
            )
            support_cls_token = self.proj(support_cls_token[:, :, None, None])[
                :, :, 0, 0
            ]

            def mixing_(support_cls, query_cls, text_embedding):
                if not self.training:
                    alpha = 0.4
                else:
                    alpha = torch.rand(1).item()
                mixed = alpha * support_cls + (1 - alpha) * text_embedding
                r = query_cls.unsqueeze(1) * mixed.unsqueeze(1)
                condition = torch.cat([r, mixed.unsqueeze(1)], dim=-1)
                condition = einops.rearrange(condition, 'b n l -> (b n) l')
                condition = self.reduction(condition)
                condition = einops.rearrange(
                    condition, '(b n) l -> b n l', b=feat.shape[0]
                )
                return condition

            condition = mixing_(
                support_cls=support_cls_token,
                query_cls=cls_token,
                text_embedding=self.text_embeddings[batch['class_id']],
            )

            # TODO, mixup text embeddings and visual cls_token. and cls_token from support set

            tmp = self.decoder_1(x=condition, context=feat, extract_dots=True)
            _all_x, _all_dots = tmp
            dot = _all_dots[-1]  # 16,8,1,1024
            dot = torch.mean(dot, dim=1, keepdim=False)  # 16,1,1024
            logits = einops.rearrange(
                dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )  # 16,1024
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'light_weight_visual_as_query_support':
            ## 先只用文本作为query
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')  # backbone的特征，进过proj

            # TODO, r 是否需要norm. 文本和text的相似性
            support_cls_token = torch.mean(
                batch['support_cls_token'], dim=1, keepdim=False
            )
            support_cls_token = self.proj(support_cls_token[:, :, None, None])[
                :, :, 0, 0
            ]

            def mixing_(support_cls, query_cls, text_embedding):
                # if not self.training:
                #     alpha = 0.4
                # else:
                #     alpha = torch.rand(1).item()

                alpha = 1
                mixed = alpha * support_cls + (1 - alpha) * text_embedding
                r = query_cls.unsqueeze(1) * mixed.unsqueeze(1)
                condition = torch.cat([r, mixed.unsqueeze(1)], dim=-1)
                condition = einops.rearrange(condition, 'b n l -> (b n) l')
                condition = self.reduction(condition)
                condition = einops.rearrange(
                    condition, '(b n) l -> b n l', b=feat.shape[0]
                )
                return condition

            condition = mixing_(
                support_cls=support_cls_token,
                query_cls=cls_token,
                text_embedding=self.text_embeddings[batch['class_id']],
            )

            # TODO, mixup text embeddings and visual cls_token. and cls_token from support set

            tmp = self.decoder_1(x=condition, context=feat, extract_dots=True)

            _all_x, _all_dots = tmp
            dot = _all_dots[-1]  # 16,8,1,1024
            dot = torch.mean(dot, dim=1, keepdim=False)  # 16,1,1024
            logits = einops.rearrange(
                dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
            )  # 16,1024
            # if not self.training:
            #     output = self.refine_output(output, k)

            # reshape output to image size
            output = resize(
                input=logits,
                size=batch['query_img'].shape[2:],
                mode='bilinear',
                align_corners=False,
            )

            loss = torch.nn.BCEWithLogitsLoss()(
                output.squeeze(1), batch['query_mask'].float()
            )

            output_after_sigmoid = F.sigmoid(output.squeeze(1))
            pred_mask_01 = torch.zeros_like(output_after_sigmoid)
            pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'light_weight_text_as_query_multiscale':
            feat = einops.rearrange(feat, 'b c h w -> b (h w) c')
            condition = self.text_embeddings[batch['class_id']].unsqueeze(1)
            # TODO, mixup text embeddings and visual cls_token. and cls_token from support set

            tmp = self.decoder_1(x=condition, context=feat, extract_dots=True)
            _all_x, _all_dots = tmp

            weights = [0.1, 0.3, 0.6]
            loss = 0.0
            for _idx, (dot, weight) in enumerate(zip(_all_dots, weights)):
                dot = torch.mean(dot, dim=1, keepdim=False)  # 16,1,1024
                logits = einops.rearrange(
                    dot, 'b c (h w) -> b c h w', h=hw_shape[0], w=hw_shape[1]
                )  # 16,1024
                # if not self.training:
                #     output = self.refine_output(output, k)

                # reshape output to image size
                output = resize(
                    input=logits,
                    size=batch['query_img'].shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )

                loss = loss + weight * torch.nn.BCEWithLogitsLoss()(
                    output.squeeze(1), batch['query_mask'].float()
                )

                if _idx == len(_all_dots) - 1:
                    output_after_sigmoid = F.sigmoid(output.squeeze(1))
                    pred_mask_01 = torch.zeros_like(output_after_sigmoid)
                    pred_mask_01[output_after_sigmoid >= 0.5] = 1

            _all = {
                'pred_logits': output_after_sigmoid,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all

        if self.decode_mode == 'clip_context_affinity_v1':

            ## 获得每一层的affinity的mask
            all_affinity_masks = []
            for _idx, A in enumerate(batch['affinity']):
                if _idx == 0:
                    continue
                else:  # 去除affinity, resize mask
                    A_ss, A_sq, A_qq = A
                    _size = int(A_ss.size()[-1] ** 0.5)
                    mask = einops.rearrange(
                        F.interpolate(
                            batch['support_masks'],
                            size=(_size, _size),
                            mode='bilinear',
                            align_corners=True,
                        ),  
                        "n c h w -> n (h w) c",
                    ).squeeze(-1)  # torch.Size([8, 2304])

                stack_ids = [0, 1, 3, 4]

                if _idx == 1:
                    tmp = self.affinity_blocks[0](
                        x=self.pe[0](A_ss),
                        context=self.pe[0](A_sq),
                    )
                    tmp = torch.softmax(
                        torch.einsum('ijk,ikl->ijl', tmp, A_qq), dim=1
                    )  # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1)
                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

                    # coarse_mask = coarse_mask * context_weight + coarse_affinity_mask * (
                    #     1 - context_weight
                elif _idx == 2:
                    tmp = self.affinity_blocks[1](
                        x=self.pe[1](A_ss),
                        context=self.pe[1](A_sq),
                    )
                    tmp = torch.softmax(
                        torch.einsum('ijk,ikl->ijl', tmp, A_qq), dim=1
                    )  # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1)
                    # coarse_mask = coarse_mask * context_weight + coarse_affinity_mask * (
                    #     1 - context_weight

                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

                elif _idx == 3:
                    tmp = self.affinity_blocks[2](
                        x=self.pe[2](A_ss),
                        context=self.pe[2](A_sq),
                    )
                    tmp = torch.softmax(
                        torch.einsum('ijk,ikl->ijl', tmp, A_qq), dim=1
                    )  # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1)
                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

            coarse_masks = []
            for affinity_mask in all_affinity_masks:
                coarse_mask_clip = F.interpolate(
                input=batch['clip_probs'],
                size=affinity_mask.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
                coarse_masks.append(torch.cat([coarse_mask_clip, affinity_mask], dim=1))
            
            coarse_masks.reverse()
            coarse_masks1, coarse_masks2, coarse_masks3 = coarse_masks

            coarse_masks1 = self.conv1(coarse_masks1 )  
            coarse_masks2 = self.conv2( coarse_masks2)  
            coarse_masks3 = self.conv3(coarse_masks3)  


                        # multi-scale cascade (pixel-wise addition)
            coarse_masks1 = F.interpolate(
                coarse_masks1,
                coarse_masks2.size()[-2:],
                mode='bilinear',
                align_corners=True,
            )
            mix = coarse_masks1 + coarse_masks2
            mix = self.conv4(mix)  # torch.Size([20, 128, 24, 24])

            mix = F.interpolate(
                mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True
            )
            mix = mix + coarse_masks3
            mix = self.conv5(mix)  # torch.Size([20, 128, 48, 48])

            # skip connect 1/8 and 1/4 features (concatenation)
            nshot=1
            if nshot == 1:
                support_feat = batch['support_feat'][1]
                 # torch.Size([20, 256, 48, 48])
            else:
                # support_feat = (
                #     torch.stack(
                #         [support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]
                #     )
                #     .max(dim=0)
                #     .values
                # )
                pass
            mix = torch.cat(
                (mix, batch['query_feat'][1], support_feat), 1
            )  # torch.Size([20, 640, 48, 48])

            upsample_size = (mix.size(-1) * 2,) * 2
            mix = F.interpolate(
                mix, upsample_size, mode='bilinear', align_corners=True
            )  # torch.Size([20, 640, 96, 96])
            if nshot == 1:
                support_feat =batch['support_feat'][0]
            else:
                pass
                # support_feat = (
                #     torch.stack(
                #         [support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]
                #     )
                #     .max(dim=0)
                #     .values
                # )
            mix = torch.cat(
                (mix,   batch['query_feat'][0] , support_feat), 1
            )  # torch.Size([8, 896, 96, 96])



            mix = torch.cat(
            (
                mix,
                F.interpolate(
                    input=batch['clip_probs'],
                    size=(mix.shape[-2], mix.shape[-1]),
                    mode='bilinear',
                    align_corners=True,
                ),
            ),
            1,
        )  # torch.Size([8, 1024, 96, 96])

            out = self.mixer1(mix)  # torch.Size([8, 64, 96, 96])
            upsample_size = (out.size(-1) * 2,) * 2
            out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)

            out = torch.cat(
                (
                    out,
                    F.interpolate(
                        input=batch['clip_probs'],
                        size=(out.shape[-2], out.shape[-1]),
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                1,
            )  # torch.Size([8, 1024, 96, 96])
            out = self.mixer2(out)  # torch.Size([8, 16, 384, 384])
            upsample_size = (out.size(-1) * 2,) * 2
            out = F.interpolate(
                out, upsample_size, mode='bilinear', align_corners=True
            )  # torch.Size([8, 16, 384, 384])

            out = torch.cat(
                (
                    out,
                    F.interpolate(
                        input=batch['clip_probs'],
                        size=(out.shape[-2], out.shape[-1]),
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                1,
            )  # torch.Size([8, 1024, 96, 96])

            logit_mask = self.mixer3(out)  # torch.Size([8, 2, 384, 384])
            loss = torch.nn.CrossEntropyLoss(ignore_index=255)(
                logit_mask, batch['query_mask'].long()
            )
            pred_mask_01 =  torch.argmax(logit_mask, dim=1)
            _all = {
                'pred_logits': logit_mask,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all


        if self.decode_mode == 'clip_context_affinity_v2':
            ## using sigmoid 激活函数
            ## 获得每一层的affinity的mask
            all_affinity_masks = []
            for _idx, A in enumerate(batch['affinity']):
                if _idx == 0:
                    continue
                else:  # 去除affinity, resize mask
                    A_ss, A_sq, A_qq = A
                    _size = int(A_ss.size()[-1] ** 0.5)
                    mask = einops.rearrange(
                        F.interpolate(
                            batch['support_masks'],
                            size=(_size, _size),
                            mode='bilinear',
                            align_corners=True,
                        ),  
                        "n c h w -> n (h w) c",
                    ).squeeze(-1)  # torch.Size([8, 2304])

                stack_ids = [0, 1, 3, 4]

                if _idx == 1:
                    tmp = self.affinity_blocks[0](
                        x=self.pe[0](A_ss),
                        context=self.pe[0](A_sq),
                    )
                    tmp =  torch.einsum('ijk,ikl->ijl', tmp, A_qq) # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.sigmoid(torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1))
                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

                    # coarse_mask = coarse_mask * context_weight + coarse_affinity_mask * (
                    #     1 - context_weight
                elif _idx == 2:
                    tmp = self.affinity_blocks[1](
                        x=self.pe[1](A_ss),
                        context=self.pe[1](A_sq),
                    )
                    tmp = torch.einsum('ijk,ikl->ijl', tmp, A_qq) # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.sigmoid(torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1))
                    # coarse_mask = coarse_mask * context_weight + coarse_affinity_mask * (
                    #     1 - context_weight

                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

                elif _idx == 3:
                    tmp = self.affinity_blocks[2](
                        x=self.pe[2](A_ss),
                        context=self.pe[2](A_sq),
                    )
                    tmp = torch.einsum('ijk,ikl->ijl', tmp, A_qq)  # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.sigmoid(torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1))
                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

            coarse_masks = []
            for affinity_mask in all_affinity_masks:
                coarse_mask_clip = F.interpolate(
                input=batch['clip_probs'],
                size=affinity_mask.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
                coarse_masks.append(torch.cat([coarse_mask_clip, affinity_mask], dim=1))
            
            coarse_masks.reverse()
            coarse_masks1, coarse_masks2, coarse_masks3 = coarse_masks

            coarse_masks1 = self.conv1(coarse_masks1 )  
            coarse_masks2 = self.conv2( coarse_masks2)  
            coarse_masks3 = self.conv3(coarse_masks3)  


                        # multi-scale cascade (pixel-wise addition)
            coarse_masks1 = F.interpolate(
                coarse_masks1,
                coarse_masks2.size()[-2:],
                mode='bilinear',
                align_corners=True,
            )
            mix = coarse_masks1 + coarse_masks2
            mix = self.conv4(mix)  # torch.Size([20, 128, 24, 24])

            mix = F.interpolate(
                mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True
            )
            mix = mix + coarse_masks3
            mix = self.conv5(mix)  # torch.Size([20, 128, 48, 48])

            # skip connect 1/8 and 1/4 features (concatenation)
            nshot=1
            if nshot == 1:
                support_feat = batch['support_feat'][1]
                 # torch.Size([20, 256, 48, 48])
            else:
                # support_feat = (
                #     torch.stack(
                #         [support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]
                #     )
                #     .max(dim=0)
                #     .values
                # )
                pass
            mix = torch.cat(
                (mix, batch['query_feat'][1], support_feat), 1
            )  # torch.Size([20, 640, 48, 48])

            upsample_size = (mix.size(-1) * 2,) * 2
            mix = F.interpolate(
                mix, upsample_size, mode='bilinear', align_corners=True
            )  # torch.Size([20, 640, 96, 96])
            if nshot == 1:
                support_feat =batch['support_feat'][0]
            else:
                pass
                # support_feat = (
                #     torch.stack(
                #         [support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]
                #     )
                #     .max(dim=0)
                #     .values
                # )
            mix = torch.cat(
                (mix,   batch['query_feat'][0] , support_feat), 1
            )  # torch.Size([8, 896, 96, 96])



            mix = torch.cat(
            (
                mix,
                F.interpolate(
                    input=batch['clip_probs'],
                    size=(mix.shape[-2], mix.shape[-1]),
                    mode='bilinear',
                    align_corners=True,
                ),
            ),
            1,
        )  # torch.Size([8, 1024, 96, 96])

            out = self.mixer1(mix)  # torch.Size([8, 64, 96, 96])
            upsample_size = (out.size(-1) * 2,) * 2
            out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)

            out = torch.cat(
                (
                    out,
                    F.interpolate(
                        input=batch['clip_probs'],
                        size=(out.shape[-2], out.shape[-1]),
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                1,
            )  # torch.Size([8, 1024, 96, 96])
            out = self.mixer2(out)  # torch.Size([8, 16, 384, 384])
            upsample_size = (out.size(-1) * 2,) * 2
            out = F.interpolate(
                out, upsample_size, mode='bilinear', align_corners=True
            )  # torch.Size([8, 16, 384, 384])

            out = torch.cat(
                (
                    out,
                    F.interpolate(
                        input=batch['clip_probs'],
                        size=(out.shape[-2], out.shape[-1]),
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                1,
            )  # torch.Size([8, 1024, 96, 96])

            logit_mask = self.mixer3(out)  # torch.Size([8, 2, 384, 384])
            loss = torch.nn.CrossEntropyLoss(ignore_index=255)(
                logit_mask, batch['query_mask'].long()
            )
            pred_mask_01 =  torch.argmax(logit_mask, dim=1)
            _all = {
                'pred_logits': logit_mask,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all


        if self.decode_mode == 'clip_context_affinity_v3':
            ## using sigmoid 激活函数
            ## 获得每一层的affinity的mask
            all_affinity_masks = []
            for _idx, A in enumerate(batch['affinity']):
                if _idx == 0:
                    continue
                else:  # 去除affinity, resize mask
                    A_ss, A_sq, A_qq = A
                    _size = int(A_ss.size()[-1] ** 0.5)
                    mask = einops.rearrange(
                        F.interpolate(
                            batch['support_masks'],
                            size=(_size, _size),
                            mode='bilinear',
                            align_corners=True,
                        ),  
                        "n c h w -> n (h w) c",
                    ).squeeze(-1)  # torch.Size([8, 2304])

                stack_ids = [0, 1, 3, 4]

                if _idx == 1:
                    tmp = self.affinity_blocks[0](
                        x=self.pe[0](A_ss),
                        context=self.pe[0](A_sq),
                    )
                    tmp =  torch.einsum('ijk,ikl->ijl', tmp, A_qq) # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.sigmoid(torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1))

                    _min = coarse_affinity_mask.min(dim=-1, keepdim=True)[0]
                    _max = coarse_affinity_mask.max(dim=-1, keepdim=True)[0]
                    coarse_affinity_mask = (coarse_affinity_mask - _min) / (_max - _min + 1e-8)


                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

                    # coarse_mask = coarse_mask * context_weight + coarse_affinity_mask * (
                    #     1 - context_weight
                elif _idx == 2:
                    tmp = self.affinity_blocks[1](
                        x=self.pe[1](A_ss),
                        context=self.pe[1](A_sq),
                    )
                    tmp = torch.einsum('ijk,ikl->ijl', tmp, A_qq) # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.sigmoid(torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1))
                    # coarse_mask = coarse_mask * context_weight + coarse_affinity_mask * (
                    #     1 - context_weight

                    _min = coarse_affinity_mask.min(dim=-1, keepdim=True)[0]
                    _max = coarse_affinity_mask.max(dim=-1, keepdim=True)[0]
                    coarse_affinity_mask = (coarse_affinity_mask - _min) / (_max - _min + 1e-8)


                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

                elif _idx == 3:
                    tmp = self.affinity_blocks[2](
                        x=self.pe[2](A_ss),
                        context=self.pe[2](A_sq),
                    )
                    tmp = torch.einsum('ijk,ikl->ijl', tmp, A_qq)  # torch.Size([8, 2304, 256])
                    coarse_affinity_mask = torch.sigmoid(torch.einsum(
                        'ij,ijk->ik', mask, tmp
                    ).unsqueeze(1))


                    _min = coarse_affinity_mask.min(dim=-1, keepdim=True)[0]
                    _max = coarse_affinity_mask.max(dim=-1, keepdim=True)[0]
                    coarse_affinity_mask = (coarse_affinity_mask - _min) / (_max - _min + 1e-8)


                    _size = int(A_ss.size()[-1] ** 0.5)
                    all_affinity_masks.append(einops.rearrange(coarse_affinity_mask, "n c (h w)-> n c h w", h=_size, w=_size))

            coarse_masks = []
            for affinity_mask in all_affinity_masks:
                coarse_mask_clip = F.interpolate(
                input=batch['clip_probs'],
                size=affinity_mask.shape[-2:],
                mode='bilinear',
                align_corners=True,
            )
                coarse_masks.append(torch.cat([coarse_mask_clip, affinity_mask], dim=1))
            
            coarse_masks.reverse()
            coarse_masks1, coarse_masks2, coarse_masks3 = coarse_masks

            coarse_masks1 = self.conv1(coarse_masks1 )  
            coarse_masks2 = self.conv2( coarse_masks2)  
            coarse_masks3 = self.conv3(coarse_masks3)  


                        # multi-scale cascade (pixel-wise addition)
            coarse_masks1 = F.interpolate(
                coarse_masks1,
                coarse_masks2.size()[-2:],
                mode='bilinear',
                align_corners=True,
            )
            mix = coarse_masks1 + coarse_masks2
            mix = self.conv4(mix)  # torch.Size([20, 128, 24, 24])

            mix = F.interpolate(
                mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True
            )
            mix = mix + coarse_masks3
            mix = self.conv5(mix)  # torch.Size([20, 128, 48, 48])

            # skip connect 1/8 and 1/4 features (concatenation)
            nshot=1
            if nshot == 1:
                support_feat = batch['support_feat'][1]
                 # torch.Size([20, 256, 48, 48])
            else:
                # support_feat = (
                #     torch.stack(
                #         [support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]
                #     )
                #     .max(dim=0)
                #     .values
                # )
                pass
            mix = torch.cat(
                (mix, batch['query_feat'][1], support_feat), 1
            )  # torch.Size([20, 640, 48, 48])

            upsample_size = (mix.size(-1) * 2,) * 2
            mix = F.interpolate(
                mix, upsample_size, mode='bilinear', align_corners=True
            )  # torch.Size([20, 640, 96, 96])
            if nshot == 1:
                support_feat =batch['support_feat'][0]
            else:
                pass
                # support_feat = (
                #     torch.stack(
                #         [support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]
                #     )
                #     .max(dim=0)
                #     .values
                # )
            mix = torch.cat(
                (mix,   batch['query_feat'][0] , support_feat), 1
            )  # torch.Size([8, 896, 96, 96])



            mix = torch.cat(
            (
                mix,
                F.interpolate(
                    input=batch['clip_probs'],
                    size=(mix.shape[-2], mix.shape[-1]),
                    mode='bilinear',
                    align_corners=True,
                ),
            ),
            1,
        )  # torch.Size([8, 1024, 96, 96])

            out = self.mixer1(mix)  # torch.Size([8, 64, 96, 96])
            upsample_size = (out.size(-1) * 2,) * 2
            out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)

            out = torch.cat(
                (
                    out,
                    F.interpolate(
                        input=batch['clip_probs'],
                        size=(out.shape[-2], out.shape[-1]),
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                1,
            )  # torch.Size([8, 1024, 96, 96])
            out = self.mixer2(out)  # torch.Size([8, 16, 384, 384])
            upsample_size = (out.size(-1) * 2,) * 2
            out = F.interpolate(
                out, upsample_size, mode='bilinear', align_corners=True
            )  # torch.Size([8, 16, 384, 384])

            out = torch.cat(
                (
                    out,
                    F.interpolate(
                        input=batch['clip_probs'],
                        size=(out.shape[-2], out.shape[-1]),
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                1,
            )  # torch.Size([8, 1024, 96, 96])

            logit_mask = self.mixer3(out)  # torch.Size([8, 2, 384, 384])
            loss = torch.nn.CrossEntropyLoss(ignore_index=255)(
                logit_mask, batch['query_mask'].long()
            )
            pred_mask_01 =  torch.argmax(logit_mask, dim=1)
            _all = {
                'pred_logits': logit_mask,
                'pred_mask_01': pred_mask_01,
                'loss': loss,
            }
            return _all















    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.text_embeddings[:, :, None, None])

        return output

    def cls_seg_v1(self, feat_i, cls_i):
        feat_i = feat_i / feat_i.norm(dim=1, keepdim=True)
        output = F.conv2d(feat_i, cls_i[:, :, None, None])

        return output

    def refine_output(self, output, k):
        if self.pd_thresh > 0:
            N, C, H, W = output.shape
            _output = F.softmax(output * 100, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < self.pd_thresh)[:, :, None, None].expand(
                N, C, H, W
            )
            output[selected_cls] = -100

        if k is not None and self.ks_thresh > 0:
            output = F.softmax(output * 100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            # softmax
            # weight = k @ k.transpose(-2, -1)
            # weight = F.softmax(weight, dim=-1)
            # L2 distance
            k = F.normalize(k, p=2)
            weight = k @ k.transpose(-2, -1)

            selected_pos = output.max(dim=-1, keepdim=True)[0] < self.ks_thresh
            selected_pos = selected_pos.expand(-1, -1, C)

            weighted_output = weight @ output
            output[selected_pos] = weighted_output[selected_pos]
            output = output.transpose(-2, -1).view(N, C, H, W)

        return output

    # def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
    #     raise RuntimeError('MaskClip is not trainable. Try MaskClip+ instead.')

    def build_conv_block(
        self, in_channel, out_channels, kernel_sizes, spt_strides, group=4
    ):
        r"""bulid conv blocks"""
        assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

        building_block_layers = []
        for idx, (outch, ksz, stride) in enumerate(
            zip(out_channels, kernel_sizes, spt_strides)
        ):
            inch = in_channel if idx == 0 else out_channels[idx - 1]
            pad = ksz // 2

            building_block_layers.append(
                nn.Conv2d(
                    in_channels=inch,
                    out_channels=outch,
                    kernel_size=ksz,
                    stride=stride,
                    padding=pad,
                )
            )
            building_block_layers.append(nn.GroupNorm(group, outch))
            building_block_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*building_block_layers)

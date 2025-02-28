import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy


def trunc_normal_init(
    module: nn.Module,
    mean: float = 0,
    std: float = 1,
    a: float = -2,
    b: float = 2,
    bias: float = 0,
) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class TPN_Decoder(TransformerDecoder):
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None:
            output = self.norm(output)

        return outputs, attns


class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1)
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]

        q = (
            self.q(xq)
            .reshape(B, Nq, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(xk)
            .reshape(B, Nk, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(xv)
            .reshape(B, Nv, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register_module()
class ATMSingleHead(BaseDecodeHead):
    def __init__(
        self,
        img_size,
        in_channels,
        embed_dims=768,
        num_layers=3,
        num_heads=8,
        use_stages=3,
        use_proj=True,
        CE_loss=False,
        crop_train=False,
        **kwargs,
    ):
        super(ATMSingleHead, self).__init__(in_channels=in_channels, **kwargs)

        self.image_size = img_size
        self.use_stages = use_stages
        self.crop_train = crop_train
        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []
        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=0.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            decoder_layer = TPN_DecoderLayer(
                d_model=dim, nhead=nhead, dim_feedforward=dim * 4
            )
            decoder = TPN_Decoder(decoder_layer, num_layers)
            self.add_module("decoder_{}".format(i + 1), decoder)
            atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders
        self.q = nn.Embedding(1, dim)

        self.class_embed = nn.Linear(dim,self.num_classes + 1)  # TODO +1 for background
        self.CE_loss = CE_loss
        delattr(self, 'conv_seg')

        self.proj = nn.Conv2d(768, 512, 1, bias=False)
        self.text_embeddings_path = (
            'repository/MaskCLIP/pretrain/voc_ViT16_clip_text.pth'
        )
        self.visual_projs_path = 'repository/MaskCLIP/pretrain/ViT16_clip_weights.pth'
        self.register_buffer('text_embeddings', torch.randn(20, 512))
        self.load_text_embeddings()
        self.init_weights()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=0.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

        if self.text_embeddings_path is None:
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.load_text_embeddings()
        self.load_visual_projs()

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings[:, :] = loaded[:, :]

    def load_visual_projs(self):
        loaded = torch.load(self.visual_projs_path, map_location='cuda')
        # attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        attrs = ['proj']

        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)

    # def forward(self, batch):
    #     inputs = batch['feat_q']
    #     x = self._transform_inputs(inputs)
    #     q, k, v, cls_token = None, None, None, None
    #     if isinstance(x, list) and len(x) == 4:
    #         x, q, k, v = x  # 现在使用的是这个
    #         if isinstance(x, list) and len(x) == 2:
    #             x, cls_token = x
    #         if v is not None:
    #             feat = self.proj(v)  # 现在使用的是这个
    #         else:
    #             feat = self.proj(x)

    #         # feat = self.proj(q)
    #         if cls_token is not None:
    #             cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]

    #     # 显示的输入变成了v 经过proj的了。
    #     x = []

    #     inputs = [feat]

    #     # for stage_ in inputs[: self.use_stages]:
    #     #     x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)

    #     for stage_ in inputs:
    #         x.append(self.d4_to_d3(stage_) if stage_.dim() > 3 else stage_)

    #     x.reverse()
    #     bs = x[0].size()[0]

    #     laterals = []
    #     attns = []
    #     maps_size = []
    #     qs = []

    #     for idx, (x_, proj_, norm_) in enumerate(
    #         zip(x, self.input_proj, self.proj_norm)
    #     ):
    #         lateral = norm_(proj_(x_))
    #         if idx == 0:
    #             laterals.append(lateral)
    #         else:
    #             if laterals[idx - 1].size()[1] == lateral.size()[1]:
    #                 laterals.append(lateral + laterals[idx - 1])
    #             else:
    #                 # nearest interpolate
    #                 l_ = self.d3_to_d4(laterals[idx - 1])
    #                 l_ = F.interpolate(l_, scale_factor=2, mode="nearest")
    #                 l_ = self.d4_to_d3(l_)
    #                 laterals.append(l_ + lateral)

    #     lateral = laterals[-1]
    #     q_text = self.text_embeddings[batch['class_id']]
    #     # concat q_text and self.q.weight
    #     q = torch.cat([q_text, self.q.weight], dim=0).repeat(bs, 1, 1).transpose(0, 1)
    #     # q = self.q.weight.repeat(bs, 1, 1).transpose(0, 1)

    #     for idx, decoder_ in enumerate(self.decoder):
    #         q_, attn_ = decoder_(q, lateral.transpose(0, 1))
    #         for q, attn in zip(q_, attn_):
    #             attn = attn.transpose(-1, -2)
    #             attn = self.d3_to_d4(attn)
    #             maps_size.append(attn.size()[-2:])
    #             qs.append(q.transpose(0, 1))
    #             attns.append(attn)
    #     qs = torch.stack(qs, dim=0)  # 3 1 2 512
    #     outputs_class = self.class_embed(qs)  # # 3 1 2 512 512*2  变成了 3 1  2 2
    #     out = {"pred_logits": outputs_class[-1]}  # 1,2,2. 1是bs, 最后一个2是类别，中间的2是什么？

    #     outputs_seg_masks = []
    #     size = maps_size[-1]

    #     for i_attn, attn in enumerate(attns):
    #         if True:
    #             # if i_attn == 0:
    #             outputs_seg_masks.append(
    #                 F.interpolate(attn, size=size, mode='bilinear', align_corners=False)
    #             )
    #         else:
    #             outputs_seg_masks.append(
    #                 outputs_seg_masks[i_attn - 1]
    #                 + F.interpolate(
    #                     attn, size=size, mode='bilinear', align_corners=False
    #                 )
    #             )

    #     out["pred_masks"] = F.interpolate(
    #         outputs_seg_masks[-1],
    #         size=(self.image_size, self.image_size),
    #         mode='bilinear',
    #         align_corners=False,
    #     )  # 这个其实是logits 1,2，224，224

    #     # out["pred"] = self.semantic_inference(
    #     #     out["pred_logits"], out["pred_masks"]
    #     # )  # 这个是1,1,224,224。  这个是什么？ 可以直接设置一个阈值，然后转换成mask吗？

    #     # if self.training:
    #     #     # [l, bs, queries, embed]
    #     #     outputs_seg_masks = torch.stack(outputs_seg_masks, dim=0)
    #     #     out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_seg_masks)
    #     # else:
    #     #     return out["pred"]

    #     return out

    def forward(self, batch):
        inputs = batch['feat_q']
        x = self._transform_inputs(inputs)
        q, k, v, cls_token = None, None, None, None
        if isinstance(x, list) and len(x) == 4:
            x, q, k, v = x  # 现在使用的是这个
            if isinstance(x, list) and len(x) == 2:
                x, cls_token = x
            if v is not None:
                feat = self.proj(v)  # 现在使用的是这个
            else:
                feat = self.proj(x)

            if cls_token is not None:
                cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]

        output = self.cls_seg(feat)

        out = {'pred_masks': output}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        if self.CE_loss:
            return super().losses(seg_logit["pred"], seg_label)

        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)
            if self.crop_train:
                # mask seg_label by crop_idx
                bs, h, w = seg_label.size()
                mask_label = (
                    seg_label.reshape(bs, h // 16, 16, w // 16, 16)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(bs, h * w // 256, 256)
                )
                empty_label = torch.zeros_like(mask_label) + self.ignore_index
                empty_label[:, self.crop_idx] = mask_label[:, self.crop_idx]
                seg_label = (
                    empty_label.reshape(bs, h // 16, w // 16, 16, 16)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(bs, h, w)
                )
            loss = self.loss_decode(
                seg_logit, seg_label, ignore_index=self.ignore_index
            )

            loss['acc_seg'] = accuracy(
                seg_logit["pred"], seg_label, ignore_index=self.ignore_index
            )
            return loss

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.text_embeddings[:, :, None, None])

        return output

    def forward_train(self, batch):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(batch)["pred_masks"]
        # losses = self.losses(seg_logits, gt_semantic_seg)  # todo
        losses = self.cross_entropy_loss(seg_logits, batch['query_mask'].long())  # todo
        return losses, seg_logits
 
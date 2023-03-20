# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from xmlrpc.client import Boolean

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.utils.weight_init import constant_init, kaiming_init, trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F

from mmseg.ops import resize
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed


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


@BACKBONES.register_module()
class VisionTransformerDPT(BaseModule):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        patch_bias=True,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=-1,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        pre_norm=False,
        final_norm=False,
        return_qkv=False,
        skip_last_attn=False,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False,
        with_cp=False,
        pretrained=None,
        init_cfg=None,
        dpt_cfg={
            'type': 'input',  # 'deep',"conditional"
            'num_token': 10,
            "num_layers": 12, # 只有deepdpt需要
        },
    ):
        super(VisionTransformerDPT, self).__init__(init_cfg=init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, (
                f'The size of image should have length 1 or 2, '
                f'but got {len(img_size)}'
            )

        if output_cls_token:
            assert with_cls_token is True, (
                f'with_cls_token must be True if'
                f'set output_cls_token to True, but got {with_cls_token}'
            )

        assert not (
            init_cfg and pretrained
        ), 'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn(
                'DeprecationWarning: pretrained is deprecated, '
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            bias=patch_bias,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        self.dpt_cfg = dpt_cfg
        if self.dpt_cfg is not None:
            
            # 非空的话，就有这个可学习的prompt
            self.prompt_embeddings = nn.Parameter(
                    torch.zeros(1, self.dpt_cfg['num_token'], embed_dims)
                )
            nn.init.uniform_(self.prompt_embeddings.data, -0.5, 0.5)

            self.prompt_process = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.Dropout(0.1),
            )
            # init prompt_process
            for m in self.prompt_process.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)


            if self.dpt_cfg['type'] == 'deep':
                total_d_layer = self.dpt_cfg["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(
                        total_d_layer,
                        self.dpt_cfg['num_token'],
                        embed_dims,
                    )
                )
                nn.init.uniform_(self.deep_prompt_embeddings.data, -0.5, 0.5)


            if self.dpt_cfg['type'] == 'conditional':
                self.text_embeddings_path='repository/MaskCLIP/pretrain/voc_ViT16_clip_text.pth'
                print('Loading text embeddings from {}'.format(self.text_embeddings_path))
                self.register_buffer('text_embeddings', torch.randn(20, 512))
                loaded = torch.load(self.text_embeddings_path, map_location='cuda')
                self.text_embeddings[:, :] = loaded[:, :]

                self.prompt_process_ChangeDim = nn.Sequential(
                        nn.Linear(512, embed_dims),
                        nn.LayerNorm(embed_dims),
                        nn.Dropout(0.1),
                    )
                # init prompt_process
                for m in self.prompt_process_ChangeDim.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, std=0.02)
                        nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)


            if self.dpt_cfg['type'] in ['conditional', 'conditional_learnable']:
                pass

        if isinstance(out_indices, int):  # 如果是int类型，类型，是指从0-11的
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices  # 如果是list或者tuple类型，这个list或者tuple
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

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
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                )
            )

        self.pre_norm = pre_norm
        if pre_norm:
            self.norm0_name, norm0 = build_norm_layer(norm_cfg, embed_dims, postfix=0)
            self.add_module(self.norm0_name, norm0)

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        self.return_qkv = [False] * num_layers  # 初始设置为都不返回返回
        if isinstance(return_qkv, bool):  # 如果是bool类型
            for out_i in self.out_indices:
                self.return_qkv[out_i] = return_qkv  # 如果是True, 返回在out_indices中的层的qkv
        elif isinstance(return_qkv, list) or isinstance(return_qkv, tuple):
            for i, out_i in enumerate(self.out_indices):  # 如果是list或者tuple，则根据对应位置来返回
                self.return_qkv[out_i] = (return_qkv[i],)
        else:
            raise TypeError('return_qkv must be type of bool, list or tuple')

        self.skip_last_attn = skip_last_attn

    @property
    def norm0(self):
        return getattr(self, self.norm0_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if (
            isinstance(self.init_cfg, dict)
            and self.init_cfg.get('type') == 'Pretrained'
        ):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu'
            )

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(
                        msg=f'Resize the pos_embed shape from '
                        f'{state_dict["pos_embed"].shape} to '
                        f'{self.pos_embed.shape}'
                    )
                    h, w = self.img_size
                    pos_size = int(math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size),
                        self.interpolate_mode,
                    )

            print(self.load_state_dict(state_dict, False))
        elif self.init_cfg is not None:
            super(VisionTransformerDPT, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.cls_token, std=0.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0.0, std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.0)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.0)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positiong embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert (
            patched_img.ndim == 3 and pos_embed.ndim == 3
        ), 'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if (
                pos_len
                == (self.img_size[0] // self.patch_size)
                * (self.img_size[1] // self.patch_size)
                + 1
            ):
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(pos_embed.shape)
                )
            pos_embed = self.resize_pos_embed(
                pos_embed, hw_shape, (pos_h, pos_w), self.interpolate_mode
            )
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w) :]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]
        ).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode
        )
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def incorporate_prompt(self, x, prompt):
        B = x.shape[0]
        if self.dpt_cfg['type']  in ['input','deep']:
            x = torch.cat(
                [
                    x[:, :1, :],
                    self.prompt_process(prompt).expand(B, -1, -1),
                    x[:, 1:, :],
                ],
                dim=1,
            )
        if self.dpt_cfg['type'] == 'conditional':
            x = torch.cat(
                [
                    x[:, :1, :],
                    prompt,
                    x[:, 1:, :],
                ],
                dim=1,
            )

        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def forward(self, inputs, **kwargs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if self.dpt_cfg is not None:
            if self.dpt_cfg['type'] == 'input' or self.dpt_cfg['type'] == 'deep':
                prompt = self.prompt_embeddings
            elif self.dpt_cfg['type'] == 'conditional':
                class_id = kwargs['class_id']
                clsss_conditional_prompt = self.prompt_process_ChangeDim(self.text_embeddings[class_id]).unsqueeze(1)
                # clsss_conditional_prompt = clsss_conditional_prompt + self.prompt_process(self.prompt_embeddings).expand(B, -1, -1)
                prompt = clsss_conditional_prompt
                
            else:
                raise ValueError(
                    'Unexpected dpt_cfg type, got {}.'.format(self.dpt_cfg['type'])
                )
            x = self.incorporate_prompt(x, prompt)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        if self.pre_norm:
            x = self.norm0(x)

        if self.dpt_cfg is None:
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

        if self.dpt_cfg['type'] == 'input':
            outs = []
            for i, layer in enumerate(self.layers):
                x, q, k, v = layer(
                    x,
                    self.return_qkv[i]
                    or (i == len(self.layers) - 1 and self.skip_last_attn),
                )
                if i == len(self.layers) - 1:  # 最后一层比较特殊，需要做一些处理
                    if self.final_norm:
                        x = self.norm1(x)
                        if self.return_qkv[i]:
                            v = self.norm1(v)
                    if self.skip_last_attn:
                        if self.with_cls_token:
                            x[:, 1:] = v[:, 1:]
                        else:
                            x = v
                if i in self.out_indices:  # 对返回值的处理

                    prompt_removed_x = torch.cat(
                        (x[:, :1, :], x[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_q = torch.cat(
                        (q[:, :1, :], q[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_k = torch.cat(
                        (k[:, :1, :], k[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_v = torch.cat(
                        (v[:, :1, :], v[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )

                    if self.with_cls_token:
                        # Remove class token and reshape token for decoder head
                        out = prompt_removed_x[:, 1:]
                    else:
                        out = prompt_removed_x
                    B, _, C = out.shape
                    out = (
                        out.reshape(B, hw_shape[0], hw_shape[1], C)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                    if self.output_cls_token:
                        out = [out, prompt_removed_x[:, 0]]
                    if self.return_qkv[i]:
                        if self.with_cls_token:
                            q = prompt_removed_q[:, 1:]
                            k = prompt_removed_k[:, 1:]
                            v = prompt_removed_v[:, 1:]
                        v = (
                            v.reshape(B, hw_shape[0], hw_shape[1], C)
                            .permute(0, 3, 1, 2)
                            .contiguous()
                        )
                        out = [out, q, k, v]
                    outs.append(out)

            return tuple(outs)

        if self.dpt_cfg['type'] == 'deep':
            outs = []
            for i, layer in enumerate(self.layers):

                if i == 0:  # 在第一层的时候，直接将prompt加入到输入中
                    x, q, k, v = layer(
                        x,
                        self.return_qkv[i]
                        or (i == len(self.layers) - 1 and self.skip_last_attn),
                    )
                else:  # 在第二层之后的，需要将prompt替换x中的东西，不能直接用
                    deep_prompt = self.prompt_process(
                        self.deep_prompt_embeddings[i - 1]
                    ).expand(B, -1, -1)

                    # # 不做处理的话，直接将prompt加入到输入中
                    # deep_prompt = self.deep_prompt_embeddings[i - 1].expand(B, -1, -1)
                    x = torch.cat(
                        [
                            x[:, :1, :],
                            deep_prompt,
                            x[:, -hw_shape[0] * hw_shape[1] :, :],
                        ],
                        dim=1,
                    )
                    x, q, k, v = layer(
                        x,
                        self.return_qkv[i]
                        or (i == len(self.layers) - 1 and self.skip_last_attn),
                    )

                if i == len(self.layers) - 1:  # 最后一层比较特殊，需要做一些处理
                    if self.final_norm:
                        x = self.norm1(x)
                        if self.return_qkv[i]:
                            v = self.norm1(v)
                    if self.skip_last_attn:
                        if self.with_cls_token:
                            x[:, 1:] = v[:, 1:]
                        else:
                            x = v
                if i in self.out_indices:  # 对返回值的处理

                    prompt_removed_x = torch.cat(
                        (x[:, :1, :], x[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_q = torch.cat(
                        (q[:, :1, :], q[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_k = torch.cat(
                        (k[:, :1, :], k[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_v = torch.cat(
                        (v[:, :1, :], v[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )

                    if self.with_cls_token:
                        # Remove class token and reshape token for decoder head
                        out = prompt_removed_x[:, 1:]
                    else:
                        out = prompt_removed_x
                    B, _, C = out.shape
                    out = (
                        out.reshape(B, hw_shape[0], hw_shape[1], C)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                    if self.output_cls_token:
                        out = [out, prompt_removed_x[:, 0]]
                    if self.return_qkv[i]:
                        if self.with_cls_token:
                            q = prompt_removed_q[:, 1:]
                            k = prompt_removed_k[:, 1:]
                            v = prompt_removed_v[:, 1:]
                        v = (
                            v.reshape(B, hw_shape[0], hw_shape[1], C)
                            .permute(0, 3, 1, 2)
                            .contiguous()
                        )
                        out = [out, q, k, v]
                    outs.append(out)

            return tuple(outs)

        if self.dpt_cfg['type'] == 'conditional':
            outs = []
            for i, layer in enumerate(self.layers):
                x, q, k, v = layer(
                    x,
                    self.return_qkv[i]
                    or (i == len(self.layers) - 1 and self.skip_last_attn),
                )
                if i == len(self.layers) - 1:  # 最后一层比较特殊，需要做一些处理
                    if self.final_norm:
                        x = self.norm1(x)
                        if self.return_qkv[i]:
                            v = self.norm1(v)
                    if self.skip_last_attn:
                        if self.with_cls_token:
                            x[:, 1:] = v[:, 1:]
                        else:
                            x = v
                if i in self.out_indices:  # 对返回值的处理

                    prompt_removed_x = torch.cat(
                        (x[:, :1, :], x[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_q = torch.cat(
                        (q[:, :1, :], q[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_k = torch.cat(
                        (k[:, :1, :], k[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )
                    prompt_removed_v = torch.cat(
                        (v[:, :1, :], v[:, -hw_shape[0] * hw_shape[1] :, :]), dim=1
                    )

                    if self.with_cls_token:
                        # Remove class token and reshape token for decoder head
                        out = prompt_removed_x[:, 1:]
                    else:
                        out = prompt_removed_x
                    B, _, C = out.shape
                    out = (
                        out.reshape(B, hw_shape[0], hw_shape[1], C)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                    if self.output_cls_token:
                        out = [out, prompt_removed_x[:, 0]]
                    if self.return_qkv[i]:
                        if self.with_cls_token:
                            q = prompt_removed_q[:, 1:]
                            k = prompt_removed_k[:, 1:]
                            v = prompt_removed_v[:, 1:]
                        v = (
                            v.reshape(B, hw_shape[0], hw_shape[1], C)
                            .permute(0, 3, 1, 2)
                            .contiguous()
                        )
                        out = [out, q, k, v]
                    outs.append(out)

            return tuple(outs)



    def train(self, mode=True):
        # super(VisionTransformerDPT, self).train(mode)
        # if mode and self.norm_eval:
        #     for m in self.modules():
        #         if isinstance(m, nn.LayerNorm):
        #             m.eval()
        pass

    def set_prompt_train_mode(self, mode):
        # super(VisionTransformerDPT, self).train(mode)
        # if mode and self.norm_eval:
        #     for m in self.modules():
        #         if isinstance(m, nn.LayerNorm):
        #             m.eval()
        if mode:
            self.eval() # 先全部调整为eval模式
            self.prompt_process.train() # 再将prompt_process调整为train模式

            # 然后设置梯度

            for name, param in self.named_parameters():
                if name.startswith('prompt_embeddings') or name.startswith('prompt_process') or name.startswith('deep_prompt_embeddings'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        else:
            for module in self.modules():
                module.train(mode)



 
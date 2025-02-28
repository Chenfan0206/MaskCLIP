# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .isa_head import ISAHead
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead
from .point_head import PointHead
from .psa_head import PSAHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .stdc_head import STDCHead
from .uper_head import UPerHead
from .maskclip_head import MaskClipHead
from .maskclip_plus_head import MaskClipPlusHead
from .aspp_headv2 import ASPPHeadV2
from .atm_head import ATMHead
from .atm_single_head import ATMSingleHead
from .tpn_atm_head import TPNATMHead
from .seg_head import SegHead
from .seg_text_as_conditi_head import SegTextAsConditionHead
from .seg_text_as_conditi_head_v1 import SegTextAsConditionHeadV1
from .seg_text_as_conditi_head_v2 import SegTextAsConditionHeadV2
from .seg_text_as_conditi_head_v3 import SegTextAsConditionHeadV3

__all__ = [
    'FCNHead',
    'PSPHead',
    'ASPPHead',
    'PSAHead',
    'NLHead',
    'GCHead',
    'CCHead',
    'UPerHead',
    'DepthwiseSeparableASPPHead',
    'ANNHead',
    'DAHead',
    'OCRHead',
    'EncHead',
    'DepthwiseSeparableFCNHead',
    'FPNHead',
    'EMAHead',
    'DNLHead',
    'PointHead',
    'APCHead',
    'DMHead',
    'LRASPPHead',
    'SETRUPHead',
    'SETRMLAHead',
    'DPTHead',
    'SETRMLAHead',
    'SegformerHead',
    'ISAHead',
    'STDCHead',
    'MaskClipHead',
    'MaskClipPlusHead',
    'ASPPHeadV2',
    'ATMHead',
    'ATMSingleHead',
    'TPNATMHead',
    'SegHead',
    'SegTextAsConditionHead',
    'SegTextAsConditionHeadV1',
    "SegTextAsConditionHeadV2",
    "SegTextAsConditionHeadV3",
]

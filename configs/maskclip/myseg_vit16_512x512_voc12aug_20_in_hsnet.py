_base_ = [
    '../_base_/models/maskclip_vit16.py',
    '../_base_/datasets/pascal_voc12_aug_20.py',
    # '../_base_/models/maskclip_vit16.py', '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py',
]
model = dict(
    decode_head=dict(
        type='SegHead',
        num_classes=20,
        text_categories=20,
        text_channels=512,
        text_embeddings_path='repository/MaskCLIP/pretrain/voc_ViT16_clip_text.pth',
        visual_projs_path='repository/MaskCLIP/pretrain/ViT16_clip_weights.pth',
        # ks_thresh=1.0,
        pd_thresh=0.5,
    ),
)

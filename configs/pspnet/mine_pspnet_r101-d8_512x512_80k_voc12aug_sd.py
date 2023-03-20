_base_ = [
    '../_base_/models/pspnet_r50-d8.py',
    '../_base_/datasets/pascal_voc12_sd_20.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py',
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=20),
    auxiliary_head=dict(num_classes=20),
)

checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', pre_eval=True)

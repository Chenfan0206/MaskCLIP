_base_ = [
    '../../_base_/models/maskclip_plus_r50.py',
    '../../_base_/datasets/pascal_sd20all_voc15_fold2.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py',
]

# suppress_labels = list(range(15, 20))
# suppress_labels = list(range(0, 5))
suppress_labels = list(range(10, 15))
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        clip_unlabeled_cats=suppress_labels,
        unlabeled_cats=suppress_labels,
        start_clip_guided=(1, 1999),
        start_self_train=(2000, -1),
    ),
)

find_unused_parameters = True

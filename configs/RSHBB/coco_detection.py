# dataset settings
dataset_type = 'CocoDataset'
#data_root = 'data/coco/'
data_root = ''

data_root_img_train=''
data_root_img_val=''
data_root_img_test=''
data_root_test_vis=''


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# for sgdet
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_pipeline =[
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1024, 1024),
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='ImageToTensor', keys=['img']),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]


# for sgdet
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]



data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + '',
        img_prefix=data_root_img_train + 'images',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '',
        img_prefix=data_root_img_val + 'images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '',
        img_prefix=data_root_img_test + 'images',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

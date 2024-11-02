_base_ = ['../yolox_s_8xb8-300e_coco.py']

data_root = 'data/AirSim_drone/'

DEPTH_RANGE = 80

img_scale = (720, 1280)
num_classes = 1
classes = ['drone',]

disp_thr_h = 1000  # remove
disp_thr_l = 0

deepen_factor = 0.33
widen_factor = 0.5

save_epoch_intervals = 5
train_batch_size_per_gpu = 8
train_num_workers = 16
val_batch_size_per_gpu = 1
val_num_workers = 2

max_epochs = 50
num_last_epochs = 5


model = dict(
    _scope_='mmyolo',
    type='YOLODetector',
    use_syncbn=False,
    data_preprocessor=dict(
        type='DetDataPreprocessor_Disparity',
        pad_size_divisor=32,
        batch_augments=None
        ),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),      
        ),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.5)),
    init_cfg=dict(
        _delete_=True,
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_8xb8-300e_coco/yolox_s_8xb8-300e_coco_20220917_030738-d7e60cb2.pth'
        )
)

pre_transform = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]

train_pipeline_stage1 = [
    *pre_transform,
    # dict(
    #     type='Mosaic',
    #     img_scale=img_scale,
    #     pad_val=114.0,
    #     pre_transform=pre_transform),
    # dict(
    #     type='mmdet.RandomAffine',
    #     scaling_ratio_range=(0.1, 2),
    #     border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='Resize_Disparity', scale=img_scale, keep_ratio=True),
    dict(
        type='YOLOXMixUp_Disparity',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip_Disparity', prob=0.5),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(
        type='PackDetInputs_Disparity',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='Resize_Disparity', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad_Disparity',
        pad_to_square=False,
        size_divisor=32,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0), disp=0, disp_mask=0)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='RandomFlip_Disparity', prob=0.5),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='PackDetInputs_Disparity')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Disp2ColorImg'),
    # dict(type='TLBRCrop', crop_size=crop_size),  # -1 denotes bottom-right of original image
    dict(type='Resize_Disparity', scale=img_scale, keep_ratio=True),
    dict(type='Pad_Disparity',
         size_divisor=32,
         pad_val=dict(img=(114.0, 114.0, 114.0), disp=0, disp_mask=0)),
    dict(
        type='PackDetInputs_Disparity',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='AirSimDroneCoco',
        data_root=data_root,
        ann_file=f'annotations/train_cocoformat_{DEPTH_RANGE}.json',
        data_prefix=dict(img_path='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline_stage1))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='AirSimDroneCoco',
        data_root=data_root,
        ann_file=f'annotations/val_cocoformat_{DEPTH_RANGE}.json',
        data_prefix=dict(img_path='val/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

# optimizer
# default 8 gpu
base_lr = 0.001 / 8 * train_batch_size_per_gpu / 1

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# learning policy
param_scheduler = [
    dict(
        # use quadratic formula to warm up 1 epochs
        # and lr is updated by iteration
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=2,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 1 to 70 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=2,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 10 epochs
        type='mmdet.ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=200),
    checkpoint=dict(
        type='CheckpointHook', interval=save_epoch_intervals, max_keep_ckpts=3, save_best='auto'))

custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    # dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)
# auto_scale_lr = dict(base_batch_size=64)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + f'annotations/val_cocoformat_{DEPTH_RANGE}.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
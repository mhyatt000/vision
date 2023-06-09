# for now
dataset_type = None
data_root = None

pipeline = [
    dict(type="DecordInit"),
    dict(type="SampleFrames", clip_len=1, frame_interval=1, num_clips=8),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, 256)),
    dict(type="RandomResizedCrop"),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False,
    ),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]

data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/train.json",
        data_prefix=data_root + "train",
        pipeline=pipeline,
    ),
    # val=dict(...),
    # test=dict(...),
)

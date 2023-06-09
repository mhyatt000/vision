from mmcv.runner import build_runner
from mmcv.parallel import MMDataParallel
from mmcv import Config
from mmcv.utils import get_root_logger






# VideoReader
video = mmcv.VideoReader('test.mp4')

print(len(video))
print(video.width, video.height, video.resolution, video.fps)

for frame in video:
    print(frame.shape)

img = video.read()
img = video[100]
img = video[5:10]

# Editing utils

mmcv.cut_video('test.mp4', 'clip1.mp4', start=3, end=10, vcodec='h264')
mmcv.concat_video(['clip1.mp4', 'clip2.mp4'], 'joined.mp4', log_level='quiet')
mmcv.resize_video('test.mp4', 'resized1.mp4', (360, 240))
mmcv.resize_video('test.mp4', 'resized2.mp4', ratio=2)

data = dict(
    videos_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/train.json",
        data_prefix=data_root + "train",
        pipeline=[
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
        ],
    ),
    # val=dict(...),
    # test=dict(...),
)


data_cfg = dict()

pipe_cfg = dict()

loader_cfg = dict()

def build_data_pipeline(cfg):
    datasets = [build_dataset(cfg.data.train)]
    data_loaders = [
        MMDataParallel(
            build_dataloader(
                dataset,
                cfg.data.videos_per_gpu,
                cfg.data.workers_per_gpu,
                cfg.gpus,
                dist=True,
            )
        )
        for dataset in datasets
    ]

    return data_loaders


def main():
    # cfg_path = "path_to_your_config_file"
    # cfg = Config.fromfile(cfg_path)
    data_loaders = build_data_pipeline(cfg)

    print(data_loaders)

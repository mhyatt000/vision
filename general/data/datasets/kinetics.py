"""kinetics 400, 600, 700 datasets"""

from general.config import cfg
import torchvision


class Kinetics(torchvision.datasets.Kinetics):
    """docstring"""

    def __init__(
        self,
        *args,
        root="/grand/EVITA/datasets/kinetics700_2020/kinetics-dataset/k700-2020",
        **kwargs
    ):
        try:
            self.root = join(cfg.DATASETS.LOC, root)
        except:
            self.root = root

        frames_per_clip = (16,)
        step_between_clips = (16,)

        super(Kinetics, self).__init__(
            root,
            frames_per_clip=frames_per_clip,
            step_between_clips=step_between_clips,
            *args,
            **kwargs
        )


class Kinetics400(Kinetics):
    """docstring"""

    def __init__(self, *args, num_classes="400", **kwargs):
        super(Kinetics400, self).__init__(*args, num_classes=num_classes, **kwargs)


class Kinetics600(Kinetics):
    """docstring"""

    def __init__(self, *args, num_classes="600", **kwargs):
        super(Kinetics600, self).__init__(*args, num_classes=num_classes, **kwargs)


class Kinetics700(Kinetics):
    """docstring"""

    def __init__(self, *args, num_classes="700", **kwargs):
        super(Kinetics700, self).__init__(*args, num_classes=num_classes, **kwargs)


"""
torchvision.datasets.Kinetics(
    root,
    frames_per_clip,
    num_classes="400",
    split="train",
    frame_rate=None,
    step_between_clips=1,
    transform=None,
    extensions=("avi", "mp4"),
    download=False,
    num_download_workers=1,
    num_workers=1,
    _precomputed_metadata=None,
    _video_width=0,
    _video_height=0,
    _video_min_dimension=0,
    _audio_samples=0,
    _audio_channels=0,
    _legacy=False,
    output_format="TCHW",
)
"""

k = Kinetics700()

quit()
for x, y in k:
    print(y)
quit()

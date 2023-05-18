"""kinetics 400, 600, 700 datasets"""

from general.config import cfg


class Kinetics(torchvision.datasets.Kinetics):
    """docstring"""

    def __init__( self, 
            root='/grand/EVITA/datasets/kinetics700_2020/kinetics-dataset/k700-2020', 
            *args, 
            **kwargs):

        try:
            self.root = join(cfg.DATASETS.LOC, root)
        except:
            self.root = root

        super.__init__(root, *args, **kwargs)

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

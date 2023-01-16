import os

from yacs.config import CfgNode as CN

MODEL = CN(
    new_allowed=True,
    init_dict=dict(
        BODY="RESNET",
    ),
)


MODEL.FFCR = CN(
    new_allowed=True,
    init_dict=dict(
        BODY="50",
        LAYERS=[3, 4, 6, 3],
        OUT_DIM=64,
        USE_SE=False,
    ),
)


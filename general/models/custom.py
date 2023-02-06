import torch
from torch import nn
from general.config import cfg

from general.models.backbone import ffcresnet

class SYMIA_FINETUNE(ffcresnet.FFCResNet):
    def __init__(self):
        super(SYMIA_FINETUNE,self).__init__(
            block=ffcresnet.Bottleneck,
            layers=cfg.MODEL.FFCR.LAYERS,
            outdim=cfg.MODEL.FFCR.OUT_DIM or 1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            norm_layer=None,
            ratio=0.5,
            lfu=True,
            use_se=cfg.MODEL.FFCR.USE_SE or False,
        )

        # self.load_state_dict(
        self.fc = nn.Linear(self.fc.in_features, 5)



CUSTOM = {
    "symia": SYMIA_FINETUNE,
}


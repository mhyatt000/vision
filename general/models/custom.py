import os

from general.config import cfg
from general.models.backbone import ffcresnet
import torch
from torch import nn

"""
def modification(self.model):
    pass

class FROM():

    def __init__(self):
        self.model = general.models.build_model(cfg.MODEL.CUSTOM.BODY)
        snap = torch.load(os.path.join(cfg.ROOT, "experiments", cfg.MODEL.CUSTOM.FROM, "snapshot.pt"))
        self.model.load_state_dict(snap["MODEL"])

        modification(self.model)

    def forward(self,*args,**kwargs):
        return self.model(*args,**kwargs)
"""


class FFC_ARC64_FINETUNE(ffcresnet.FFCResNet):
    def __init__(self):
        super(FFC_ARC64_FINETUNE, self).__init__(
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

        snap = torch.load(
            os.path.join(cfg.ROOT, "experiments", "ffc_arc64", "snapshot.pt")
        )
        self.load_state_dict(snap["MODEL"])

        self.fc = nn.Sequential(
            nn.Linear(self.fc.in_features, cfg.MODEL.CUSTOM.ODIM),
            nn.Softmax(),
        )


CUSTOM = {
    "ffc_arc64": FFC_ARC64_FINETUNE,
}

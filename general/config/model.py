import os

from yacs.config import CfgNode as CN

MODEL = CN(
    new_allowed=True,
    init_dict=dict(
        RPN_ONLY=False,
        BOX_ON=True,
        MASK_ON=False,
        KEYPOINT_ON=False,
        DEVICE="cuda",
        META_ARCHITECTURE="GeneralizedRCNN",
        RPN_ARCHITECTURE="RPN",
        DEBUG=False,  # add debug flag
        ONNX=False,  # add onnx flag
        # If the WEIGHT starts with a catalog=//, like =R-50, the code will look for
        # the path in paths_catalog. Else, it will use it as the specified absolute
        # path
        WEIGHT="",
        PRETRAIN_NAME="",  # If LINEAR_PROB= True, only the last linear layers in rpn and roi_head are trainable
        LINEAR_PROB=False,
        MULTITASK=CN(new_allowed=True),  # Multitask Training / Test specific parameters
        #
        #
        BODY="RESNET",
        #
        ODIM=1,
    ),
)

MODEL.MLP = CN(
    new_allowed=True,
    init_dict=dict(
        IDIM=512,
        HDIM=64,
        ODIM=64,
        # ACTIVATION="GELU", # prob shouldnt import torch into config file
        DROPOUT=0.0,
    ),
)


MODEL.CONV = CN(
    new_allowed=True,
    init_dict=dict(
        ICHANNEL=1,
        OCHANNEL=1,
        KERNEL=3,
        RESHAPE=True,
    ),
)


MODEL.FFCR = CN(
    new_allowed=True,
    init_dict=dict(
        BODY=50,
        LAYERS=[3, 4, 6, 3],
        OUT_DIM=64,
        USE_SE=False,
    ),
)

temp = CN(
    new_allowed=True,
    init_dict=dict(
        temp=None,
    ),
)

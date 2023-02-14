import os
from yacs.config import CfgNode as CN
from .model import MODEL

"""
Convention about Training / Test specific parameters
arguments for train/test will be post-fixed by a _TRAIN or _TEST 
"""

_C = CN(
    new_allowed=True,
    init_dict=dict(
        DEVICE="cuda",
        AMP=True,
        SEED=0,
    ),
)

# experiment
_C.EXP = CN( 
    new_allowed=True,
    init_dict=dict(
        BODY="DEFAULT",
        TRAIN= True,
        TEST= False,
    ),
)

# temp
_C.SCHEDULER = CN(
    new_allowed=True,
    init_dict=dict(
        BODY= "POLY",
        GAMMA=0.99,
        WARMUP= 0,
    ),
)

_C.MODEL = MODEL

_C.LOADER = CN(
    new_allowed=True,
    init_dict=dict(
        LEAVE_OUT=None,
        SWAP=0,
        X=False,

        NUM_WORKERS=4, # Number of data loading threads
        # each collated batch_size % SIZE_DIVISIBILITY should == 0 
        SIZE_DIVISIBILITY=0,
        USE_RANDOM_SEED=False, # Use random sampler during training
        DISTRIBUTE_CHUNK_AMONG_NODE=False,

    ),
)


_C.MODEL.IRESNET = CN(
    new_allowed=True,
    init_dict=dict(
        BODY="IR50",
        OUT_DIM=512,
    ),
)


_C.INPUT = CN(
    init_dict=dict(
        # Size of the smallest side of the image during training
        MIN_SIZE_TRAIN=800,  # (800,)
        # Maximum size of the side of the image during training
        MAX_SIZE_TRAIN=1333,
        # Size of the smallest side of the image during testing
        MIN_SIZE_TEST=800,
        # Maximum size of the side of the image during testing
        MAX_SIZE_TEST=1333,
        # Values to be used for image normalization
        PIXEL_MEAN=[102.9801, 115.9465, 122.7717],
        # Values to be used for image normalization
        PIXEL_STD=[1.0, 1.0, 1.0],
        # Convert image to BGR format (for Caffe2 models), in range 0-255
        TO_BGR255=True,
        FORMAT="",
        FIX_RES=False,
    )
)

_C.AUGMENT = CN(
    init_dict=dict(
        USE_RA=0,
        FLIP_PROB_TRAIN=0.5,
        VERTICAL_FLIP_PROB_TRAIN=0.0,
        MULT_MIN_SIZE_TRAIN=(),
        BRIGHTNESS=0.0,
        CONTRAST=0.0,
        SATURATION=0.0,
        HUE=0.0,
        CROP_PROB=0.5,
        CROP_MIN_IOUS=(0.1, 0.3, 0.5, 0.7, 0.9),
        CROP_MIN_SIZE=0.3,
    )
)

_C.DATASETS = CN(
    init_dict=dict(
        LOC=os.path.join(os.path.expanduser("~"), "cs", ".datasets"),  # location
        # List of the dataset names for training, as present in paths_catalog.py
        TRAIN=(),
        # List of the dataset names for testing, as present in paths_catalog.py
        TEST=(),
        # Use is_crowd label
        USE_CROWD=False,
        CLASS_AGNOSTIC=False,
        CLASS_CONCAT=False,
        MAX_BOX=-1,
        SAMPLE_RATIO=0.0,
        FEW_SHOT=0,
        # SHUFFLE_SEED != 0 means shuffle the dataset in the few shot setting
        SHUFFLE_SEED=0,
        PREDEFINED_TEXT="",
        ALTERNATIVE_TRAINING=False,
        MULTISTAGE_TRAINING=False,
        REGISTER=CN(new_allowed=True),
        BOX_THRESHOLD=0.1,
        # Duplicate Dataset
        COCO_COPY=1,
        LVIS_COPY=1,
        FLICKR_COPY=1,
        MIXED_COPY=1,
        OBJECT365_COPY=1,
        VG_COPY=1,
        OI_COPY=1,
        IN_COPY=1,
        GENERAL_COPY=-1,
        GENERAL_COPY_TEST=-1,
        # OD to Grounding
        RANDOM_SAMPLE_NEG=-1,
        ADD_DET_PROMPT=False,
        ADD_DET_PROMPT_ADVANCED=False,
        USE_OD_AUG=False,
        USE_COCO_FORMAT=False,
        CONTROL_PROB=(),
        DISABLE_SHUFFLE=False,
        PROMPT_VERSION="",
        PROMPT_LIMIT_NEG=-1,
        POS_QUESTION_PROB=0.6,
        NEG_QUESTION_PROB=0.8,
        FULL_QUESTION_PROB=0.5,
        ONE_HOT=False,
        NO_MINUS_ONE_FOR_ONE_HOT=False,
        DISABLE_CLIP_TO_IMAGE=False,
        SEPARATION_TOKENS=" ",
        # LVIS
        LVIS_USE_NORMAL_AP=False,
        SPECIAL_SAFEGUARD_FOR_COCO_GROUNDING=False,
        # Caption
        BING_INDEX_LIST=[],
        CAPTION_MIN_BOX=1,
        REPLACE_CLEAN_LABEL=False,
        FURTHER_SCREEN=False,
        CAPTION_CONF=0.9,
        CAPTION_NMS=0.9,
        PACK_RANDOM_CAPTION_NUMBER=0,
        INFERENCE_CAPTION=False,
        SAMPLE_NEGATIVE_FOR_GROUNDING_DATA=-1.0,
        RANDOM_PACK_PROB=-1.0,
        NO_RANDOM_PACK_PROBABILITY=0.0,
        SAFEGUARD_POSITIVE_CAPTION=True,
        CAPTION_FORMAT_VERSION="v1",
        LOCAL_DEBUG=False,
        # Od in the wild
        # PREDEFINED_TEXT= None,
        TRAIN_DATASETNAME_SUFFIX="",
        TEST_DATASETNAME_SUFFIX="",
        OVERRIDE_CATEGORY=None,
        USE_OVERRIDE_CATEGORY=False,
        SUPRESS_QUERY=None,
        USE_SUPRESS_QUERY=False,
        USE_CAPTION_PROMPT=False,
        CAPTION_PROMPT=None,
        FLICKR_GT_TYPE="separate",
        # VQA
        DIVER_BOX_FOR_VQA=False,
    )
)

_C.DATALOADER = CN(
    init_dict=dict(
        # Number of data loading threads
        NUM_WORKERS=4,
        # If > 0, this enforces that each collated batch should have a size divisible
        # by SIZE_DIVISIBILITY
        SIZE_DIVISIBILITY=0,
        # If True, each batch should contain only images for which the aspect ratio
        # is compatible. This groups portrait images together, and landscape images
        # are not batched with portrait images.
        ASPECT_RATIO_GROUPING=True,
        # Define min number of keypoints required from GT, for example 10 out of 17
        MIN_KPS_PER_IMS=0,
        # Use random sampler during training
        USE_RANDOM_SEED=False,
        DISTRIBUTE_CHUNK_AMONG_NODE=False,
    )
)

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.VISION = CN(
    new_allowed=True,
    init_dict=dict(
        # The backbone conv body to use
        # The string must match a function that is imported in modeling.model_builder
        # (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
        # backbone)
        CONV_BODY="R-50-C4",
        FREEZE_CONV_BODY_AT=2,  # Add StopGrad at a specified stage so the bottom layers are frozen
        FREEZE=False,
        GROUP=1,
        OUT_CHANNELS=256 * 4,
        RESET_BN=False,  # Option to reset bn running statics
        NORM_LEVEL=3,  # Backbone Normalization Level
        USE_BN=False,  # BN for backbone
        USE_SYNCBN=False,  # Sync BN for backbone
        USE_NSYNCBN=False,
        USE_GN=False,  # GN for backbone
        USE_EN=False,  # Evo Norm for backbone
        # Layers for backbone
        USE_DFCONV=False,
        USE_DYRELU=False,
        USE_SE=False,
        LAYER_SETUP=(3, 4, 6, 3),
        LAYER_SEARCH=CN(new_allowed=True),
        OUT_FEATURES=("stage2", "stage3", "stage4", "stage5"),
        FPN_LAYER=(),
        USE_CHECKPOINT=False,
        # Add JF efficient det cfgs
        EFFICIENT_DET_START_FROM=3,
        EFFICIENT_DET_COMPOUND=0,
        EFFICIENT_DET_BIFPN_VERSION=0,
    ),
)
_C.MODEL.LANG = CN(
    init_dict=dict(
        WEIGHT="",
        FREEZE=False,
        USE_CHECKPOINT=False,
        TOKENIZER="bert-base-uncased",
        BODY="bert-base-uncased",
        LANG_DIM=768,
        MAX_QUERY_LEN=256,
        N_LAYERS=1,
        UNUSED_TOKEN=106,
        MASK_SPECIAL=False,
        RNN_TYPE="lstm",
        VARIABLE_LENGTH=True,
        WORD_EMBEDDING_SIZE=512,
        WORD_VEC_SIZE=512,
        HIDDEN_SIZE=512,
        BIDIRECTIONAL=True,
        INPUT_DROPOUT_P=0.5,
        DROPOUT_P=0.2,
        CORPUS_PATH="",
        VOCAB_SIZE=0,
        PAD_MAX=True,
    )
)

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN(
    init_dict=dict(
        FREEZE=False,
        USE_GN=False,
        USE_RELU=False,
        USE_DYRELU=False,
        DROP_BLOCK=True,
        DROP_PROB=0.3,
        DROP_SIZE=3,
        USE_SPP=False,
        USE_PAN=False,
        USE_DYHEAD=False,
        RETURN_SWINT_FEATURE_BEFORE_FUSION=False,
    )
)
# ---------------------------------------------------------------------------- #
# BIFPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.BIFPN = CN(
    init_dict=dict(
        NUM_REPEATS=1,
        USE_ATTENTION=True,
    )
)

# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN(
    init_dict=dict(
        # Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
        DIM_PER_GP=-1,
        # Number of groups in GroupNorm (-1 if using DIM_PER_GP)
        NUM_GROUPS=16,
        # GroupNorm's small constant in the denominator
        EPSILON=1e-5,
    )
)

# ---------------------------------------------------------------------------- #
# Evo Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.EVO_NORM = CN(
    init_dict=dict(
        # Number of groups in EvoNorm (-1 if using DIM_PER_GP)
        NUM_GROUPS=8,
        # EvoNorm's small constant in the denominator
        EPSILON=1e-5,
    )
)

# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN(
    init_dict=dict(
        # This is the number of foreground classes and background.
        NUM_CLASSES=81,
        # Convolutions to use in the cls and bbox tower
        # NOTE= this doesn't include the last conv for logits
        NUM_CONVS=4,
        # During inference, #locs to select based on cls score before NMS is performed
        # per FPN level
        PRE_NMS_TOP_N=1000,
        # Prior prob for the positives at the beginning of training. This is used to set
        # the bias init for the logits layer
        PRIOR_PROB=0.01,
        # Inference cls score threshold, anchors with score > INFERENCE_TH are
        # considered for inference
        INFERENCE_TH=0.05,
        # NMS threshold used in RetinaNet
        NMS_TH=0.4,
        DETECTIONS_PER_IMG=100,
    )
)

# ---------------------------------------------------------------------------- #
# Focal Loss Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.FOCAL = CN(
    init_dict=dict(
        # Weight for bbox_regression loss
        BBOX_REG_WEIGHT=4.0,
        # Smooth L1 loss beta for bbox regression
        BBOX_REG_BETA=0.11,
        # IoU overlap ratio for labeling an anchor as positive
        # Anchors with >= iou overlap are labeled positive
        FG_IOU_THRESHOLD=0.5,
        # IoU overlap ratio for labeling an anchor as negative
        # Anchors with < iou overlap are labeled negative
        BG_IOU_THRESHOLD=0.4,
        # Focal loss parameter= alpha
        LOSS_ALPHA=0.25,
        # Focal loss parameter= gamma
        LOSS_GAMMA=2.0,
    )
)

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN(
    init_dict=dict(
        NUM_CLASSES=81,  # the number of classes including background
        FPN_STRIDES=[8, 16, 32, 64, 128],
        PRIOR_PROB=0.01,
        INFERENCE_TH=0.05,
        NMS_TH=0.6,
        PRE_NMS_TOP_N=1000,
        NUM_CONVS=4,  # the number of convolutions used in the cls and bbox tower
        USE_DFCONV=False,  # if use deformable conv to align features
        # if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
        CENTER_SAMPLING_RADIUS=0.0,
        # IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
        IOU_LOSS_TYPE="iou",
        NORM_REG_TARGETS=False,
        CENTERNESS_ON_REG=False,
        USE_GT_CENTER=False,
        DETECTIONS_PER_IMG=100,
        USE_GN=False,
        USE_BN=False,
        INFERENCE_TH_TRAIN=0.0,
        PRE_NMS_TOP_N_TRAIN=3000,
        POST_NMS_TOP_N_TRAIN=1000,
    )
)

# ---------------------------------------------------------------------------- #
# ATSS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ATSS = CN(
    init_dict=dict(
        NUM_CLASSES=81,  # the number of classes including background
        PRIOR_PROB=0.01,
        INFERENCE_TH=0.05,
        NMS_TH=0.6,
        PRE_NMS_TOP_N=1000,
        NUM_CONVS=4,  # the number of convolutions used in the cls and bbox tower
        # the channels of convolutions used in the cls and bbox tower
        CHANNELS=128,
        USE_DFCONV=False,  # if use deformable conv to align features
        TOPK=9,  # topk for selecting candidate positive samples from each level
        # Weight for bbox_regression loss
        REG_LOSS_WEIGHT=2.0,
        DETECTIONS_PER_IMG=100,
        USE_GN=False,
        USE_BN=False,
        USE_DYRELU=False,
        USE_SE=False,
        INFERENCE_TH_TRAIN=0.0,
        PRE_NMS_TOP_N_TRAIN=3000,
        POST_NMS_TOP_N_TRAIN=1000,
    )
)

# ---------------------------------------------------------------------------- #
# DYHEAD Options
# ---------------------------------------------------------------------------- #
_C.MODEL.DYHEAD = CN(
    init_dict=dict(
        NUM_CLASSES=81,  # the number of classes including background
        PRIOR_PROB=0.01,
        NUM_CONVS=4,  # the number of convolutions used in the cls and bbox tower
        # the channels of convolutions used in the cls and bbox tower
        CHANNELS=128,
        GROUPS=1,
        USE_DFCONV=False,  # if use deformable conv to align features
        TOPK=9,  # topk for selecting candidate positive samples from each level
        SCORE_AGG="MEAN",  # MEAN or MAX, for binary focal loss score aggregation
        LOG_SCALE=0.0,  # temperature (dot product)
        SHALLOW_LOG_SCALE=0.0,  # temperature (shallow contrastive)
        USE_GN=False,
        USE_NSYNCBN=False,
        USE_SYNCBN=False,
        USE_DYFUSE=False,
        USE_DYRELU=False,
        CONV_FUNC="",
        # CosineSimOutputLayers= https=//github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/modeling/roi_heads/fast_rcnn.py#L448-L464
        COSINE_SCALE=-1.0,
        FUSE_CONFIG=CN(
            init_dict=dict(
                EARLY_FUSE_ON=False,
                TYPE="",
                JOINT_EMB_SIZE=256,
                JOINT_OUT_SIZE=256,
                JOINT_EMB_DROPOUT=0.1,
                JOINT_MLP_LAYERS=2,
                USE_CLASSIFICATION_LOSS=False,
                USE_TOKEN_LOSS=False,
                TOKEN_LOSS_WEIGHT=1.0,
                TOKEN_GAMMA=2.0,
                TOKEN_ALPHA=0.25,
                USE_DOT_PRODUCT_TOKEN_LOSS=False,
                USE_CONTRASTIVE_ALIGN_LOSS=False,
                CONTRASTIVE_HIDDEN_DIM=64,
                CONTRASTIVE_ALIGN_LOSS_WEIGHT=1.0,
                DOT_PRODUCT_TOKEN_LOSS_WEIGHT=1.0,
                USE_LAYER_SCALE=True,
                SEPARATE_BIDIRECTIONAL=False,
                STABLE_SOFTMAX_2D=False,
                DO_LANG_PROJ_OUTSIDE_CHECKPOINT=False,
                USE_FUSED_FEATURES_DOT_PRODUCT=False,
                # Controls for
                CLAMP_MIN_FOR_UNDERFLOW=False,
                CLAMP_MAX_FOR_OVERFLOW=False,
                CLAMP_BERTATTN_MIN_FOR_UNDERFLOW=False,
                CLAMP_BERTATTN_MAX_FOR_OVERFLOW=False,
                CLAMP_DOT_PRODUCT=False,
                # MLM Loss
                MLM_LOSS=False,
                MLM_LOSS_FOR_ONLY_POSITIVES=True,
                NO_MASK_FOR_OD=False,
                NO_MASK_FOR_GOLD=False,
                MLM_LOSS_COEF=1.0,
                MLM_OBJ_FOR_ONLY_POSITIVE=False,
                # Shallow Contrastive Loss (FPN)
                USE_SHALLOW_CONTRASTIVE_LOSS=False,
                SHALLOW_MAX_POSITIVE_ANCHORS=100,
                USE_SHALLOW_ZERO_PADS=False,
                SHALLOW_CONTRASTIVE_HIDDEN_DIM=64,
                SHALLOW_CONTRASTIVE_LOSS_WEIGHT=1.0,
                # Shallow Contrastive Loss (BACKBONE)
                USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS=False,
                ADD_LINEAR_LAYER=False,
            )
        ),
        # use checkpoint to save memory
        USE_CHECKPOINT=False,
    )
)
# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN(
    init_dict=dict(
        USE_FPN=False,
        # Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
        ANCHOR_SIZES=(32, 64, 128, 256, 512),
        # Stride of the feature map that RPN is attached.
        # For FPN, number of strides should match number of scales
        ANCHOR_STRIDE=(16,),
        ASPECT_RATIOS=(0.5, 1.0, 2.0),  # RPN anchor aspect ratios
        ANCHOR_SHIFT=(0.0, 0.0, 0.0, 0.0),  # Anchor shift away ration from the center for r,t,l,d
        USE_RELATIVE_SIZE=False,  # Use center to decide anchor size
        # Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
        # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
        STRADDLE_THRESH=0,
        OCTAVE=2.0,  # Anchor scales per octave for complex anchors
        SCALES_PER_OCTAVE=3,
        # Minimum overlap required between an anchor and ground-truth box for the
        # (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
        # ==> positive RPN example)
        FG_IOU_THRESHOLD=0.7,
        # Maximum overlap allowed between an anchor and ground-truth box for the
        # (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
        # ==> negative RPN example)
        BG_IOU_THRESHOLD=0.3,
        BATCH_SIZE_PER_IMAGE=256,  # Total number of RPN examples per image
        POSITIVE_FRACTION=0.5,  # Target fraction of foreground (positive) examples per RPN minibatch
        # Number of top scoring RPN proposals to keep before applying NMS
        # When FPN is used, this is *per FPN level* (not total)
        PRE_NMS_TOP_N_TRAIN=12000,
        PRE_NMS_TOP_N_TEST=6000,
        # Number of top scoring RPN proposals to keep after applying NMS
        POST_NMS_TOP_N_TRAIN=2000,
        POST_NMS_TOP_N_TEST=1000,
        NMS_THRESH=0.7,  # NMS threshold used on RPN proposals
        # Proposal height and width both need to be greater than RPN_MIN_SIZE
        # (a the scale used during training or inference)
        MIN_SIZE=0,
        # Number of top scoring RPN proposals to keep after combining proposals from
        # all FPN levels
        FPN_POST_NMS_TOP_N_TRAIN=2000,
        FPN_POST_NMS_TOP_N_TEST=2000,
        RPN_HEAD="SingleConvRPNHead",  # Custom rpn head, empty to use default conv or separable conv
        FREEZE=False,
        FORCE_BOXES=False,
        RETURN_FUSED_FEATURES=False,
    )
)

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN(
    init_dict=dict(
        USE_FPN=False,
        # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
        FG_IOU_THRESHOLD=0.5,
        # Overlap threshold for an RoI to be considered background
        # (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
        BG_IOU_THRESHOLD=0.5,
        # Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
        # These are empirically chosen to approximately lead to unit variance targets
        BBOX_REG_WEIGHTS=(10.0, 10.0, 5.0, 5.0),
        # RoI minibatch size *per image* (number of regions of interest [ROIs])
        # Total number of RoIs per training minibatch =
        #   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
        # E.g., a common configuration is= 512 * 2 * 8 = 8192
        BATCH_SIZE_PER_IMAGE=512,
        # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
        POSITIVE_FRACTION=0.25,
        # Only used on test mode
        # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
        # balance obtaining high recall with not having too many low precision
        # detections that will slow down inference post processing steps (like NMS)
        SCORE_THRESH=0.05,
        # Overlap threshold used for non-maximum suppression (suppress boxes with
        # IoU >= this threshold)
        NMS=0.5,
        # Maximum number of detections to return per image (100 is based on the limit
        # established for the COCO dataset)
        DETECTIONS_PER_IMG=100,
    )
)

_C.MODEL.ROI_BOX_HEAD = CN(
    init_dict=dict(
        FEATURE_EXTRACTOR="ResNet50Conv5ROIFeatureExtractor",
        PREDICTOR="FastRCNNPredictor",
        POOLER_RESOLUTION=14,
        POOLER_SAMPLING_RATIO=0,
        POOLER_SCALES=(1.0 / 16,),
        NUM_CLASSES=81,
        MLP_HEAD_DIM=1024,  # Hidden layer dimension when using an MLP for the RoI box head
        USE_GN=False,
        DILATION=1,
        CONV_HEAD_DIM=256,
        NUM_STACKED_CONVS=4,
        POOLER_ALIGNED=False,  # Use D2 style ROIAlignV2
    )
)

_C.MODEL.ROI_MASK_HEAD = CN(
    init_dict=dict(
        FEATURE_EXTRACTOR="ResNet50Conv5ROIFeatureExtractor",
        PREDICTOR="MaskRCNNC4Predictor",
        POOLER_RESOLUTION=14,
        POOLER_SAMPLING_RATIO=0,
        POOLER_SCALES=(1.0 / 16,),
        MLP_HEAD_DIM=1024,
        CONV_LAYERS=(256, 256, 256, 256),
        RESOLUTION=14,
        SHARE_BOX_FEATURE_EXTRACTOR=True,
        # Whether or not resize and translate masks to the input image.
        POSTPROCESS_MASKS=False,
        POSTPROCESS_MASKS_THRESHOLD=0.5,
        DILATION=1,
        USE_GN=False,
        HG_SCALE=1,
    )
)

_C.MODEL.ROI_KEYPOINT_HEAD = CN(
    init_dict=dict(
        FEATURE_EXTRACTOR="KeypointRCNNFeatureExtractor",
        PREDICTOR="KeypointRCNNPredictor",
        POOLER_RESOLUTION=14,
        POOLER_SAMPLING_RATIO=0,
        POOLER_SCALES=(1.0 / 16,),
        MLP_HEAD_DIM=1024,
        CONV_LAYERS=tuple(512 for _ in range(8)),
        RESOLUTION=14,
        NUM_CLASSES=17,
        KEYPOINT_NAME=(),  # If left empty, use default names
        SHARE_BOX_FEATURE_EXTRACTOR=True,
    )
)

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = dict(ResNet, ResNeXt)
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN(
    init_dict=dict(
        USE_STEM3X3=False,
        WITH_SE=False,
        USE_AVG_DOWN=False,
        # Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
        NUM_GROUPS=1,
        WIDTH_PER_GROUP=64,  # Baseline width of each group
        # Place the stride 2 conv on the 1x1 filter
        # Use True only for the original MSRA ResNet; use False for C2 and Torch models
        STRIDE_IN_1X1=True,
        # Residual transformation function
        TRANS_FUNC="BottleneckWithFixedBatchNorm",
        # ResNet's stem function (conv1 and pool1)
        STEM_FUNC="StemWithFixedBatchNorm",
        RES5_DILATION=1,  # Apply dilation in stage "res5"
        BACKBONE_OUT_CHANNELS=256 * 4,
        RES2_OUT_CHANNELS=256,
        STEM_OUT_CHANNELS=64,
        REVISION="resnet_light",
        STAGE_WITH_DCN=(False, False, False, False),  # Deformable convolutions
        WITH_MODULATED_DCN=False,
        DEFORMABLE_GROUPS=1,
    )
)

# ---------------------------------------------------------------------------- #
# Swin Transformer
# ---------------------------------------------------------------------------- #
_C.MODEL.SWINT = CN(
    init_dict=dict(
        EMBED_DIM=96,
        OUT_CHANNELS=(96, 192, 384, 768),
        DEPTHS=(2, 2, 6, 2),
        NUM_HEADS=(3, 6, 12, 24),
        WINDOW_SIZE=7,
        MLP_RATIO=4,
        DROP_PATH_RATE=0.2,
        APE=False,
        VERSION="v1",
        OUT_NORM=True,
        LAYER_SCALE=0,
    )
)

# ---------------------------------------------------------------------------- #
# CVT SPEC
# ---------------------------------------------------------------------------- #
_C.MODEL.SPEC = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# CLIP SPEC
# ---------------------------------------------------------------------------- #
_C.MODEL.CLIP = CN(
    init_dict=dict(
        CONTEXT_LENGTH=256,  # default 77
        WIDTH=512,
        LAYERS=12,
        HEADS=8,
        DROP_PATH=0.0,
        TOKENIZER="clip",
        VOCAB_SIZE=49408,
    )
)

# ---------------------------------------------------------------------------- #
# SEARCH
# ---------------------------------------------------------------------------- #
_C.SEARCH = CN(
    init_dict=dict(
        MAX_EPOCH=20,
        SELECT_NUM=20,
        POPULATION_NUM=64,
        MUTATION_NUM=24,
        CROSSOVER_NUM=24,
        MUTATION_PROB=0.1,
    )
)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #

#NOTE: it seems like solver is a combo of loss, optim, and scheduler

_C.SOLVER = CN(
    new_allowed=True,
    init_dict=dict(
        USE_AMP=False,
        MAX_ITER=40000,
        MULTI_MAX_ITER=(),  # set different max epoch for different stage
        MAX_EPOCH=0,  # any epoch number>0 will overwrite max_iter
        MULTI_MAX_EPOCH=(),  # set different max epoch for different stage
        OPTIMIZER="SGD",  # "ADAMW"
        BASE_LR=0.001,
        LANG_LR=0.00001,
        BACKBONE_BODY_LR_FACTOR=1.0,
        BIAS_LR_FACTOR=2,
        GRAD_CLIP=0.0,
        GRAD_ACC_EVERY = 1, # accumulate every epoch if possible
        # D2 gradient clip
        CLIP_GRADIENTS=CN(
            init_dict=dict(
                ENABLED=False,
                CLIP_VALUE=0.0,
                CLIP_TYPE="full_model",
                NORM_TYPE=2.0,
            )
        ),
        MODEL_EMA=0.0,
        MOMENTUM=0.9,
        WEIGHT_DECAY=0.0005,
        WEIGHT_DECAY_BIAS=0.0,
        WEIGHT_DECAY_NORM_FACTOR=1.0,
        USE_COSINE=False,  # use cosine lr to replace default multistage
        MIN_LR=0.000001,
        GAMMA=0.1,
        STEPS=(30000,),
        USE_AUTOSTEP=False,
        STEP_PATIENCE=5,
        WARMUP_FACTOR=1.0 / 3,
        WARMUP_ITERS=500,
        WARMUP_METHOD="linear",
        CHECKPOINT_PERIOD=2500,
        CHECKPOINT_PER_EPOCH=-1.0,
        TEST_WITH_INFERENCE=False,
        AUTO_TERMINATE_PATIENCE=-1,
        # Number of images per batch
        # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
        # see 2 images per batch
        IMS_PER_BATCH=16,
        MAX_NEG_PER_BATCH=0.1,  # This is the max negative ratio allowed per batch
        SEED=0,
        DISABLE_OUTPUT_DISTRIBUTED=False,
        PROMPT_PROBING_LEVEL=-1.0,
        # -1 means tuning the whole model;
        # 1 means tuning the whole language model; 1.5 means tuning the box head as well
        FIND_UNUSED_PARAMETERS=True,
        DATASET_LENGTH=-1,  # Just for logging purpose
        TUNING_HIGHLEVEL_OVERRIDE=None,
        USE_EMA_FOR_MONITOR=False,
        WEIGHT_DECAY_SCHEDULE=False,
        WEIGHT_DECAY_SCHEDULE_RATIO=0.667,
    ),
)


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN(
    init_dict=dict(
        EXPECTED_RESULTS=[],
        EXPECTED_RESULTS_SIGMA_TOL=4,
        DURING_TRAINING=False,
        # Number of images per batch
        # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
        # see 2 images per batch
        IMS_PER_BATCH=16,
        # Special Test Configuration
        USE_MULTISCALE=False,
        # _C.TEST.SCALES = (400, 600, 800, 1000, 1200, 1400)
        # _C.TEST.RANGES = ((96, 10000), (64, 10000), (0, 10000), (0, 10000), (0, 256), (0, 192))
        SCALES=(400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800),
        RANGES=(
            (96, 10000),
            (96, 10000),
            (64, 10000),
            (64, 10000),
            (64, 10000),
            (0, 10000),
            (0, 10000),
            (0, 256),
            (0, 256),
            (0, 192),
            (0, 192),
            (0, 96),
        ),
        MAX_SIZE=2500,
        FLIP=True,
        SPECIAL_NMS="none",  # ('none', 'soft-nms', 'vote', 'soft-vote')
        TH=0.6,  # threshold for nms or vote
        PRE_NMS_TOP_N=1000,
        NUM_CLASSES=81,
        SELECT_CLASSES=(),
        EVAL_TASK="",
        SUBSET=-1,
        CHUNKED_EVALUATION=-1,
        MDETR_STYLE_AGGREGATE_CLASS_NUM=-1,
    )
)


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "OUTPUT"

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

_C.TENSORBOARD_EXP = "OUTPUT"  # TensorBoard experiment location

_C.GLIPKNOW = CN(
    init_dict=dict(
        KNOWLEDGE_FILE="",
        KNOWLEDGE_TYPE="",
        MAX_NUM_CLASSES_PER_BATCH_TRAIN=-1,
        PARALLEL_LANGUAGE_INPUT=False,
        LAN_FEATURE_AGG_TYPE="first",
        GPT3_NUM=5,
        WIKI_AND_GPT3=False,
    )
)

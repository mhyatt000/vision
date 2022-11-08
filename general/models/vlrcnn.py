import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from general.config import cfg

from .backbone import build_backbone
from .head import build_head
from .lang import build_lm, build_tokenizer
from .rpn import build_rpn


class VLRCNN(nn.Module):
    """general VL RCNN"""

    def __init__(self):
        super(VLRCNN, self).__init__()

        self.backbone = build_backbone()

        self.tokenizer = build_tokenizer()
        self.tokenizer_vocab = self.tokenizer.get_vocab()
        self.tokenizer_vocab_ids = [v for k, v in self.tokenizer_vocab.items()]

        self.lm = build_lm()
        self.rpn = build_rpn()
        self.head = build_head()

        # options
        self.DEBUG = cfg.MODEL.DEBUG

        """"
        since these variables are stored in the cfg
        there is no point in making them object variables
        !!!
        """

        self.freeze_backbone = cfg.MODEL.VISION.FREEZE
        self.freeze_fpn = cfg.MODEL.FPN.FREEZE
        self.freeze_rpn = cfg.MODEL.RPN.FREEZE
        self.add_linear_layer = cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER

        self.force_boxes = cfg.MODEL.RPN.FORCE_BOXES

        """
        YOU SHOULD REALLY LOOK THROUGH THIS CODE AND UNDERSTAND IT vvv
        """

        if cfg.MODEL.LINEAR_PROB:
            assert cfg.MODEL.BACKBONE.FREEZE, "For linear probing, backbone should be frozen!"
            if hasattr(self.backbone, "fpn"):
                assert cfg.MODEL.FPN.FREEZE, "For linear probing, FPN should be frozen!"
        self.linear_prob = cfg.MODEL.LINEAR_PROB
        self.freeze_cls_logits = cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # disable cls_logits
            if hasattr(self.rpn.head, "cls_logits"):
                for p in self.rpn.head.cls_logits.parameters():
                    p.requires_grad = False

        self.freeze_lm = cfg.MODEL.LANG.FREEZE
        if self.freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        self.use_mlm_loss = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS
        self.mlm_loss_for_only_positives = cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES

        if cfg.GLIPKNOW.KNOWLEDGE_FILE:
            from maskrcnn_benchmark.data.datasets.tsv import \
                load_from_yaml_file

            self.class_name_to_knowledge = load_from_yaml_file(cfg.GLIPKNOW.KNOWLEDGE_FILE)
            self.class_name_list = sorted([k for k in self.class_name_to_knowledge])

    def train(self, mode=True):
        """
        puts model into training mode ...
        keeps layers frozen
        """

        pass

    def forward(self, images, targets=None, captions=None):
        """docstring"""

    def _forward_language_parallel(
        self, captions=None, targets=None, device=None, positive_map=None
    ):
        """
        docstring
        """

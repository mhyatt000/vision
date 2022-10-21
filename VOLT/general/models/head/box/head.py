import torch
from torch import nn

from .feature_extractor import build_box_feature_extractor
from .predictor import build_box_predictor
from .inference import build_box_post_processor
from .loss import build_box_loss_evaluator

# from maskrcnn_benchmark.utils.amp import custom_fwd, custom_bwd


class BoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()

        self.feature_extractor = build_box_feature_extractor(cfg)
        self.predictor = build_box_predictor(cfg)
        self.post_processor = build_box_post_processor(cfg)
        self.loss_evaluator = build_box_loss_evaluator(cfg)

        self.onnx = cfg.MODEL.ONNX

    # @custom_fwd(cast_inputs=torch.float32)
    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.
        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)

        if self.onnx:
            return x, (class_logits, box_regression, [box.bbox for box in proposals]), {}

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression])
        loss_dict = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        return (x, proposals, loss_dict)

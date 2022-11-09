import math

import torch

from general.config import cfg


class CombinedMarginLoss(torch.nn.Module):
    """
    m: margins for:
        m1: multiplicative angular margin
        m2: additive angular margin
        m3: additive cosine margin
    s: scale
    inter_thresh: interclass filtering threshold
    """

    def __init__(self):
        super().__init__()

        self.s = cfg.LOSS.AAM.S
        self.m1, self.m2, self.m3 = cfg.LOSS.AAM.M
        self.inter_thresh = cfg.LOSS.AAM.INTER_THRESH

        self.arcface = ArcFace(s=self.s, margin=self.m2)
        self.cosface = CosFace(s=self.s, margin=self.m3)

        # For ArcFace
        # self.cos_m = math.cos(self.m2)
        # self.sin_m = math.sin(self.m2)
        # self.theta = math.cos(math.pi - self.m2)
        # self.sinmm = math.sin(math.pi - self.m2) * self.m2
        # self.easy_margin = False

    def forward(self, logits, labels):

        # index of positive samples
        index = torch.where(labels != -1)[0]

        if self.inter_thresh > 0:
            with torch.no_grad():
                dirty = logits > self.inter_thresh
                dirty = dirty.float()
                mask = torch.ones([index.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index], 0)
                dirty[index] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        tgt = logits[index, labels[index].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            logits = self.arcface(tgt, labels)
        elif self.m3:
            logits = self.cosface(tgt, labels)
        else:
            raise

        return logits


class ArcFace(torch.nn.Module):
    """
    ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()

        self.s = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits, labels):
        index = torch.where(labels != -1)[0]
        tgt = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(tgt, 2))
        cos_theta_m = tgt * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        final_tgt = apply_easy_margin(tgt)

        logits[index, labels[index].view(-1)] = final_tgt
        logits = logits * self.s
        return logits

    def apply_easy_margin(self, tgt):
        return torch.where(
            tgt > (0 if self.easy_margin else self.theta),
            cos_theta_m,
            tgt - (0 if self.easy_margin else self.sinm),
        )


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()

        self.s = s
        self.margin = m

    def forward(self, logits, labels):
        index = torch.where(labels != -1)[0]
        tgt = logits[index, labels[index].view(-1)]
        final_tgt = tgt - self.margin

        logits[index, labels[index].view(-1)] = final_tgt
        logits = logits * self.s
        return logits

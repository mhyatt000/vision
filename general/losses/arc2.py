import math

import torch
from torch import nn
import torch.nn.functional as F

from general.config import cfg


class CombinedMarginLoss(nn.Module):
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


class ArcFace(nn.Module):
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

        """ try to rewrite the loss from their paper 
        L2 from paper
        L3 is added margin
        """

        C = set(labels.tolist())
        print('C',C)
        for c in C:
            index = torch.where(labels == c)[0]
            X = logits[index.view(-1)]
            Y = labels[index.view(-1)]

            # X is a collection of vector directions now
            # scaled by their magnitude
            X = torch.div(X, torch.sqrt(torch.sum(torch.pow(X,2), -1)).reshape(-1,1))
            W = torch.mean(X, -2)
            # W is the average of the directions of the others
            W = torch.div(W, torch.sqrt(torch.sum(torch.pow(W,2), -1)).reshape(-1,1))

            print(W.repeat(X.shape[0], 1).shape == X.shape)
            loss = nn.CrossEntropyLoss()(X, W.repeat(X.shape[0], 1))
            print(loss)

        """
        index = torch.where(labels != -1)[0]
        print(labels[index].view(-1))
        print(logits)
        tgt = logits[labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(tgt, 2))
        print(sin_theta)
        cos_theta_m = tgt * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        print(cos_theta_m)
        cos_theta_m = tgt * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        print(cos_theta_m)
        final_tgt = self.apply_easy_margin(tgt, cos_theta_m)
        print(final_tgt, final_tgt.shape)
        print(logits.shape)
        print(index.shape)
        print(labels.shape, labels[index].shape, labels[index].view(-1).shape)
        logits[labels[index].view(-1)] = final_tgt
        print(logits)
        logits = logits * self.s
        print(logits)
        return logits
        """

    def apply_easy_margin(self, tgt, cos_theta_m):
        return torch.where(
            tgt > (0 if self.easy_margin else self.theta),
            cos_theta_m,
            tgt - (0 if self.easy_margin else self.sinmm),
        )


class CosFace(nn.Module):
    def __init__(self, s=64.0, margin=0.40):
        super(CosFace, self).__init__()

        self.s = s
        self.margin = margin

    def forward(self, logits, labels):
        index = torch.where(labels != -1)[0]
        tgt = logits[index, labels[index].view(-1)]
        final_tgt = tgt - self.margin

        logits[index, labels[index].view(-1)] = final_tgt
        logits = logits * self.s
        return logits

if __name__ == '__main__': 

    loss = CombinedMarginLoss()

    arcface = loss.arcface
    print(arcface.cos_m)
    print(arcface.sin_m)
    print(arcface.theta)
    print(arcface.sinmm)

    x = torch.Tensor(torch.rand([5,2]))
    y = torch.Tensor(torch.rand([5])).int()
    print(x.dtype,y.dtype)

    arcface(x,y)

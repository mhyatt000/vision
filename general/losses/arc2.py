import torch
import itertools
from torch import nn
import math
from general.config import cfg
import torch.nn.functional as F


class ArcFace(torch.nn.Module):
    """ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):"""

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (cfg.LOSS.PFC.NCLASSES, cfg.LOSS.PFC.EMBED_DIM))
        )

    def get_l5(self, embeddings, labels):
        """
        get L5 term from arcface paper
        intra cluster compactness
        """

        norm = F.normalize
        logits = F.linear(norm(embeddings), norm(self.weight))
        theta = torch.acos(logits[labels.view(-1).long()])

        loss = theta.mean() / 3.14
        return loss

    def get_l6(self):
        """
        get L6 term from arcface paper
        inter cluseter distance
        """

        loss = 0
        pairs = itertools.combinations(list(range(cfg.LOSS.PFC.NCLASSES)), 2)
        norm = lambda x: F.normalize(self.weight[x], dim=0)
        for a, b in pairs:
            loss += torch.acos(F.linear(norm(a), norm(b)))
        loss /= -3.14 * (cfg.LOSS.PFC.NCLASSES - 1)
        return loss

    def apply_margin(self, embeddings, labels):

        norm = F.normalize
        logits = F.linear(norm(embeddings), norm(self.weight))
        logits = logits.clamp(-1, 1)

        # print(embeddings)
        # print(self.weight)
        # quit()

        # these logits are X•W ... X•W = cos(theta)
        # X•W is dimension R(batch*nclasses) cos(theta) for each class?

        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1).long()]

        # use sin_theta to calculate cos_theta_m
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m

        # feel like this could cause problems for outlier images ...
        # but in theory the softmax CE should smooth things out in the long run
        # why does target_logit have to be greater in order to replace?

        if self.easy_margin:
            final = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final = torch.where(target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        # this line fixes the problem suggested above
        # only replace cos_theta_m for sample at the specified class
        logits[index, labels[index].view(-1).long()] = final
        logits = logits * self.scale
        return logits

    def forward(self, logits, labels):
        """docstring"""

        margin_logits = self.apply_margin(logits, labels)

        # print(margin_logits)
        # print()
        # print(labels)
        loss = F.cross_entropy(margin_logits, labels.view(-1).long())
        # print(loss.detach())
        # quit()
        # l5 = self.get_l5(logits, labels)
        l6 = 10 * self.get_l6() # this could be very important... so 10x it?

        return loss + l6

import torch
import torch.nn as nn
import torch.nn.functional as F
from general.config import cfg


class AngularPenaltySM(nn.Module):
    def __init__(
        self,
        in_dim=cfg.LOSS.PFC.EMBED_DIM,
        out_dim=cfg.LOADER.NCLASSES,
        loss_type="arcface",
        eps=1e-7,
        s=None,
        m=None,
    ):
        """
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        """
        super(AngularPenaltySM, self).__init__()

        loss_type = loss_type.lower()
        assert loss_type in ["arcface", "sphereface", "cosface"]

        if loss_type == "arcface":
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == "sphereface":
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == "cosface":
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m

        self.loss_type = loss_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        """input shape (N, in_dim)"""

        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_dim

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)  # gets the norm of the vector i think...
        wf = self.fc(x)
        labels = labels.long()

        if self.loss_type == "cosface":
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == "arcface":
            numerator = self.s * torch.cos(
                torch.acos(
                    torch.clamp(
                        torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps
                    )
                )
                + self.m
            )
        if self.loss_type == "sphereface":
            numerator = self.s * torch.cos(
                self.m
                * torch.acos(
                    torch.clamp(
                        torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps
                    )
                )
            )

        excl = torch.cat(
            [torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)],
            dim=0,
        )
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

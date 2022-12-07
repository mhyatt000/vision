import torch.nn as nn
from general.models.layers.ffc import *

from general.config import cfg

__all__ = [
    "FFCResNet",
    "FFCR18",
    "FFCR34",
    "FFCR26",
    "FFCR50",
    "FFCR101",
    "FFCR152",
    "FFCR200",
    "FFCRX50_32x4d",
    "FFCRX101_32x8d",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        ratio_gin=0.5,
        ratio_gout=0.5,
        lfu=True,
        use_se=False,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1

        self.conv1 = FFC_BNA(
            inplanes,
            width,
            kernel_size=3,
            padding=1,
            stride=stride,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU,
            enable_lfu=lfu,
        )
        self.conv2 = FFC_BNA(
            width,
            planes * self.expansion,
            kernel_size=3,
            padding=1,
            ratio_gin=ratio_gout,
            ratio_gout=ratio_gout,
            norm_layer=norm_layer,
            enable_lfu=lfu,
        )
        self.se_block = FFC_SE(planes * self.expansion, ratio_gout) if use_se else nn.Identity()

        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        ratio_gin=0.5,
        ratio_gout=0.5,
        lfu=True,
        use_se=False,
    ):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = FFC_BNA(
            inplanes,
            width,
            kernel_size=1,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            activation_layer=nn.ReLU,
            enable_lfu=lfu,
        )
        self.conv2 = FFC_BNA(
            width,
            width,
            kernel_size=3,
            ratio_gin=ratio_gout,
            ratio_gout=ratio_gout,
            stride=stride,
            padding=1,
            groups=groups,
            activation_layer=nn.ReLU,
            enable_lfu=lfu,
        )
        self.conv3 = FFC_BNA(
            width,
            planes * self.expansion,
            kernel_size=1,
            ratio_gin=ratio_gout,
            ratio_gout=ratio_gout,
            enable_lfu=lfu,
        )
        self.se_block = FFC_SE(planes * self.expansion, ratio_gout) if use_se else nn.Identity()
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.downsample is None else self.downsample(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_l, x_g = self.se_block(x)

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


class FFCResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=cfg.MODEL.FFCR.NUM_CLASSES or 1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        norm_layer=None,
        ratio=0.5,
        lfu=True,
        use_se=cfg.MODEL.FFCR.USE_SE or False,
    ):
        super(FFCResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        inplanes = 64
        # TODO add ratio-inplanes-groups assertion

        self.inplanes = inplanes
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.lfu = lfu
        self.use_se = use_se
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, inplanes * 1, layers[0], stride=1, ratio_gin=0, ratio_gout=ratio
        )
        self.layer2 = self._make_layer(
            block, inplanes * 2, layers[1], stride=2, ratio_gin=ratio, ratio_gout=ratio
        )
        self.layer3 = self._make_layer(
            block, inplanes * 4, layers[2], stride=2, ratio_gin=ratio, ratio_gout=ratio
        )
        self.layer4 = self._make_layer(
            block, inplanes * 8, layers[3], stride=2, ratio_gin=ratio, ratio_gout=0
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, ratio_gin=0.5, ratio_gout=0.5):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or ratio_gin == 0:
            downsample = FFC_BNA(
                self.inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                ratio_gin=ratio_gin,
                ratio_gout=ratio_gout,
                enable_lfu=self.lfu,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                self.dilation,
                ratio_gin,
                ratio_gout,
                lfu=self.lfu,
                use_se=self.use_se,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    ratio_gin=ratio_gout,
                    ratio_gout=ratio_gout,
                    lfu=self.lfu,
                    use_se=self.use_se,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x[0])

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def FFCR18(**kwargs):
    model = FFCResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def FFCR34(**kwargs):
    model = FFCResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def FFCR26(**kwargs):
    model = FFCResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model


def FFCR50(**kwargs):
    model = FFCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def FFCR101(**kwargs):
    model = FFCResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def FFCR152(**kwargs):
    model = FFCResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def FFCR200(**kwargs):
    model = FFCResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def FFCRX50_32x4d(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    model = FFCResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def FFCRX101_32x8d(**kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    model = FFCResNet(Bottleneck, [3, 4, 32, 3], **kwargs)

    return model


models = {
    "18": FFCR18,
    "26": FFCR26,
    "34": FFCR34,
    "50": FFCR50,
    "101": FFCR101,
    "152": FFCR152,
    "200": FFCR200,
}


def FFCR():
    """default builder for ffc resnet variants"""

    """TODO include resNEXT variants"""
    return models[cfg.MODEL.FFCR.BODY]()

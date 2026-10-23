"""Modernised GoogLeNet for the sports-balls dataset.

Blocks keep the 2014 GoogLeNet filter budget. Three variants are available so
the notebook can change exactly one thing at a time:

    modern      5x5 branch factored into 3x3 -> 3x3, batch norm in every branch
    bn5x5       real 5x5 branch, batch norm in every branch
    v1          real 5x5 branch, no batch norm (the original notebook)

The stem and the classifier head are identical across all three.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1x1, 3x3-reduce, 3x3, 5x5-reduce, 5x5-out, pool-proj
BLOCK_CONFIG = {
    "3a": (64, 96, 128, 16, 32, 32),
    "3b": (128, 128, 192, 32, 96, 64),
    "4a": (192, 96, 208, 16, 48, 64),
    "4b": (160, 112, 224, 24, 64, 64),
    "4c": (128, 128, 256, 24, 64, 64),
    "4d": (112, 144, 288, 32, 64, 64),
    "4e": (256, 160, 320, 32, 128, 128),
    "5a": (256, 160, 320, 32, 128, 128),
    "5b": (384, 192, 384, 48, 128, 128),
}

STAGE_PLAN = [
    ("3a", "3b"),
    ("4a", "4b", "4c", "4d", "4e"),
    ("5a", "5b"),
]

AUX_AFTER = "4d"

VARIANTS = {
    "modern": dict(factor_5x5=True, use_bn=True),
    "bn5x5": dict(factor_5x5=False, use_bn=True),
    "v1": dict(factor_5x5=False, use_bn=False),
}


def conv_bn_relu(in_ch, out_ch, kernel, stride=1, padding=None, use_bn=True):
    if padding is None:
        padding = kernel // 2
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding,
                  bias=not use_bn)
    ]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fpool,
                 factor_5x5=True, use_bn=True):
        super().__init__()
        self.branch1 = conv_bn_relu(in_ch, f1, 1, padding=0, use_bn=use_bn)

        self.branch3 = nn.Sequential(
            conv_bn_relu(in_ch, f3r, 1, padding=0, use_bn=use_bn),
            conv_bn_relu(f3r, f3, 3, use_bn=use_bn),
        )

        if factor_5x5:
            # Two 3x3 convs give the same receptive field as one 5x5 at
            # roughly half the parameters and one more nonlinearity.
            self.branch5 = nn.Sequential(
                conv_bn_relu(in_ch, f5r, 1, padding=0, use_bn=use_bn),
                conv_bn_relu(f5r, f5, 3, use_bn=use_bn),
                conv_bn_relu(f5, f5, 3, use_bn=use_bn),
            )
        else:
            self.branch5 = nn.Sequential(
                conv_bn_relu(in_ch, f5r, 1, padding=0, use_bn=use_bn),
                conv_bn_relu(f5r, f5, 5, padding=2, use_bn=use_bn),
            )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_bn_relu(in_ch, fpool, 1, padding=0, use_bn=use_bn),
        )
        self.out_channels = f1 + f3 + f5 + fpool

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)],
            dim=1,
        )


class AuxHead(nn.Module):
    """Deep-supervision tap. Returns raw logits."""

    def __init__(self, in_ch, num_classes, hidden=1024, dropout=0.7):
        super().__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(5, stride=3),
            conv_bn_relu(in_ch, 128, 1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=15, aux_weight=0.3, dropout=0.4, in_channels=3,
                 factor_5x5=True, use_bn=True):
        super().__init__()
        self.aux_weight = aux_weight

        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, padding=1),
            conv_bn_relu(64, 64, 1, padding=0),
            conv_bn_relu(64, 192, 3),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stages = nn.ModuleList()
        self.aux_tap = None
        self.aux_channels = None
        in_ch = 192

        for stage in STAGE_PLAN:
            # Pooling stages land in the same list, so record the real index.
            stage_index = len(self.stages)
            layers = []
            for name in stage:
                block = InceptionBlock(in_ch, *BLOCK_CONFIG[name],
                                       factor_5x5=factor_5x5, use_bn=use_bn)
                layers.append(block)
                in_ch = block.out_channels
                if name == AUX_AFTER:
                    self.aux_tap = (stage_index, len(layers) - 1)
                    self.aux_channels = block.out_channels
            self.stages.append(nn.ModuleList(layers))
            if stage != STAGE_PLAN[-1]:
                self.stages.append(
                    nn.ModuleList([nn.MaxPool2d(3, stride=2, padding=1)])
                )

        self.final_channels = in_ch

        # Centres the descriptor before pooling. Without it the last ReLU leaves
        # a large positive offset and the initial logits come out far too big.
        self.final_norm = nn.BatchNorm2d(self.final_channels)

        self.aux = None
        if aux_weight > 0 and self.aux_tap is not None:
            self.aux = AuxHead(self.aux_channels, num_classes)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.final_channels, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _run_stages(self, x, capture_tap=False):
        x = self.stem(x)
        tap = None
        for si, stage in enumerate(self.stages):
            for bi, layer in enumerate(stage):
                x = layer(x)
                if capture_tap and (si, bi) == self.aux_tap:
                    tap = x
        return x, tap

    def forward_features(self, x):
        """Conv feature map going into the classifier, shape (B, 1024, 7, 7)."""
        features, _ = self._run_stages(x)
        return self.final_norm(features)

    def forward(self, x, with_aux=None):
        if with_aux is None:
            with_aux = self.training and self.aux is not None

        features, tap = self._run_stages(x, capture_tap=with_aux)
        features = self.final_norm(features)
        logits = self.head(features)

        if with_aux:
            return logits, self.aux(tap)
        return logits

    def grad_cam_layer(self):
        """Module to hook for Grad-CAM: the last conv feature map."""
        return self.final_norm


def build_model(num_classes=15, aux_weight=0.3, dropout=0.4, variant="modern"):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, choose from {list(VARIANTS)}")
    return GoogLeNet(num_classes, aux_weight, dropout, **VARIANTS[variant])


def inception_loss(outputs, target, aux_weight, label_smoothing=0.0):
    """Cross entropy on the main head plus the weighted auxiliary term."""
    if isinstance(outputs, tuple):
        logits, aux_logits = outputs
        main = F.cross_entropy(logits, target, label_smoothing=label_smoothing)
        aux = F.cross_entropy(aux_logits, target, label_smoothing=label_smoothing)
        return main + aux_weight * aux, main.detach(), aux.detach()

    main = F.cross_entropy(outputs, target, label_smoothing=label_smoothing)
    return main, main.detach(), torch.zeros((), device=main.device)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

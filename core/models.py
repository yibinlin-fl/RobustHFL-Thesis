"""CIFAR architectures checked against RAHFL's stage schedules and stems."""
from torch import nn
from torch.nn import functional as F
import torch


def conv_bn(cin, cout, kernel, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout))


class Classifier(nn.Module):
    def __init__(self, stages, features, classes):
        super().__init__()
        self.backbone = nn.Sequential(*stages)
        self.feature_dim = features
        self.classifier = nn.Linear(features, classes)

    def forward(self, x, return_feat=False):
        features = F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)
        logits = self.classifier(features)
        if return_feat:
            return logits, F.normalize(features, dim=1)
        return logits


class Residual(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(conv_bn(cin, cout, 3, stride), nn.ReLU(), conv_bn(cout, cout, 3))
        self.skip = nn.Identity() if cin == cout and stride == 1 else conv_bn(cin, cout, 1, stride)

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


class ResNet(Classifier):
    def __init__(self, blocks, classes):
        channels = (64, 128, 256, 512)
        stages = [conv_bn(3, channels[0], 3), nn.ReLU()]
        cin = channels[0]
        for stage, (cout, count) in enumerate(zip(channels, blocks)):
            stages.append(Residual(cin, cout, 1 if stage == 0 else 2))
            stages.extend(Residual(cout, cout, 1) for _ in range(count - 1))
            cin = cout
        super().__init__(stages, cin, classes)


def ResNet10(num_classes=10):
    return ResNet((1, 1, 1, 1), num_classes)


def ResNet12(num_classes=10):
    return ResNet((2, 1, 1, 1), num_classes)


class MobileBlock(nn.Module):
    def __init__(self, cin, cout, expansion, stride):
        super().__init__()
        width = cin * expansion
        self.body = nn.Sequential(conv_bn(cin, width, 1), nn.ReLU(),
                                  conv_bn(width, width, 3, stride, width), nn.ReLU(), conv_bn(width, cout, 1))
        self.residual = stride == 1
        self.skip = nn.Identity() if cin == cout or not self.residual else conv_bn(cin, cout, 1)

    def forward(self, x):
        result = self.body(x)
        return result + self.skip(x) if self.residual else result


class MobileNetV2(Classifier):
    def __init__(self, num_classes=10):
        cin = 32
        stages = [conv_bn(3, cin, 3), nn.ReLU()]
        schedule = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
                    (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
        for expansion, cout, count, stride in schedule:
            stages.append(MobileBlock(cin, cout, expansion, stride))
            stages.extend(MobileBlock(cout, cout, expansion, 1) for _ in range(count - 1))
            cin = cout
        stages.extend([conv_bn(cin, 1280, 1), nn.ReLU()])
        super().__init__(stages, 1280, num_classes)


class ShuffleBlock(nn.Module):
    def __init__(self, cin, cout, stride, groups):
        super().__init__()
        self.stride = stride
        self.groups = groups
        out = cout - cin if stride == 2 else cout
        mid = out // 4
        self.reduce = conv_bn(cin, mid, 1, groups=1 if cin == 24 else groups)
        self.expand = nn.Sequential(conv_bn(mid, mid, 3, stride, mid), nn.ReLU(), conv_bn(mid, out, 1, groups=groups))
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        z = F.relu(self.reduce(x))
        n, c, h, w = z.shape
        z = z.reshape(n, self.groups, c // self.groups, h, w).transpose(1, 2).reshape(n, c, h, w)
        z = self.expand(z)
        return F.relu(torch.cat([z, self.pool(x)], dim=1) if self.stride == 2 else F.relu(z + x))


class ShuffleNet(Classifier):
    def __init__(self, num_classes=10):
        cin = 24
        stages = [conv_bn(3, cin, 1), nn.ReLU()]
        for cout, count in [(200, 4), (400, 8), (800, 4)]:
            stages.append(ShuffleBlock(cin, cout, 2, 2))
            stages.extend(ShuffleBlock(cout, cout, 1, 2) for _ in range(count - 1))
            cin = cout
        super().__init__(stages, cin, num_classes)


MODELS = {model.__name__: model for model in (ResNet10, ResNet12, ShuffleNet, MobileNetV2)}


def build_model(name):
    return MODELS[name]()

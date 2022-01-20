import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

from config import *

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function

class transition_layer(nn.Sequential):
    def __init__(self, num_features, compress_r):
        super(transition_layer, self).__init__()
        self.add_module("bn", nn.BatchNorm2d(num_features=num_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(in_channels=num_features, out_channels=int(int(num_features)*compress_r),
                                                    kernel_size=1, stride=1, bias=False))
        self.add_module("avgpool", nn.AvgPool2d(kernel_size=2, stride=2))

class dense_layer(nn.Module):
    def __init__(self, num_features, drop_r):
        super(dense_layer, self).__init__()
        self.drop_rate = drop_r
        self.add_module("bn1", nn.BatchNorm2d(num_features=num_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels=num_features, out_channels=num_features,
                                                   kernel_size=1, stride=1))

        self.add_module("bn2", nn.BatchNorm2d(num_features=num_features))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(in_channels=num_features, out_channels=num_features,
                                                   kernel_size=3, stride=1, padding=1))


    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.bn1, self.relu1, self.conv1)
        if any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.bn2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class MyDenseBlock(nn.Module):
    def __init__(self, num_features, layerNum):
        super(MyDenseBlock, self).__init__()
        for i in range(layerNum):
            layer = dense_layer(num_features*(2**i), drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)
    def forward(self, input_features):
        features = [input_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class MyDenseNet(nn.Module):
    def __init__(self):
        super(MyDenseNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        for i, num_layers in enumerate(block_cfg):
            block = MyDenseBlock(
                num_features=num_features,
                layerNum = num_layers,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            trans = transition_layer(int(num_features*(1+(1-2**num_features)/(-1))), 1/(1+(1-2**num_features)/(-1)))
            self.features.add_module('transition%d' % (i + 1), trans)
        block = MyDenseBlock(
            num_features=num_features,
            layerNum = last_layerNum,
        )

        self.features.add_module('denseblock%d' % (len(block_cfg)+1), block)
        self.features.add_module('norm_final', nn.BatchNorm2d(int(num_features*(1+(1-2**num_features)/(-1)))))

        self.classifier = nn.Linear(int(num_features*(1+(1-2**num_features)/(-1))), num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
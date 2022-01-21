import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from config import *

class U_conv(nn.Sequential):
    def __init__(self, in_features, out_features):
        super(U_conv, self).__init__()
        self.add_module('conv',nn.Conv2d(in_channels=in_features,out_channels=out_features,
                                          kernel_size=3, stride=1))
        self.add_module('relu',nn.ReLU(inplace=True))

class DownBlock(nn.Module):
    def __init__(self, in_features):
        super(DownBlock, self).__init__()
        self.features = nn.Sequential(OrderedDict([]))
        conv1 = U_conv(in_features, in_features*2)
        self.features.add_module('conv1',conv1)
        conv2 = U_conv(in_features*2, in_features*2)
        self.features.add_module('conv2',conv2)
        max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.features.add_module('maxpool',max_pool)
    def forward(self, input_features):
        return self.features(input_features)

class UpBlock(nn.Module):
    def __init__(self, in_features):
        super(UpBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=in_features,out_channels=int(in_features/2),
                                          kernel_size=1, stride=1)
        self.conv1 = U_conv(in_features, int(in_features/4))
        self.conv2 = U_conv(int(in_features/4), int(in_features/4))
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.drop_out = nn.Dropout2d(p=0.5)

    def forward(self, copy_features, up_features):
        copy_features_w = copy_features.shape[-1]
        up_features_w = up_features.shape[-1]
        crop_features = copy_features[:,:, copy_features_w/2-up_features_w/2:copy_features_w/2+up_features_w/2,
                                           copy_features_w/2-up_features_w/2:copy_features_w/2+up_features_w/2]
        up_features = self.up_sample(up_features)
        output = self.conv0(up_features)
        concate_features = torch.cat([crop_features, output], 1)
        output = self.drop_out(concate_features)
        output = self.conv1(output)
        output = self.conv2(output)

        return output

class U_net(nn.Module):
    def __init__(self, in_features):
        super(U_net, self).__init__()
        # down
        self.down_block0 = nn.Sequential(OrderedDict([
                ('input_conv0', U_conv(input_channel, in_features)),
                ('input_conv1', U_conv(in_features, in_features)),
                ('input_maxpool', nn.MaxPool2d(kernel_size=2,stride=2)),
                ]))
        self.down_block1 = DownBlock(in_features * (2 ** 0))
        self.down_block2 = DownBlock(in_features * (2 ** 1))
        self.down_block3 = DownBlock(in_features * (2 ** 2))
        # bottom
        self.bottom_block = nn.Sequential(OrderedDict([
                ('bottom_conv0', U_conv(in_features * (2 ** 3), in_features * (2 ** 4))),
                ('bottom_conv1', U_conv(in_features * (2 ** 4), in_features * (2 ** 4))),
                ]))
        # up
        self.up_block0 = UpBlock(in_features * (2 ** 4))
        self.up_block1 = UpBlock(in_features * (2 ** 3))
        self.up_block2 = UpBlock(in_features * (2 ** 2))
        self.up_block3 = UpBlock(in_features * (2 ** 1))
        # output
        self.output_conv = nn.Conv2d(in_channels=in_features,out_channels=class_num,
                                          kernel_size=1, stride=1)
    def forward(self, input_images):
        down_features0 = self.down_block0(input_images)
        down_features1 = self.down_block1(down_features0)
        down_features2 = self.down_block2(down_features1)
        down_features3 = self.down_block3(down_features2)
        bottom_features = self.bottom_block(down_features3)
        up_features0 = self.up_block0(down_features3, bottom_features)
        up_features1 = self.up_block1(down_features2, up_features0)
        up_features2 = self.up_block2(down_features1, up_features1)
        up_features3 = self.up_block3(down_features0, up_features2)
        output = self.output_conv(up_features3)
        return output
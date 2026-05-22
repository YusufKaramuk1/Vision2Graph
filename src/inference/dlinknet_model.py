"""D-LinkNet34 mimarisi (DeepGlobe Road Extraction Challenge).

log01_dink34.th checkpoint'i bu mimariye yuklenir.
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nonlinearity = partial(F.relu, inplace=True)


class Dblock(nn.Module):
    """Genisleyen (dilated) konvolusyonlarla merkez bloku."""

    def __init__(self, channel):
        super().__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

    def forward(self, x):
        d1 = nonlinearity(self.dilate1(x))
        d2 = nonlinearity(self.dilate2(d1))
        d3 = nonlinearity(self.dilate3(d2))
        d4 = nonlinearity(self.dilate4(d3))
        return x + d1 + d2 + d3 + d4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = nonlinearity(self.norm1(self.conv1(x)))
        x = nonlinearity(self.norm2(self.deconv2(x)))
        x = nonlinearity(self.norm3(self.conv3(x)))
        return x


class DinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=None)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = nonlinearity(self.finaldeconv1(d1))
        out = nonlinearity(self.finalconv2(out))
        out = self.finalconv3(out)
        return torch.sigmoid(out)

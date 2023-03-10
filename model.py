# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """Base block in CNN"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not bn_act)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            x = self.silu(self.bn(self.conv(x)))

            return x
        else:
            return self.conv(x)

class BottleNeckBlock(nn.Module):
    def __init__(self, channels, short_cut=True):
        super().__init__()
        self.short_cut = short_cut
        self.Conv = nn.Sequential(CNNBlock(channels, channels//2, 3, 1, 1),
                                  CNNBlock(channels//2, channels, 3, 1, 1))

    def forward(self, x):
        if self.short_cut:
            return self.Conv(x) + x
        else:
            return self.Conv(x)

class C2FBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Conv_end = CNNBlock(int(0.5*(1+2)*out_channels), out_channels, kernel_size=1, stride=1, padding=0)
        self.BottleNeck = BottleNeckBlock(out_channels//2, **kwargs)

    def forward(self, x):
        x = self.Conv(x)
        x, x1 = torch.split(x, self.out_channels//2, dim=1)
        x2 = self.BottleNeck(x1)
        x = torch.cat([x, x1, x2], dim=1)
        x = self.Conv_end(x)
        return x

class C2F_2_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Conv_end = CNNBlock(int(0.5*(2+2)*out_channels), out_channels, kernel_size=1, stride=1, padding=0)
        self.BottleNeck = BottleNeckBlock(out_channels//2, **kwargs)

    def forward(self, x):
        x = self.Conv(x)
        x, x1 = torch.split(x, self.out_channels//2, dim=1)
        x2 = self.BottleNeck(x1)
        x3 = self.BottleNeck(x2)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.Conv_end(x)
        return x

class SPPFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Conv = CNNBlock(channels, channels, kernel_size=1, stride=1, padding=0)
        self.Conv_end = CNNBlock(4*channels, channels, kernel_size=1, stride=1, padding=0)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.Conv(x)
        x = torch.cat([x, self.MaxPool(x), self.MaxPool(self.MaxPool(x)), self.MaxPool(self.MaxPool(self.MaxPool(x)))],
                      dim=1)
        x = self.Conv_end(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes=500):
        super().__init__()
        self.Conv = nn.Sequential(CNNBlock(512, 1280, kernel_size=1, stride=1, padding=0))
        self.Flatten = nn.Flatten()
        self.Linear = nn.Sequential(nn.Linear(62720, num_classes))

    def forward(self, x):
        x = self.Conv(x)
        x = self.Flatten(x)
        x = self.Linear(x)
        return x

class Yolov8_cls(nn.Module):
    """Model architecture based page: https://blog.roboflow.com/whats-new-in-yolov8/
       and the ONNX file of yolov8_cls.onnx"""

    def __init__(self, in_channels, num_classes=500):
        super().__init__()
        self.Block1 = nn.Sequential(CNNBlock(in_channels, 32, 3, 2, 1),
                                    CNNBlock(32, 64, 3, 2, 1))

        self.Block2 = C2FBlock(64, 64)

        self.Block3 = nn.Sequential(CNNBlock(64, 128, 3, 2, 1),
                                    C2F_2_Block(128, 128))

        self.Block4 = nn.Sequential(CNNBlock(128, 256, 3, 2, 1),
                                    C2F_2_Block(256, 256))

        self.Block5 = nn.Sequential(CNNBlock(256, 512, 3, 2, 1),
                                    C2F_2_Block(512, 512))

        self.Block6 = Classifier(num_classes)

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x = self.Block5(x)
        x = self.Block6(x)
        return x



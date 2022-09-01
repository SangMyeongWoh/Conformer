import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import pdb
import numpy as np
import time
import numpy as np
import pytorch_lightning as pl
from CustomDataSet import CocoDataset
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(1536, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.Tanh())

        self.conv1 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                #    nn.BatchNorm2d(512),
                                   nn.Tanh())
        self.conv1_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=1, dilation=1, bias=False),
                                #    nn.BatchNorm2d(512),
                                   nn.Tanh())
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                #    nn.BatchNorm2d(512),
                                   nn.Tanh())
        self.conv2_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=1, dilation=1, bias=False),
                                #    nn.BatchNorm2d(512),
                                   nn.Tanh())
        self.conv3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1, dilation=1, bias=False),
                                #    nn.BatchNorm2d(128),
                                   nn.Tanh())

        self.conv4 = nn.Sequential(nn.Conv2d(128, 3, kernel_size=1, padding=0, dilation=1, bias=False),
                                    nn.Tanh())


    def forward(self, x):
        x = self.conv0(x)
        x = F.upsample(self.conv1(x), size=(56, 56), mode='bilinear', align_corners=True)
        x = self.conv1_2(x)
        x = F.upsample(self.conv2(x), size=(112, 112), mode='bilinear', align_corners=True)
        x = self.conv2_2(x)
        x = F.upsample(self.conv3(x), size=(224, 224), mode='bilinear', align_corners=True)
        x = self.conv4(x)

        return x

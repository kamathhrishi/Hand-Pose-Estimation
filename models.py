import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HandSegmentationModel(nn.Module):
    def __init__(self):
        super(HandSegmentationModel, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((13, 13))
        self.conv2_t = nn.ConvTranspose2d(512, 256, 3, padding=1)
        self.conv3_t = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv4_t = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv5_t = nn.ConvTranspose2d(256, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = F.interpolate(self.conv2_t(x), scale_factor=2.0)
        x = F.interpolate(self.conv3_t(x), scale_factor=2.0)
        x = F.interpolate(self.conv4_t(x), scale_factor=2.0)
        x = F.interpolate(self.conv5_t(x), scale_factor=2.465)
        return torch.sigmoid(x)


class KeyPointDetectionModel(nn.Module):
    def __init__(self):
        super(KeyPointDetectionModel, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((13, 13))
        self.conv2_t = nn.ConvTranspose2d(512, 256, 3, padding=1)
        self.conv3_t = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv4_t = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv5_t = nn.ConvTranspose2d(256, 21, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = F.interpolate(
            self.conv2_t(x), scale_factor=2.0, recompute_scale_factor=True
        )
        x = F.interpolate(
            self.conv3_t(x), scale_factor=2.0, recompute_scale_factor=True
        )
        x = F.interpolate(
            self.conv4_t(x), scale_factor=2.0, recompute_scale_factor=True
        )
        x = F.interpolate(
            self.conv5_t(x), scale_factor=1.93, recompute_scale_factor=True
        )
        return torch.sigmoid(x)

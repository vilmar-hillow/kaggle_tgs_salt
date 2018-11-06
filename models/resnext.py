from .blocks import DecoderBlock, ConvRelu
from torch import nn
import torch
import pretrainedmodels


class ResNext(nn.Module):
    def __init__(self, num_filters=32, pretrained=False, is_deconv=False, **kwargs):
        super(ResNext, self).__init__()
        num_classes = 1

        self.pool = nn.MaxPool2d(2, 2)

        model_name = 'se_resnext50_32x4d'
        if pretrained:
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        else:
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)

        self.conv1 = model.layer0

        self.conv2 = model.layer1

        self.conv3 = model.layer2

        self.conv4 = model.layer3

        self.conv5 = model.layer4

        self.center = DecoderBlock(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        x_out = self.final(dec0)

        return x_out

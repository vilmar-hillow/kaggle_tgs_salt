from .blocks import DecoderBlock, ConvRelu
from torch import nn
import torch
from modules.wider_resnet import WiderResNet
from modules.bn import ABN
from collections import OrderedDict


class WideResnet(nn.Module):
    def __init__(self, num_filters=32, pretrained=False, is_deconv=False, num_input_channels=3, **kwargs):
        super(WideResnet, self).__init__()
        num_classes = 1

        if 'norm_act' not in kwargs:
            norm_act = ABN
        else:
            norm_act = kwargs['norm_act']

        self.pool = nn.MaxPool2d(2, 2)

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=1000, norm_act=norm_act)
        encoder_dict = encoder.state_dict()

        if pretrained:
            state = torch.load("wide_resnet38_ipabn_lr_256.pth.tar")["state_dict"]
            state = {k.replace('module.', ''): v for k, v in state.items()}
            state = {k: v for k, v in state.items() if k in encoder_dict}
            encoder_dict.update(state)
            encoder.load_state_dict(state)

        self.conv1 = nn.Sequential(
            OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5
        self.conv6 = encoder.mod6
        self.conv7 = encoder.mod7

        self.center = DecoderBlock(4096, num_filters * 32, num_filters * 32, is_deconv=is_deconv)
        self.dec7 = DecoderBlock(4096 + num_filters * 32, num_filters * 32, num_filters * 32, is_deconv=is_deconv)
        self.dec6 = DecoderBlock(2048 + num_filters * 32, num_filters * 16, num_filters * 16, is_deconv=is_deconv)
        self.dec5 = DecoderBlock(1024 + num_filters * 16, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        conv6 = self.conv6(self.pool(conv5))
        conv7 = self.conv7(self.pool(conv6))


        center = self.center(self.pool(conv7))

        dec7 = self.dec7(torch.cat([center, conv7], 1))

        dec6 = self.dec6(torch.cat([dec7, conv6], 1))
        dec5 = self.dec5(torch.cat([dec6, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)

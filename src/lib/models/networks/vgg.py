# coding: utf-8
# @Author: oliver
# @Date:   2019-07-31 14:42:16

import os
import math
import logging

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models

BN_MOMENTUM = 0.1


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class VGG(nn.Module):
    def __init__(self, model_name, heads, head_conv):
    	super(VGG, self).__init__()
        self.inplanes = 512
        self.heads = heads
        self.deconv_with_bias = False

        self.model_name = model_name
        self._features = getattr(models, self.model_name)().features

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]

            #fc = DCN(self.inplanes, planes, 
            #        kernel_size=(3,3), stride=1,
            #        padding=1, dilation=1, deformable_groups=1)
            fc = nn.Conv2d(self.inplanes, planes,
                    kernel_size=3, stride=1, 
                    padding=1, dilation=1, bias=False)
            fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self._features(x)
        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self):
        if 1:
            url = model_urls[self.model_name]
            pretrained_state_dict = model_zoo.load_url(url)
            model_state_dict = self.state_dict()
            for k in pretrained_state_dict:
                if k in model_state_dict:
                    if pretrained_state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                            k, model_state_dict[k].shape, pretrained_state_dict[k].shape))
                        pretrained_state_dict[k] = model_state_dict[k]

            self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

def get_model(model_name, heads, head_conv=256):
  model = VGG(model_name, heads, head_conv=head_conv)
  model.init_weights()
  return model

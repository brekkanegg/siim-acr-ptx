# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


import math

from models import resnet




model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class double_conv(nn.Module):
    '''(conv => BN => ELU) * 2'''

    def __init__(self, in_ch, out_ch, stride=1):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        # self.mpconv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     double_conv(in_ch, out_ch)
        # )
        self.mpconv = double_conv(in_ch, out_ch, stride=2)


    def forward(self, x):
        x = self.mpconv(x)


        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, scale_factor=2):
        super(up, self).__init__()

        self.scale_factor = scale_factor

        if bilinear:
            self.up = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

        else:
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])

            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=self.scale_factor, padding=1, output_padding=1)  # fixme: kernel is 2

        self.conv = double_conv(in_ch, out_ch)


    def copy_crop(self, x1, x2):
        x2_cx, x2_cy = x2.shape[2] // 2, x2.shape[3] // 2
        x1_px, x1_py = x1.shape[2] // 2, x1.shape[3] // 2

        x2 = x2[:, :, x2_cx-x1_px:x2_cx+x1_px, x2_cy-x1_py:x2_cy+x1_py]

        return x2


    def forward(self, x1, x2):

        x1 = self.up(x1)

        x2 = self.copy_crop(x1, x2)
        x = torch.cat([x2, x1], dim=1)  # fixme: 일대일로 concat 할 것인지 - deeplabv3+ 참고
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x





class UNet(nn.Module):
    def __init__(self, n_classes, stride=1, bilinear=True):
        super(UNet, self).__init__()
        if bilinear == 'bi':
            bilinear = True
        elif bilinear == 'transpose':
            bilinear = False

        self.scale_factors = [2 for _ in range(4)]
        for sf in range(int(math.log(stride, 2))):
            self.scale_factors[sf] = 1

        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=bilinear, scale_factor=self.scale_factors[0])
        self.up2 = up(512, 128, bilinear=bilinear, scale_factor=self.scale_factors[1])
        self.up3 = up(256, 64, bilinear=bilinear, scale_factor=self.scale_factors[2])
        self.up4 = up(128, 64, bilinear=bilinear, scale_factor=self.scale_factors[3])
        self.outc = outconv(64, n_classes)

        self._init_weight()


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UNetRes50(nn.Module):
    def __init__(self, n_channels, n_classes, stride=1, bilinear=True, pretrained=True):
        super(UNetRes50, self).__init__()

        self.resnet50encoder = resnet.ResNetEncoder(resnet.Bottleneck, [3 ,4, 6, 3], n_channels)

        self.scale_factors = [2 for _ in range(4)]
        for sf in range(int(math.log(stride, 2))):
            self.scale_factors[sf] = 1

        self.up1 = up(3072, 512, bilinear=bilinear, scale_factor=self.scale_factors[0])
        self.up2 = up(1024, 256, bilinear=bilinear, scale_factor=self.scale_factors[1])
        self.up3 = up(512, 64, bilinear=bilinear, scale_factor=self.scale_factors[2])
        self.up4 = up(128, 64, bilinear=bilinear, scale_factor=self.scale_factors[3]*2)
        self.outc = outconv(64, n_classes)

        self._init_weight()

        # pretrained=True should be after than self._init_weight()
        if pretrained:
            model_dict = self.resnet50encoder.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            in_conv_weight =  pretrained_dict['conv1.weight']
            pretrained_dict['conv1.weight'] = torch.cat((torch.mean(in_conv_weight, dim=1).unsqueeze(1), )*n_channels, 1)


            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.resnet50encoder.load_state_dict(model_dict)






    def forward(self, x):

        x1, x2, x3, x4, x5 = self.resnet50encoder(x)
        # print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)

        x = self.up1(x5, x4)  # 3072 -> 512
        x = self.up2(x, x3)  # 1024 -> 256
        x = self.up3(x, x2)  # 512 -> 64
        x = self.up4(x, x1)  # 128 -> 64
        x = self.outc(x)
        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
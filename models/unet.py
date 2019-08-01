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
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



class double_conv(nn.Module):
    '''(conv => BN => ELU) * 2'''

    def __init__(self, in_ch, out_ch1, out_ch2, stride, dilation=1, drop_ratio=0.0, in_norm=False):
        super(double_conv, self).__init__()
        # padding = stride

        if in_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch1, 3, stride, padding=dilation, dilation=dilation),
                nn.InstanceNorm2d(out_ch1),
                nn.ELU(inplace=True),
                nn.Dropout2d(p=drop_ratio),
                nn.Conv2d(out_ch1, out_ch2, 3, stride=1, padding=dilation, dilation=dilation),
                nn.InstanceNorm2d(out_ch2),
                nn.ELU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch1, 3, stride, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_ch1),
                nn.ELU(inplace=True),
                nn.Dropout2d(p=drop_ratio),
                nn.Conv2d(out_ch1, out_ch2, 3, stride=1, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(out_ch2),
                nn.ELU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, up_in_ch, up_out_ch, stride=2, drop_ratio=0.2, in_norm=False):
        super(up, self).__init__()

        self.stride = stride
        self.transpose_conv = nn.ConvTranspose2d(in_channels=up_in_ch, out_channels=up_out_ch, kernel_size=3,
                                                 stride=self.stride, padding=1, output_padding=1)

        self.conv = double_conv(in_ch, out_ch, out_ch, 1, drop_ratio=drop_ratio, in_norm=in_norm)


    def copy_crop(self, x1, x2):
        x2_cx, x2_cy = x2.shape[2] // 2, x2.shape[3] // 2
        x1_px, x1_py = x1.shape[2] // 2, x1.shape[3] // 2

        x2 = x2[:, :, x2_cx-x1_px:x2_cx+x1_px, x2_cy-x1_py:x2_cy+x1_py]
        # not copy_crop on z-axis

        return x2


    def forward(self, x1, x2):

        # x1 = F.interpolate(x1, scale_factor=self.stride, mode='bilinear', align_corners=True)
        # fixme - check
        x1 = self.transpose_conv(x1)
        x2 = self.copy_crop(x1, x2)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



# class VanilaUNet(nn.Module):
#     def __init__(self, n_classes):
#         super(VanilaUNet, self).__init__()
#         self.inc = double_conv(1, 64, 64, stride=1)
#         self.down1 = double_conv(64, 128, 128, stride=2)
#         self.down2 = double_conv(128, 256, 256, stride=2)
#         self.down3 = double_conv(256, 512, 512, stride=2)
#         self.down4 = double_conv(512, 512, 512, stride=2)
#         self.up1 = up(1024, 256, 256, stride=2)
#         self.up2 = up(512, 128, 128, stride=2)
#         self.up3 = up(256, 64, 64, stride=2)
#         self.up4 = up(128, 64, 64, stride=2)
#         self.outc = nn.Conv2d(64, n_classes, 3, stride=1, padding=1)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outc(x)
#         return x





class UNet(nn.Module): # 2d U-Net + dilation + spatial dropout
    def __init__(self, n_classes, filters=32, pretrained=True):
        super(UNet, self).__init__()

        self.inc = double_conv(1, filters, filters*2, stride=1, dilation=1)
        self.down1 = double_conv(filters*2, filters*2, filters*4, stride=2, dilation=1)
        self.down2 = double_conv(filters*4, filters*4, filters*8, stride=2, dilation=2)
        self.down3 = double_conv(filters*8, filters*8, filters*16, stride=2, dilation=2)


        self.up1 = up(filters*16, filters*8, filters*16, filters*8, stride=2, drop_ratio=0.2)
        self.up2 = up(filters*8, filters*4, filters*8, filters*4, stride=2, drop_ratio=0.2)
        self.up3 = up(filters*4, filters*2, filters*4, filters*2, stride=2, drop_ratio=0.0)

        self.outc = nn.Conv2d(filters*2, n_classes, 3, stride=1, padding=1)

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(512, 256, 1, stride=1),
                                        nn.BatchNorm2d(256),
                                        nn.ELU(inplace=True),
                                        nn.Dropout2d(p=0.2),
                                        nn.Conv2d(256, 64, 1, stride=1),
                                        nn.BatchNorm2d(64),
                                        nn.ELU(inplace=True),
                                        nn.Dropout2d(p=0.2),
                                        nn.Conv2d(64, 2, 1, stride=1),
                                        )

        self._init_weight()

        if pretrained:
            pretrained_dict_dir = './ckpt/pretrain/chest14/epoch_4.pth.tar'
            print('using pretrained weight: ', pretrained_dict_dir)
            pretrained_dict = torch.load(pretrained_dict_dir)
            self._load_weight(self.inc, pretrained_dict)
            self._load_weight(self.down1, pretrained_dict)
            self._load_weight(self.down2, pretrained_dict)
            self._load_weight(self.down3, pretrained_dict)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)  # x4 shape - [b, 512, 64, 64]

        c = self.classifier(x4)
        c = torch.squeeze(torch.squeeze(c, -1), -1)

        # x5 = self.down4(x4)
        # print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x, x1)
        x = self.outc(x)
        return x, c


    def _load_weight(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)



    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class UNetRes50(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(UNetRes50, self).__init__()

        self.resnetencoder = resnet.ResNetEncoder(resnet.Bottleneck, [3 ,4, 6, 3], 1)


        # self.up1 = up(3072, 512, 512, stride=2) # up(3072, 512, 512, stride=2)
        # self.up2 = up(1024, 256, 256, stride=2)
        # self.up3 = up(512, 64, 64, stride=2)
        # self.up4 = up(128, 64, 64, stride=2)
        # self.up5 = up(65, 32, 32, stride=2)
        # self.outc = nn.Conv2d(32, n_classes, 3, stride=1, padding=1)

        self.up1 = up(2048, 1024, 2048, 1024, stride=2) # up(3072, 512, 512, stride=2)
        self.up2 = up(1024, 512, 1024, 512, stride=2)
        self.up3 = up(512, 256, 512, 256, stride=2)
        self.up4 = up(128, 64, 256, 64, stride=2)
        self.up5 = up(33, 16, 64, 32, stride=2)
        self.outc = nn.Conv2d(16, n_classes, 3, stride=1, padding=1)

        # self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                                 nn.Conv2d(2048, 64, 1, stride=1),
        #                                 nn.BatchNorm2d(64),
        #                                 nn.ELU(inplace=True),
        #                                 nn.Dropout2d(p=0.2),
        #                                 nn.Conv2d(64, 2, 1, stride=1),
        #                                 )


        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(2048, 256, 1, stride=1),
                                        nn.BatchNorm2d(256),
                                        nn.ELU(inplace=True),
                                        nn.Dropout2d(p=0.2),
                                        nn.Conv2d(256, 64, 1, stride=1),
                                        nn.BatchNorm2d(64),
                                        nn.ELU(inplace=True),
                                        nn.Dropout2d(p=0.2),
                                        nn.Conv2d(64, 2, 1, stride=1),
                                        )

        self._init_weight()


        # pretrained=True should be after than self._init_weight()
        if pretrained:
            model_dict = self.resnetencoder.state_dict()
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            in_conv_weight =  pretrained_dict['conv1.weight']
            pretrained_dict['conv1.weight'] = torch.cat((torch.mean(in_conv_weight, dim=1).unsqueeze(1), )*1, 1)

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.resnetencoder.load_state_dict(model_dict)


    def forward(self, x):
        x0 = x
        x1, x2, x3, x4, x5 = self.resnetencoder(x)
        # print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # x5 shape = (4, 2048, 16, 16)
        c = self.classifier(x5)
        c = torch.squeeze(torch.squeeze(c, -1), -1)

        x = self.up1(x5, x4)  # 3072 -> 512
        x = self.up2(x, x3)  # 1024 -> 256
        x = self.up3(x, x2)  # 512 -> 64
        x = self.up4(x, x1)  # 128 -> 64
        x = self.up5(x, x0)

        x = self.outc(x)

        return x, c


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



# ## todo: 뭐가 잘 못 되었는지 모르겠다..
# class UNetRes34(nn.Module):
#     def __init__(self, n_classes, pretrained=True):
#         super(UNetRes34, self).__init__()
#
#         self.resnetencoder = resnet.ResNetEncoder(resnet.Bottleneck, [3 ,4, 6, 3], 1)
#
#
#         self.up1 = up(3072, 512, 512, stride=2)
#         self.up2 = up(1024, 256, 256, stride=2)
#         self.up3 = up(512, 64, 64, stride=2)
#         self.up4 = up(128, 64, 64, stride=2)
#         self.up5 = up(65, 32, 32, stride=2)
#
#         self.outc = nn.Conv2d(32, n_classes, 3, stride=1, padding=1)
#
#         self._init_weight()
#
#         # pretrained=True should be after than self._init_weight()
#         if pretrained:
#             model_dict = self.resnetencoder.state_dict()
#             pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#
#             in_conv_weight =  pretrained_dict['conv1.weight']
#             pretrained_dict['conv1.weight'] = torch.cat((torch.mean(in_conv_weight, dim=1).unsqueeze(1), )*1, 1)
#
#             # 2. overwrite entries in the existing state dict
#             model_dict.update(pretrained_dict)
#             # 3. load the new state dict
#             self.resnetencoder.load_state_dict(model_dict)
#
#
#     def forward(self, x):
#         x0 = x
#         x1, x2, x3, x4, x5 = self.resnetencoder(x)
#         print(x.shape, x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
#
#         x = self.up1(x5, x4)  # 3072 -> 512
#         x = self.up2(x, x3)  # 1024 -> 256
#         x = self.up3(x, x2)  # 512 -> 64
#         x = self.up4(x, x1)  # 128 -> 64
#         x = self.up5(x, x0)
#
#         x = self.outc(x)
#
#         return x
#
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

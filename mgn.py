import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck

#def make_model(args):
#    return MGN(args)

class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.
    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        # global cross-channel averaging
        x = x.mean(1, keepdim=True)
        # 3-by-3 conv
        x = self.conv1(x)
        # bilinear resizing
        x = F.upsample(x, (x.size(2)*2, x.size(3)*2), mode='bilinear', align_corners=True)
        # scaling conv
        x = self.conv2(x)
        return x


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    Aim: Spatial Attention + Channel Attention
    Output: attention maps with shape identical to input.
    """
    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""
    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        return y_soft_attn

class MGN(nn.Module):
    def __init__(self):
        super(MGN, self).__init__()
        num_classes =739
        num_feats = 256

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        self.ha1 = HarmAttn(1024)
        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        #if args.pool == 'max':
            #pool2d = nn.MaxPool2d
        #elif args.pool == 'avg':
        pool2d = nn.AvgPool2d
        #else:
        #    raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, num_feats, 1, bias=False), nn.BatchNorm2d(num_feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        #self.fc_id_2048_0 = nn.Linear(2048, num_classes)
        self.fc_id_2048_0 = nn.Linear(num_feats, num_classes)
        self.fc_id_2048_1 = nn.Linear(num_feats, num_classes)
        self.fc_id_2048_2 = nn.Linear(num_feats, num_classes)

        self.fc_id_256_1_0 = nn.Linear(num_feats, num_classes)
        self.fc_id_256_1_1 = nn.Linear(num_feats, num_classes)
        self.fc_id_256_2_0 = nn.Linear(num_feats, num_classes)
        self.fc_id_256_2_1 = nn.Linear(num_feats, num_classes)
        self.fc_id_256_2_2 = nn.Linear(num_feats, num_classes)
        self.fc_g = nn.Linear(num_feats * 8, num_classes)


        self.fc_id_2048_0_w = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())
        self.fc_id_2048_1_w = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())
        self.fc_id_2048_2_w = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())

        self.fc_id_256_1_0_w = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())
        self.fc_id_256_1_1_w = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())
        self.fc_id_256_2_0_w = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())
        self.fc_id_256_2_1_w = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())
        self.fc_id_256_2_2_w = nn.Sequential(nn.Linear(num_feats, 1), nn.Sigmoid())



        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)
        self._init_fc(self.fc_g)

        """
        self._init_fc(self.fc_id_2048_0_w)
        self._init_fc(self.fc_id_2048_1_w)
        self._init_fc(self.fc_id_2048_2_w)
        self._init_fc(self.fc_id_256_1_0_w)
        self._init_fc(self.fc_id_256_1_1_w)
        self._init_fc(self.fc_id_256_2_0_w)
        self._init_fc(self.fc_id_256_2_1_w)
        self._init_fc(self.fc_id_256_2_2_w)
        """
    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        #nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)
        x_attention = self.ha1(x)
        #x_attn, x_theta = self.ha1(x)
        x = x * x_attention
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''
        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)
        #print("self.fc_id_2048_0_w(fg_p1).shape:",self.fc_id_2048_0_w(fg_p1).shape)
        #print("fg_p1.shape:", fg_p1.shape)
        lfg_p1 = self.fc_id_2048_0_w(fg_p1) * fg_p1
        #print("lfg_p1.shape:", lfg_p1.shape)
        lfg_p2 = self.fc_id_2048_1_w(fg_p2) * fg_p2
        lfg_p3 = self.fc_id_2048_2_w(fg_p3) * fg_p3

        lf0_p2 = self.fc_id_256_1_0_w(f0_p2) * f0_p2
        lf1_p2 = self.fc_id_256_1_1_w(f1_p2) * f1_p2
        lf0_p3 = self.fc_id_256_2_0_w(f0_p3) * f0_p3
        lf1_p3 = self.fc_id_256_2_1_w(f1_p3) * f1_p3
        lf2_p3 = self.fc_id_256_2_2_w(f2_p3) * f2_p3


        predict = torch.cat([lfg_p1, lfg_p2, lfg_p3, lf0_p2, lf1_p2, lf0_p3, lf1_p3, lf2_p3], dim=1)
        g1 = self.fc_g(predict)
        #predict = torch.cat([lfg_p1, lf0_p2, lf1_p2, lf0_p3, lf1_p3, lf2_p3], dim=1)
        #return predict
        return g1

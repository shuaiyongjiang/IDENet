#!/usr/bin/python3
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from pvtv2_encoder import pvt_v2_b2

class BaseCBR(nn.Module):
    def __init__(self, dim_in, dim_out, ker_size=3):
        super(BaseCBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=ker_size, stride=1, padding=(ker_size - 1) // 2, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=(3, 3), s=(1, 1), p=(1, 1), g=1, d=(1, 1), bias=False, bn=True,
                 relu=True, dw=False, sig=False):
        super(convbnrelu, self).__init__()
        if dw:
            conv = [nn.Conv2d(in_channel, in_channel, k, s, p, dilation=d, groups=in_channel, bias=bias),
                    nn.Conv2d(in_channel, out_channel, 1, s, 0, dilation=d, bias=bias)]
        else:
            conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.GELU())
        if sig:
            conv.append(nn.Sigmoid())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
        

class BarCBR(nn.Module):
    def __init__(self, dim_in, dim_out, ker_size=(3, 3), padding=(1, 1)):
        super(BarCBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=ker_size, stride=1, padding=padding, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DwCBR(nn.Module):
    def __init__(self, dim_in, dim_out, ker_size=7, groups=8):
        super(DwCBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=ker_size, stride=1, padding=(ker_size - 1) // 2, bias=True,
                      groups=dim_in),
            nn.Conv2d(dim_in, dim_out, 1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        spa = torch.mean(x, 1, True)
        spa = nn.Sigmoid()(self.conv(spa))
        x = torch.mul(spa, x) + x
        return x

class CA(nn.Module):
    def __init__(self, ch):
        super(CA, self).__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        gap = nn.AdaptiveAvgPool2d((1, 1))(x)
        gap = nn.Sigmoid()(self.conv(gap))
        x = torch.mul(gap, x) + x
        return x

# Depth-wise Pyramid Pooling
class DwPP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(DwPP, self).__init__()
        self.branch1 = BaseCBR(dim_in, dim_out, 1)
        self.branch2 = BaseCBR(dim_in, dim_out, 3)
        self.branch3 = BaseCBR(dim_in, dim_out, 5)
        self.branch4 = DwCBR(dim_in, dim_out, 9, dim_in // dim_out)
        self.branch5 = DwCBR(dim_in, dim_out, 13, dim_in // dim_out)
        self.branch6 = BaseCBR(dim_in, dim_out, 1)
        self.conv_cat = BaseCBR(dim_out * 6, dim_out, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        conv1x = self.branch1(x)
        conv3x = self.branch2(x)
        conv5x = self.branch3(x)
        conv9x = self.branch4(x)
        conv13x = self.branch5(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch6(global_feature)
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x, conv3x, conv5x, conv9x, conv13x, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

# Edge Pyramid Pooling
class EdPP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(EdPP, self).__init__()
        self.branch1 = BaseCBR(dim_in, dim_out, 1)
        self.branch2 = BaseCBR(dim_in, dim_out, 3)
        self.branch3 = BarCBR(dim_in, dim_out, (3, 1), (1, 0))
        self.branch4 = BarCBR(dim_in, dim_out, (1, 3), (0, 1))
        self.branch5 = BaseCBR(1, dim_out, 1)
        self.conv_cat = BaseCBR(dim_out * 5, dim_out, 1)

    def forward(self, x):
        _, _, h, w = x.size()
        conv1 = self.branch1(x)
        conv2 = self.branch2(x)
        conv3 = self.branch3(x)
        conv4 = self.branch4(x)
        spa_feature = torch.mean(x, 1, True)
        spa_feature = self.branch5(spa_feature)

        feature_cat = torch.cat([conv1, conv2, conv3, conv4, spa_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class AxialAttention(nn.Module):
    def __init__(self, channel):
        super(AxialAttention, self).__init__()
        self.LN = nn.LayerNorm(channel)
        self.q = nn.Linear(channel, channel, bias=False)
        self.k = nn.Linear(channel, channel, bias=False)
        self.v = nn.Linear(channel, channel, bias=False)

    def forward(self, x):
        B,A1,A2,C = x.shape
        
        x = self.LN(x)
        q = self.q(x).view(B, A1, A2, C)
        k = self.k(x).view(B, A1, A2, C)
        v = self.v(x).view(B, A1, A2, C)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (A1*C)**2
        attn = attn.softmax(dim=1)
        x = torch.matmul(attn, v).view(B,A1,A2,C)
        return x


class IDEGM(nn.Module):
    def __init__(self, channel):
        super(IDEGM, self).__init__()
        self.x_horz_conv_init = convbnrelu(channel, channel, (1, 1), p=(0, 0))
        self.x_vert_conv_init = convbnrelu(channel, channel, (1, 1), p=(0, 0))
        self.y_horz_conv_init = convbnrelu(channel, channel, (1, 1), p=(0, 0))
        self.y_vert_conv_init = convbnrelu(channel, channel, (1, 1), p=(0, 0))

        self.xy_horz_atten = AxialAttention(channel)
        self.xy_vert_atten = AxialAttention(channel)

        self.x_conv_out = convbnrelu(channel, channel, (1, 1), p=(0, 0))
        self.y_conv_out = convbnrelu(channel, channel, (1, 1), p=(0, 0))

    def forward(self, x, y):  # x,y [B,C,H,W] -> [B,C,2H,W]/[B,C,H,2W] -> [B,W,2H,C]/[B,H,2W,C]
        size = x.shape[-1]
        x_horz = self.x_horz_conv_init(x)
        x_vert = self.x_vert_conv_init(x)
        y_horz = self.y_horz_conv_init(y)
        y_vert = self.y_vert_conv_init(y)

        xy_horz = torch.cat((x_horz, y_horz), dim=3)
        l_token1 = F.pad(xy_horz, [0, 0, 1, 0])
        l_token2 = F.pad(xy_horz, [0, 0, 0, 1])
        xy_horz = torch.cat((l_token1, l_token2), dim=3)

        xy_horz = xy_horz.permute(0, 2, 3, 1)
        xy_horz = self.xy_horz_atten(xy_horz)
        xy_horz = xy_horz.permute(0, 3, 1, 2)

        l_token1, l_token2 = xy_horz[:, :, :, :2 * size], xy_horz[:, :, :, 2 * size:]

        l_token1 = l_token1[:, :, 1:, :]
        l_token2 = l_token2[:, :, :-1, :]
        xy_horz = l_token1 + l_token2

        xy_vert = torch.cat((x_vert, y_vert), dim=2)
        h_token1 = F.pad(xy_vert, [1, 0, 0, 0])
        h_token2 = F.pad(xy_vert, [0, 1, 0, 0])
        xy_vert = torch.cat((h_token1, h_token2), dim=2)

        xy_vert = xy_vert.permute(0, 3, 2, 1)
        xy_vert = self.xy_vert_atten(xy_vert)
        xy_vert = xy_vert.permute(0, 3, 2, 1)

        h_token1, h_token2 = xy_vert[:, :, :2 * size, :], xy_vert[:, :, 2 * size:, :]
        h_token1 = h_token1[:, :, :, 1:]
        h_token2 = h_token2[:, :, :, :-1]
        xy_vert = h_token1 + h_token2

        x_horz, y_horz = xy_horz[:, :, :, :size], xy_horz[:, :, :, size:]
        x_vert, y_vert = xy_vert[:, :, :size, :], xy_vert[:, :, size:, :]

        out_x = self.x_conv_out(x_horz + x_vert) + x
        out_y = self.y_conv_out(y_horz + y_vert) + y
        return out_x, out_y



# interaction module
class InterModule(nn.Module):
    def __init__(self):
        super(InterModule, self).__init__()
        self.edge_init = BaseCBR(64, 64)
        self.sal_init = BaseCBR(64, 64)

        self.idegm = IDEGM(64)

        self.edge_out = BaseCBR(64, 64)
        self.sal_out = BaseCBR(64, 64)

    def forward(self, sal, edge):
        edge = self.edge_init(edge)
        sal = self.sal_init(sal)

        edge, sal = self.idegm(edge, sal)

        edge = self.edge_out(edge)
        sal = self.sal_out(sal)
        return sal, edge

# attention fusion module
class AF(nn.Module):
    def __init__(self, type='s'):
        super(AF, self).__init__()
        
        if type == 's':
            self.att = SA()
        else:
            self.att = CA(192)

        self.cbr1up = BaseCBR(64, 64)
        self.cbr1lo = BaseCBR(64, 64)

        self.cbr2up = BaseCBR(192, 64)
        self.cbr2lo = BaseCBR(192, 64)

        self.cbr3up = BaseCBR(64, 64)
        self.cbr3lo = BaseCBR(64, 64)

        self.cbrfuse = BaseCBR(128, 64)

    def forward(self, up, low):
        if low.size()[2:] != up.size()[2:]:
            low = F.interpolate(low, size=up.size()[2:], mode='bilinear', align_corners=True)

        fea1up = self.cbr1up(up)
        fea1lo = self.cbr1lo(low)

        mul = torch.mul(fea1up, fea1lo)
        fuse = torch.cat((fea1up, fea1lo), 1)
        fuse = torch.cat((fuse, mul), 1)

        fea_weighted = self.att(fuse)

        fea2up = self.cbr2up(fea_weighted)
        fea2lo = self.cbr2lo(fea_weighted)

        fea3up = self.cbr3up(fea2up + fea1up)
        fea3lo = self.cbr3lo(fea2lo + fea1lo)

        fea = torch.cat((fea3up, fea3lo), 1)
        fea = self.cbrfuse(fea)
        return fea


# channel-attention fusion module (CAF)
class CAF(nn.Module):
    def __init__(self):
        super(CAF, self).__init__()
        self.fusemodule = AF(type='c')

    def forward(self, up, low):
        if low.size()[2:] != up.size()[2:]:
            low = F.interpolate(low, size=up.size()[2:], mode='bilinear', align_corners=True)
        out = self.fusemodule(up, low)
        return out


# spatial-attention fusion module (SAF)
class SAF(nn.Module):
    def __init__(self):
        super(SAF, self).__init__()
        self.fusemodule = AF(type='s')

    def forward(self, up, low):
        if low.size()[2:] != up.size()[2:]:
            low = F.interpolate(low, size=up.size()[2:], mode='bilinear', align_corners=True)
        out = self.fusemodule(up, low)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # sod
        self.caf34 = CAF()
        self.caf23 = CAF()
        self.caf12 = CAF()
        # edge
        self.saf34 = SAF()
        self.saf23 = SAF()
        self.saf12 = SAF()
        # interaction
        self.im4 = InterModule()
        self.im3 = InterModule()
        self.im2 = InterModule()

    def forward(self, s1, s2, s3, s4, e1, e2, e3, e4):
        s4, e4 = self.im4(s4, e4)
        s3 = self.caf34(s3, s4)
        e3 = self.saf34(e3, e4)

        s3, e3 = self.im3(s3, e3)
        s2 = self.caf23(s2, s3)
        e2 = self.saf23(e2, e3)

        s2, e2 = self.im2(s2, e2)
        s1 = self.caf12(s1, s2)
        e1 = self.saf12(e1, e2)
        return s1, s2, s3, s4, e1, e2, e3, e4


# select and enhance
class SEBlock(nn.Module):
    def __init__(self):
        super(SEBlock, self).__init__()
        self.edge_init = BaseCBR(64, 64)
        self.sal_init = BaseCBR(64, 64)

        self.adv_global = nn.AdaptiveAvgPool2d(1)

        self.edge_out = BaseCBR(64, 64)
        self.sal_out = BaseCBR(64, 64)

    def forward(self, sal, edge):
        edge = self.edge_init(edge)
        sal = self.sal_init(sal)

        spatial_weight = torch.mean(edge, 1, True)
        channel_weight = self.adv_global(sal)

        edge = channel_weight * edge * spatial_weight
        sal = edge.detach() * channel_weight * sal

        edge = self.edge_out(edge)
        sal = self.sal_out(sal)
        return sal, edge


class IDENet(nn.Module):
    def __init__(self, cfg):
        super(IDENet, self).__init__()
        self.cfg = cfg
        self.backbone_pvt = pvt_v2_b2()
        self.backbone_pvt.load_state_dict(torch.load("/home/jsy/IDENet/models/pvt_v2_b2.pth"),strict=False)

        chs = [64,128,320,512]

        self.dwpp4 = DwPP(chs[3], 64)
        self.dwpp3 = DwPP(chs[2], 64)
        self.dwpp2 = DwPP(chs[1], 64)
        self.dwpp1 = DwPP(chs[0], 64)

        self.edpp4 = EdPP(chs[3], 64)
        self.edpp3 = EdPP(chs[2], 64)
        self.edpp2 = EdPP(chs[1], 64)
        self.edpp1 = EdPP(chs[0], 64)

        self.decoder = Decoder()

        self.se1 = SEBlock()
        self.se2 = SEBlock()

        self.linears1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linears2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linears3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linears4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # edge branch
        self.lineare1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineare2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineare3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineare4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x, dep, shape=None):
        shape = x.size()[2:] if shape is None else shape
        r = x[:,0].unsqueeze(1)
        m, _ = x[:,1:].max(dim=1)
        d = dep[:, 0].unsqueeze(1)
        x = torch.cat((r, m.unsqueeze(1), d), dim=1)

        pvt1, pvt2, pvt3, pvt4 = self.backbone_pvt(x)

        e1, e2, e3, e4 = self.edpp1(pvt1), self.edpp2(pvt2), self.edpp3(pvt3), self.edpp4(pvt4)
        s1, s2, s3, s4 = self.dwpp1(pvt1), self.dwpp2(pvt2), self.dwpp3(pvt3), self.dwpp4(pvt4)
        
        s1, s2, s3, s4, e1, e2, e3, e4 = self.decoder(s1, s2, s3, s4, e1, e2, e3, e4)

        s1, e1 = self.se1(s1, e1)
        s2, e2 = self.se2(s2, e2)

        s1 = F.interpolate(self.linears1(s1), size=shape, mode='bilinear', align_corners=True)
        s2 = F.interpolate(self.linears2(s2), size=shape, mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.linears3(s3), size=shape, mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.linears4(s4), size=shape, mode='bilinear', align_corners=True)

        e1 = F.interpolate(self.lineare1(e1), size=shape, mode='bilinear', align_corners=True)
        e2 = F.interpolate(self.lineare2(e2), size=shape, mode='bilinear', align_corners=True)
        e3 = F.interpolate(self.lineare3(e3), size=shape, mode='bilinear', align_corners=True)
        e4 = F.interpolate(self.lineare4(e4), size=shape, mode='bilinear', align_corners=True)
        return s1, s2, s3, s4, e1, e2, e3, e4

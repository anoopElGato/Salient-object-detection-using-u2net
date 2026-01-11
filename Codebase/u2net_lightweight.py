"""
Lightweight U2-Net Model variant with reduced parameters: U2NET-LITE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class REBNCONV(nn.Module):
    """Convolution -> Batch Normalization -> ReLU"""
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def upsample_like(src, tar):
    """Upsample src to match tar's spatial dimensions"""
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)


class RSU4_Lite(nn.Module):
    """RSU with 4 layers and reduced intermediate channels"""
    def __init__(self, in_ch=3, mid_ch=10, out_ch=3):
        super(RSU4_Lite, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class U2NETLITE(nn.Module):
    """
    Lightweight U2-Net:
    - 4 encoder stages instead of 6
    - Reduced channel dimensions
    - Shorter RSU blocks
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETLITE, self).__init__()

        # Encoder: 4 stages
        self.stage1 = RSU4_Lite(in_ch, 10, 32)  # 32 channels
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU4_Lite(32, 12, 64)  # 64 channels
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU4_Lite(64, 15, 128)  # 128 channels
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4_Lite(128, 20, 128)  # 128 channels

        # Decoder: 3 stages
        self.stage3d = RSU4_Lite(256, 15, 64)
        self.stage2d = RSU4_Lite(128, 12, 32)
        self.stage1d = RSU4_Lite(64, 10, 32)

        # Side outputs
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(128, out_ch, 3, padding=1)

        # Fusion
        self.outconv = nn.Conv2d(4*out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx4up = upsample_like(hx4, hx3)

        # Decoder
        hx3d = self.stage3d(torch.cat((hx4up, hx3), 1))
        hx3dup = upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # Side outputs
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = upsample_like(d3, d1)

        d4 = self.side4(hx4)
        d4 = upsample_like(d4, d1)

        # Fusion
        d0 = self.outconv(torch.cat((d1, d2, d3, d4), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Estimate model size in MB"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return size_all_mb

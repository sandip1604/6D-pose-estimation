import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, inputChannels, outChannels, hiddenChannels=None):
        super().__init__()
        if not hiddenChannels:
            hiddenChannels = outChannels
        self.doubleConv = nn.Sequential(
            nn.Conv2d(inputChannels, hiddenChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hiddenChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hiddenChannels, outChannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.doubleConv(x)


class DownSizing(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, inputChannels, outChannels):
        super().__init__()
        self.maxpoolConv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(inputChannels, outChannels)
        )

    def forward(self, x):
        return self.maxpoolConv(x)


class UpSizing(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = lambda x: torch.tensor([x2.size()[x] - x1.size()[x]])
        diffY, diffX = diff(2), diff(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
        
class UNet(nn.Module):
    def __init__(self, n_channels = 3, out_channels_id = 9, out_channels_uv = 256, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.out_channels_id = out_channels_id
        self.out_channels_uv = out_channels_uv
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownSizing(64, 128)
        self.down2 = DownSizing(128, 256)
        self.down3 = DownSizing(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownSizing(512, 1024 // factor)


        #ID MASK
        self.up1Id = UpSizing(1024, 512, bilinear)
        self.up2Id = UpSizing(512, 256, bilinear)
        self.up3Id = UpSizing(256, 128, bilinear)
        self.up4Id = UpSizing(128, 64 * factor, bilinear)
        self.outcId = OutConv(64, out_channels_id)

        #U Mask
        self.up1U = UpSizing(1024, 512, bilinear)
        self.up2U = UpSizing(512, 512, bilinear)
        self.outcU1 = OutConv(256, out_channels_uv)
        self.outcU2 = OutConv(256, out_channels_uv)
        self.outcU3 = OutConv(256, out_channels_uv)
        self.outcU4 = OutConv(256, out_channels_uv)
        self.up3U = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4U = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #V Mask
        self.up1V = UpSizing(1024, 512, bilinear)
        self.up2V = UpSizing(512, 512, bilinear)
        self.outcV1 = OutConv(256, out_channels_uv)
        self.outcV2 = OutConv(256, out_channels_uv)
        self.outcV3 = OutConv(256, out_channels_uv)
        self.outcV4 = OutConv(256, out_channels_uv)
        self.up3V = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4V = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # ID mask
        xId = self.up1Id(x5, x4)
        xId = self.up2Id(xId, x3)
        xId = self.up3Id(xId, x2)
        xId = self.up4Id(xId, x1)
        logits_id = self.outcId(xId)

        # U mask
        xU = self.up1U(x5, x4)
        xU = self.up2U(xU, x3)
        xU = self.outcU1(xU)
        xU = self.outcU2(xU)
        xU = self.outcU3(xU)
        xU = self.up3U(xU)
        xU = self.up4U(xU)
        logits_u = self.outcU4(xU)

        # V mask
        xV = self.up1V(x5, x4)
        xV = self.up2V(xV, x3)
        xV = self.outcV1(xV)
        xV = self.outcV2(xV)
        xV = self.outcV3(xV)
        xV = self.up3V(xV)
        xV = self.up4V(xV)
        logits_v = self.outcV4(xV)
        
        return self.outcId(xId), self.outcU4(xU), self.outcV4(xV)
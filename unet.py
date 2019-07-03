import torch
from torch import nn
from torch.nn import functional as F

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, dilation=1, first=False):
        super().__init__()
        tmp = self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3,padding=padding, dilation=dilation),
            nn.ReLU()
        )

        if first:
            self.layer = tmp
        else:
            self.layer = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(in_channels),
                tmp
            )

    def forward(self, x):
        return self.layer(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, dilation=1, crop=True):
        """
        :param padding: Padding to be used in convolution
        :param dilation: dilation to be used in convolution
        :param crop: True if crop input1 before concatenating,
                    False if pad input2 before concatenating.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.ReLU()
        )
        self.crop = crop

    def forward(self, x_prev, x_cur):
        x_cur = self.up(x_cur)

        hc, wc = x_cur.shape[-2:]
        hp, wp = x_prev.shape[-2:]
        diffH = hc-hp
        diffW = wc-wp
        if self.crop:
            h_p = diffH//2
            w_p = diffW//2
            x_prev = F.pad(x_prev, (w_p, diffW-w_p, h_p, diffH-h_p))
        else:
            diffH *= -1
            diffW *= -1
            h_p = diffH//2
            w_p = diffW//2
            x_cur = F.pad(x_cur, (w_p, diffW-w_p, h_p, diffH-h_p))

        x = torch.cat([x_cur, x_prev], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, pad_before, down_dilation=[2,2,2,2,2],
                up_dilation=[2,2,2,2], p = 0.4, crop=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(pad_before)
        self.down1 = DownConv(in_channels, 64, 0, down_dilation[0], True)
        self.down2 = DownConv(64, 128, 0, down_dilation[1])
        self.down3 = DownConv(128, 256, 0, down_dilation[2])
        self.down4 = DownConv(256, 512, 0, down_dilation[3])
        self.down5 = DownConv(512, 1024, 0, down_dilation[4])
        self.dropout = nn.Dropout(p=p)
        self.up1 = UpConv(1024, 512, 0, up_dilation[0])
        self.up2 = UpConv(512, 256, 0, up_dilation[1])
        self.up3 = UpConv(256, 128, 0, up_dilation[2])
        self.up4 = UpConv(128, 64, 0, up_dilation[3])
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x0 = x
        x = self.pad(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x5 = self.dropout(x5)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final(x)
        return x

"""For this version, the idea is that for first few layers
    if input has two similar images the output should be
    almost similar. Hopefully this should achive consistency
    across time :)
"""

import torch
from torch import nn
from torch.nn import functional as F

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, dilation=1, first=False):
        super().__init__()
        tmp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
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
    def __init__(self, in_channels, out_channels, pad_before,
                 down_dilation=[2, 2, 2, 2, 2],
                 up_dilation=[2, 2, 2, 2], p=0.4,
                 return_initial=0):
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
        self.final = nn.Conv2d(64+in_channels, out_channels, 1)

        self.return_initial = return_initial

    def forward(self, img1, img2):
        x0 = img1
        img1 = self.pad(img1)

        processed = 0
        if self.return_initial:
            img2 = self.pad(img2)
            to_return = []

        x1 = self.down1(img1)
        if processed < self.return_initial:
            processed += 1
            img2 = self.down1(img2)
            to_return.append((x1, img2))

        x2 = self.down2(x1)
        if processed < self.return_initial:
            processed += 1
            img2 = self.down1(img2)
            to_return.append((x2, img2))

        x3 = self.down3(x2)
        if processed < self.return_initial:
            processed += 1
            img2 = self.down1(img2)
            to_return.append((x3, img2))

        x4 = self.down4(x3)
        if processed < self.return_initial:
            processed += 1
            img2 = self.down1(img2)
            to_return.append((x4, img2))

        x5 = self.down5(x4)
        if processed < self.return_initial:
            processed += 1
            img2 = self.down1(img2)
            to_return.append((x5, img2))

        x5 = self.dropout(x5)
        img1 = self.up1(x4, x5)
        img1 = self.up2(x3, img1)
        img1 = self.up3(x2, img1)
        img1 = self.up4(x1, img1)

        h_current, w_current = img1.shape[-2:]
        h_orig, w_orig = x0.shape[-2:]
        diff_h = h_current-h_orig
        diff_w = w_current-w_orig
        h = diff_h//2
        w = diff_w//2
        x0 = F.pad(x0, (w, diff_w-w, h, diff_h-h))
        img1 = torch.cat([img1, x0], dim=1)
        img1 = self.final(img1)
        return img1

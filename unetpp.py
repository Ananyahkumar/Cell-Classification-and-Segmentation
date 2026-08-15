import torch
import torch.nn as nn

class NestedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetPP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        n1, n2, n3, n4, n5 = 32, 64, 128, 256, 512

        # Encoder
        self.conv0_0 = NestedConvBlock(in_ch, n1)
        self.conv1_0 = NestedConvBlock(n1, n2)
        self.conv2_0 = NestedConvBlock(n2, n3)
        self.conv3_0 = NestedConvBlock(n3, n4)
        self.conv4_0 = NestedConvBlock(n4, n5)

        # Decoder (nested blocks)
        self.conv0_1 = NestedConvBlock(n1+n2, n1)
        self.conv1_1 = NestedConvBlock(n2+n3, n2)
        self.conv2_1 = NestedConvBlock(n3+n4, n3)
        self.conv3_1 = NestedConvBlock(n4+n5, n4)

        self.conv0_2 = NestedConvBlock(n1*2+n2, n1)
        self.conv1_2 = NestedConvBlock(n2*2+n3, n2)
        self.conv2_2 = NestedConvBlock(n3*2+n4, n3)

        self.conv0_3 = NestedConvBlock(n1*3+n2, n1)
        self.conv1_3 = NestedConvBlock(n2*3+n3, n2)

        self.conv0_4 = NestedConvBlock(n1*4+n2, n1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = lambda ch: nn.ConvTranspose2d(ch, ch, 2, stride=2)

        self.up1_0 = self.upsample(n2)
        self.up2_0 = self.upsample(n3)
        self.up3_0 = self.upsample(n4)
        self.up4_0 = self.upsample(n5)

        self.up1_1 = self.upsample(n2)
        self.up2_1 = self.upsample(n3)
        self.up3_1 = self.upsample(n4)

        self.up1_2 = self.upsample(n2)
        self.up2_2 = self.upsample(n3)

        self.up1_3 = self.upsample(n2)

        self.final = nn.Conv2d(n1, out_ch, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x4_0 = self.conv4_0(self.maxpool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 1))

        return torch.sigmoid(self.final(x0_4))

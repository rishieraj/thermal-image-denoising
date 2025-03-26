import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(x)

class NUCNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.down_doubleconv1 = DoubleConv(1, features//16)
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(features//16, features//16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features//16),
            nn.ReLU()
        )
        self.down_doubleconv2 = DoubleConv(features//16, features//8)
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(features//8, features//8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features//8),
            nn.ReLU()
        )
        self.down_doubleconv3 = DoubleConv(features//8, features//4)
        self.down_conv3 = nn.Sequential(
            nn.Conv2d(features//4, features//4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features//4),
            nn.ReLU()
        )
        self.down_doubleconv4 = DoubleConv(features//4, features//2)
        self.down_conv4 = nn.Sequential(
            nn.Conv2d(features//2, features//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU()
        )

        self.bottleneck = DoubleConv(features//2, features)

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(features, features//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(features//2),
            nn.ReLU()
        )
        self.up_doubleconv1 = DoubleConv(features, features//2)

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(features//2, features//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(features//4),
            nn.ReLU()
        )
        self.up_doubleconv2 = DoubleConv(features//2, features//4)

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(features//4, features//8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(features//8),
            nn.ReLU()
        )
        self.up_doubleconv3 = DoubleConv(features//4, features//8)

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(features//8, features//16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(features//16),
            nn.ReLU()
        )
        self.up_doubleconv4 = DoubleConv(features//8, features//16)

        self.out = nn.Sequential(
            nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_conv1 = self.down_doubleconv1(x)
        x = self.down_conv1(x_conv1)
        x_conv2 = self.down_doubleconv2(x)
        x = self.down_conv2(x_conv2)
        x_conv3 = self.down_doubleconv3(x)
        x = self.down_conv3(x_conv3)
        x_conv4 = self.down_doubleconv4(x)
        x = self.down_conv4(x_conv4)

        x = self.bottleneck(x)

        x = self.up_conv1(x)
        x = torch.cat([x_conv4, x], dim=1)
        x = self.up_doubleconv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x_conv3, x], dim=1)
        x = self.up_doubleconv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x_conv2, x], dim=1)
        x = self.up_doubleconv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x_conv1, x], dim=1)
        x = self.up_doubleconv4(x)

        x = self.out(x)

        return x
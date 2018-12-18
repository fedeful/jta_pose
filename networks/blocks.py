import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class SpecialDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SpecialDownBlock, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x1, x2):
        x1 = self.mpconv(x1)
        xf = torch.cat([x1, x2], dim=1)
        return xf


class UpBlock(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, middle_ch, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(middle_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_X = x1.size()[2] - x2.size()[2]
        diff_Y = x1.size()[3] - x2.size()[4]
        x2 = F.pad(x2, (diff_X // 2, int(diff_X / 2),
                        diff_Y // 2, int(diff_Y / 2)))

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.conv2 = DoubleConv(out_ch*2, out_ch*2)

    def forward(self, x):
        x = self.conv(x)
        return x


class SpecialInConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SpecialInConv, self).__init__()
        self.conv_1 = DoubleConv(in_ch, out_ch)
        self.conv_2 = DoubleConv(out_ch*2, out_ch*2)

    def forward(self, x1, x2):
        x1 = self.conv_1(x1)
        x_int = torch.cat([x1, x2], dim=1)
        x_int = self.conv_2(x_int)
        return x_int


class SpecialOutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SpecialOutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch//2, 1)
        self.conv2 = nn.Conv2d(in_ch//2, out_ch, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GeneratorSimpleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GeneratorSimpleBlock, self).__init__()
        self.sb = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 5, stride=1, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.sb(x)
        return x


class GeneratorConvTransBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GeneratorConvTransBlock, self).__init__()
        self.sb = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.sb(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalization):
        super(DiscriminatorBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))
        if normalization:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.sq = nn.Sequential(
            *layers
        )

    def forward(self, x):
        return self.sq(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):

        super(ResidualBlock, self).__init__()
        # First Mini-Block
        self.conv_1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(ch)
        self.relu_1 = nn.ReLU()

        # Second  Mini-Block
        self.conv_2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(ch)
        self.relu_2 = nn.ReLU()

        self.relu_f = nn.ReLU()

    def forward(self, x):

        x1 = self.relu_1(self.bn_1(self.conv_1(x)))
        x1 = self.relu_2(self.bn_2(self.conv_2(x1)))

        return self.relu_f(x + x1)


class SimpleDoubleCo(nn.Module):
    def __init__(self, channel_in, channel_out, keep_dimension=True, batch_normalization=True, activation=True):
        super(SimpleDoubleCo, self).__init__()
        self.ci = channel_in
        self.co = channel_out
        self.kd = keep_dimension
        self.bn = batch_normalization
        self.a = activation

        if self.a:
            self.relu_0 = nn.ReLU()
            self.relu_1 = nn.ReLU()

        if self.bn:
            self.batch_norm_0 = nn.BatchNorm2d(channel_out)
            self.batch_norm_1 = nn.BatchNorm2d(channel_out)

        if self.kd:
            self.conv_0 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1)
            self.conv_1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_0 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=0)
            self.conv_1 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        if self.a and self.bn:
            return self.relu_1(self.batch_norm_1(self.conv_1(self.relu_0(self.batch_norm_0(self.conv_0(x))))))
        elif self.a:
            return self.relu_1(self.conv_1(self.relu_0(self.conv_0(x))))
        elif self.bn:
            return self.batch_norm_1(self.conv_1(self.batch_norm_0(self.conv_0(x))))
        else:
            return self.conv_1(self.conv_0(x))


class SimpleUnetDownBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(SimpleUnetDownBlock, self).__init__()

        self.ci = channel_in
        self.co = channel_out

        self.conv_down = nn.Conv2d(self.ci, self.ci, kernel_size=4, stride=2, padding=1)
        self.scb = SimpleDoubleCo(self.ci, self.co)

    def forward(self, x):
        x = self.scb(self.conv_down(x))
        return x


class SimpleUnetUpBlock(nn.Module):
    def __init__(self, channel_in, channel_out, final=False):
        super(SimpleUnetUpBlock, self).__init__()

        self.ci = channel_in
        self.co = channel_out

        if final:
            self.conv_up = nn.ConvTranspose2d(self.ci, self.ci, kernel_size=2, stride=2)
            self.sdc = SimpleDoubleCo(self.ci*2, self.co)
        else:
            self.conv_up = nn.ConvTranspose2d(self.ci, self.co, kernel_size=2, stride=2)
            self.sdc = SimpleDoubleCo(self.ci, self.co)

    def forward(self, x1, x2):

        x1 = self.conv_up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))

        x = torch.cat([x2, x1], dim=1)
        x = self.sdc(x)
        return x

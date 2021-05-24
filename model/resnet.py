import torch
import torch.nn as nn
import torch.nn.functional as F


assert torch.cuda.is_available(), "DCN doesn't work on cpu"


class BasicLayer(nn.Module):
    """Basic residual layer used in ResNet18 & 34"""
    def __init__(self, inC, outC, halve_size=False):
        """BasicLayer's constructor

        Args:
            inC (int): number of channels of layer's input
            outC (int): number of channels of layer's output
            halve_size (bool): output has half the size of input or not. Default: False
        """
        super(BasicLayer, self).__init__()
        self.halve_size = halve_size
        # components of main pass
        if self.halve_size == False:
            self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(outC)
            self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(outC)
        else:
            self.conv1 = nn.Conv2d(inC, outC, kernel_size=3, stride=2, padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(outC)
            self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=1, padding=1,bias=False)
            self.bn2 = nn.BatchNorm2d(outC)

        # components of skip connection
        self.conv_skip = None
        self.bn_skip = None
        if self.halve_size:
            self.conv_skip = nn.Conv2d(inC, outC, kernel_size=1, stride=2, bias=False)
            self.bn_skip = nn.BatchNorm2d(outC)

    def forward(self, input):
        """Forward pass


        Args:
            input (torch.Tensor): shape (N, inC, H, W)
        Returns:
            torch.Tensor: shape (N, outC, H', W'), H' = H/2 if halve_size else H
        """
        # main pass
        out = self.conv1(input)  # shape (N, outC, H', W')
        out = F.relu(self.bn1(out))

        out = self.conv2(out)  # shape (N, outC, H', W')
        out = self.bn2(out)

        # skip connection
        if self.halve_size:
            residual = self.conv_skip(input)  # shape (N, outC, H/2, W/2)
            residual = self.bn_skip(residual)
        else:
            residual = input  # shape (N, inC, H, W)

        out = F.relu(out + residual)  # shape (N, outC, H', W')
        return out


class UpSampleLayer(nn.Module):
    """A module made of a Deformable Convolution layer & a ConvTranspose2d for upsampling a feature map"""
    def __init__(self, inC, outC):
        """Constructor of UpSampleLayer

        Args:
            inC (int): number of input's channels
            outC (int): number of output's channels
        """
        super(UpSampleLayer, self).__init__()
        self.defconv = nn.Conv2d(inC, outC, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outC)
        self.convtrans = nn.ConvTranspose2d(outC, outC, kernel_size=4, stride=2, padding=1, output_padding=0,
                                            bias=False)
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self, input):
        """Forward pass


        Args:
            input (torch.Tensor): shape (N, inC, H, W)
        Return:
            torch.Tensor: upsampled tensor, shape (N, outC, 2*H, 2*W)
        """
        out = self.bn1(self.defconv(input))
        out = self.bn2(self.convtrans(out))
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     stride=2, padding=1)
        # NOTE: after passing through maxpool1, image size is reduced by 4

        # down-sampling path made of ResNet18
        self.conv2_1 = BasicLayer(64, 64, False)
        self.conv2_2 = BasicLayer(64, 64, False)

        self.conv3_1 = BasicLayer(64, 128, True)
        self.conv3_2 = BasicLayer(128, 128, False)

        self.conv4_1 = BasicLayer(128, 256, True)
        self.conv4_2 = BasicLayer(256, 256, False)

        self.conv5_1 = BasicLayer(256, 512, True)
        self.conv5_2 = BasicLayer(512, 512, False)

        # up-sampling path
        # NOTE: compared to Figure.6 UpSampleLayer below are indexed from bottom to top
        # (meaning self.up3 is the last one)
        self.up1 = UpSampleLayer(512, 256)
        self.up2 = UpSampleLayer(256, 128)
        self.up3 = UpSampleLayer(128, 64)

    def forward(self, input):
        """Forward pass


        Args:
            input (torch.Tensor): shape (N, 3, 384, 384)
        Returns:
            torch.Tensor: shape (N, 64, 96, 96)
        """
        out = self.bn1(self.conv1(input))
        out = self.maxpool1(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.up1(out)
        out = self.up2(out)
        out = self.up3(out)
        return out
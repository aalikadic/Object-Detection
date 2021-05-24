import torch
import torch.nn as nn

from collections import OrderedDict


def conv_block(inC, midC, outC):
    """Construct a block of 2 conv2d layers. This block makes a branch in the head of an architecture
        in CenterNet family

    Args:
        inC (int): number of input channels
        midC (int): number of output channels of 1st conv2d layer
        outC (int): number of output channels of 2nd conv2d layer
    Returns:
        torch.nn.Module: a block of 2 conv2d layers
    """
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(inC, midC, kernel_size=3,
                            stride=1, padding=1, bias=True)),
        ('relu', nn.ReLU()),
        ('conv2', nn.Conv2d(midC, outC, kernel_size=1,
                            stride=1, padding=0, bias=True))
    ]))


class CenterNetHead(nn.Module):
    """Head of an architecture defined in CenterNet framework"""
    def __init__(self, inC, nClasses, midC):
        """Constructor of CenterNetHead

        Args:
            inC (int): number of channels of output of backbone
            nClasses (int): number of classes of objects in the dataset
            midC (int): number of output channels of 1st conv2d layer of a branch of head
        """
        super(CenterNetHead, self).__init__()
        self.heatmap = conv_block(inC, midC, 20)
        self.wh = conv_block(inC, midC, 2)
        self.offset = conv_block(inC, midC, 2)

    def forward(self, input):
        """Forward passs
        Args:
            input (torch.Tensor): output of backone, shape (N, inC, H, W)
        """
        hm = self.heatmap(input)  # shape (N, nClasses, H, W) -
        wh = self.wh(input)  # shape (N, 2, H, W) -
        offset = self.offset(input)  # shape (N, 2, H, W) -

        out = torch.cat([hm, wh, offset], dim=1)  # shape (N, nClasses + 4, H, W)
        return out



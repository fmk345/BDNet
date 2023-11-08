import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------
# basic blocks
# --------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, bias=True, norm=False, act='relu', transpose=False, output_padding=0, groups=1):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        if act == 'relu':
            act = nn.ReLU(inplace=True)

        layers = list()
        padding = kernel_size // 2
        if transpose:
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups))
        if norm:
            layers.append(norm)
        elif act:
            layers.append(act)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, n_feat, kernel_size=[3, 3]):
        super(ResBlock, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        layers = [BasicConv(n_feat, n_feat, kernel_size=sz, act=nn.ReLU(
            inplace=True)) for sz in kernel_size[:-1]]
        layers.append(BasicConv(n_feat, n_feat,
                      kernel_size=kernel_size[-1], act=False))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x) + x


class ResBlockSeq(nn.Module):
    """
    Sequential residual blocks
    """

    def __init__(self, in_channel, out_channel, num_res=3, kernel_size=3, Conv=BasicConv, ResBlock=ResBlock):
        super(ResBlockSeq, self).__init__()

        layers = [Conv(in_channel, out_channel, kernel_size)] + \
            [ResBlock(out_channel, [kernel_size, kernel_size])
             for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

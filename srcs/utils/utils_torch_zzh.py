import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.signal import convolve2d, correlate2d

# ===============
# model info: refer to `utils_eval_zzh.py`
# ===============

# ===============
# basic operation
# ===============

# --------------------------------------------
# padding and crop with pytorch
# --------------------------------------------


def pad2same_size(x1, x2):
    '''
    pad x1 or x2 to the same size (to the size of the larger one)
    '''
    diffX = x2.size()[3] - x1.size()[3]
    diffY = x2.size()[2] - x1.size()[2]

    if diffX == 0 and diffY == 0:
        return x1, x2

    if diffX >= 0 and diffY >= 0:
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))
    elif diffX < 0 and diffY < 0:
        x2 = nn.functional.pad(
            x2, (-diffX // 2, -diffX - (-diffX)//2, -diffY // 2, -diffY - (-diffY)//2))
    elif diffX >= 0 and diffY < 0:
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2, 0, 0))
        x2 = nn.functional.pad(
            x2, (0, 0, -diffY // 2, -diffY - (-diffY)//2))
    elif diffX < 0 and diffY >= 0:
        x1 = nn.functional.pad(x1, (0, 0, diffY // 2, diffY - diffY//2))
        x2 = nn.functional.pad(
            x2, (-diffX // 2, -diffX - (-diffX)//2, 0, 0))

    return x1, x2


def pad2size(x, size):
    '''
    pad x to given size
    x: N*C*H*W
    size: H'*W'
    '''
    diffX = size[1] - x.size()[3]
    diffY = size[0] - x.size()[2]

    if diffX == 0 and diffY == 0:
        return x

    if diffX >= 0 and diffY >= 0:
        x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2,
                                  diffY // 2, diffY - diffY//2))
    elif diffX < 0 and diffY < 0:
        x = x[:, :, -diffY // 2: diffY + (-diffY) //
              2, -diffX // 2: diffX + (-diffX)//2]
    elif diffX >= 0 and diffY < 0:
        x = x[:, :, -diffY // 2: diffY + (-diffY)//2, :]
        x = nn.functional.pad(x, (diffX // 2, diffX - diffX//2, 0, 0))
    elif diffX < 0 and diffY >= 0:
        x = x[:, :, :, -diffX // 2: diffX + (-diffX)//2]
        x = nn.functional.pad(x, (0, 0, diffY // 2, diffY - diffY//2))
    return x


# --------------------------------------------
# real convolution implemented with pytorch
# --------------------------------------------


def torch_real_conv2d(tensor, psf, mode='circular'):
    '''
    tensor:     N*C*H*W, torch.float32 tensor
    psf:        n*C*h*w, torch.float32 tensor
    mode:       conlolution mode, 'circular' | 'same' | 'valid'
    $return:    Nn*C*H'*W', each 2D H*W image will convolve with $n 2D h*w psf
    Note: this code will cause very large CPU memory if not use GPU/CUDA
    '''
    # padding for 'circular' mode
    x_pad_len, y_pad_len = psf.shape[-2]-1, psf.shape[-1]-1
    if mode == 'circular':
        pad_width = (y_pad_len//2, y_pad_len-y_pad_len//2,
                     x_pad_len//2, x_pad_len-x_pad_len//2)
        tensor = F.pad(tensor, pad_width, "circular")
        mode = 'valid'  # after padding, use valid to conduct convolve

    # flip psf (F.conv2d is corr in fact)
    psf = torch.flip(psf, [-2, -1])

    # reshape
    n, c, h, w = list(psf.shape)
    psf = psf.permute(1, 0, 2, 3).reshape(n*c, 1, h, w)

    # conv
    conv_tensor = F.conv2d(tensor, psf, groups=tensor.shape[1], padding=mode)

    # reshape
    N, C, H, W = list(conv_tensor.shape)
    conv_tensor = conv_tensor.reshape(c, n*N, H, W).permute(1, 0, 2, 3)

    return conv_tensor

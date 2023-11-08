
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from scipy.signal import convolve2d, correlate2d
# --------------------------------------------
# padding
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


def pad4conv(tensor, psf_sz, mode='circular', value=0.0):
    '''
    padding image for convolution
    
    tensor (torch tensor):  4D image tensor ([N,C,H,W] ) to be padded
    psf_sz (list[int]): 2D psf size ([H,W) in convolution
    mode (str): padding mode, 'circular' (default) | 'constant' | 'reflect' | 'replicate'
    
    refer to F.pad for specific parameter setting
    '''
    x_pad_len, y_pad_len = psf_sz[0]-1, psf_sz[1]-1
    pad_width = (y_pad_len//2, y_pad_len-y_pad_len//2,
                 x_pad_len//2, x_pad_len-x_pad_len//2)
    tensor = F.pad(tensor, pad_width, mode=mode, value=value)
    return tensor


# --------------------------------------------
# normalization with pytorch
# --------------------------------------------
def NormLayer(norm_type, ch_width):
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer



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

# --------------------------------------------
# real convolutional layer implemented with scipy
# --------------------------------------------


class ScipyConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, filter, bias):
        # detach so we can cast to NumPy
        input, filter, bias = input.detach(), filter.detach(), bias.detach()
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        if bias is not None:
            result += bias.numpy()
        ctx.save_for_backward(input, filter, bias)
        return torch.as_tensor(result, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter, bias = ctx.saved_tensors
        grad_output = grad_output.numpy()
        grad_bias = np.sum(grad_output, keepdims=True)
        grad_input = convolve2d(grad_output, filter.numpy(), mode='full')
        # the previous line can be expressed equivalently as:
        # grad_input = correlate2d(grad_output, flip(flip(filter.numpy(), axis=0), axis=1), mode='full')
        grad_filter = correlate2d(input.numpy(), grad_output, mode='valid')
        return torch.from_numpy(grad_input), torch.from_numpy(grad_filter).to(torch.float), torch.from_numpy(grad_bias).to(torch.float)


class ScipyConv2d(Module):
    def __init__(self, filter_width, filter_height, bias=None):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(filter_width, filter_height))

        if bias:
            self.bias = Parameter(torch.randn(1, 1))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter, self.bias)

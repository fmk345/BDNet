# -*- coding: utf-8 -*-
import numpy as np
import scipy
import cv2
from math import cos, sin
from numpy import zeros, ones, prod, array, pi, log, maximum, mod, arange, sum, mgrid, exp, pad, round, ceil, floor
from numpy.random import randn, rand, randint, uniform
from scipy.signal import convolve2d
import torch
import os
import random
from os.path import join as opj
import torch.nn.functional as F
from scipy.signal import fftconvolve
from scipy import ndimage

'''
some codes are copied/modified from 
    https://github.com/cszn
    https://github.com/twhui/SRGAN-pyTorch
    https://github.com/xinntao/BasicSR
    https://gitlab.mpi-klsb.mpg.de/jdong/dwdn

Last modified: 2021-10-28 Zhihong Zhang 
'''

# ===============
# Fourier transformation
# ===============


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        # otf: NxCxHxWx2
        otf: NxCxHxW
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fft2(otf, dim=(-2, -1))
    return otf

# ===============
# image blurring
# ===============


def img_blur(img, psf, noise_level=0.01, mode='circular', cval=0):
    """
    blur image with blur kernel

    Args:
        img (ndarray): gray or rgb sharp image,[H, W <,C>]
        psf (ndarray): blur kernel,[H, W <,C>]
        noise_level (scalar): gaussian noise std (0-1)
        mode (str): convolution mode, 'circular' ('wrap') | 'constant' | ...
        cval: padding value for 'constant' padding

        refer to ndimage.filters.convolve for specific param setting

    Returns:
        x: blurred image
    """
    # convolution
    if mode == 'circular':
        # in ndimage.filters.convolve, 'circular'=='wrap'
        mode = 'wrap'
    if img.ndim == 3 and psf.ndim == 2:
        # rgb image
        psf = np.expand_dims(psf, axis=2)

    blur_img = ndimage.filters.convolve(
        img, psf, mode=mode, cval=cval)

    # add Gaussian noise
    blur_noisy_img = blur_img + \
        np.random.normal(0, noise_level, blur_img.shape)
    return blur_noisy_img.astype(np.float32)


def img_blur_torch(img, psf, noise_level=0.0, conv_mode='conv', pad_mode='circular', pad_value=0.0):
    """
    a sharp image blurred by a blur kernel (torch version) 

    Args:
        img (torch tensor): 4D image batch, [N,C,H,W]
        psf (torch tensor): 4D blur kernel batch [N,C,H,W]
        noise_level (scalar): gaussian noise level (0-1)
        conv_mode: convolution mode,'corr' | 'conv', implemented by flip the blur kernel
        pad_mode (str): padding mode
        pad_value (int): padding value for 'constant' padding mode


    Returns:
        blur_img (torch tensor): 4D blurry image batch
    """

    N1, C1, H1, W1 = img.shape
    N2, C2, H2, W2 = psf.shape
    assert N1 == N2, 'img and psf should have the same batch size'

    # match dimension
    if C1 != C2:
        # expand 1 channel psf to 3 channel
        psf = psf.expand([N2, C1, H2, W2]).contiguous()

    # flip psf
    if conv_mode=='conv':
        psf = psf.flip(-2, -1)
    
    # sharp image padding
    psf_sz = psf.shape[-2:]
    img_pad = pad4conv(img, psf_sz, mode=pad_mode, value=pad_value)

    # convolvolution of sharp image and kernel
    blur_img = torch.zeros_like(img)
    for k in range(N1*C1):
        blur_img[k//3, k % 3] = F.conv2d(img_pad[k//3, k % 3].unsqueeze(
            0).unsqueeze(0), psf[k//3, k % 3].unsqueeze(0).unsqueeze(0), padding='valid').squeeze()

    # add Gaussian noise
    if noise_level > 0:
        blur_img = blur_img + \
            torch.tensor(np.random.normal(0, noise_level,
                                          blur_img.shape), dtype=torch.float32)
    return blur_img

# ===============
# circular padding
# ===============


def pad4conv(tensor, psf_sz, mode='circular', value=0.0):
    '''
    padding image for convolution

    tensor (torch tensor):  4D image tensor ([N,C,H,W] ) to be padded
    psf_sz (list[int]): 2D psf size ([H,W) in convolution
    mode (str): padding mode, 'circular' (default) | 'constant' | 'reflect' | 'replicate'
    value (float): padding value for 'constant' padding mode
    refer to F.pad for specific parameter setting
    '''
    x_pad_len, y_pad_len = psf_sz[0]-1, psf_sz[1]-1
    pad_width = (y_pad_len//2, y_pad_len-y_pad_len//2,
                 x_pad_len//2, x_pad_len-x_pad_len//2)
    tensor = F.pad(tensor, pad_width, mode=mode, value=value)
    return tensor


def pad_circular(x, pad):
    """
    2D image circular padding
    :param x: img, shape [H, W]
    :param pad: pad size,int >= 0
    :return:
    """
    x = torch.cat([x, x[0:pad]], dim=0)
    x = torch.cat([x, x[:, 0:pad]], dim=1)
    x = torch.cat([x[-2 * pad:-pad], x], dim=0)
    x = torch.cat([x[:, -2 * pad:-pad], x], dim=1)

    return x


def pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :param dim: the dimension over which the tensors are padded
    :return:
    """

    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        if d >= len(x.shape):
            raise IndexError(f"dim {d} out of range")

        idx = tuple(slice(0, None if s != d else pad, 1)
                    for s in range(len(x.shape)))
        x = torch.cat([x, x[idx]], dim=d)

        idx = tuple(slice(None if s != d else -2 * pad, None if s !=
                    d else -pad, 1) for s in range(len(x.shape)))
        x = torch.cat([x[idx], x], dim=d)
        pass

    return x

# ===============
# edge taper
# ===============
# Implementation from https://github.com/teboli/polyblur


def edgetaper(img, kernel, n_tapers=3):
    if type(img) == np.ndarray:
        return edgetaper_np(img, kernel, n_tapers)
    else:
        return edgetaper_torch(img, kernel, n_tapers)


def pad_for_kernel_np(img, kernel, mode):
    p = [(d-1)//2 for d in kernel.shape]
    padding = [p, p] + (img.ndim-2)*[(0, 0)]
    return np.pad(img, padding, mode)


def crop_for_kernel_np(img, kernel):
    p = [(d-1)//2 for d in kernel.shape]
    r = [slice(p[0], -p[0]), slice(p[1], -p[1])] + (img.ndim-2) * [slice(None)]
    return img[r]


def edgetaper_alpha_np(kernel, img_shape):
    v = []
    for i in range(2):
        z = np.fft.fft(np.sum(kernel, 1-i), img_shape[i]-1)
        z = np.real(np.fft.ifft(np.square(np.abs(z)))).astype(np.float32)
        z = np.concatenate([z, z[0:1]], 0)
        v.append(1 - z / np.max(z))
    return np.outer(*v)


def edgetaper_np(img, kernel, n_tapers=3):
    alpha = edgetaper_alpha_np(kernel, img.shape[0:2])
    _kernel = kernel
    if 3 == img.ndim:
        kernel = kernel[..., np.newaxis]
        alpha = alpha[..., np.newaxis]
    for i in range(n_tapers):
        blurred = fftconvolve(pad_for_kernel_np(
            img, _kernel, 'wrap'), kernel, mode='valid')
        img = alpha*img + (1-alpha)*blurred
    return img


def edgetaper_alpha_torch(kernel, img_shape):
    z = torch.fft.fft(torch.sum(kernel, -1), img_shape[0]-1)
    z = torch.real(torch.fft.ifft(torch.abs(z)**2)).float()
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v1 = 1 - z / torch.max(z)

    z = torch.fft.fft(torch.sum(kernel, -2), img_shape[1] - 1)
    z = torch.real(torch.fft.ifft(torch.abs(z) ** 2)).float()
    z = torch.cat([z, z[..., 0:1]], dim=-1)
    v2 = 1 - z / torch.max(z)

    return v1.unsqueeze(-1) * v2.unsqueeze(-2)


def edgetaper_torch(img, kernel, n_tapers=3):
    h, w = img.shape[-2:]
    alpha = edgetaper_alpha_torch(kernel, (h, w))
    _kernel = kernel
    ks = _kernel.shape[-1] // 2
    for i in range(n_tapers):
        img_padded = F.pad(img, [ks, ks, ks, ks], mode='circular')
        K = p2o(kernel, img_padded.shape[-2:])
        I = torch.fft.fft2(img_padded)
        blurred = torch.real(torch.fft.ifft2(K * I))[..., ks:-ks, ks:-ks]
        img = alpha * img + (1 - alpha) * blurred
    return img


if __name__ == '__main__':
    import torch
    x = torch.zeros(4, 3, 7, 7)
    p = torch.zeros(4, 3, 2, 2)

    xp = img_blur_torch(x, p)

    print(xp.shape)

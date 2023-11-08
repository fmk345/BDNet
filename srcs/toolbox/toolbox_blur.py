# -*- coding: utf-8 -*-
from functools import reduce
import numpy as np
import scipy
from scipy import ndimage
import os
import sys
from math import cos, sin
from numpy import zeros, ones,  array, pi,  min, max, maximum, sum, mgrid, exp, pad, round, ceil, floor
from numpy.random import randn, rand, randint, uniform
from scipy.signal import convolve2d
import cv2
from os.path import join as opj
import scipy.io as sio
import random

'''
some codes are copied/modified from 
    Kai Zhang (github: https://github.com/cszn)
    https://github.com/twhui/SRGAN-pyTorch
    https://github.com/xinntao/BasicSR

Last modified: 2021-10-28 Zhihong Zhang 
'''

# ===============
# blur kernels generation
# ===============

# ----- fspecial -----


def fspecial_gauss(size, sigma):
    x, y = mgrid[-size // 2 + 1: size // 2 + 1, -size // 2 + 1: size // 2 + 1]
    g = exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1),
                         np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h

# ----- (coded) Linear Motion Blur -----


def linearMotionBlurKernel(motion_len=[15, 35], theta=[0, 2*pi], psf_sz=50):
    '''
    linear motion blur kernel
    kernel_len: kernel length range (pixel)
    theta: motion direction range (rad)
    psf_sz: psf size (pixel)
    '''
    if not isinstance(psf_sz, list):
        psf_sz = [psf_sz, psf_sz]
    if not isinstance(motion_len, list):
        motion_len = [motion_len, motion_len]
    if not isinstance(theta, list):
        theta = [theta, theta]

    motion_len = uniform(*motion_len)
    theta = uniform(*theta)

    motion_len_n = ceil(motion_len*2).astype(int)  # num of points

    # get random trajectory
    try_times = 0
    while(True):
        x = zeros((2, motion_len_n))
        Lx = motion_len*cos(theta)
        Ly = motion_len*sin(theta)
        x[0] = round(np.linspace(0, Lx, motion_len_n))
        x[1] = round(np.linspace(0, Ly, motion_len_n))
        x = x.astype(int)
        # traj threshold judge
        x[0], x[1] = x[0]-min(x[0]), x[1]-min(x[1])  # move to first quadrant
        x_thr = [max(x[0])+1, max(x[1]+1)]
        if ((np.array(psf_sz) - np.array(x_thr)) > 0).all():
            break  # proper trajectory with length < psf_size

        try_times = try_times+1
        assert try_times < 10, 'Error: MOTION_LEN > PSF_SZ'

    # get kernel
    k = zeros(x_thr)
    for x_i in x.T:
        k[x_i[0], x_i[1]] = 1

    # padding
    pad_width = (psf_sz[0] - x_thr[0], psf_sz[1] - x_thr[1])
    pad_width = ((pad_width[0]//2, pad_width[0]-pad_width[0]//2),
                 (pad_width[1]//2, pad_width[1]-pad_width[1]//2))

    assert (np.array(pad_width) > 0).all(), 'Error: MOTION_LEN > PSF_SZ'
    k = pad(k, pad_width, 'constant')

    # guassian blur
    k = np.rot90(k, -1)
    k = k/sum(k)
    k = convolve2d(k, fspecial_gauss(2, 1), "same")  # gaussian blur
    k = k/sum(k)

    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    return k


def codedLinearMotionBlurKernel(motion_len=[15, 35], theta=[0, 2*pi], psf_sz=50, code=None):
    '''
    linear motion blur kernel
    kernel_len: kernel length range (pixel)
    theta: motion direction range (rad)
    psf_sz: psf size (pixel)
    code: flutter shutter code
    '''
    if isinstance(psf_sz, int):
        psf_sz = [psf_sz, psf_sz]
    if isinstance(motion_len, (int, float)):
        motion_len = [motion_len, motion_len]
    if isinstance(theta, (int, float)):
        theta = [theta, theta]

    motion_len = uniform(*motion_len)
    theta = uniform(*theta)

    # get coded trajectory
    # code matching to motion length
    if code is None:
        motion_len_n = ceil(motion_len*3).astype(int)  # num of points
        code_n = ones((1, motion_len_n))
    else:
        motion_len_n = ceil(maximum(motion_len, len(code))*3).astype(int)
        code_n = [code[floor(k*len(code)/motion_len_n).astype(int)]
                  for k in range(motion_len_n)]
    # print(code, '\n', code_n)

    # get random trajectory
    try_times = 0
    while(True):
        x = zeros((2, motion_len_n))
        Lx = motion_len*cos(theta)
        Ly = motion_len*sin(theta)
        x[0] = round(np.linspace(0, Lx, motion_len_n))
        x[1] = round(np.linspace(0, Ly, motion_len_n))
        x = x*code_n  # coded traj
        x = x.astype(int)
        # traj threshold judge
        x[0], x[1] = x[0]-min(x[0]), x[1]-min(x[1])  # move to first quadrant
        x_thr = [max(x[0])+1, max(x[1]+1)]
        if ((np.array(psf_sz) - np.array(x_thr)) > 0).all():
            break  # proper trajectory with length < psf_size

        try_times = try_times+1
        assert try_times < 10, 'Error: MOTION_LEN > PSF_SZ'

    # get kernel
    k = zeros(x_thr)
    for x_i in x.T:
        k[x_i[0], x_i[1]] = 1

    # padding
    pad_width = (psf_sz[0] - x_thr[0], psf_sz[1] - x_thr[1])
    pad_width = ((pad_width[0]//2, pad_width[0]-pad_width[0]//2),
                 (pad_width[1]//2, pad_width[1]-pad_width[1]//2))

    k = pad(k, pad_width, 'constant')

    # guassian blur
    k = np.rot90(k, -1)
    k = k/sum(k)
    k = convolve2d(k, fspecial_gauss(2, 1), "same")  # gaussian blur
    k = k/sum(k)

    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    return k


def linearTrajectory(T_value):
    '''
    get a linear trajectory coordinate sequence with length of T pixels (or length belong to [T[0],T[1]])
    '''
    if isinstance(T_value, (int, float)):
        T_value = T_value
    else:
        T_value = randint(T_value[0], T_value[1])

    # original point = [0,0], direction = theta
    theta = rand()*2*pi
    Tx = T_value*cos(theta)
    Ty = T_value*sin(theta)

    x = zeros((2, T_value))
    x[0] = np.linspace(0, Tx, T_value)
    x[1] = np.linspace(0, Ty, T_value)

    return x


# ----- (coded) Random Motion Blur -----

def codedRandomMotionBlurKernelPair(motion_len_r=[15, 35],  psf_sz=50, code=None):
    '''
    a pair of coded and non-coded random motion blur kernel
    kernel_len: kernel length range (pixel)
    psf_sz: psf size (pixel)
    code: flutter shutter code
    '''
    if isinstance(psf_sz, int):
        psf_sz = [psf_sz, psf_sz]
    if isinstance(motion_len_r, (int, float)):
        motion_len_r = [motion_len_r, motion_len_r]

    motion_len_v = uniform(*motion_len_r)

    # get coded trajectory
    # code matching to motion length
    if code is None:
        # num of sampling points
        motion_len_n = ceil(motion_len_v*3).astype(int)
        code_n = ones((1, motion_len_n))
    else:
        code = np.array(code, dtype=np.float32)
        motion_len_n = ceil(maximum(motion_len_v, len(code))*3).astype(int)
        code_n = [code[floor(k*len(code)/motion_len_n).astype(int)]
                  for k in range(motion_len_n)]
    # print(code, '\n', code_n)

    # get random trajectory
    x = getRandomTrajectory(motion_len_r, motion_len_n,
                            psf_sz, curvature_param=1)
    k = traj2kernel(x, psf_sz, traj_v=code_n)
    gaussian_kernel_width = random.choice([2, 3])
    k = convolve2d(k, fspecial_gauss(gaussian_kernel_width, 1),
                   "same")  # gaussian blur
    k = k/sum(k)

    k_orig = traj2kernel(x, psf_sz, traj_v=1)
    k_orig = convolve2d(k_orig, fspecial_gauss(
        gaussian_kernel_width, 1), "same")  # gaussian blur
    k_orig = k_orig/sum(k_orig)
    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    return k, k_orig


def getRandomTrajectory(motion_len_r, motion_len_n, motion_thr, curvature_param=1, max_try_times=100):
    '''
    generate random traj (MOTION_LEN_N points) with length belong to MOTION_LEN and range within MOTION_THR
    motion_len_r: kernel length range (pixel)
    motion_len_n: num of traj points (will affect the traj shape)
    motion_thr:   threshold of the trajectory's bounding box's size
    curvature_param: curvature control parameter (will affect the traj shape)
    max_try_times: maximum retry times
    '''
    if isinstance(motion_thr, int):
        motion_thr = [motion_thr, motion_thr]
    if isinstance(motion_len_r, (int, float)):
        motion_len_r = [motion_len_r, motion_len_r]

    try_times = 0
    while(True):
        x = zeros((3, motion_len_n))
        v = zeros((3, motion_len_n))
        r = zeros((3, motion_len_n))
        trans_delta = 1
        rot_delta = 2 * pi / motion_len_n

        for t in range(1, motion_len_n):
            trans_n = randn(3)/(t+1)
            # rot_n = r[:, t - 1] + randn(3)/t # original code
            rot_n = r[:, t - 1] + randn(3)/t*curvature_param
            # Keep the inertia of volecity
            v[:, t] = v[:, t - 1] + trans_delta * trans_n
            # Keep the inertia of direction
            r[:, t] = r[:, t - 1] + rot_delta * rot_n

            st = rot3D(v[:, t], r[:, t])
            x[:, t] = x[:, t - 1] + st

        # calc trajectory  length and rescale
        x[0], x[1] = x[0]-min(x[0]), x[1]-min(x[1])  # move to first quadrant
        x_len = np.sum(np.array([np.sqrt(np.sum((x[:, k+1]-x[:, k])**2))
                                 for k in range(motion_len_n-1)]))

        if not (motion_len_r[0] < x_len < motion_len_r[1]):
            # rescale traj to desired length
            x = x*np.random.uniform(*motion_len_r)/x_len
            x_len = np.sum(np.array([np.sqrt(np.sum((x[:, k+1]-x[:, k])**2))
                                     for k in range(motion_len_n-1)]))

        # calc trajectory threshold and judge
        x_thr = [max(x[0])+1, max(x[1]+1)]
        if motion_thr[0] > x_thr[0] and motion_thr[1] > x_thr[1] and motion_len_r[0] < x_len < motion_len_r[1]:
            break  # proper trajectory with length < psf_size and length in MOTION_LEN range
        try_times = try_times+1
        assert try_times < max_try_times, 'Error: MOTION_LEN and PSF_SZ is not proper'
    return x


def traj2kernel(x, psf_sz, traj_v=1):
    '''
    convert trajectory to blur kernel
    x: traj coord, 2*N
    psf_sz: psf size
    traj_v: value of trajectory points, scalar | 1*N list
    '''

    if isinstance(psf_sz, int):
        psf_sz = [psf_sz, psf_sz]
    if isinstance(traj_v, (int, float)):
        traj_v = [traj_v]*x.shape[1]
    traj_v = np.array(traj_v).squeeze()

    x = x.astype(int)
    x[0], x[1] = x[0]-min(x[0]), x[1]-min(x[1])  # move to first quadrant
    x_thr = [max(x[0])+1, max(x[1]+1)]
    psf = zeros(x_thr)
    for n, x_i in enumerate(x.T):
        psf[x_i[0], x_i[1]] += traj_v[n]

    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    # padding
    pad_width = (psf_sz[0] - x_thr[0], psf_sz[1] - x_thr[1])
    pad_width = ((pad_width[0]//2, pad_width[0]-pad_width[0]//2),
                 (pad_width[1]//2, pad_width[1]-pad_width[1]//2))

    # assert (np.array(pad_width) > 0).all(), 'Error: MOTION_LEN > PSF_SZ'
    psf = pad(psf, pad_width, 'constant')

    # normalize
    # k = np.rot90(k, -1)
    psf = psf/sum(np.array(traj_v))

    return psf


def rot3D(x, r):
    Rx = array([[1, 0, 0], [0, cos(r[0]), -sin(r[0])],
               [0, sin(r[0]), cos(r[0])]])
    Ry = array([[cos(r[1]), 0, sin(r[1])], [
               0, 1, 0], [-sin(r[1]), 0, cos(r[1])]])
    Rz = array([[cos(r[2]), -sin(r[2]), 0],
               [sin(r[2]), cos(r[2]), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    x = R @ x
    return x

#  --------- util ----------


def psf_blur_img(img, psf, noise_level=0):
    """
    coded exposure psf blurred image

    Args:
        img (ndarray): sharp image
        psf (ndarray): coded exposure psf
        noise_level (scalar): noise level

    Returns:
        x: [description]
    """
    coded_blur_img = ndimage.filters.convolve(
        img, np.expand_dims(psf, axis=2), mode='wrap')
    # add Gaussian noise
    coded_blur_img = coded_blur_img + \
        np.random.normal(0, noise_level, coded_blur_img.shape)
    return coded_blur_img.astype(np.float32)


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    import matplotlib.pyplot as plt
    from utils import utils_deblur_kair
    from utils.utils_image_zzh import augment_img

# %% Funciton list
    FLAG_traj_psf = False
    FLAG_psf_blur_image = True



# %% use random motion trajectory (or load existing traj) to generate corresponding box&ce psf
if FLAG_traj_psf:
    # params
    motion_len_r = [60, 100]
    psf_sz = 64
    # ce_code = [1,0,1,0,0,1]
    ce_code = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0,
               0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1]  # [raskar2006CodedExposure]
    iter = 200
    load_traj = False
    traj_dir = './dataset/benchmark/pair_traj_psf1/traj/'
    save_dir = './outputs/tmp/'

    # generate dir
    os.makedirs(save_dir+'box_psf', exist_ok=True)
    os.makedirs(save_dir+'ce_psf', exist_ok=True)
    os.makedirs(save_dir+'traj', exist_ok=True)

    # pre calc
    ce_code = np.array(ce_code, dtype=np.float32)
    motion_len_v = uniform(*motion_len_r)
    # motion_len_n = ceil(maximum(motion_len_v, len(ce_code))*3).astype(int)
    motion_len_n = 300 # assign the trajectory pixel length
    code_n = [ce_code[floor(k*len(ce_code)/motion_len_n).astype(int)]
              for k in range(motion_len_n)]
    # run
    for m in range(iter):
        if load_traj:
            traj = sio.loadmat(traj_dir+'traj%02d.mat' % (k+1))
            traj = traj['traj']
        else:
            traj = getRandomTrajectory(motion_len_r, motion_len_n,
                                       psf_sz, curvature_param=1).astype(np.int32)

        gaussian_kernel_width = random.choice([2, 3])
        
        k = traj2kernel(traj, psf_sz, traj_v=code_n)
        k = convolve2d(k, fspecial_gauss(gaussian_kernel_width, 1),
                       "same")  # gaussian blur
        k = k/sum(k)

        k_orig = traj2kernel(traj, psf_sz, traj_v=1)
        k_orig = convolve2d(k_orig, fspecial_gauss(
            gaussian_kernel_width, 1), "same")  # gaussian blur
        k_orig = k_orig/sum(k_orig)

        ce_psf_png = k/np.max(k)*255
        box_psf_png = k_orig/np.max(k)*255

        print('PSF Num.', m+1)
        sio.savemat(opj(save_dir, 'traj/traj%03d.mat' % (m+1)), {'traj': traj})
        cv2.imwrite(opj(save_dir, 'ce_psf/ce_psf%03d.png'
                    % (m+1)), ce_psf_png, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(opj(save_dir, 'box_psf/box_psf%03d.png'
                    % (m+1)), box_psf_png, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# %% blur image using existing psf
if FLAG_psf_blur_image:
    # param & path
    noise_level = 0
    # corresponding idx, -1 means all files
    img_idxs = -1  # list(range(100))
    # corresponding idx, -1 means psf_idxs=img_idxs
    psf_idxs = -1  # list(range(100))

    psf_dir = './dataset/benchmark/pair_traj_psf/box_psf/'  # ce-raskar_psf | box_psf
    img_dir = './dataset/benchmark/CBSD68/orig/'
    save_dir = './outputs/tmp/blur_pair_traj_psf_box/'  # ce-raskar | box

    # get path & gen dir
    img_names = sorted(os.listdir(img_dir))
    psf_names = sorted(os.listdir(psf_dir))
    img_num = len(img_names)
    psf_num = len(psf_names)

    if img_idxs == -1:
        img_idxs = list(range(img_num))
    if psf_idxs == -1:
        psf_idxs = img_idxs

    assert img_num >= len(img_idxs) and psf_num >= len(
        psf_idxs), f'Error: Given idx num: img {len(img_idxs)}, psf {len(psf_idxs)} > Total file num: img {img_num}, psf {psf_num}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # blur image
    cnt = 0
    for psf, t in zip(psf_idxs, img_idxs):
        print('--> PSF-%d, Image-%d' % (psf+1, t+1))
        # psf
        psf_k = cv2.imread(opj(psf_dir, psf_names[psf]))
        assert psf_k is not None, 'psf_%d is None' % psf
        psf_k = cv2.cvtColor(psf_k, cv2.COLOR_BGR2GRAY)
        psf_k = psf_k.astype(np.float32)/np.sum(psf_k)

        # image
        img_t = cv2.imread(opj(img_dir, img_names[t]))
        assert img_t is not None, 'img_%d is None' % t
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        img_t = img_t.astype(np.float32)/255

        # blur & noise
        blur_tmp = psf_blur_img(img_t, psf_k)
        if noise_level == 0:
            noisy_blur_tmp = blur_tmp
        else:
            noisy_blur_tmp = blur_tmp + \
                np.random.normal(0, noise_level, blur_tmp.shape)

        noisy_blur_tmp = noisy_blur_tmp[..., ::-1]*255

        cnt += 1
        cv2.imwrite(opj(save_dir, 'blur_img%03d.png' % cnt),
                    noisy_blur_tmp, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# %%

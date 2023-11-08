# -*- coding: utf-8 -*-
import numpy as np
import scipy
from math import cos, sin
from numpy import zeros, ones, array, pi, log, maximum, arange, sum, mgrid, exp, pad, round, ceil, floor
from numpy.random import randn, rand, randint, uniform
from scipy.signal import convolve2d
import torch
import random
import os
import cv2
from os.path import join as opj


'''
some codes are copied/modified from 
    https://github.com/cszn
    https://github.com/twhui/SRGAN-pyTorch
    https://github.com/xinntao/BasicSR
    https://gitlab.mpi-klsb.mpg.de/jdong/dwdn

More kernel generation methods:
    - https://github.com/donggong1/learn-optimizer-rgdn/blob/master/data/make_kernel.py

Last modified: 2021-10-28 Zhihong Zhang 
'''

# ===============
# blur kernels generation
# ===============
#%% ------ basic function


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


# %% ----- fspecial -----


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


def fspecial_gaussian_zzh(hsize, sigma):
    # zzh: extend to h!=w
    hsize = hsize if isinstance(hsize, list) else [hsize, hsize]
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

# %% ----- (coded) Linear Motion Blur -----


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
    k = convolve2d(k, fspecial_gauss(3, 1), "same")  # gaussian blur
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
    k = convolve2d(k, fspecial_gauss(3, 1), "same")  # gaussian blur
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


# %% ----- Random Motion Blur -----
# ------------------------------------------- Boracchi's method >>> ------------------------------------------
def create_random_psf(psf_size=64, trajSize=64, anxiety=0.005, num_samples=2000, max_total_length=64, exp_time=[1]):
    """
    PSFs are obtained by sampling the continuous trajectory TrajCurve on a regular pixel grid using linear interpolation at subpixel level

    Args:
        % trajectory_size: size (in pixels) of the square support of the %tory curve
        % anxiety: amount of shake, which scales random vector added at each sample
        % num_samples: number of samples where the Trajectory is sampled
        % max_total_length: maximum trajectory length computed as sum of all distanced between consecutive points
        % psf_size   Size of the PFS where the TrajCurve is sampled
        % exp_time         Vector of exposure times: for each of them a PSF will be generated, default = [1]

    Returns:
        blur kernel as NumPy array of shape [PSFsize, PSFsize]

    Modified: Zhihong Zhang
    Reference: [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from Motion Blur"
    """

    x, _, _ = create_random_trajectory(
        trajSize, anxiety, num_samples, max_total_length)
    psf_size = (psf_size, psf_size)

    if isinstance(exp_time, (int, float)):
        exp_time = [exp_time]

    # PSFnumber = len(exp_time)
    numt = len(x)

    # center with respect to baricenter
    x = x - np.mean(x) + (psf_size[1] + 1j * psf_size[0] + 1 + 1j) / 2

    #    x = np.max(1, np.min(PSFsize[1], np.real(x))) + 1j*np.max(1, np.min(PSFsize[0], np.imag(x)))

    # generate PSFs
    PSFs = []
    PSF = np.zeros(psf_size)

    def triangle_fun(d):
        return max(0, (1 - np.abs(d)))

    def triangle_fun_prod(d1, d2):
        return triangle_fun(d1) * triangle_fun(d2)

    # set the exposure time
    for jj in range(len(exp_time)):
        try_times = 0
        while(try_times < 200):
            try_times = try_times+1
            if jj == 0:
                prevT = 0
            else:
                prevT = exp_time[jj - 1]

            # sample the trajectory until time exp_time
            for t in range(len(x)):
                if (exp_time[jj] * numt >= t) and (prevT * numt < t - 1):
                    t_proportion = 1
                elif (exp_time[jj] * numt >= t - 1) and (prevT * numt < t - 1):
                    t_proportion = exp_time[jj] * numt - t + 1
                elif (exp_time[jj] * numt >= t) and (prevT * numt < t):
                    t_proportion = t - prevT * numt
                elif (exp_time[jj] * numt >= t - 1) and (prevT * numt < t):
                    t_proportion = (exp_time[jj] - prevT) * numt
                else:
                    t_proportion = 0

                m2 = min(psf_size[1] - 2, max(1, int(np.floor(np.real(x[t])))))
                M2 = m2 + 1
                m1 = min(psf_size[0] - 2, max(1, int(np.floor(np.imag(x[t])))))
                M1 = m1 + 1

                # linear interp. (separable)
                PSF[m1, m2] = PSF[m1, m2] + t_proportion * \
                    triangle_fun_prod(np.real(x[t]) - m2, np.imag(x[t]) - m1)
                PSF[m1, M2] = PSF[m1, M2] + t_proportion * \
                    triangle_fun_prod(np.real(x[t]) - M2, np.imag(x[t]) - m1)
                PSF[M1, m2] = PSF[M1, m2] + t_proportion * \
                    triangle_fun_prod(np.real(x[t]) - m2, np.imag(x[t]) - M1)
                PSF[M1, M2] = PSF[M1, M2] + t_proportion * \
                    triangle_fun_prod(np.real(x[t]) - M2, np.imag(x[t]) - M1)
            if(not np.any(np.isnan(PSF))):
                break

        PSFs.append(PSF / PSF.sum())
        if len(PSFs) == 1:
            PSFs = PSFs[0]
    return PSFs


def create_random_trajectory(trajectory_size=64, anxiety=0.005, num_samples=2000, max_total_length=64):
    """
    Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012].
    Each trajectory consists of a complex-valued vector determining the discrete positions of a particle following a
    2-D random motion in continuous domain. The particle has an initial velocity vector which, at each iteration, is
    affected by a Gaussian perturbation and by a deterministic inertial component, directed toward the previous
    particle position. In addition, with a small probability, an impulsive (abrupt) perturbation aiming at inverting
    the particle velocity may arises, mimicking a sudden movement that occurs when the user presses the camera
    button or tries to compensate the camera shake. At each step, the velocity is normalized to guarantee that
    trajectories corresponding to equal exposures have the same length. Each perturbation (Gaussian, inertial, and
    impulsive) is ruled by its own parameter. Rectilinear Blur as in [Boracchi and Foi 2011] can be obtained by
    setting anxiety to 0 (when no impulsive changes occurs)

    :param trajectory_size: size (in pixels) of the square support of the Trajectory curve
    :param anxiety: amount of shake, which scales random vector added at each sample
    :param num_samples: number of samples where the Trajectory is sampled
    :param max_total_length: maximum trajectory length computed as sum of all distanced between consecutive points

    Modified: Zhihong Zhang
    Reference: [Boracchi and Foi 2012] Giacomo Boracchi and Alessandro Foi, "Modeling the Performance of Image Restoration from Motion Blur"
    """

    abruptShakesCounter = 0
    totalLength = 0
    # term determining, at each sample, the strength of the component leading towards the previous position
    centripetal = 0.7 * np.random.rand()

    # term determining, at each sample, the random component of the new direction
    gaussianTerm = 10 * np.random.rand()

    # probability of having a big shake, e.g. due to pressing camera button or abrupt hand movements
    freqBigShakes = 0.2 * np.random.rand()

    # v is the initial velocity vector, initialized at random direction
    init_angle = 2*np.pi * np.random.rand()

    # initial velocity vector having norm 1
    v0 = np.cos(init_angle) + 1j * np.sin(init_angle)

    # the speed of the initial velocity vector
    v = v0 * max_total_length/(num_samples-1)

    if anxiety > 0:
        v = v0 * anxiety
    # initialize the trajectory vector
    x = np.zeros(num_samples, dtype=np.complex)

    for t in range(num_samples-1):
        # determine if there is an abrupt (impulsive) shake
        if np.random.rand() < freqBigShakes * anxiety:
            # if yes, determine the next direction which is likely to be opposite to the previous one
            nextDirection = 2 * v * \
                (np.exp(1j * (np.pi + (np.random.rand() - 0.5))))
            abruptShakesCounter = abruptShakesCounter + 1
        else:
            nextDirection = 0

        # determine the random component motion vector at the next step
        dv = nextDirection + anxiety * (gaussianTerm * (np.random.randn(
        ) + 1j * np.random.randn()) - centripetal * x[t]) * (max_total_length / (num_samples - 1))
        v = v + dv

        # velocity vector normalization
        v = (v / np.abs(v)) * max_total_length / (num_samples - 1)
        # update particle position
        x[t + 1] = x[t] + v

        # compute total length
        totalLength = totalLength + np.abs(x[t + 1] - x[t])

    # Center the Trajectory

    # Set the lowest position in zero
    x = x - 1j * np.min(np.imag(x)) - np.min(np.real(x))

    # Center the Trajectory
    x = x - 1j * \
        np.remainder(np.imag(x[0]), 1) - \
        np.remainder(np.real(x[0]), 1) + 1 + 1j
    x = x + 1j * np.ceil((trajectory_size - np.max(np.imag(x))) / 2) + \
        np.ceil((trajectory_size - np.max(np.real(x))) / 2)
    return x, totalLength, abruptShakesCounter

# ------------------------------------------ Boracchi's method <<< -------------------------------------------


# ------------------------------------------- Schmidt's method >>> ------------------------------------------
# Schmidt's method (similar to KAIR's implementation - KaiZhang, but add control params and no gaussian kernek and resize op)
def randomBlurKernelSynthesis(motion_len_r=[1, 100], motion_len_n=250, psf_sz=37, curvature_param=1):
    '''
    generate random motion blur kernel

    motion_len_r: kernel length range (pixel)
    motion_len_n: num of traj points (will affect the traj shape)
    psf_sz: psf size (pixel)
    curvature_param: curvature control parameter for traj generation
    reference: Uwe Schmidt et. al. “Cascades of Regression Tree Fields for Image Restoration”
    '''
    if isinstance(psf_sz, int):
        psf_sz = [psf_sz, psf_sz]
    if isinstance(motion_len_r, (int, float)):
        motion_len_r = [motion_len_r, motion_len_r]

    # get random trajectory
    x = randomTrajectory(motion_len_r, motion_len_n,
                         psf_sz,  curvature_param=curvature_param)

    # traj 2 kernel
    k = kernelFromTrajectory(x, psf_sz)
    gaussian_kernel_width = 2
    k = convolve2d(k, fspecial_gauss(gaussian_kernel_width, 1),
                   "same")  # gaussian blur
    k = k/sum(k)

    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    return k


def randomTrajectory(motion_len_r, motion_len_n, motion_thr, curvature_param=1, max_try_times=100):
    '''
    generate random traj (MOTION_LEN_N points) with length belong to MOTION_LEN_R and range within MOTION_THR
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


def kernelFromTrajectory(x, psf_sz, traj_v=1):
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


# ------------------------------------------- Schmidt's method <<< ------------------------------------------

# ------------------------------------------- Kair's method >>> ------------------------------------------

def blurkernel_synthesis_kair(h=37, w=None):
    # function: randomly generate different type of kernels (motion blur, gaussian, )
    # https://github.com/tkkcc/prior/blob/879a0b6c117c810776d8cc6b63720bf29f7d0cc4/util/gen_kernel.py
    # Modified by Zhihong Zhang
    # zzh fixup small bugs
    w = h if w is None else w
    x = randomTrajectory_kair(250)
    k = None
    while k is None:
        k = kernelFromTrajectory_kair(x)
    kdims = k.shape

    # judge [h,w] v.s. kdims
    if h < kdims[0]:
        k = k[0:h, :]
        kdims[0] = h

    if w < k.shape[1]:
        k = k[:, 0:w]
        kdims[1] = w

    # center pad to kdims
    pad_width = ((h - kdims[0]) // 2, (w - kdims[1]) // 2)

    # zzh: be cautious about aliquant case
    pad_width = [(pad_width[0], h - kdims[0]-pad_width[0]),
                 (pad_width[1], w - kdims[1]-pad_width[1])]

    k = pad(k, pad_width, "constant")
    h, w = k.shape
    if np.random.randint(0, 4) == 1:
        k = cv2.resize(k, (random.randint(h, 5*h),
                       random.randint(w, 5*w)), interpolation=cv2.INTER_LINEAR)
        m, n = k.shape
        k = k[(m-h)//2: (m-h)//2+h, (n-w)//2: (n-w)//2+w]

    #zzh: gaussian kernel (why?)
    if sum(k) < 0.1:
        k = fspecial_gaussian_zzh([h, w], 0.1+6*np.random.rand(1))
    k = k / sum(k)
    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()
    return k


def kernelFromTrajectory_kair(x):
    h = 5 - log(rand()) / 0.15
    h = round(min([h, 27])).astype(int)
    h = h + 1 - h % 2
    w = h
    k = zeros((h, w))

    xmin = min(x[0])
    xmax = max(x[0])
    ymin = min(x[1])
    ymax = max(x[1])
    xthr = arange(xmin, xmax, (xmax - xmin) / w)
    ythr = arange(ymin, ymax, (ymax - ymin) / h)

    for i in range(1, xthr.size):
        for j in range(1, ythr.size):
            idx = (
                (x[0, :] >= xthr[i - 1])
                & (x[0, :] < xthr[i])
                & (x[1, :] >= ythr[j - 1])
                & (x[1, :] < ythr[j])
            )
            k[i - 1, j - 1] = sum(idx)
    if sum(k) == 0:
        return
    k = k / sum(k)
    gaussian_kernel_sz = 2
    k = convolve2d(k, fspecial_gauss(gaussian_kernel_sz, 1), "same")
    k = k / sum(k)
    return k


def randomTrajectory_kair(T):
    '''
    get a random trajectory coordinate sequence with length T
    '''
    x = zeros((3, T))  # traj coord
    v = randn(3, T)
    r = zeros((3, T))
    trv = 1 / 1
    trr = 2 * pi / T
    for t in range(1, T):
        F_rot = randn(3) / (t + 1) + r[:, t - 1]
        F_trans = randn(3) / (t + 1)
        r[:, t] = r[:, t - 1] + trr * F_rot
        v[:, t] = v[:, t - 1] + trv * F_trans
        st = v[:, t]
        st = rot3D(st, r[:, t])
        x[:, t] = x[:, t - 1] + st
    return x

# ------------------------------------------- Kair's method <<< ------------------------------------------

#%% ----- (non) Coded Random Motion Blur -----
# Based on Schmidt's method


def codedRandomMotionBlurKernel(motion_len_r=[15, 35],  psf_sz=50, code=None, motion_len_n=None, gaussian_kernel=[2,3]):
    '''
    a pair of coded and non-coded random motion blur kernel
    kernel_len: kernel length range (pixel)
    psf_sz: psf size (pixel)
    code: flutter shutter code
    motion_len_n: pixel number of the trajectory, default=None, i.e. auto-determined
    gaussian_kernel: gaussian kernel width candidates
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
        if motion_len_n is None:
            motion_len_n = ceil(motion_len_v*3).astype(int)
        code_n = ones((1, motion_len_n))
    else:
        code = np.array(code, dtype=np.float32)
        if motion_len_n is None:
            motion_len_n = ceil(maximum(motion_len_v, len(code))*3).astype(int)
        code_n = [code[floor(k*len(code)/motion_len_n).astype(int)]
                  for k in range(motion_len_n)]
    # print(code, '\n', code_n)

    traj = getRandomTrajectory(motion_len_r, motion_len_n,
                            psf_sz, curvature_param=1)
    # traj 2 kernel
    traj = traj[0:2]
    k = traj2kernel(traj, psf_sz, traj_v=code_n)
    
    # random choose gaussian kernel
    gaussian_kernel_width = random.choice(gaussian_kernel)
    k = convolve2d(k, fspecial_gauss(gaussian_kernel_width, 1),
                   "same")  # gaussian blur
    k = k/sum(k)

    # import matplotlib.pyplot as plt
    # plt.imshow(k, interpolation="nearest", cmap="gray")
    # plt.show()

    return k, traj


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





if __name__ == '__main__':
    from utils_image_zzh import img_matrix
    save_dir = './outputs/tmp/test/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_kernel = 20
    ce_code = [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    psfs = []
    for k in range(n_kernel):
        # psf = randomBlurKernelSynthesis()
        # psf = randomBlurKernelSynthesis(
        #     motion_len_n=250,  curvature_param=1)
        psf = codedRandomMotionBlurKernel(
            motion_len_r=[60, 100],  psf_sz=64, code=ce_code)
        psf = psf/np.max(psf)*255
        psfs.append(psf)
        print("kernel_%02d" % (k+1))
        cv2.imwrite(opj(save_dir, 'psf%02d.png' % (k+1)),
                    psf, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    psf_matirx = img_matrix(psfs, 2, 10, 0)
    cv2.imwrite(opj(save_dir, '_all_psf.png'),
                psf_matirx, [cv2.IMWRITE_PNG_COMPRESSION, 0])

'''
pytorch version 2D patches extraction and combination (without overlapping) based on basic torch operations
'''

import torch


# ==================================================
# patch processing based on pytorch (without overlap)
# -----
# example

#     input_re, batch_list = window_partitionx(input_data, win_size)
#     restored = model_restoration(input_re)
#     restored = window_reversex(restored, win, Hx, Wx, batch_list)

# -----
# Copyright https://github.com/INVOKERer/DeepRFT
# ==================================================

def window_partitionx(x, window_size, keep_edge=True):
    """
    split a full image into multi-patch (could compensate for aliquant edge patch with keep_edge=True)

    Args:
        x: (B, C, H, W)
        window_size (int): window size
        keep_edge (bool): whether keep the aliquant edges, default=True

    Returns:
        windows: (num_windows*B, C, window_size, window_size), 
        batch_list: cumulative patch num list for aliquot & aliquant patches
    """
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if keep_edge:
        if h == H and w == W:
            return x_main, [b_main]
        if h != H and w != W:
            x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
            b_r = x_r.shape[0] + b_main
            x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
            b_d = x_d.shape[0] + b_r
            x_dd = x[:, :, -window_size:, -window_size:]
            b_dd = x_dd.shape[0] + b_d
            # batch_list = [b_main, b_r, b_d, b_dd]
            return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
        if h == H and w != W:
            x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
            b_r = x_r.shape[0] + b_main
            return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
        if h != H and w == W:
            x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
            b_d = x_d.shape[0] + b_main
            return torch.cat([x_main, x_d], dim=0), [b_main, b_d]
    else:
        return x_main, [b_main]


def window_reversex(windows, window_size, H, W, batch_list, keep_edge=True):
    """
    reverse multi-patch back to a full image (compensate for aliquant edge patch)

    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        batch_list (list[int]): cumulative patch num list for aliquot & aliquant patches
        keep_edge (bool): whether keep the aliquant edges, default=True

    Returns:
        x: (B, C, H, W)
    """
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    if keep_edge:
        B, C, _, _ = x_main.shape
        # print('windows: ', windows.shape)
        # print('batch_list: ', batch_list)
        res = torch.zeros([B, C, H, W], device=windows.device)
        res[:, :, :h, :w] = x_main
        if h == H and w == W:
            return res
        if h != H and w != W and len(batch_list) == 4:
            x_dd = window_reverses(
                windows[batch_list[2]:, ...], window_size, window_size, window_size)
            res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
            x_r = window_reverses(
                windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
            res[:, :, :h, w:] = x_r[:, :, :, w - W:]
            x_d = window_reverses(
                windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
            res[:, :, h:, :w] = x_d[:, :, h - H:, :]
            return res
        if w != W and len(batch_list) == 2:
            x_r = window_reverses(
                windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
            res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        if h != H and len(batch_list) == 2:
            x_d = window_reverses(
                windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
            res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    else:
        res = x_main
    return res


# -------------------------------------------------------------------------


def window_partitions(x, window_size):
    """
    split a full image into multi-patch (discard aliquant edge patch)

    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size,
               W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous(
    ).view(-1, C, window_size, window_size)
    return windows

def window_reverses(windows, window_size, H, W):
    """
    reverse multi-patch back to a full image (discard aliquant edge patch)

    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    C = windows.shape[1]
    x = windows.view(-1, H // window_size, W // window_size,
                     C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


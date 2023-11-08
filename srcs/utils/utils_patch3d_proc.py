'''
pytorch version 3D patch extraction and combination (w/wo overlapping) according to the conv layer's sliding window rule 
Copyright https://discuss.pytorch.org/t/creating-non-overlapping-patches-and-reconstructing-image-back-from-the-patches/88417/8
'''

import torch


def extract_patches_3ds(x, kernel_size, padding=0, stride=1):
    """
    The simlified version of `extract_patches_3d` based on `torch.Tensor.unfold` (without the dilation parameter).
    Note: the padding parameter is a little different with `extract_patches_2d`
    Args:
        x (tensor): torch tensor with shape of (B,C,D,H,W)
        kernel_size (int or tuple_3): patch size
        padding (int or tuple_3, optional): padding size for `x`. Defaults to 0.  (following F.pad's rule)
        stride (int or tuple_3, optional): stride of the patch slicing. Defaults to 1.
        dilation (int or tuple_3, optional): dilation of the samping kernel Defaults to 1.

    Returns:
        tensor of the patches with shape of (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    channels = x.shape[1]

    x = torch.nn.functional.pad(x, padding)
    # (B, C, D, H, W)
    x = x.unfold(2, kernel_size[0], stride[0]).unfold(
        3, kernel_size[1], stride[1]).unfold(4, kernel_size[2], stride[2])
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])
    x = x.contiguous().view(-1, channels,
                            kernel_size[0], kernel_size[1], kernel_size[2])
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
    return x


def combine_patches_3dx(x, kernel_size, output_shape, padding=0, stride=1, dilation=1, overlap_op="mean"):
    """
    the extended version of `combine_patches_3d` with the `overlap_op` parameter to choose mean/sum operation in overlapped areas
    """
    if overlap_op == "sum":
        return combine_patches_3d(x, kernel_size, output_shape, padding, stride, dilation)
    elif overlap_op == "mean":
        all_one = torch.ones(output_shape, dtype=x.dtype, device=x.device)
        # mask = extract_patches_3ds(all_one, kernel_size, padding, stride)
        mask = extract_patches_3d(
            all_one, kernel_size, padding, stride, dilation)
        mask_full = combine_patches_3d(mask, kernel_size, output_shape,
                                       padding, stride, dilation)
        mask_full = torch.where(
            mask_full == 0, torch.ones_like(mask_full), mask_full)
        x_full = combine_patches_3d(
            x, kernel_size, output_shape, padding, stride, dilation)
        return x_full/mask_full
    else:
        raise ValueError(
            f"`overlap_op` should be 'mean' | 'sum' but get {overlap_op}")


# --------------------------------------------------


def extract_patches_3d(x, kernel_size, padding=0, stride=1, dilation=1):
    """
    extract 3d patches for a batch of 3D tensor

    Args:
        x (tensor): torch tensor with shape of (B,C,D,H,W)
        kernel_size (int or tuple_3): patch size
        padding (int or tuple_3, optional): padding size for `x`. Defaults to 0.  (following conv2d's rule)
        stride (int or tuple_3, optional): stride of the patch slicing. Defaults to 1.
        dilation (int or tuple_3, optional): dilation of the samping kernel Defaults to 1.

    Returns:
        tensor of the patches with shape of (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation *
                   (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]

    d_dim_in = x.shape[2]
    h_dim_in = x.shape[3]
    w_dim_in = x.shape[4]
    d_dim_out = get_dim_blocks(
        d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_out = get_dim_blocks(
        h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_out = get_dim_blocks(
        w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    # (B, C, D, H, W)
    x = x.view(-1, channels, d_dim_in, h_dim_in * w_dim_in)
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[0], 1), padding=(
        padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
    # (B, C * kernel_size[0], d_dim_out * H * W)

    x = x.view(-1, channels * kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[1], kernel_size[2]), padding=(
        padding[1], padding[2]), stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))
    # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

    x = x.view(-1, channels, kernel_size[0], d_dim_out,
               kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)
    # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.contiguous().view(-1, channels,
                            kernel_size[0], kernel_size[1], kernel_size[2])
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    return x


def combine_patches_3d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    """
    combine 3D patches to restore the original large tensor (sum overlapped area)

    Args:
        x (tensor): tensor of the 3D patches with shape of (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])
        kernel_size (int or tuple_3): patch size
        output_shape (tuple): shape of original large 3D tensor (B,C,D,H,W)
        padding (int or tuple_3, optional): padding size for `x`. Defaults to 0.  (following conv2d's rule)
        stride (int or tuple_3, optional): stride of the patch slicing. Defaults to 1.
        dilation (int or tuple_3, optional): dilation. Defaults to 1.

    Note: The values in the overlapped areas will be summed up when combining.

    Returns:
        original large 3D tensor with shape of (B,C,D,H,W)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation *
                   (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    d_dim_in = get_dim_blocks(
        d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_in = get_dim_blocks(
        h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_in = get_dim_blocks(
        w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    x = x.view(-1, channels, d_dim_in, h_dim_in, w_dim_in,
               kernel_size[0], kernel_size[1], kernel_size[2])
    # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

    x = x.contiguous().view(-1, channels *
                            kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(kernel_size[1], kernel_size[2]), padding=(
        padding[1], padding[2]), stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out * w_dim_out), kernel_size=(
        kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
    # (B, C, D, H * W)

    x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, D, H, W)

    return x


# ----------------------------------------------------

if __name__ == '__main__':

    shape = (1, 2, 4, 4, 4)
    a = torch.arange(1, 129, dtype=torch.float).view(*shape)

    print(a.shape)
    # print(a)

    b = extract_patches_3ds(a, kernel_size=3, padding=1, stride=2)

    print(b.shape)

    # print(b)
    c = combine_patches_3dx(
        b, kernel_size=3, output_shape=shape, padding=1, stride=2, overlap_op='mean')  #

    print(c.shape)
    # print(c)

    # compare
    print(torch.all(a == c))

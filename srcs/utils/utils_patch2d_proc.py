'''
pytorch version 2D patch extraction and combination (w/wo overlapping) according to the conv layer's sliding window rule 
Copyright https://discuss.pytorch.org/t/creating-non-overlapping-patches-and-reconstructing-image-back-from-the-patches/88417/8
'''

import torch
from math import ceil


def extract_patches_2ds(x, kernel_size, padding=0, stride=1):
    """
    The simlified version of `extract_patches_2d` based on `torch.Tensor.unfold` (without the dilation parameter).
    Note: the padding parameter is a little different with `extract_patches_2d`
    Args:
        x (tensor): torch tensor with shape of (B,C,H,W)
        kernel_size (int or tuple_2): patch size
        padding (int or tuple_2, optional): padding size for `x`. Defaults to 0. (following F.pad's rule)
        stride (int or tuple_2, optional): stride of the patch slicing. Defaults to 1.

    Returns:
        tensor of the patches with shape of (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    channels = x.shape[1]

    x = torch.nn.functional.pad(x, padding)
    # (B, C, H, W)
    x = x.unfold(2, kernel_size[0], stride[0]).unfold(
        3, kernel_size[1], stride[1])
    # (B, C, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1])
    x = x.contiguous().view(-1, channels,
                            kernel_size[0], kernel_size[1])
    # (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
    return x


def combine_patches_2dx(x, kernel_size, output_shape, padding=0, stride=1, dilation=1, overlap_op="mean"):
    """
    the extended version of `combine_patches_2d` with the `overlap_op` parameter to choose mean/sum operation in overlapped areas
    """
    if overlap_op == "sum":
        return combine_patches_2d(x, kernel_size, output_shape, padding, stride, dilation)
    elif overlap_op == "mean":
        all_one = torch.ones(output_shape, dtype=x.dtype, device=x.device)
        mask = extract_patches_2d(
            all_one, kernel_size, padding, stride, dilation)
        x_full = combine_patches_2d(
            x, kernel_size, output_shape, padding, stride, dilation)
        mask_full = combine_patches_2d(
            mask, kernel_size, output_shape, padding, stride, dilation)
        mask_full = torch.where(
            mask_full == 0, torch.ones_like(mask_full), mask_full)
        return x_full/mask_full
    else:
        raise ValueError(
            f"`overlap_op` should be 'mean' | 'sum' but get {overlap_op}")


# --------------------------------------------------


def extract_patches_2d(x, kernel_size, padding=0, stride=1, dilation=1):
    """
    extract 2d patches for a batch of 2D tensor

    Args:
        x (tensor): torch tensor with shape of (B,C,H,W)
        kernel_size (int or tuple_2): patch size
        padding (int or tuple_2, optional): padding size for `x`. Defaults to 0. (following conv2d's rule)
        stride (int or tuple_2, optional): stride of the patch slicing. Defaults to 1.
        dilation (int or tuple_2, optional): dilation of the samping kernel Defaults to 1.

    Returns:
        tensor of the patches with shape of (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
    """

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation *
                   (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    h_dim_in = x.shape[2]
    w_dim_in = x.shape[3]

    h_dim_out = get_dim_blocks(
        h_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    w_dim_out = get_dim_blocks(
        w_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])

    x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[0], kernel_size[1]), padding=(
        padding[0], padding[1]), stride=(stride[0], stride[1]), dilation=(dilation[0], dilation[1]))
    # (B, C * kernel_size[0] * kernel_size[1], h_dim_out, w_dim_out)

    x = x.view(-1, channels, kernel_size[0],
               kernel_size[1], h_dim_out, w_dim_out)
    # (B, C, kernel_size[0], kernel_size[1], h_dim_out, w_dim_out)

    x = x.permute(0, 1, 4, 5, 2, 3)
    # (B, C, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1])

    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1])
    # (B * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])

    return x


def combine_patches_2d(x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
    """
    combine 2D patches to restore the original large tensor (sum overlapped area)

    Args:
        x (tensor): tensor of the 2D patches with shape of (B *h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1])
        kernel_size (int or tuple_2): patch size
        output_shape (tuple): shape of original large 2D tensor (B,C,H,W)
        padding (int or tuple_4, optional): padding size for `x`. Defaults to 0. (following conv2d's rule)
        stride (int or tuple_2, optional): stride of the patch slicing. Defaults to 1.
        dilation (int or tuple_2, optional): dilation. Defaults to 1.

    Note: The values in the overlapped areas will be summed up when combining.

    Returns:
        original large 2D tensor with shape of (B,C,H,W)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation *
                   (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    h_dim_out, w_dim_out = output_shape[2:]
    h_dim_in = get_dim_blocks(
        h_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
    w_dim_in = get_dim_blocks(
        w_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])

    x = x.view(-1, channels, h_dim_in, w_dim_in,
               kernel_size[0], kernel_size[1])
    # (B, C, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1])

    x = x.permute(0, 1, 4, 5, 2, 3)
    # (B, C, kernel_size[0], kernel_size[1],h_dim_in, w_dim_in)

    x = x.contiguous().view(-1, channels *
                            kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)
    # (B, C * kernel_size[0] * kernel_size[1], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(kernel_size[0], kernel_size[1]), padding=(
        padding[0], padding[1]), stride=(stride[0], stride[1]), dilation=(dilation[0], dilation[1]))
    # (B, C * kernel_size[0], H, W)

    x = x.view(-1, channels, h_dim_out, w_dim_out)
    # (B, C, H, W)

    return x


# ----------------------------------------------------
def calc_padding(img_sz, patch_sz, overlap):
    """
    calc the mininal padding size to cover the whole image when extracting patches
    """
    if isinstance(patch_sz, int):
        patch_sz = (patch_sz, patch_sz)
    strides = (patch_sz[0] - overlap, patch_sz[1] - overlap)
    img_SZ = (ceil((img_sz[0]-patch_sz[0])/strides[0])*strides[0] + patch_sz[0],
              ceil((img_sz[1]-patch_sz[1])/strides[1])*strides[1] + patch_sz[1])
    pad_x, pad_y = ceil((img_SZ[0]-img_sz[0]) //
                        2), ceil((img_SZ[1]-img_sz[1])//2)
    return pad_x, pad_y

# ----------------------------------------------------


if __name__ == '__main__':
    import cv2
    a = cv2.imread(
        '/ssd/0/zzh/tmp/CED/outputs/code_dev/phpnet/test/2022-11-14_23-27-29/outputs/test21_0001_gt_img.jpg')
    cv2.imwrite('a.jpg', a)
    a = torch.tensor(a, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    print(a.shape)

    shape = a.shape
    stride = 44
    kernel_size = 64
    padding = calc_padding(shape[2:], kernel_size, overlap=kernel_size-stride)
    print(padding)

    b = extract_patches_2d(a, kernel_size=kernel_size,
                           padding=padding, stride=stride)
    print(b.shape)

    c = combine_patches_2dx(b, kernel_size=kernel_size, output_shape=shape,
                            padding=padding, stride=stride, overlap_op='mean')  #
    print(c.shape)

    cc = c[0].permute(1, 2, 0).numpy()
    cv2.imwrite('cc.jpg', cc)

    # compare
    print(torch.all(a == c))

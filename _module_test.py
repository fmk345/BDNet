import torch
import cv2
import torch.nn.functional as F

if __name__ == '__main__':
    import cv2
    from einops import rearrange
    a = cv2.imread(
        '/ssd/0/zzh/tmp/CED/outputs/code_dev/phpnet/test/2022-11-16_16-16-55/outputs/test01_0001_gt_img.jpg')
    b = cv2.imread(
        '/ssd/0/zzh/tmp/CED/outputs/code_dev/phpnet/test/2022-11-16_16-16-55/outputs/test05_0001_in_img.jpg')
    cv2.imwrite('a.jpg', a[0:512, 0:512, ...])
    cv2.imwrite('b.jpg', b[0:512, 0:512, ...])

    a = torch.tensor(a, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    a = a[:, :, 0:512, 0:512]  # [1,3,h,w]
    b = torch.tensor(b, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    b = b[:, :, 0:512, 0:512]  # [1,3,h,w]
    aa = torch.cat((a, b), dim=0)
    print(aa.shape)

    a_sf = F.pixel_unshuffle(aa, downscale_factor=2)
    print(a_sf.shape)
    # a_sf = a_sf.view(1, 3, 4, 256, 256)  # [1,3*4,h//2,w//2]
    a_sf = rearrange(a_sf, 'b (c n) h w -> (b n) c h w', n=4)
    print(a_sf.shape)

    a_sf_ = a_sf[5, :, ...].permute(1, 2, 0).numpy()
    print(a_sf_.shape)
    cv2.imwrite('cc5.jpg', a_sf_)

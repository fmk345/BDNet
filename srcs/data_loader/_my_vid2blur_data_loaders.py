import sys
import torch.distributed as dist
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
import cv2
import os
import numpy as np
from tqdm import tqdm
from os.path import join as opj

# =================
# loading multiple frames from a video and average them to form a blur image
#
# dataset for loading multiple video frames
# data dir structure:
#     data_dir
#     ├─ vid_dir1
#     |  ├─ frame1
#     |  ├─ frame2
#     |  ├─ ...
#     ├─ vid_dir2
#     |  ├─ frame1
#     |  ├─ frame2
#     |  ├─ ...
#     ├─ ...
# =================

# =================
# basic functions
# =================


def input_data_gen(frames, ce_code, noise_level=0):
    """
    generate input data

    Args:
        frames (ndarray): high frame rate frames  (value [0,1])
        ce_code (ndarray): exposure code sequence
        noise_level: Gaussian noise sigma
    """
    frame_sz = frames.shape
    _ce_code = ce_code[:, None, None]
    _ce_code = np.tile(_ce_code, frame_sz[1:])
    # print(code.shape)
    coded_meas = np.sum(_ce_code*frames, axis=0)/np.sum(ce_code)

    # add Gaussian noise
    if noise_level>0:
        coded_meas = coded_meas + \
            np.random.normal(0, noise_level, coded_meas.shape).astype(np.float32)

    return coded_meas.clip(0, 1)


def init_network_input(coded_blur_img, ce_code):
    """
    calculate the initial input of the network


    Args:
        coded_blur_img (ndarray): coded measurement
        ce_code (ndarray): encoding code
    """

    # rescale the input to normal light condition
    coded_blur_img_rescale = coded_blur_img*len(ce_code)/sum(ce_code)
    return coded_blur_img_rescale


def transform(vid, prob=0.5, tform_op=['all']):
    """
    video data transform (data augment) with a $op chance

    Args:
        vid ([ndarray]): [shape: N*H*W*C]
        prob (float, optional): [probility]. Defaults to 0.5.
        op (list, optional): ['flip' | 'rotate' | 'reverse']. Defaults to ['all'].
    """
    if 'flip' in tform_op or 'all' in tform_op:
        # flip left-right or flip up-down
        if np.random.rand() < prob:
            vid = vid[:, :, ::-1, :]
        if np.random.rand() < prob:
            vid = vid[:, ::-1, :, :]
    if 'rotate' in tform_op or 'all' in tform_op:
        # rotate 90 / -90 degrees
        if prob/4 < np.random.rand() <= prob/2:
            np.transpose(vid, axes=(0, 2, 1, 3))[:, ::-1, ...]  # -90
        elif prob/2 < np.random.rand() <= prob:
            vid = np.transpose(
                vid[:, ::-1, :, :][:, :, ::-1, :], axes=(0, 2, 1, 3))[:, ::-1, ...]  # 90

    if 'reverse' in tform_op or 'all' in tform_op:
        if np.random.rand() < prob:
            vid = vid[::-1, ...]

    return vid.copy()

# =================
# Video-averaged blur image dataset
# =================


class VidBlur_Dataset(Dataset):
    """
    dataset for loading multiple video frames, load one batch before each iter
    """

    def __init__(self, data_dir, ce_code, patch_sz=None, tform_op=None, sigma_range=0, stride=1):
        super(VidBlur_Dataset, self).__init__()
        self.ce_code = np.array(ce_code, dtype=np.float32)
        self.sigma_range = sigma_range
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.vid_length = len(ce_code)
        self.img_paths = []
        self.vid_idx = []
        self.stride = stride  # stride of the starting frame

        # get image paths
        img_nums = []
        vid_paths = []
        if isinstance(data_dir, str):
            # single dataset
            vid_names = sorted(os.listdir(data_dir))
            vid_paths = [opj(data_dir, vid_name) for vid_name in vid_names]
        else:
            # multiple dataset
            for data_dir_n in sorted(data_dir):
                vid_names_n = sorted(os.listdir(data_dir_n))
                vid_paths_n = [opj(data_dir_n, vid_name_n)
                               for vid_name_n in vid_names_n]
                vid_paths.extend(vid_paths_n)

        for vid_path in vid_paths:
            img_names = sorted(os.listdir(vid_path))
            img_nums.append(len(img_names))
            self.img_paths.extend(
                [opj(vid_path, img_name) for img_name in img_names])

        counter = 0
        for img_num in img_nums:
            self.vid_idx.extend(
                list(range(counter, counter+img_num-self.vid_length+1, stride)))
            counter = counter+img_num

    def __getitem__(self, idx):
        # load video frames
        vid = []
        for k in range(self.vid_idx[idx], self.vid_idx[idx]+self.vid_length):
            # read image
            img = cv2.imread(self.img_paths[k])
            assert img is not None, 'Image read falied'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.patch_sz:
                if k == self.vid_idx[idx]:
                    # set the random crop point
                    img_sz = img.shape
                    assert (img_sz[0] >= self.patch_sz[0]) and (img_sz[1] >= self.patch_sz[1]
                                                                ), 'error PATCH_SZ larger than image size'
                    xmin = np.random.randint(0, img_sz[1]-self.patch_sz[1])
                    ymin = np.random.randint(0, img_sz[0]-self.patch_sz[0])

                # crop to patch size
                img_crop = img[ymin:ymin+self.patch_sz[0],
                               xmin:xmin+self.patch_sz[1], :]
            else:
                img_crop = img

            vid.append(img_crop)

        vid = np.array(vid, dtype=np.float32)/255  # [vid_num, h, w, c]

        # data augment
        if self.tform_op:
            vid = transform(vid, tform_op=self.tform_op)

        # noise level
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)

        # calc coded measurement
        coded_meas = input_data_gen(vid, self.ce_code, noise_level)

        # calc middle video frame
        sharp_img = vid[self.ce_code.shape[0]//2, ...]

        return vid.transpose(0, 3, 1, 2), coded_meas.transpose(2, 0, 1)

    def __len__(self):
        return len(self.vid_idx)


class VidBlur_Dataset_all2CPU(Dataset):
    """
    dataset for loading multiple video frames,, load entire dataset to CPU to speed up the data load process
    """

    def __init__(self, data_dir, ce_code, patch_sz=None, tform_op=None, sigma_range=0, stride=1):
        super(VidBlur_Dataset_all2CPU, self).__init__()
        self.ce_code = np.array(ce_code, dtype=np.float32)
        self.sigma_range = sigma_range
        self.patch_sz = [patch_sz] * \
            2 if isinstance(patch_sz, int) else patch_sz
        self.tform_op = tform_op
        self.vid_length = len(ce_code)
        self.img_paths = []
        self.vid_idx = []  # start frame index of each video
        self.imgs = []
        self.stride = stride  # stride of the starting frame

        # get image paths and load images
        img_nums = []
        vid_paths = []
        if isinstance(data_dir, str):
            # single dataset
            vid_names = sorted(os.listdir(data_dir))
            vid_paths = [opj(data_dir, vid_name) for vid_name in vid_names]
        else:
            # multiple dataset
            for data_dir_n in sorted(data_dir):
                vid_names_n = sorted(os.listdir(data_dir_n))
                vid_paths_n = [opj(data_dir_n, vid_name_n)
                               for vid_name_n in vid_names_n]
                vid_paths.extend(vid_paths_n)

        for vid_path in vid_paths:
            img_names = sorted([os.path.join(vid_path, name) for name in os.listdir(vid_path) if name.endswith('png')])
            img_nums.append(len(img_names))
            self.img_paths.extend(img_names)

        # img_shape = None
        for img_path in tqdm(self.img_paths, desc='⏳ Loading dataset to Memory'):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            assert img is not None, 'Image read falied'
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgs.append(img)
            # assert (
            #     img_shape is None) or img.shape == img_shape, 'Please make sure the images in the datasets have the same size'
            # img_shape = img.shape
        # self.imgs = np.array(self.imgs, dtype=img.dtype)  # [vid_num, h, w, c], double cpu memory

        counter = 0
        for img_num in img_nums:
            self.vid_idx.extend(
                list(range(counter, counter+img_num-self.vid_length+1, stride)))
            counter = counter+img_num

    def __getitem__(self, idx):
        # load video frames
        vid = self.imgs[self.vid_idx[idx]:self.vid_idx[idx]+self.vid_length]
        vid = np.array(vid, dtype=np.float32)/255

        img_sz = vid[0].shape
        # crop to patch size
        if self.patch_sz:
            assert (img_sz[0] >= self.patch_sz[0]) and (img_sz[1] >= self.patch_sz[1]
                                                        ), 'error PATCH_SZ larger than image size'
            xmin = np.random.randint(0, img_sz[1]-self.patch_sz[1])
            ymin = np.random.randint(0, img_sz[0]-self.patch_sz[0])
            vid = vid[:, ymin:ymin+self.patch_sz[0],
                      xmin:xmin+self.patch_sz[1], :]
        # data augment
        if self.tform_op:
            vid = transform(vid, tform_op=self.tform_op)

        # noise level
        if isinstance(self.sigma_range, (int, float)):
            noise_level = self.sigma_range
        else:
            noise_level = np.random.uniform(*self.sigma_range)

        coded_meas = input_data_gen(vid, self.ce_code, noise_level)

        # calc middle video frame
        # sharp_img = vid[self.ce_code.shape[0]//2, ...]

        # [debug] test
        # multi_imsave(vid*255, 'vid')
        # cv2.imwrite('./outputs/tmp/test/coded_meas.jpg', coded_meas[:,:,::-1]*255)
        # cv2.imwrite('./outputs/tmp/test/clear.jpg', sharp_img[:, :, ::-1]*255)

        return vid, coded_meas

    def __len__(self):
        return len(self.vid_idx)


class VidBlur_Dataset_RealExp:
    """
    Datasetfor real test
    """
    pass


# =================
# get dataloader
# =================

def get_data_loaders(data_dir, ce_code, batch_size, patch_size=None, tform_op=None, sigma_range=0, shuffle=True, validation_split=0.1, status='train', num_workers=8, pin_memory=False, prefetch_factor=2, all2CPU=True):
    if status == 'train':
        if all2CPU:
            dataset = VidBlur_Dataset_all2CPU(
                data_dir, ce_code, patch_size, tform_op, sigma_range)
        else:
            dataset = VidBlur_Dataset(
                data_dir, ce_code, patch_size, tform_op, sigma_range)
    elif status == 'test':
        if all2CPU:
            dataset = VidBlur_Dataset_all2CPU(
                data_dir, ce_code, patch_size, tform_op, sigma_range, len(ce_code))
        else:
            dataset = VidBlur_Dataset(
                data_dir, ce_code, patch_size, tform_op, sigma_range, len(ce_code))
    elif status == 'real_test':
        dataset = VidBlur_Dataset_RealExp(
            data_dir, ce_code, patch_size)
    else:
        raise NotImplementedError(
            f"status ({status}) should be 'train' | 'test' ")

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory
    }

    if status == 'train' or status == 'debug':
        # split dataset into train and validation set
        num_total = len(dataset)
        if isinstance(validation_split, int):
            assert validation_split > 0
            assert validation_split < num_total, "validation set size is configured to be larger than entire dataset."
            num_valid = validation_split
        else:
            num_valid = int(num_total * validation_split)
        num_train = num_total - num_valid

        train_dataset, valid_dataset = random_split(
            dataset, [num_train, num_valid])

        train_sampler, valid_sampler = None, None
        if dist.is_initialized():
            loader_args['shuffle'] = False
            train_sampler = DistributedSampler(train_dataset)
            valid_sampler = DistributedSampler(valid_dataset)
        return DataLoader(train_dataset, sampler=train_sampler, **loader_args), \
            DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
    else:
        return DataLoader(dataset, **loader_args)


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__) + os.sep + '../')
    # from utils.utils_image_zzh import multi_imsave

    data_dir = './dataset/'
    # data_dir = '/ssd/2/zzh/dataset/GoPro/GOPRO_Large_all/small_test/'
    # ce_code = [0, 1, 1, 0, 1]
    # ce_code = [1, 0, 1, 1, 1, 0, 0, 1, 0, 1]
    ce_code = [1,0,1,0,1,1,1,1,1,1,0,0,1,0,1,0,1,0,0,0,0,1,1,1,0,1,0,0,0,0,1]
    val_dataloader = get_data_loaders(
        data_dir, ce_code, batch_size=1, num_workers=8, shuffle=False, all2CPU=True, status='test')
    # train_dataloader, val_dataloader = get_data_loaders(
    #     data_dir, ce_code, patch_size=512, tform_op=['all'], batch_size=2, num_workers=8, all2CPU=True,status='test')

    k = 0
    for sharp_img, coded_meas in val_dataloader:
        k += 1
        coded_meas = coded_meas.numpy()[0, ...]*255
        sharp_img = sharp_img.numpy()[0, 0, ...]*255
        # init_input = init_input.numpy()[0, ::-1, ...].transpose(1, 2, 0)*255

        if not os.path.exists('./outputs/tmp/test/'):
            os.makedirs('./outputs/tmp/test/')

        cv2.imwrite(f'./outputs/tmp/test/{k:03d}coded_meas.jpg', coded_meas)
        cv2.imwrite(f'./outputs/tmp/test/{k:03d}clear.jpg', sharp_img)
        # cv2.imwrite('./outputs/tmp/test/init_input.jpg', init_input)

        if k % 1 == 0:
            print('k = ', k)

import logging
import os
import cv2
import torch
import time
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
from srcs.utils.util import instantiate
from srcs.utils.utils_image_kair import tensor2uint, imsave
from srcs.utils.utils_patch_proc import window_partitionx, window_reversex
import torch.nn.functional as F
from srcs.utils.utils_eval_zzh import gpu_inference_time

def testing(gpus, config):
    test_worker(gpus, config)


def test_worker(gpus, config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # logger & dir setting
    logger = logging.getLogger('test')
    os.makedirs(config.outputs_dir,exist_ok=True)

    # prepare model & checkpoint for testing
    # load checkpoint
    logger.info(f"💡 Loading checkpoint: {config.checkpoint} ...")
    checkpoint = torch.load(config.checkpoint)
    logger.info(f"💡 Checkpoint loaded: epoch {checkpoint['epoch']}.")

    # select config file
    if 'config' in checkpoint:
        loaded_config = OmegaConf.create(checkpoint['config'])
    else:
        loaded_config = config

    # instantiate model
    model = instantiate(loaded_config.arch)
    logger.info(model)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)

    # load weight
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # setup data_loader instances
    data_loader = instantiate(config.test_data_loader)

    # test
    metrics = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(data_loader, model,
               device, metrics, config)
    logger.info(log)


def test(data_loader, model,  device, metrics, config):
    '''
    test step
    '''

    # init dir
    res_dir_input = os.path.join(config.outputs_dir, 'input')
    res_dir_output = os.path.join(config.outputs_dir, 'output')
    os.makedirs(res_dir_input,exist_ok=True)
    os.makedirs(res_dir_output,exist_ok=True)
    
    # init model
    model = model.to(device)
    model_deblur = model.DeBlurNet  # deblur model

    # init param
    ce_weight = model.BlurNet.ce_weight.detach()
    ce_code = ((torch.sign(ce_weight)+1)/2).int()
    time_idx = torch.tensor(range(len(ce_code)))/(len(ce_code)-1)
    time_idx = time_idx.unsqueeze(1).to(device)
    scale_fc = 8/5

    model_deblur.eval()
    time_start = time.time()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc='⏳ Testing')):
            # data = torch.flip(data.to(device), [2, 3])
            data = data.to(device).float()/255/scale_fc
            N, C, Hx, Wx = data.shape
            
            # direct
            output = model_deblur(ce_blur=data, time_idx=time_idx, ce_code=ce_code)

            # sliding window - patch processing
            # _, _, Hx, Wx = data.shape
            # sliding window - patch processing
            # data_re, batch_list = window_partitionx(_data, config.win_size)
            # output = model_deblur(data_re)
            # output = window_reversex(
            #     output, config.win_size, Hx, Wx, batch_list)

            # pad & crop
            # sf = 4
            # HX, WX = int((Hx+sf-1)/sf)*sf, int((Wx+sf-1)/sf) * \
            #     sf  # pad to a multiple of scale_factor (sf)
            # pad_h, pad_w = HX-Hx, WX-Wx
            # data_pad = F.pad(_data/scale_fc, [0, pad_w, 0, pad_h])
            # output = model_deblur(data_pad)
            # output = output[:, :, :Hx, :Wx]

            # clamp to 0-1
            output = torch.clamp(output, 0, 1)

            # save some sample images
            for k, (in_img, out_img) in enumerate(zip(data, output)):
                in_img = tensor2uint(in_img*scale_fc)
                imsave(
                    in_img, f'{res_dir_input}/ce-blur#{i*N+k+1:04d}.jpg')
                for j in range(len(ce_code)):
                    out_img_j = tensor2uint(out_img[j])
                    imsave(
                        out_img_j, f'{res_dir_output}/frame#{i*N+k+1:04d}-{j:04d}.jpg')

    time_end = time.time()
    time_cost = time_end-time_start
    n_samples = len(data_loader.sampler)
    log = {'time/sample': time_cost/n_samples}
    return log

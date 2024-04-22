import numpy as np
import os
import hydra
import torch
import warnings
from omegaconf import OmegaConf
from importlib import import_module
# torch.autograd.set_detect_anomaly(True) # for debug
from srcs.model.bd_model import BDNeRV
# fix random seeds for reproducibility
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['NUMEXPR_MAX_THREADS'] = r'1'
torch.backends.cudnn.enabled = False
# ignore warning
warnings.filterwarnings('ignore')

# my_train | cebd_train

@hydra.main(config_path='conf/', config_name='cebd_train')
def main(config):
    # GPU setting
    if not config.gpus or config.gpus == -1:
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = config.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
    n_gpu = len(gpus)
    assert len(gpus) <= torch.cuda.device_count(
    ), f'There are {torch.cuda.device_count()} GPUs on this machine, but you assigned $gpus={gpus}.'
    
    # resume
    config_v = OmegaConf.to_yaml(config, resolve=True)
    # import pdb;pdb.set_trace()
    # show config

    print('='*40+'\n', config_v, '\n'+'='*40+'\n')

    # training
    trainer_name = 'srcs.trainer.%s' % config.trainer_name
    training_module = import_module(trainer_name)
    training_module.trainning(gpus, config)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()

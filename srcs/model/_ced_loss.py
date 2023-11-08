import torch.nn.functional as F
import torch
from srcs.model.fswdnet_modules import LBSign

# light thtoughput loss for FSWDNet


def light_throughput_loss(ce_weight, target_throughput=0.5):
    ce_code = (LBSign.apply(ce_weight)+1)/2
    light_throught = torch.mean(ce_code)
    return F.mse_loss(light_throught, target_throughput)

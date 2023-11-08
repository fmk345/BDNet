import cv2
import numpy as np
import torch 

tensor = torch.zeros(16,3,256,256,device='cuda:0') # a tensor for test
# tensor = xxx

# save N*3*H*W tensor to RGB image: save $tensor[0]
tmp_ = tensor[0].detach().cpu().numpy()[::-1].transpose(1, 2, 0)
tmp_ = 255*(tmp_ - tmp_.min())/(tmp_.max() - tmp_.min())
cv2.imwrite('tensor_0.png', tmp_.astype(np.uint8))

# save N*C*H*W tensor to gray image: save $tensor[0,0]
tmp_ = tensor[0, 8].detach().cpu().numpy()
tmp_ = 255*(tmp_ - tmp_.min())/(tmp_.max() - tmp_.min())
cv2.imwrite('tensor_0.png', tmp_.astype(np.uint8))

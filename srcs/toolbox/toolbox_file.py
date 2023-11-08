import os
from os.path import join as opj

#%% file rename

# %% rename file
dir = '/hdd/1/zzh/project/CED-Net/dataset/benchmark/pair_box_random_psf1'

file_names = sorted(os.listdir(dir))
for k, file_name in enumerate(file_names):
    os.rename(opj(dir, file_name), opj(dir, 'box_psf%02d.png' % (k+1)))

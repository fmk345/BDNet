# run multi tasks sequentially
python train.py -m optimizer.lr=0.001,0.002

# resume training (change ckp, gpu, run_dir in the config file first)
python train.py --config-path outputs/Analysis/ce_duty/train/2023-05-17_00-59-55-todo-len04/.hydra --config-name config  hydra.run.dir=outputs/Analysis/ce_duty/train/2023-05-17_00-59-55-todo-len04
import torch
import torch.distributed as dist
from torchvision.utils import make_grid
import platform
from omegaconf import OmegaConf
from .base import BaseTrainer
from srcs.utils.util import collect, instantiate, get_logger
from srcs.logger import BatchMetrics
import torch.nn.functional as F
from functools import reduce
from srcs.utils.utils_eval_zzh import gpu_inference_time, model_complexity
from srcs.model._basic_binary_modules import STEBinary_fc

#======================================
# Trainer: modify '_train_epoch'
#======================================


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, ce_opt_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.ce_opt_epoch = ce_opt_epoch
        self.lr_scheduler = lr_scheduler
        self.limit_train_iters = config['trainer'].get(
            'limit_train_iters', len(self.data_loader))
        if not self.limit_train_iters or self.limit_train_iters > len(self.data_loader):
            self.limit_train_iters = len(self.data_loader)
        self.limit_valid_iters = config['trainer'].get(
            'limit_valid_iters', len(self.valid_data_loader))
        if not self.limit_valid_iters or self.limit_valid_iters > len(self.valid_data_loader):
            self.limit_valid_iters = len(self.valid_data_loader)
        self.log_weight = config['trainer'].get('log_weight', False)
        args = ['loss', *[m.__name__ for m in self.metric_ftns]]
        self.train_metrics = BatchMetrics(
            *args, postfix='/train', writer=self.writer)
        self.valid_metrics = BatchMetrics(
            *args, postfix='/valid', writer=self.writer)
        self.n_levels = 3  # model scale levels
        # self.scale = 0.5   # model scale
        self.grad_clip = 0.5  # optimizer gradient clip value
        self.losses = self.config['loss']
        self.light_throughput = self.config.get('light_throughput', None)
        self.opt_cecode = self.config['arch'].get('opt_cecode', False)
        # zzh: for ce optim
        # if self.opt_cecode:
        #     self.ce_optimizer = torch.optim.SGD(
        #         self.model.BlurNet.parameters(), lr=0.001)
        #     self.ce_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #         self.ce_optimizer, 10, gamma=0.5, last_epoch=-1)

    def clip_gradient(self, optimizer, grad_clip=0.5):
        """
        Clips gradients computed during backpropagation to avoid explosion of gradients.

        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
        """
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def _light_throughput_loss(self, ce_weight, light_throughput):
        ce_code = STEBinary_fc(ce_weight)
        now_light_throught = torch.mean(ce_code)
        target_light_throughput = torch.tensor(
            light_throughput).to(now_light_throught.device)
        return 100*F.mse_loss(now_light_throught, target_light_throughput)
    
    def _ce_reblur(self, output):
        # frame_n should equal to ce_code_n cases
        ce_weight = self.model.BlurNet.ce_weight.detach().squeeze()
        ce_code = ((torch.sign(ce_weight)+1)/2)
        ce_code_ = torch.tensor(ce_code).view(1, -1, 1, 1, 1)
        ce_output = torch.sum(torch.mul(output, ce_code_), dim=1)/len(ce_code)
        return ce_output
    
    def _after_iter(self, epoch, batch_idx, phase, loss, metrics, image_tensors: dict):
        # hook after iter
        self.writer.set_step(
            (epoch - 1) * getattr(self, f'limit_{phase}_iters') + batch_idx, speed_chk=f'{phase}')

        loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
        getattr(self, f'{phase}_metrics').update('loss', loss_v)

        for k, v in metrics.items():
            getattr(self, f'{phase}_metrics').update(k, v.item()) # `v` is a torch tensor

        for k, v in image_tensors.items():
            self.writer.add_image(
                f'{phase}/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        interp_scale = 1

        # control ce code and optimization status
        if self.opt_cecode and (self.ce_opt_epoch is not None):
            if self.ce_opt_epoch[0] <= epoch <= self.ce_opt_epoch[1]:
                self.model.BlurNet.ce_weight.requires_grad = True
            else:
                self.model.BlurNet.ce_weight.requires_grad = False
            self.logger.info(
                f'Current CE Opt: {self.model.BlurNet.ce_weight.requires_grad}')


        for batch_idx, vid in enumerate(self.data_loader):  # video_dataloader
            
            vid = vid.to(self.device).float()/255 
            target = vid[:,::interp_scale]

            output, data, data_noisy = self.model(vid)
            output_ = torch.flatten(output, end_dim=1)
            target_ = torch.flatten(target, end_dim=1)

            # main loss
            loss = self.losses['main_loss'] * \
                self.criterion['main_loss'](output_, target_)


            self.optimizer.zero_grad()
            # if self.opt_cecode:
            # optce optim
            #     self.ce_optimizer.zero_grad()
            loss.backward()

            # [debug]
            # ce_weight = self.model.BlurNet.ce_weight
            # print('ce_weight: ', ce_weight.data.squeeze())
            # print('ce_weight_grad: ', ce_weight.grad.data.squeeze())

            # clip gradient note: for OptCE this should be careful
            # self.clip_gradient(self.optimizer, self.grad_clip)

            self.optimizer.step()
            # if self.opt_cecode:
            #     self.ce_optimizer.step()

            # iter record
            if batch_idx % self.logging_step == 0 or (batch_idx+1) == self.limit_train_iters:
                # loss
                loss_v = loss.item() if self.config.n_gpu == 1 else collect(loss)
                self.writer.set_step(
                    (epoch - 1) * self.limit_train_iters + batch_idx, speed_chk='train')
                self.train_metrics.update('loss', loss_v)
                # iter metrics
                iter_metrics = {}
                for met in self.metric_ftns:
                    if self.config.n_gpu > 1:
                        # average metric between processes
                        metric_v = collect(met(output_, target_))
                    else:
                        # print(output.shape, target.shape)
                        metric_v = met(output_, target_)
                    iter_metrics.update({met.__name__: metric_v})

                # iter images
                frame_num = output.shape[1]
                image_tensors = {
                    'input': data_noisy[0, ...], 'output': output[0,0::frame_num//4, ...], 'target': target[0,0::frame_num//4, ...]}
                # image_tensors = {} # donnot save image in tensorboard in training epoch to save space
                # aftet iter hook
                self._after_iter(epoch, batch_idx, 'train',
                                 loss, iter_metrics, {})
                # iter log
                self.logger.info(
                    f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss:.6f} Lr: {self.optimizer.param_groups[0]["lr"]:.3e}')

            if (batch_idx+1) == self.limit_train_iters:
                # save demo images to tensorboard after trainig epoch
                self.writer.set_step(epoch)
                for k, v in image_tensors.items():
                    self.writer.add_image(
                        f'train/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
                break
        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # if self.opt_cecode and self.ce_lr_scheduler is not None:
        #     self.ce_lr_scheduler.step()

        # add result metrics on entire epoch to tensorboard
        self.writer.set_step(epoch)
        for k, v in log.items():
            self.writer.add_scalar(k + '/epoch', v)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        interp_scale = 1
        self.valid_metrics.reset()
        with torch.no_grad():
            # show ce code update when ceopt
            if self.opt_cecode:
                # ce weight and ce code
                ce_weight = self.model.BlurNet.ce_weight.detach().squeeze()
                ce_code = ((torch.sign(ce_weight)+1)/2).int()
                self.logger.info('-'*70+'\nCurrent CE Weight: ' + reduce(lambda x,
                                 y: x+y, ['%.4f ' % x for x in ce_weight.tolist()]))
                self.logger.info('Current CE Code: ' + str(ce_code.tolist()))

            for batch_idx, vid in enumerate(self.valid_data_loader):
                vid = vid.to(self.device).float()/255
                target = vid[:,::interp_scale]

                # forward
                output, data, data_noisy = self.model(vid)
                output_ = torch.flatten(output, end_dim=1)
                target_ = torch.flatten(target, end_dim=1)

                # main loss
                loss = self.losses['main_loss'] * \
                    self.criterion['main_loss'](output_, target_)
                # reblur loss: frame_n should equal to ce_code_n cases
                if 'reblur_loss' in self.losses:
                    # coded_output = self.model.BlurNet(output)
                    ce_output = self._ce_reblur(output)
                    loss = loss + self.losses['reblur_loss'] *self.criterion['reblur_loss'](ce_output, data)

                # iter metrics
                iter_metrics = {}
                for met in self.metric_ftns:
                    if self.config.n_gpu > 1:
                        # average metric between processes
                        metric_v = collect(met(output_, target_))
                    else:
                        # print(output.shape, target.shape)
                        metric_v = met(output_, target_)
                    iter_metrics.update({met.__name__: metric_v})

                # iter images
                frame_num = output.shape[1]
                image_tensors = {
                    'input': data_noisy[0, ...], 'output': output[0, 0::frame_num//4, ...], 'target': target[0, 0::frame_num//4, ...]}

                # aftet iter hook
                self._after_iter(epoch, batch_idx, 'valid',
                                 loss, iter_metrics, {})

                if (batch_idx+1) == self.limit_valid_iters:
                    # save demo images to tensorboard after valid epoch
                    self.writer.set_step(epoch)
                    for k, v in image_tensors.items():
                        self.writer.add_image(
                            f'valid/{k}', make_grid(image_tensors[k][0:8, ...].cpu(), nrow=2, normalize=True))
                    break

        # add histogram of model parameters to the tensorboard
        if self.log_weight:
            for name, p in self.model.BlurNet.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        try:
            # epoch-based training
            # total = len(self.data_loader.dataset)
            total = self.data_loader.batch_size * self.limit_train_iters
            current = batch_idx * self.data_loader.batch_size
            if dist.is_initialized():
                current *= dist.get_world_size()
        except AttributeError:
            # iteration-based training
            total = self.limit_train_iters
            current = batch_idx
        return base.format(current, total, 100.0 * current / total)


#======================================
# Trainning: run Trainer for trainning
#======================================


def trainning(gpus, config):
    # enable access to non-existing keys
    OmegaConf.set_struct(config, False)
    n_gpu = len(gpus)
    config.n_gpu = n_gpu
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    if n_gpu > 1:
        torch.multiprocessing.spawn(
            multi_gpu_train_worker, nprocs=n_gpu, args=(gpus, config))
    else:
        train_worker(config)


def train_worker(config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    logger = get_logger('train')
    # setup data_loader instances
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    logger.info(model)

    # calc MACs & Param. Num
    model_complexity(model=model, input_shape=(8, 3, 256, 256), logger=logger)

    # get function handles of loss and metrics
    # criterion = instantiate(config.loss)
    # get function handles of loss and metrics
    criterion = {}
    if 'main_loss' in config.loss:
        criterion['main_loss'] = instantiate(config.main_loss)
    if 'reblur_loss' in config.loss:
        criterion['reblur_loss'] = instantiate(
            config.reblur_loss)
    # metrics = [instantiate(met, is_func=True) for met in config['metrics']]
    metrics = [instantiate(met) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    # optimizer = instantiate(config.optimizer, model.DeBlurNet.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler, ce_opt_epoch=config.ce_opt_epoch)
    trainer.train()


def multi_gpu_train_worker(rank, gpus, config):
    """
    Training with multiple GPUs

    Args:
        rank ([type]): [description]
        gpus ([type]): [description]
        config ([type]): [description]

    Raises:
        RuntimeError: [description]
    """
    # initialize training config
    config.local_rank = rank
    if(platform.system() == 'Windows'):
        backend = 'gloo'
    elif(platform.system() == 'Linux'):
        backend = 'nccl'
    else:
        raise RuntimeError('Unknown Platform (Windows and Linux are supported')
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:34567',
        world_size=len(gpus),
        rank=rank)
    torch.cuda.set_device(gpus[rank])

    # start training processes
    train_worker(config)

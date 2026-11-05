# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics
import time
import torch
import tqdm
from tensorboardX import SummaryWriter
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--workers', default=4, type=int,)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # parser.add_argument('--wandb_project', default='v2x_dataset',)
    parser.add_argument('--wandb_project', default=None,)
    parser.add_argument('--tag', default=None, type=str)
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.msig_load_yaml(opt.hypes_yaml, opt)
    hypes['flag_train'] = True

    # update name with tag
    if opt.tag is not None:
        hypes['name'] = hypes['name'] + '_' + opt.tag

    multi_gpu_utils.init_distributed_mode(opt)
    if not opt.distributed:
        opt.rank = 0

    local_rank, world_size = multi_gpu_utils.get_dist_info()
    
    print('==============================')
    print('INFO | RANK: ', local_rank)
    print('INFO | world_size: ', world_size)
    print('==============================')

    print('-----------------Dataset Building------------------')
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    
    # val_dataset = build_dataset(hypes, visualize=False, train=False)
    # val_dataset.set_flag_compute_validation_loss()

    if opt.distributed:
        sampler_train = DistributedSampler(train_dataset)
        # sampler_val = multi_gpu_utils.OpenPCDetDistributedSampler(
        #     val_dataset, world_size, local_rank, shuffle=False
        # )

        batch_size = hypes['train_params']['batch_size_per_gpu']
        print(f'INFO | Training in distributed mode : total batch size is {batch_size * world_size}')

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  num_workers=opt.workers,
                                  shuffle=False,  # shuffle is done by DistributedSampler
                                  collate_fn=train_dataset.collate_batch_train,
                                  drop_last=True,
                                  sampler=sampler_train)
        
        # val_loader = DataLoader(val_dataset,
        #                         batch_size=batch_size,
        #                         num_workers=opt.workers,
        #                         shuffle=False,
        #                         drop_last=True,
        #                         collate_fn=val_dataset.collate_batch_test,
        #                         sampler=sampler_val)
    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=opt.workers,
                                  collate_fn=train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        # val_loader = DataLoader(val_dataset,
        #                         batch_size=hypes['train_params']['batch_size'],
        #                         num_workers=opt.workers,
        #                         collate_fn=val_dataset.collate_batch_test,
        #                         shuffle=False,
        #                         pin_memory=False,
        #                         drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank % torch.cuda.device_count()]
        )
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # add wandb logger
    if opt.rank == 0 and opt.wandb_project is not None:
        # hava to create this before calling tensorboard
        wandb.init(project=opt.wandb_project,
                    name=f"{hypes['name']}",
                    sync_tensorboard=True,
                    )
        wandb.config.update(opt)
        wandb.config.update(hypes)

    # record training
    writer = SummaryWriter(saved_path) if opt.rank == 0 else None

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    if opt.rank == 0 and opt.wandb_project is not None:
        wandb.config.update({"total_step": len(train_loader) * epoches})
        # wandb.config.update(cfg)
        wandb.watch(model)

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if opt.distributed:
            sampler_train.set_epoch(epoch)

        if local_rank == 0:
            pbar_train = tqdm.tqdm(total=len(train_loader), leave=local_rank == 0, desc=f'train ep {epoch}')

        _time_data = time.time()
        for i, batch_data in enumerate(train_loader):
            _data_time = time.time() - _time_data
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected

            # case2 : early fusion train --> all data projected to ego

            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs as well
            if not opt.half:
                _time_forward = time.time()
                
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
                
                _forward_time = time.time() - _time_forward
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])

            _time_backward = time.time()
            if not opt.half:
                final_loss.backward()
                # clip must run after backward - gradients don't exist yet
                # before it, so clipping here used to be a no-op
                clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            _backward_time = time.time() - _time_backward

            time_infos = {
                'data_time': _data_time,
                'forward_time': _forward_time,
                'backward_time': _backward_time
            }
            
            if local_rank == 0:
                criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar_train, time_infos=time_infos)
                pbar_train.update(1)

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

            # ================= end of 1 training iter
            del batch_data
            _time_data = time.time()
        
        # ================= end of 1 training epoch
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step()

        for param_group in optimizer.param_groups:
            # print('learning rate %.7f' % param_group["lr"])
            if writer is not None:
                writer.add_scalar('learning_rate', param_group["lr"], epoch)

        if epoch % hypes['train_params']['save_freq'] == 0 and local_rank == 0:
            torch.save(
                model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1))
            )

        # if epoch % hypes['train_params']['eval_freq'] == 0:
        #     valid_ave_loss = []

        #     if local_rank == 0:
        #         pbar_val = tqdm.tqdm(total=len(val_loader), leave=local_rank == 0, desc=f'val ep {epoch}')

        #     for i, batch_data in enumerate(val_loader):
        #         model.eval()
        #         batch_data = train_utils.to_device(batch_data, device)
                
        #         with torch.no_grad():
        #             ouput_dict = model(batch_data['ego'])

        #         final_loss = criterion(
        #             ouput_dict,
        #             batch_data['ego']['label_dict']
        #         )

        #         valid_ave_loss.append(final_loss.item())

        #         if local_rank == 0:
        #             pbar_val.update(1)
            
        #     valid_ave_loss = statistics.mean(valid_ave_loss)

        #     print('\n=========================================================')
        #     print('At epoch %d, the validation loss is %f' % (epoch,
        #                                                       valid_ave_loss))
        #     print('\n=========================================================')

        #     if writer is not None:
        #         writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()

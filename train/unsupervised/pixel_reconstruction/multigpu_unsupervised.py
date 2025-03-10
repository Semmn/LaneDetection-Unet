import sys
sys.path.append('/work')

from pixel_reconstruction.dataset import MaskedCULaneDataset, MaskedMultiDataset

from train.utils.loss import MaskedPixelReconstructLoss
from train.utils.metrics import get_pixel_reconst, get_masked_pixel_reconst
from train.utils.weight_load import load_weight, load_weight_with_cp, set_weights
from train.utils.logger import set_logger_for_training, checkdir
from train.utils.distributed_training import setup_for_distributed, init_for_distributed
from train.utils.plotting import plot_figure
from train.utils.dataset import read_culane_segdata, read_multi_segdata
from train.utils.create_model import create_unet
from train.utils.utils import copy_files
from train.utils.recorder import Recorder

import numpy as np
import argparse
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import logging
import yaml
from datetime import timedelta

import torch
import torch.nn as nn

# for data augmentation
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# for ConsineSchedulerLR
import timm.scheduler

# for pytorch distributed training
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# set argument parser
def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training') # batch size
    parser.add_argument('--val_batch_size', default=32, type=int, help='batch size for validation') # batch size for validation
    
    # path for scheduler configuration file: each configuration files includes specific parameter settings of scheduler
    parser.add_argument('--model_config', type=str, default='/work/train/unsupervised/pixel_reconstruction/model_config/unet.yaml', help='unet configuration file')
    parser.add_argument('--train_config', type=str, default='/work/train/unsupervised/pixel_reconstruction/train_config/patch_masked_multidata.yaml', help='training configuration file')
        
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio for masking operation')
    parser.add_argument('--mask_window_h', type=int, default=56, help='mask window height size')
    parser.add_argument('--mask_window_w', type=int, default=168, help='mask window width size')
    parser.add_argument('--loss_type', type=str, default='mse', help='loss type for training. mse or masked_mse')
    parser.add_argument('--multi_data_mode', action=argparse.BooleanOptionalAction) # flag for multi data training mode (bdd100k, LLaMAS, TuSimple, CuLane)
    
    parser.add_argument('--project_name', type=str, default=f'convnext_tiny_unet/culane', help='project name used for defining save path')
    parser.add_argument('--weight_dir', type=str, default='/work/checkpoints', help='basic path for saving model weights and files')
    parser.add_argument('--plot_dir', type=str, default='/work/plots', help='basic path for saving plots')
    parser.add_argument('--save_step', type=int, default=10, help='steps for model saving')
    parser.add_argument('--save_plot_step', type=int, default=5, help='steps for plot saving')
    parser.add_argument('--load_weight_path', type=str, default='None')
    
    parser.add_argument('--rank', type=int, default=0, help='rank of the process')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank of the process')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers for multi-gpu training')
    parser.add_argument('--gpu_ids', nargs='+', default=['0', '1', '2', '3', '4', '5', '6', '7'], help='gpu ids for training') 
    parser.add_argument('--world_size', type=int, default=8)

    return parser


def main(opts):
    init_for_distributed(opts)
    print("Distributed training Initialized")
    local_gpu_id = opts.gpu
    
    # only rank 0 will copy the file and create directory
    if opts.rank == 0:
        currtime = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y_%m_%d_%H%M%S") # need to plus 9 hours to utc time (ktz=utc+9)
        opts.project_name = f'{opts.project_name}/{currtime}'
            
        opts.save_path = opts.weight_dir + '/' + opts.project_name
        opts.save_plot_path = opts.plot_dir + '/' + opts.project_name

        checkdir(opts.save_path)
        checkdir(opts.save_plot_path)
        
        copy_file_list = [('/work/model/backbone/convnext_se/convnext_se.py', f'{opts.save_path}/convnext_se.py'),
                          ('/work/model/head/convnext_se_unet.py', f'{opts.save_path}/convnext_se_unet.py'),
                          ('/work/train/unsupervised/pixel_reconstruction/multigpu_unsupervised.py', f'{opts.save_path}/multigpu_unsupervised.py'),
                          (opts.model_config, f'{opts.save_path}/model_config.yaml'),
                          (opts.lr_policy_config, f'{opts.save_path}/lr_policy.yaml'),
                          ('/work/scripts/convnext_unet_unsupervised.sh', f'{opts.save_path}/convnext_unet_unsupervised.sh')]    
        
        copy_files(copy_file_list)
        train_logger, val_logger, process_logger = set_logger_for_training(opts)
        
    convnext_unet = create_unet(opts.model_config)
    convnext_unet = convnext_unet.to(local_gpu_id)
    convnext_unet = DDP(module=convnext_unet, device_ids=[local_gpu_id])
    
    if not opts.multi_data_mode:
        # CULaneSegDataset is the dataset that returns (image, mask) pair for training and validation
        # mask of the returned dataset excludes background channel and only contains foreground lane information
        train_dataset = MaskedCULaneDataset(cu_train_x, img_size=(224, 672), mask_ratio=opts.mask_ratio, mask_window_size=(opts.mask_window_h, opts.mask_window_w))
        val_dataset = MaskedCULaneDataset(cu_val_x, img_size=(224, 672), mask_ratio=opts.mask_ratio, mask_window_size=(opts.mask_window_h, opts.mask_window_w))
    else:
        # MaskedMultiDataset for multi-data training
        # currently, mask size for all dataset are same for (opts.mask_window_w, opts.mask_window_h)
        train_dataset = MaskedMultiDataset(unsuper_train_x, img_size=[(224, 672)]*4, target_size=[(224, 672)]*4, mask_window_size=[(opts.mask_window_h, opts.mask_window_w)]*4, mask_ratio=opts.mask_ratio)
        val_dataset = MaskedMultiDataset(unsuper_val_x, img_size=[(224, 672)]*4, target_size=[(224, 672)]*4, mask_window_size=[(opts.mask_window_h, opts.mask_window_w)]*4, mask_ratio=opts.mask_ratio)
    
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    val_sampler = DistributedSampler(dataset=val_dataset, shuffle=True)
    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opts.batch_size,
                                                   shuffle=False,
                                                   num_workers=int(opts.num_workers/opts.world_size),
                                                   sampler=train_sampler,
                                                   pin_memory=False)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=opts.val_batch_size,
                                                 shuffle=False,
                                                 num_workers=int(opts.num_workers/opts.world_size),
                                                 sampler=val_sampler,
                                                 pin_memory=False)
    
    
    convnext_unet, criterion, optimizer, scheduler, start_epoch, end_epoch, scheduler_type = set_weights(opts, convnext_unet, train_dataloader, process_logger)
        
    recorder = Recorder()
    recorder.register(['train_loss', 'train_pixel_reconst', 'val_loss', 'val_pixel_reconst']) # register each metric name for recording
    recorder.register_scalar(['running_loss', 'running_pixel_reconst']) # register temporal scalar value for recording

    train_num_prints = 25
    val_num_prints = 10

    total_train_iter = len(train_dataloader)
    total_val_iter = len(val_dataloader)

    print_threshold = int(total_train_iter/train_num_prints)
    val_print_threshold = int(total_val_iter/val_num_prints)
            
    if opts.loss_type=='mse':
        print_loss_str = 'mse loss'
        print_metrics_str = 'masked pixel-reconstruct'
    elif opts.loss_type=='masked_mse':
        print_loss_str = 'masked mse loss'
        print_metrics_str = 'pixel-reconstruct'

    for epoch in range(start_epoch, end_epoch): # epochs follows 0-index ordering
        convnext_unet.train()
        train_sampler.set_epoch(epoch)
        
        recorder.initialize(['train_loss', 'train_pixel_reconst']) # initialize the metrics for recording (training)
        recorder.initialize_scalar(['running_loss', 'running_pixel_reconst']) # initialize the running loss and lane score
        
        if opts.rank==0:
          train_logger.info(f'Epoch {epoch+1} started!')
        
        for i, batch in enumerate(train_dataloader, 1): # batch, i follows 1-index ordering
            image, mask = batch, batch.clone()
            image = image.to(local_gpu_id)
            mask = mask.to(local_gpu_id)
            
            mask_location = (image==0).all(dim=1) # (B, C, H, W) -> channel-wisely when all pixel-values are zero, the pixel is masked.
            
            optimizer.zero_grad()
            prediction_mask = convnext_unet(image)
            if opts.loss_type == 'mse':
                loss = criterion(prediction_mask, mask) # when use torch.nn.mse_loss()
            elif opts.loss_type =='masked_mse':
                loss = criterion(prediction_mask, mask, mask_location) # when use PixelReconstructLoss()
                
            loss.backward()
            optimizer.step()
            
            if opts.loss_type == 'mse':
                pixel_reconst = get_masked_pixel_reconst(prediction_mask, mask, mask_location)
            elif opts.loss_type == 'masked_mse':
                pixel_reconst = get_pixel_reconst(prediction_mask, mask)
                
            recorder.append('train_loss', loss.detach().item())
            recorder.append('train_pixel_reconst', pixel_reconst.detach().cpu())
                
            recorder.add('running_loss', loss.detach().item())
            recorder.add('running_pixel_reconst', pixel_reconst.detach().cpu())
            
            if scheduler_type in ['one_cycle']: # in case of pytorch OneCycleLR scheduler, it update itself step-wise (only training process will update the scheduler)
                scheduler.step()
                
            if i % print_threshold == 0:
                running_loss = recorder.get_scalar('running_loss')
                running_pixel_reconst = recorder.get_scalar('running_pixel_reconst')
                print_str = f"training [{epoch+1}:{i:5d}/{total_train_iter}] {print_loss_str}: {running_loss/print_threshold:.5f} {print_metrics_str}: {running_pixel_reconst/print_threshold:.5f} lr: {optimizer.param_groups[0]['lr']}"
                print(print_str)
                if opts.rank==0:
                    train_logger.info(print_str)
                recorder.initialize_scalar(['running_loss', 'running_pixel_reconst'])

        recorder.collect_statistic(['train_loss', 'train_pixel_reconst'])
        recorder.append_statistic('lr', optimizer.param_groups[0]['lr'])
        
        loss = recorder.get_statistic('train_loss')[-1]
        pixel_reconst = recorder.get_statistic('train_pixel_reconst')[-1]
        
        print_str = '=' * 50 + f"\nTrain Epoch: {epoch+1}, Loss: {loss:.5f}, Pixel-Reconstruct: {pixel_reconst:.5f}, LR: {optimizer.param_groups[0]['lr']}\n" + '=' * 50
        print(print_str)
        
        if opts.rank == 0:
            train_logger.info(print_str)
        
        if opts.rank==0: # only rank 0 will run validation
            convnext_unet.eval()
            recorder.initialize(['val_loss', 'val_pixel_reconst']) # initialize the metrics for recording (training)
            recorder.initialize_scalar(['running_loss', 'running_pixel_reconst']) # initialize the running loss and lane score
            
            val_logger.info(f'epoch: {epoch+1} start!')
            
            with torch.no_grad():
                for i, batch in enumerate(val_dataloader, 1):
                    image, mask = batch, batch.clone()
                    image = image.to(opts.rank)
                    mask = mask.to(opts.rank)
                    
                    mask_location = (image==0).all(dim=1) # (B, C, H, W) -> channel-wisely when all pixel-values are zero, the pixel is masked.
                    
                    prediction_mask = convnext_unet(image)
                    
                    if opts.loss_type == 'mse':
                        loss = criterion(prediction_mask, mask) # when using torch.nn.MseLoss() as loss
                    elif opts.loss_type =='masked_mse':
                        loss = criterion(prediction_mask, mask, mask_location)

                    if opts.loss_type == 'mse':
                        pixel_reconst = get_masked_pixel_reconst(prediction_mask, mask, mask_location)
                    elif opts.loss_type == 'masked_mse':
                        pixel_reconst = get_pixel_reconst(prediction_mask, mask)
                    
                    recorder.append('val_loss', loss.detach().item())    
                    recorder.add('running_loss', loss.detach().item())
                    
                    recorder.append('val_pixel_reconst', pixel_reconst.detach().cpu())
                    recorder.add('running_pixel_reconst', pixel_reconst.detach().cpu())
                    
                    if i % val_print_threshold == 0:
                        running_loss = recorder.get_scalar('running_loss')
                        running_pixel_reconst = recorder.get_scalar('running_pixel_reconst')
                        print_str = f"Validation [{epoch+1}:{i:5d}/{total_val_iter}] {print_loss_str}: {running_loss/val_print_threshold:.5f} {print_metrics_str}: {running_pixel_reconst/val_print_threshold:.5f} "
                        print(print_str)
                        
                        val_logger.info(print_str)
                        recorder.initialize_scalar(['running_loss', 'running_pixel_reconst'])

            recorder.collect_statistic(['val_loss', 'val_pixel_reconst'])
            loss = recorder.get_statistic('val_loss')[-1]
            pixel_reconst = recorder.get_statistic('val_pixel_reconst')[-1]
            
            print_str = '='*50 + f"\nVal Epoch: {epoch+1}, Loss: {loss:.5f}, Pixel-Reconstruction: {pixel_reconst:.5f}, LR: {optimizer.param_groups[0]['lr']}\n" + '='*50
            print(print_str)
            val_logger.info(print_str)
            
        
        # cosine_warmup_restart uses timm library whoose schedulers are updated by epoch-wise
        if scheduler_type in ['cosine_warmup_restart']:    
            # update scheduler
            scheduler.step(epoch)
        
        if opts.rank == 0: # only rank 0 will save the model and plots
            checkdir(opts.save_path+'/weights')
            if ((epoch+1) % opts.save_step) == 0 or (epoch+1)==end_epoch:
                print("+"*50)
                print(f"Saving Model to {opts.save_path}/weights...")
                print("+"*50)
                
                currtime = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y_%m_%d_%H%M%S")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict' : convnext_unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict(),
                    'cosumed_batch': (epoch+1) * len(train_dataloader)
                }, f'{opts.save_path}/weights/{currtime}_epoch{epoch+1}.pth') # epochs are saved as 1-index ordering (index 0 means initial state)
                
            if ((epoch+1) % opts.save_plot_step) == 0 or (epoch+1)==end_epoch:
                recorder.save_plots(key_list=[['train_loss', 'val_loss'], ['train_pixel_reconst', 'val_pixel_reconst'], ['lr']], 
                                    title_list=['Training and Validation Loss', 'Training and Validation Masked Pixel Reconstruction', 'Learning Rate'],
                                    x_ticks_list=[1,1,1],
                                    x_label_list=['Epochs', 'Epochs', 'Epochs'],
                                    y_label_list=['Loss', 'Normalized Pixel Reconstruction', 'LR'],
                                    save_path_list=[opts.save_plot_path+'/loss', opts.save_plot_path+'/reconst', opts.save_plot_path+'/lr'],
                                    fname_list=[f'{epoch}', f'{epoch}', f'{epoch}'])


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Unsupervised Training', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    # you can directly set the world size and number of worker
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    
    if opts.multi_data_mode: # if multi_data_mode is used, multiple lane dataset will be used for self-supervised training
        CULANE_PATH='/work/dataset/CULane'
        TUSIMPLE_PATH='/work/dataset/Tusimple'
        LLAMAS_PATH='/work/dataset/LLAMAS'
        BDD100K_PATH='/work/dataset/BDD100K'
        PATH_DICT={'CULANE':CULANE_PATH, 'TUSIMPLE': TUSIMPLE_PATH, 'LLAMAS': LLAMAS_PATH, 'BDD100K': BDD100K_PATH}
        
        data_dict = read_multi_segdata(PATH_DICT)
        cu_train_x, cu_train_y, cu_val_x, cu_val_y, cu_test_x = data_dict['CULANE']
        tu_trainval_x, tu_trainval_y, tu_test_x, tu_test_y = data_dict['TUSIMPLE']
        llamas_train_x, llamas_train_y, llamas_val_x, llamas_val_y, llamas_test_x = data_dict['LLAMAS']
        bdd_train_x, bdd_train_y, bdd_val_x, bdd_val_y, bdd_test_x = data_dict['BDD100K']
        
        
        # unsupervised settings
        # test data cannot be used in this case because there is no label available
        unsuper_train_x = {"culane": cu_train_x, "tusimple": tu_trainval_x,
                   "llamas": llamas_train_x + llamas_test_x,
                   "bdd": bdd_train_x + bdd_test_x}
        unsuper_val_x = {"culane": cu_val_x, "tusimple": tu_test_x, 
                        "llamas": llamas_val_x,
                        "bdd": bdd_val_x}
            
    else:
        CULANE_PATH='/work/dataset/CULane'
        train, val, test, lane_info = read_culane_segdata(CULANE_PATH)
        cu_train_x, cu_train_y = train
        cu_val_x, cu_val_y = val
        cu_test_x, _ = test
        cu_train_lane, cu_val_lane = lane_info

    main(opts)
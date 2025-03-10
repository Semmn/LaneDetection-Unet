import sys
sys.path.append('/work')

from train.segmentation.dataset import CULaneSegDataset

from train.utils.loss import DICELoss, LaneDICELoss
from train.utils.metrics import get_dice_score, get_iou_score, get_lane_score
from train.utils.weight_load import load_weight, load_weight_with_cp, set_weights
from train.utils.logger import set_logger_for_training, checkdir
from train.utils.distributed_training import setup_for_distributed, init_for_distributed
from train.utils.plotting import plot_figure
from train.utils.dataset import read_culane_segdata
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
import json
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


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for training') # batch size
    parser.add_argument('--val_batch_size', default=32, type=int, help='batch size for validation') # batch size for validation
    
    # path for scheduler configuration file: each configuration files includes specific parameter settings of scheduler
    parser.add_argument('--model_config', type=str, default='/work/train/segmentation/model_config/unet.yaml', help='unet configuration file')
    parser.add_argument('--train_config', type=str, default='/work/train/segmentation/train_config/segmentation_culane.yaml', help='training configuration file')
        
    parser.add_argument('--project_name', type=str, default=f'convnext_tiny_unet/culane', help='project name used for defining save path')
    parser.add_argument('--weight_dir', type=str, default='/work/checkpoints', help='basic path for saving model weights and files')
    parser.add_argument('--plot_dir', type=str, default='/work/plots', help='basic path for saving plots')
    parser.add_argument('--save_step', type=int, default=10, help='steps for model saving')
    parser.add_argument('--save_plot_step', type=int, default=5, help='steps for plot saving')
    parser.add_argument('--load_weight_path', type=str, help='None') # settings middle-saved checkpoint file path
    parser.add_argument('--pretrained_weight_path', type=str, help='None') # setting pre-trained weight path (will not load optimizer, scheduler state and set start_epoch to 0)
    
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
                          ('/work/train/segmentation/multigpu.py', f'{opts.save_path}/multigpu.py'),
                          (opts.model_config, f'{opts.save_path}/model_config.yaml'),
                          (opts.lr_policy_config, f'{opts.save_path}/lr_policy.yaml'),
                          ('/work/scripts/convnext_unet_training.sh', f'{opts.save_path}/convnext_unet_training.sh')]
            
        copy_files(copy_file_list)
        
        train_logger, val_logger, process_logger = set_logger_for_training(opts)

    convnext_unet = create_unet(opts.model_config)
    convnext_unet = convnext_unet.to(local_gpu_id)
    convnext_unet = DDP(module=convnext_unet, device_ids=[local_gpu_id])
    
    # instead of using albumentation Normalize(), we simply divide the given image by 255.
    train_transforms = A.Compose([
        A.Resize(224, 672),
        A.ColorJitter(brightness=(0.5, 1.5), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=1, translate_percent=(0.05, 0.05), rotate=(2,2), shear=(0, 0), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), normalization='standard', p=1.0), # normalize images
        ToTensorV2(p=1.0) # convert to pytorch Tensor format (HWC -> CHW)
    ])

    val_transforms = A.Compose([
        A.Resize(224, 672),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), normalization='standard', p=1.0),
        ToTensorV2(p=1.0)
    ])

    # CULaneSegDataset is the dataset that returns (image, mask) pair for training and validation
    # mask of the returned dataset excludes background channel and only contains foreground lane information
    train_dataset = CULaneSegDataset(train_x, train_y, transforms = train_transforms, num_classes=5, is_test=False)
    val_dataset = CULaneSegDataset(val_x, val_y, transforms=val_transforms, num_classes=5, is_test=False)
    # test_dataset = CULaneSegDataset(test_x, None, transforms=val_transforms, num_classes=5, is_test=True)
    
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    val_sampler = DistributedSampler(dataset=val_dataset, shuffle=False)
    # test_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
    
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
    # test_dataloader = torch.utils.data.DataLoader(test_dataset,
    #                                               batch_size=opts.val_batch_size,
    #                                               shuffle=False,
    #                                               num_workers=int(opts.num_workers/opts.world_size),
    #                                               sampler=test_sampler,
    #                                               pin_memory=True)
    

    convnext_unet, criterion, optimizer, scheduler, start_epoch, end_epoch, scheduler_type = set_weights(opts, convnext_unet, train_dataloader, process_logger)
            
    recorder = Recorder()
    recorder.register(['train_loss', 'train_dice', 'train_lane', 'val_loss', 'val_dice', 'val_lane']) # register each metric name for recording
    recorder.register_scalar(['running_loss', 'running_lane']) # register temporal scalar value for recording
    
    train_num_prints = 25 # number of prints during training process
    val_num_prints = 10 # number of prints during validation process

    total_train_iter = len(train_dataloader)
    total_val_iter = len(val_dataloader)

    print_threshold = int(total_train_iter/train_num_prints)
    val_print_threshold = int(total_val_iter/val_num_prints)
    
    for epoch in range(start_epoch, end_epoch): # epochs follows 0-index ordering
        convnext_unet.train()
        train_sampler.set_epoch(epoch)
        recorder.initialize_scalar(['running_loss', 'running_lane']) # initialize the running loss and lane score
        recorder.initialize(['train_loss', 'train_dice', 'train_lane']) # initialize the metrics for recording (training)
                
        if opts.rank == 0:
            train_logger.info(f'epoch: {epoch+1} start!')

        for i, batch in enumerate(train_dataloader, 1): # batch, i follows 1-index ordering
            image, mask = batch
            image = image.to(local_gpu_id)
            mask = mask.to(local_gpu_id)
            
            optimizer.zero_grad()
            
            prediction_mask = convnext_unet(image)
            
            loss = criterion(prediction_mask, mask)
            loss.backward()
            optimizer.step()
            
            dice_score = get_dice_score(prediction_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
            lane_score = get_lane_score(prediction_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
            
            recorder.append('train_loss', loss.detach().item())
            recorder.append('train_dice', dice_score)
            recorder.append('train_lane', lane_score)
            
            recorder.add('running_loss', loss.detach().item())
            recorder.add('running_lane', lane_score)

            if scheduler_type in ['one_cycle']: # in case of pytorch OneCycleLR scheduler, it update itself step-wise (only training process will update the scheduler)
                scheduler.step()
            
            if i % print_threshold == 0:
                running_loss = recorder.get_scalar('running_loss')
                running_lane = recorder.get_scalar('running_lane')
                print(f"training [{epoch+1}:{i:5d}/{total_train_iter}] loss: {running_loss/print_threshold:.5f} lane_score: {running_lane/print_threshold:.5f} lr: {optimizer.param_groups[0]['lr']}")
                if opts.rank==0:
                    train_logger.info(f"training [{epoch+1}:{i:5d}/{total_train_iter}] loss: {running_loss/print_threshold:.5f} lane_score: {running_lane/print_threshold:.5f} lr: {optimizer.param_groups[0]['lr']}")
                recorder.initialize_scalar(['running_loss', 'running_lane']) # set 'running_loss' and 'running_lane' as 0 value
        
        recorder.collect_statistic(['train_loss', 'train_dice', 'train_lane']) # collect statistic from registry (default to 'average')
        recorder.append_statistic('lr', optimizer.param_groups[0]['lr'])
        
        loss = recorder.get_statistic('train_loss')[-1]
        dice_coeff = recorder.get_statistic('train_dice')[-1]
        lane_coeff = recorder.get_statistic('train_lane')[-1]
        
        print("="*50)
        print(f"Train Epoch: {epoch+1}, Dice Score: {dice_coeff:.5f}, Lane Score: {lane_coeff:.5f}, Loss: {loss:.5f}, LR: {optimizer.param_groups[0]['lr']}")
        print("="*50)
        
        if opts.rank == 0:
            train_logger.info('=' * 50 + f"\nTrain Epoch: {epoch+1}, Dice Score: {dice_coeff:.3f}, Lane Score: {lane_coeff:.3f}, Loss: {loss:.3f}, LR: {optimizer.param_groups[0]['lr']}\n" + '=' * 50)
        
        if opts.rank==0: # only rank 0 will run validation
            convnext_unet.eval()
            recorder.initialize_scalar(['running_loss', 'running_lane']) # initialize the running loss and lane score
            recorder.initialize(['val_loss', 'val_dice', 'val_lane']) # initialize the metrics for recording (validating)
            
            val_logger.info(f'epoch: {epoch+1} start!')
            with torch.no_grad():
                for i, batch in enumerate(val_dataloader, 1):
                    image, mask = batch
                    image = image.to(opts.rank)
                    mask = mask.to(opts.rank)
                    
                    prediction_mask = convnext_unet(image)
                    loss = criterion(prediction_mask, mask)
                    
                    dice_score = get_dice_score(prediction_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
                    lane_score = get_lane_score(prediction_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
                    
                    recorder.append('val_loss', loss.detach().item())    
                    recorder.append('val_dice', dice_score)
                    recorder.append('val_lane', lane_score)
                    
                    recorder.add('running_loss', loss.detach().item())
                    recorder.add('running_lane', lane_score)
                    
                    if i % val_print_threshold == 0:
                        running_loss = recorder.get_scalar('running_loss')
                        running_lane = recorder.get_scalar('running_lane')
                        print(f"Validation [{epoch+1}:{i:5d}/{total_val_iter}] loss: {running_loss/val_print_threshold:.5f} lane_score: {running_lane/val_print_threshold:.5f}")
                        val_logger.info(f"Validation [{epoch+1}:{i:5d}/{total_val_iter}] loss: {running_loss/val_print_threshold:.5f} lane_score: {running_lane/val_print_threshold:.5f}")
                        recorder.initialize_scalar(['running_loss', 'running_lane']) # set 'running_loss' and 'running_lane' as 0 value
        
            recorder.collect_statistic(['val_loss', 'val_dice', 'val_lane']) # collect statistic from registry (default to 'average')
            loss = recorder.get_statistic('val_loss')[-1]
            dice_coeff = recorder.get_statistic('val_dice')[-1]
            lane_coeff = recorder.get_statistic('val_lane')[-1]
            
            print("="*50)
            print(f"Val Epoch: {epoch+1}, Dice Score: {dice_coeff:.5f}, Lane Score: {lane_coeff:.5f}, Loss: {loss:.5f}, LR: {optimizer.param_groups[0]['lr']}")
            print("="*50)
            val_logger.info('='*50 + f"\nVal Epoch: {epoch+1}, Dice Score: {dice_coeff:.5f}, Lane Score: {lane_coeff:.5f}, Loss: {loss:.5f}, LR: {optimizer.param_groups[0]['lr']}\n" + '='*50)
        
        # cosine_warmup_restart uses timm library whoose schedulers are updated to epoch-wise
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
                    'consumed_batch': (epoch+1) * len(train_dataloader) # number of total consumed iteration (number of total consumed batches during training)
                }, f'{opts.save_path}/weights/{currtime}_epoch{epoch+1}.pth') # epochs are saved as 1-index ordering (index 0 means initial state)
                
            if ((epoch+1) % opts.save_plot_step) == 0 or (epoch+1)==end_epoch:
                recorder.save_plots(key_list=[['train_loss', 'val_loss'], ['train_dice', 'val_dice'], ['train_lane', 'val_lane'], ['lr']], 
                                    title_list=['Training and Validation Loss', 'Training and Validation Dice Score', 'Training and Validation Lane Score', 'Learning Rate'],
                                    x_ticks_list=[1,1,1,1],
                                    x_label_list=['Epochs', 'Epochs', 'Epochs', 'Epochs'],
                                    y_label_list=['Loss', 'Dice Score', 'Lane Dice Score', 'LR'],
                                    save_path_list=[opts.save_plot_path+'/loss', opts.save_plot_path+'/dice', opts.save_plot_path+'/lane', opts.save_plot_path+'/lr'],
                                    fname_list=[f'{epoch}', f'{epoch}', f'{epoch}', f'{epoch}'])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Segmentation Training', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    # you can directly set the world size and number of worker
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
        
    BASEPATH='/work/dataset/CULane'
    train, val, test, lane_info = read_culane_segdata(BASEPATH)
    train_x, train_y = train
    val_x, val_y = val
    test_x, _ = test
    train_y_lane, val_y_lane = lane_info

    main(opts)
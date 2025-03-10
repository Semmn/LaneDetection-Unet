# append python search path
import sys, os

sys.path.append('/work/model/')
from head.convnext_se_unet import Encoder, DBlock, DStage, DStemNx, DStemStacked, DStemStaged, Decoder, UNet
from utils.conv_2d import adjust_padding_for_strided_output, DepthWiseSepConv
from utils.stochastic_depth_drop import create_linear_p, create_uniform_p
sys.path.append('/work/utils/')
from loss import DICELoss, LaneDICELoss
from metrics import get_dice_score, get_iou_score, get_lane_score

from dataset import CULaneSegDataset

import numpy as np
import argparse
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import json

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

# for using deepspeed library
import deepspeed
from deepspeed.accelerator import get_accelerator


def add_argument():
    parser = argparse.ArgumentParser(description='CULane')
    
    parser.add_argument('--val_batch_size', default=32, type=int, help='batch size for validation')
    parser.add_argument('--epoch', type=int, default=100, help='total number of epochs for training')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch of training')
    parser.add_argument('--local-rank', type=int, default=0, help='local rank of the process')
    parser.add_argument('--rank', type=int, default=0, help='rank of the process')
    parser.add_argument('--ds_config_path', type=str, default='/work/train/CULane/ds_config.json', help='deepspeed configuration json file path')
    
    parser.add_argument('--project_name', type=str, default=f'convnext_tiny_unet/culane', help='project name used for defining save path')
    parser.add_argument('--weight_dir', type=str, default='/work/checkpoints', help='basic path for saving model weights and files')
    parser.add_argument('--plot_dir', type=str, default='/work/plots', help='basic path for saving plots')
    parser.add_argument('--save_step', type=int, default=10, help='steps for model saving')
    parser.add_argument('--save_plot_step', type=int, default=5, help='steps for plot saving')
    
    parser.add_argument('--ckpt_id', type=str, help='checkpoint id for loading model. in this case, use saved epoch number')
    
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    return args

# check whether the directory exists or not
def checkdir(path):
    if not os.path.exists(path):
        print("creating directories: ", path)
        os.makedirs(path)
        
def save_plots(train_history, val_history, train_dice, val_dice, train_lane, val_lane, epoch, save_plot_path):
    checkdir(save_plot_path)
    checkdir(save_plot_path+'/loss')
    checkdir(save_plot_path+'/dice')
    checkdir(save_plot_path+'/lane')
    
    
    try:
        # plot loss
        plt.figure(figsize=(12, 8))
        plt.title('Training and Validation Loss')
        plt.plot(range(len(train_history)), train_history, label='train loss')
        plt.plot(range(len(val_history)), val_history, label='val loss')
        
        tick_step = 1
        plt.xticks(range(0, len(train_history)+1, tick_step))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{save_plot_path}/loss/{epoch}.png')
        
        # plot dice score
        plt.figure(figsize=(12, 8))
        plt.title('Training and Validation Dice Score')
        plt.plot(range(len(train_dice)), train_dice, label='train dice coeff')
        plt.plot(range(len(val_dice)), val_dice, label='val dice coeff')
        plt.xticks(range(0, len(train_dice)+1, tick_step))
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.savefig(f'{save_plot_path}/dice/{epoch}.png')
        
        # plot lane dice score
        plt.figure(figsize=(12, 8))
        plt.title('Training and Validation Lane-Dice Score')
        plt.plot(range(len(train_lane)), train_lane, label='train lane-dice coeff')
        plt.plot(range(len(val_lane)), val_lane, label='val lane-dice coeff')
        plt.xticks(range(0, len(train_lane)+1, tick_step))
        plt.xlabel('Epochs')
        plt.ylabel('Lane Dice Score')
        plt.legend()
        plt.savefig(f'{save_plot_path}/lane/{epoch}.png')
        
        # need to close all plots to save memory
        plt.close('all')
    except:
        print("[Error]: Error while saving plots.....")


def get_ds_config(args):
    """Get the DeepSpeed configuration dictionary."""
    
    with json.open(args.ds_config_path, 'r') as fr:
        ds_config = json.load(fr)
            
    return ds_config


def main(opts):
    
    deepspeed.init_distributed()
    print("Distributed training Initialized")
    _local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(_local_rank)
    
    if opts.rank == 0:
        currtime = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y_%m_%d_%H%M%S") # need to plus 9 hours to utc time (ktz=utc+9)
        opts.project_name = f'{opts.project_name}/{currtime}'
            
        opts.save_path = opts.weight_dir + '/' + opts.project_name
        opts.save_plot_path = opts.plot_dir + '/' + opts.project_name

        checkdir(opts.save_path)
        checkdir(opts.save_plot_path)
            
        # copying files for saving experiments settings
        os.system(f'cp /work/model/backbone/convnext_se/convnext_se.py {opts.save_path}/convnext_se.py')
        os.system(f'cp /work/model/head/convnext_se_unet.py {opts.save_path}/convnext_se_unet.py')
        os.system(f'cp /work/train/CULane/deepspeed.py {opts.save_path}/deepspeed.py')
        os.system(f'cp /work/train/CULane/ds_config.json {opts.save_path}/ds_config.json')
        os.system(f'cp /work/scripts/convnext_unet_deepspeed.sh {opts.save_path}/convnext_unet_deepspeed.sh')

    #########################################################
    ################# initialize encoder ####################
    #########################################################
    dp_list, dp_mode = create_linear_p([3,3,6,3], 'batch', 0.5) # creates linearly decaying stochastic depth drop probability
    # dp_list, dp_mode = create_uniform_p([3,3,9,3], 'batch', 0.001) # create constant stochastic depth drop probability
        
    unet_encoder = Encoder(num_blocks=[3,3,6,3], input_channels=3, stem_kersz=(4,4), stem_stride=(4,4), img_hw=[(56, 168), (28, 84), (14, 42), (7,21)], main_channels=[48, 96, 192, 384], expansion_dim=[48*4, 96*4, 192*4, 384*4],
                                kernel_sz=[(7,7),(7,7),(7,7),(7,7)], stride=[(1,1),(1,1),(1,1),(1,1)], padding=['same', 'same', 'same', 'same'], dilation=[1,1,1,1], groups=[1,1,1,1], droprate=dp_list, drop_mode=dp_mode,
                                use_se=[True, True, True, True], squeeze_ratio=16, transition_kersz=[-1, (2,2),(2,2),(2,2)], transition_stride=[-1, (2,2), (2,2), (2,2)], norm_mode='layer_norm', device='cuda')

    ############################################################
    ################# intialize decoder & head #################
    ############################################################

    # class DStem4x(torch.nn.Module):
    #     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
    #                  dilation=1, device='cuda'):

    # class DStemStacked(torch.nn.Module):
    #     def __init__(self, in_channels:list, out_channels:list, kernel_size:list, stride:list, padding:list, output_padding:list, groups:list,
    #                  dilation:list, device):

    # class DStemStaged(torch.nn.ModuleList):
    #     def __init__(self, num_blocks:list, img_hw:list, input_channels, main_channels, expansion_channels, kernel_sz, stride, padding, dilation, groups, droprate, drop_mode,
    #                  use_se, squeeze_ratio, transition_kersz, transition_stride, transition_padding, transition_out_padding, device='cuda'):

    # head_4x = DStemNx(in_channels=96, out_channels=128, kernel_size=(16, 24), stride=(8, 16), padding=(1,0), output_padding=(0,0), groups=1, dilation=1, device='cuda')
    # head_stack = DStemStacked(in_channels=[96, 128], out_channels=[128, 256], kernel_size=[(4,4), (12,12)], stride=[(2,4), (4,4)], padding=[(1,0), (1,0)], output_padding=[(0,0), (0,0)], 
    #                           groups=[1,1], dilation=[1,1], device='cuda')

    # droprate, drop_mode = create_linear_p([2,2,2,2,1], dp_mode='batch', last_p=0.25)
    # head_stage = DStemStaged(num_blocks=[2, 1], img_hw=[(112, 336), (224, 672)], input_channels=48, main_channels=[48, 48], expansion_channels=[48*4, 48*4],
    #                          kernel_sz=[(7,7), (7,7)], stride=[(1,1), (1,1)], padding=['same', 'same'], dilation=[1, 1], groups=[1,1], 
    #                          droprate=droprate[3:], drop_mode=drop_mode[3:],
    #                          use_se=[True, True], squeeze_ratio=16, transition_kersz=[(7,7)]*2, transition_stride=[(2,2)]*2, transition_padding=[(3,3)]*2,
    #                          transition_out_padding=[(1,1)]*2, norm_mode='layer_norm', device='cuda')

    droprate, drop_mode = create_linear_p([2,2,2], dp_mode='batch', last_p=0.25)
    head_4x = DStemNx(in_channels=96, out_channels=48, kernel_size=(4,4), stride=(4,4), padding=(0,0), output_padding=(0,0), groups=1, dilation=1, device='cuda')
    unet_decoder = Decoder(num_blocks=[2,2,2], img_hw=[(14, 42), (28, 84), (56, 168)], main_channels=[384, 192, 96], expansion_dim=[384*4, 192*4, 96*4],
                    kernel_sz=[(7,7)]*3, stride=[(1,1)]*3, padding=['same']*3, dilation=[1]*3, groups=[1]*3, droprate=droprate, drop_mode=drop_mode,
                    use_se=[True]*3, squeeze_ratio=16, encoder_channels=[384, 192, 96, 48], transition_kersz=[(7,7)]*3, transition_stride=[(2,2)]*3,
                    transition_padding=[(3,3)]*3, transition_out_padding=[(1,1)]*3, norm_mode='layer_norm', head=head_4x, device='cuda')


    ############################################################
    ################# intialize unet model #####################
    ############################################################
    # if num_cls==2 -> binary segmentation. to classify lane position on the road separately, 
    # we need to at least make 5 masks for 4 lanes each presenting the lane position of the road. (0: background, 1: lane1, 2: lane2, 3: lane3, 4: lane4)
    convnext_unet = UNet(encoder=unet_encoder, decoder=unet_decoder, num_cls=5, output_mode='probs')

        
    train_transforms = A.Compose([
        A.Resize(224, 672),
        A.ColorJitter(brightness=(0.5, 1.5), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=1, translate_percent=(0.05, 0.05), rotate=(2,2), shear=(0, 0), p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), normalization='standard', p=1.0), # normalize images
        ToTensorV2(p=1.0) # convert to pytorch Tensor format (HWC -> CHW)
    ])

    val_transforms = A.Compose([
        A.Resize(224, 672),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), normalization='standard', p=1.0),
        ToTensorV2(p=1.0)
    ])

    train_dataset = CULaneSegDataset(train_x, train_y, transforms = train_transforms, num_classes=5, is_test=False)
    val_dataset = CULaneSegDataset(val_x, val_y, transforms=val_transforms, num_classes=5, is_test=False)
    # test_dataset = CULaneSegDataset(test_x, None, transforms=val_transforms, num_classes=5, is_test=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.val_batch_size, shuffle=True)
    
    # initialize deepspeed model
    # 1) distributed model
    # 2) distributed data loader
    # 3) deepspeed optimizer
    ds_config = get_ds_config(opts)
    parameters = filter(lambda p: p.requires_grad, convnext_unet.parameters())
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(args=opts, model=convnext_unet, model_parameters=parameters, training_data=train_dataset, config=ds_config,)
    
    if opts.start_epoch != 0: # if start epoch of training is not zero
        # load checkpoints
        _, client_sd = model_engine.load_checkpoint(opts.save_path + '/weights', opts.ckpt_id) # ckpt_id is epoch (1-index ordered)
        epoch = client_sd['epoch']
        if opts.start_epoch != epoch: # opts.start_epoch should also be 1-index ordered.
            raise Exception("saved epoch and opts.start_epoch are not matched!")
    else:
        client_sd = {}
        
    #get the local device name(str) and local rank (int)
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank
    
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half
        
    # criterion = torch.nn.CrossEntropyLoss() # cross entropy loss for training
    # criterion = DICELoss(weights=[], num_cls=5) for weighted Dice Loss
    # criterion = DICELoss(num_cls=4) # custom dice loss for segmentation tasks
    criterion = LaneDICELoss(num_cls=5) # background information will be treated as 0-th channel

    train_history = []
    val_history = []

    train_dice = []
    val_dice = []
    
    train_lane = []
    val_lane = []
    
    train_num_prints = 25
    val_num_prints = 10

    total_train_iter = len(train_dataloader)
    total_val_iter = len(val_dataloader)

    print_threshold = int(total_train_iter/train_num_prints)
    val_print_threshold = int(total_val_iter/val_num_prints)

    for epoch in range(opts.start_epoch, opts.epoch): # epochs follows 0-index ordering
        model_engine.train()
        running_loss = 0.0
        running_lane = 0.0
        loss_container = []
        dice_container = []
        lane_container = []
        for i, batch in enumerate(train_dataloader, 1): # batch, i follows 1-index ordering
            image, mask = batch
            image = image.to(local_device)
            mask = mask.to(local_device)
            
            if target_dtype != None:
                image = image.to(target_dtype)
            
            pred_mask = model_engine(image)
            loss = criterion(pred_mask, mask)
            
            model_engine.backward(loss)
            model_engine.step()
            
            loss_container.append(loss.detach().item())
            
            dice_coeff = get_dice_score(pred_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
            dice_container.append(dice_coeff)
            
            lane_coeff = get_lane_score(pred_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
            lane_container.append(lane_coeff)
            
            running_loss += loss.detach().item()
            running_lane += lane_coeff
            if local_rank == 0 and i % print_threshold == 0:
                print(f"training [{epoch+1}:{i:5d}/{total_train_iter}] loss: {running_loss/print_threshold:.3f} lane-dice: {running_lane/print_threshold:.3f}")
                running_loss = 0.0
                running_lane = 0.0
            
        loss = np.mean(loss_container)
        dice_coeff = np.mean(dice_container)
        lane_coeff = np.mean(lane_container)
        train_history.append(loss)
        train_dice.append(dice_coeff)
        train_lane.append(lane_coeff)
        
        if local_rank == 0:
            print("="*50)
            print(f"Train Epoch: {epoch+1}, Dice Score: {dice_coeff:.3f}, Loss: {loss:.3f}, Lane Dice: {lane_coeff:.3f}")
            print("="*50)
            
        model_engine.eval()
        running_loss = 0.0
        running_lane = 0.0
        loss_container = []
        dice_container = []
        lane_container = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader, 1):
                image, mask = batch
                image = image.to(local_device)
                mask = mask.to(local_device)
                
                if target_dtype != None:
                    image = image.to(target_dtype)
                
                pred_mask = model_engine(image)
                loss = criterion(pred_mask, mask)
                
                loss_container.append(loss.detach().item())
                
                dice_coeff = get_dice_score(pred_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
                dice_container.append(dice_coeff)
                
                lane_coeff = get_lane_score(pred_mask.cpu().detach().numpy(), mask.cpu().detach().numpy())
                lane_container.append(lane_coeff)
                
                running_loss += loss.detach().item()
                running_lane += lane_coeff 
                if local_rank == 0 and i % val_print_threshold == 0:
                    print(f"Validation [{epoch+1}:{i:5d}/{total_val_iter}], loss: {running_loss/val_print_threshold:.3f}, lane-dice: {running_lane/print_threshold:.3f}")
                    running_loss = 0.0
                    running_lane = 0.0
        
        loss = np.mean(loss_container)
        dice_coeff = np.mean(dice_container)
        lane_coeff = np.mean(lane_container)
        val_history.append(loss)
        val_dice.append(dice_coeff)
        val_lane.append(lane_coeff)
        
        if local_rank == 0:
            print("="*50)
            print(f"Val Epoch: {epoch+1}, Dice Score: {dice_coeff:.3f}, Loss: {loss:.3f}, Lane Dice: {lane_coeff:.3f}")
            print("="*50)
            checkdir(opts.save_path+'/weights')
            
            if ((epoch+1) % opts.save_step) == 0 or (epoch+1)==opts.epoch:
                print("+"*50)
                print(f"Saving Model to {opts.save_path}/weights...")
                print("+"*50)
                
                client_sd['epoch'] = epoch
                ckpt_id = epoch+1 # checkpoint id is epoch following 1-index ordering
                model_engine.save_checkpoint(opts.save_path+'/weights', ckpt_id, client_sd=client_sd)
                
            if ((epoch+1) % opts.save_plot_step) == 0 or (epoch+1)==opts.epoch:
                save_plots(train_history, val_history, train_dice, val_dice, train_lane, val_lane, epoch+1, opts.save_plot_path) # epochs as 1-index ordering


if __name__ == '__main__':
    
    # enabled torch.backends.cudnn    
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True

    opts = add_argument()
    
    # you can directly set the world size and number of worker
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4
    
    BASE_PATH='/work/dataset/CULane'
    with open(BASE_PATH+'/list/train_gt.txt', 'r') as fr:
        train_list = fr.readlines()
    with open(BASE_PATH+'/list/val_gt.txt', 'r') as fr:
        val_list = fr.readlines()
    with open(BASE_PATH+'/list/test.txt', 'r') as fr:
        test_list = fr.readlines()


    train_x, train_y, train_y_lane = [], [], [] # train_y_lane denotes whether lane exist or not.
    val_x, val_y, val_y_lane = [], [], []
    test_x = []
    BASE_PATH = '/work/dataset/CULane'
    for path in train_list:
        tokens = path.split(' ')
        train_x.append(BASE_PATH+tokens[0])
        train_y.append(BASE_PATH+tokens[1])
        train_y_lane.append([int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5].strip())])

    for path in val_list:
        tokens = path.split(' ')
        val_x.append(BASE_PATH+tokens[0])
        val_y.append(BASE_PATH+tokens[1])
        val_y_lane.append([int(tokens[2]), int(tokens[3]), int(tokens[4]), int(tokens[5].strip())])

    for path in test_list:
        test_x.append(BASE_PATH+path.strip())

    main(opts)
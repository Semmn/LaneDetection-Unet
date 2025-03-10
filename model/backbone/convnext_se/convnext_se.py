import numpy as np
import torch
import torch.nn as nn
import torchvision



def adjust_padding_for_strided_output(kernel_size, stride):
        # range of padding: (kernel_size - stride)/2 <= padding < (kernel_size)/2
        # this ensures that output convolution channel size are 1/(transition_stride) of the input size
        bottom_inclusive_h = (kernel_size[0] -stride[0])/2
        ceil_exclusive_h = (kernel_size[0])/2
        
        bottom_inclusive_w = (kernel_size[1] - stride[1])/2
        ceil_exclusive_w = (kernel_size[1])/2
        
        padding_h = -1
        padding_w = -1
        
        nearest_nat = np.floor(bottom_inclusive_h)
        if nearest_nat == bottom_inclusive_h: # in case of (1 <= p < 1.5) -> p == np.rint(1)
            padding_h = int(nearest_nat)
        else:
            if nearest_nat < bottom_inclusive_h: # in case of (1.5 <= p < 2.5) -> p == np.rint(1.5)+1
                padding_h = int(nearest_nat+1)
            else:
                raise Exception("Invalid transition stride and kernel height size. Check if kernel_size >= stride.")
        
        nearest_nat = np.floor(bottom_inclusive_w)
        if nearest_nat == bottom_inclusive_w: # in case of (1 <= p < 1.5) -> p == np.rint(1)
            padding_w = int(nearest_nat)
        else:
            if nearest_nat < bottom_inclusive_w: # in case of (1.5 <= p < 2.5) -> p == np.rint(1.5)+1
                padding_w = int(nearest_nat + 1)
            else:
                raise Exception("Invalid transition stride and kernel width size. Check if kernel_size >= stride.")
        return (padding_h, padding_w)
    

# create linearly decaying stochastic depth drop probability list, mode
# earlier layers should be more reliably present as it extract low-level features that will be used by later layers
def create_linear_p(block_nlist:list, dp_mode:str, last_p:float):
    
    droprate = []  # probability of stochastic drop depth for all blocks in all stages
    drop_mode = [] # mode of stochastic drop depth for all blocks in all stages
    
    l_count = 0
    num_of_l = sum(block_nlist)
    
    for s_index, num_block in enumerate(block_nlist,0): # starts from stage 0
        p_list = []
        mode_list = []
        for b_index in range(num_block): # starts from block 0
            p = (l_count/num_of_l) * (1-last_p)
            p_list.append(p)
            mode_list.append(dp_mode)
            l_count+=1
        droprate.append(p_list)
        drop_mode.append(mode_list)
    return droprate, drop_mode

# create uniform stochastic depth drop probability list, mode
def create_uniform_p(block_nlist:list, dp_mode:str, uniform_p:int):
    droprate = []
    drop_mode = []
    
    for s_index, num_block in enumerate(block_nlist, 0):
        p_list = []
        mode_list = []
        for b_index in range(num_block):
            p_list.append(uniform_p)
            mode_list.append(dp_mode)
        droprate.append(p_list)
        drop_mode.append(mode_list)

    return droprate, drop_mode


class DepthWiseSepConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sz, stride, padding, dilation, device='cuda'):
        super().__init__()
        self.depthwise_inc = in_channels
        self.depthwise_ouc = in_channels
        self.depthwise_kersz = kernel_sz
        self.depthwise_std = stride
        self.depthwise_pad = padding
        self.depthwise_dil = dilation
        self.depthwise_grp = in_channels
        
        self.pointwise_inc = in_channels
        self.pointwise_ouc = out_channels
        self.pointwise_kersz = (1,1)
        self.pointwise_std = (1,1)
        self.pointwise_pad = 0
        self.pointwise_dil = 1
        self.pointwise_grp = 1
        
        self.device = device
        self.depthwise_conv = torch.nn.Conv2d(in_channels=self.depthwise_inc, out_channels=self.depthwise_ouc, kernel_size=self.depthwise_kersz,
                                              stride=self.depthwise_std, padding=self.depthwise_pad, dilation=self.depthwise_dil, groups=self.depthwise_grp, bias=True, padding_mode='zeros',
                                              device=self.device)
        self.pointwise_conv = torch.nn.Conv2d(in_channels=self.pointwise_inc, out_channels=self.pointwise_ouc, kernel_size=self.pointwise_kersz, stride=self.pointwise_std,
                                              padding=self.pointwise_pad, dilation=self.pointwise_dil, groups=self.pointwise_grp, bias=True, padding_mode='zeros',
                                              device=self.device)
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        
        return x


class Stem(torch.nn.Module):
    def __init__(self, stem_in_channels, stem_out_channels, stem_kernel_sz, stem_stride, stem_padding, stem_dilation, stem_groups, device='cuda'):
        super().__init__()
        self.in_channels = stem_in_channels
        self.out_channels = stem_out_channels
        
        self.kernel_sz = stem_kernel_sz
        self.stride = stem_stride
        self.padding = stem_padding
        
        self.dilation = stem_dilation
        self.groups = stem_groups
        
        self.device=device
        
        self.stem_conv = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_sz, stride=self.stride, padding=self.padding,
                                         dilation=self.dilation, groups=self.groups, bias=True, padding_mode='zeros', device=self.device)
    
    def forward(self, x):
        x = self.stem_conv(x)
        return x
        
class Block(torch.nn.Module):
    def __init__(self, img_hw, in_channels, out_channels, kernel_sz, stride, padding, dilation, groups, droprate, drop_mode, 
                 use_se:bool, squeeze_ratio, transition=False, transition_channels=-1, transition_kersz=-1, transition_stride=-1,
                 norm_mode='layer_norm', device='cuda'):
        super().__init__()
        
        self.img_h = img_hw[0]
        self.img_w = img_hw[1]
        
        self.block_channels = in_channels
        self.block_out_channels = out_channels
        self.kernel_sz = kernel_sz
        self.stride = stride
        self.padding = padding
        self.dilation = dilation 
        self.groups = groups # this controls the number of groups for pointwise convolution
        
        self.droprate = droprate # stochastic depth drop rate
        if drop_mode not in ['row', 'batch']:
            raise Exception("drop_mode must be 'row' or 'batch'")
        self.drop_mode = drop_mode
        
        self.use_se = use_se # whether to use se operation in the block
        self.squeeze_r = squeeze_ratio
        
        if norm_mode not in ['batch_norm', 'layer_norm']:
            raise Exception(f"Unsupported Normaliztion method: {norm_mode}. Normalization method must be either 'batch_norm', 'layer_norm'.")
        self.norm_mode = norm_mode
        
        self.device = device
        self.transition = transition
        
        self.transition_channels = transition_channels # number of last channels before transitioning
        self.transition_kersz = transition_kersz # kernel size for transitioning
        self.transition_stride = transition_stride # number of strides for shape matching in transitioning process
        
        if self.transition:
            self.depthwisesep_conv = DepthWiseSepConv(in_channels=self.transition_channels, out_channels=self.block_channels,
                                                    kernel_sz=self.kernel_sz, stride=self.transition_stride, padding=adjust_padding_for_strided_output(self.kernel_sz, self.transition_stride), dilation=1, device=self.device)
            self.transition_conv = torch.nn.Conv2d(in_channels=self.transition_channels, out_channels=self.block_channels, kernel_size=self.transition_kersz, stride=self.transition_stride, 
                                                    padding=adjust_padding_for_strided_output(self.transition_kersz, self.transition_stride), dilation=1, groups=1, device=self.device)
        else:
            self.depthwisesep_conv = DepthWiseSepConv(in_channels=self.block_channels, out_channels=self.block_channels, kernel_sz=self.kernel_sz, stride=self.stride, padding=self.padding, dilation=self.dilation, device=self.device)
        self.pointwise_1 = torch.nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=1, groups=self.groups, device=self.device)
        self.pointwise_2 = torch.nn.Conv2d(in_channels=self.block_out_channels, out_channels=self.block_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=1, groups=self.groups, device=self.device)
        
        if self.use_se:
            self.fc1 = torch.nn.Conv2d(in_channels=self.block_channels, out_channels=int(self.block_channels/self.squeeze_r), kernel_size=(1,1), stride=(1,1), padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Conv2d(in_channels=int(self.block_channels/self.squeeze_r), out_channels=self.block_channels, kernel_size=(1,1), stride=(1,1), padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device)
            self.sigmoid = torch.nn.Sigmoid()

        # use batch normalization when dealing with large input size
        if self.norm_mode == 'layer_norm':
            self.normalization = torch.nn.LayerNorm([self.block_channels, self.img_h, self.img_w], device=self.device)
        elif self.norm_mode == 'batch_norm':
            self.normalization = torch.nn.BatchNorm2d(num_features=self.block_channels, device=self.device)
        self.gelu = torch.nn.GELU()
        
    def forward(self, x):
        
        highway = x
        
        x = self.depthwisesep_conv(x)
        x = self.normalization(x)
        x = self.pointwise_1(x)
        x = self.gelu(x)
        x = self.pointwise_2(x)
        
        if self.use_se:
            squeeze = torch.nn.functional.avg_pool2d(x, x.size()[2:]) # same as global average pooling
            excitation = self.fc1(squeeze)
            excitation = self.relu(excitation)
            excitation = self.fc2(excitation)
            attention = self.sigmoid(excitation) # (B, C, 1, 1)
            x = x * attention
            
        if self.transition: # if it is in transitioning process
            highway = self.transition_conv(highway)
        
        highway = highway + torchvision.ops.stochastic_depth(x, p=self.droprate, mode=self.drop_mode, training=self.training) # stochastic depth drop for residuals
        return highway
    

class Stage(torch.nn.Module):
    def __init__(self, transition_flag, num_blocks:int, img_hw:list, in_channels, out_channels, kernel_sz, stride, padding, dilation, groups, droprate:list, drop_mode:list, use_se:bool, squeeze_ratio:int, 
                 transition_channels=-1, transition_kersz=-1, transition_stride=-1, norm_mode='layer_norm', device='cuda'):
        super().__init__()
        
        self.transition_flag = transition_flag # whether stage includes transition=True blocks
        self.num_blocks = num_blocks
        self.img_h = img_hw[0] # image height, width corresponding to specific stage
        self.img_w = img_hw[1]
        self.stage_channels = in_channels
        self.stage_out_channels = out_channels
        self.kernel_sz = kernel_sz
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.droprate=droprate
        for i in range(len(drop_mode)):
            if drop_mode[i] not in ['row', 'batch']:
                raise Exception("drop_mode must be 'row' or 'batch'")
        self.drop_mode=drop_mode
        self.use_se=use_se # whether to use se operation in the blocks
        self.squeeze_ratio=squeeze_ratio
        
        if norm_mode not in ['layer_norm', 'batch_norm']:
            raise Exception(f"Unsupported normalization method: {norm_mode}. Must be either 'layer_norm' or 'batch_norm'")
        self.norm_mode = norm_mode
        self.device = device
        
        self.transition_channels = transition_channels
        self.transition_kersz = transition_kersz
        self.transition_stride = transition_stride
        
        
        self.blocks = []
        for i in range(num_blocks):
            if i == 0 and self.transition_flag: # the start block of stage process transitioning.
                self.blocks.append(Block([self.img_h, self.img_w], in_channels=self.stage_channels, out_channels=self.stage_out_channels, kernel_sz=self.kernel_sz, stride=self.stride, padding=self.padding, dilation=self.dilation,
                                            groups=self.groups, droprate=self.droprate[i], drop_mode=self.drop_mode[i], use_se=self.use_se, squeeze_ratio=self.squeeze_ratio, transition=True, transition_channels=self.transition_channels, 
                                            transition_kersz=self.transition_kersz, transition_stride=self.transition_stride, norm_mode=self.norm_mode, device=self.device))
            else:
                self.blocks.append(Block([self.img_h, self.img_w], in_channels=self.stage_channels, out_channels=self.stage_out_channels, kernel_sz=self.kernel_sz, stride=self.stride, padding=self.padding, dilation=self.dilation,
                                            groups=self.groups, droprate=self.droprate[i], drop_mode=self.drop_mode[i], use_se=self.use_se, squeeze_ratio=self.squeeze_ratio, norm_mode=self.norm_mode, device=self.device))
        self.blocks = torch.nn.ModuleList(self.blocks)
    
    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
        return x


class ConvNextV1(torch.nn.Module):
    def __init__(self, num_blocks:list, input_channels:int, stem_kersz:tuple, stem_stride:tuple, 
                 img_hw:list, main_channels:list, expansion_dim:list, kernel_sz:list, stride:list, padding:list, dilation:list, groups:list, droprate:list, drop_mode:list, use_se:list, squeeze_ratio:int,
                 transition_kersz:list, transition_stride:list, norm_mode:str, device='cuda'):
        
        super().__init__()
        
        self.num_blocks = num_blocks # number of blocks for each stage
        self.input_channels = input_channels # mostly RGB 3 channel images
        self.stem_kersz = stem_kersz # kernel size for stem layer
        self.stem_stride = stem_stride # stride for stem layer
        self.img_hw = img_hw # representative image height and width for each stage
        self.main_channels = main_channels # main channels for each stage
        self.expansion_dim = expansion_dim # expansion dimension for each stage
        self.kernel_sz = kernel_sz # kernel size for each stage
        self.stride=stride # stride for each stage
        self.padding=padding # padding for each stage
        self.dilation=dilation # dilation for each stage
        self.groups=groups # number of groups for each stage
        self.droprate=droprate # constant droprate for all stage
        self.drop_mode=drop_mode # drop_mode is same for all stage
        self.use_se=use_se # flag for using se operation in each stage
        self.squeeze_ratio=squeeze_ratio # squeeze_ratio is same for all stage
        self.transition_kersz=transition_kersz # transition kernel size for each stage
        self.transition_stride=transition_stride # transition stride for each stage
        
        if norm_mode not in ['batch_norm', 'layer_norm']:
            raise Exception(f"Unsupported normalization method: {norm_mode}. Must be either 'layer_norm' or 'batch_norm'")
        self.norm_mode = norm_mode # normalization mode (layer_norm: torch.nn.LayerNorm, batch_norm: torch.nn.BatchNorm2d)
        
        self.device=device
        self.num_stages = len(num_blocks)
        
        self.stem = Stem(stem_in_channels=self.input_channels, stem_out_channels=main_channels[0], stem_kernel_sz=self.stem_kersz, stem_stride=self.stem_stride,
                         stem_padding=adjust_padding_for_strided_output(self.stem_kersz, self.stem_stride), stem_dilation=1, stem_groups=1, device=self.device)
        self.stages = []
        for i in range(self.num_stages):
            if i == 0:
                self.stages.append(Stage(transition_flag=False, num_blocks=self.num_blocks[i], img_hw=self.img_hw[i], in_channels=self.main_channels[i], out_channels=self.expansion_dim[i],
                                    kernel_sz=self.kernel_sz[i], stride=self.stride[i], padding=self.padding[i], dilation=self.dilation[i], groups=self.groups[i], droprate=self.droprate[i],
                                    drop_mode=self.drop_mode[i], use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, norm_mode=self.norm_mode, device=self.device))
            else:
                self.stages.append(Stage(transition_flag=True, num_blocks=self.num_blocks[i], img_hw=self.img_hw[i], in_channels=self.main_channels[i], out_channels=self.expansion_dim[i],
                                        kernel_sz=self.kernel_sz[i], stride=self.stride[i], padding=self.padding[i], dilation=self.dilation[i], groups=self.groups[i], droprate=self.droprate[i],
                                        drop_mode=self.drop_mode[i], use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, transition_channels=self.main_channels[i-1], transition_kersz=self.transition_kersz[i],
                                        transition_stride=self.transition_stride[i], norm_mode=self.norm_mode, device=self.device))
        self.stages = torch.nn.ModuleList(self.stages)
        
    def forward(self, x):
        x = self.stem(x)
        for i in range(self.num_stages):
            x = self.stages[i](x)
        return x


class ConvNextV1CLS(torch.nn.Module):
    def __init__(self, backbone:ConvNextV1, num_cls=1000, output_mode='logits', device='cuda'):
        super().__init__()
        
        self.backbone=backbone
        self.num_cls=num_cls
        if output_mode not in ['logits', 'activation']:
            raise Exception("output_mode must be 'logits' or 'activation'")
        self.output_mode=output_mode
        self.device=device
        
        self.cls_head = torch.nn.Linear(in_features=backbone.main_channels[-1], out_features=self.num_cls, device=self.device)
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])
        x = x.flatten(start_dim=1, end_dim=3)
        x = self.cls_head(x)
        if self.output_mode=='logits':
            return x
        elif self.output_mode=='probs':
            return torch.nn.functional.softmax(x)



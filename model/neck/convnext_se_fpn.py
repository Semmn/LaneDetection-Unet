from backbone.convnext_se.convnext_se import ConvNextV1, Stage, Stem
from utils.conv_2d import adjust_padding_for_strided_output, DepthWiseSepConv
from utils.stochastic_depth_drop import create_linear_p, create_uniform_p

import torch
import torch.nn as nn


# Convnext_se for detector backbone (lane point regression + bg/fg classification for detection task)
class DetectorBackbone(torch.nn.Module):
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
        
        if norm_mode not in ['layer_norm', 'batch_norm']:
            raise Exception(f"Unsupported normalization method: {norm_mode}. Must be either 'layer_norm' or 'batch_norm'")
        self.norm_mode = norm_mode
        
        self.device=device
        self.num_stages = len(num_blocks)
        
        self.stem = Stem(stem_in_channels=self.input_channels, stem_out_channels=main_channels[0], stem_kernel_sz=self.stem_kersz, stem_stride=self.stem_stride,
                         stem_padding=adjust_padding_for_strided_output(self.stem_kersz, self.stem_stride), stem_dilation=1, stem_groups=1, device=self.device)
        self.stages = []
        for i in range(self.num_stages):
            if i == 0:
                self.stages.append(Stage(transition_flag=False, num_blocks=self.num_blocks[i], img_hw=self.img_hw[i], in_channels=self.main_channels[i], out_channels=self.expansion_dim[i],
                                    kernel_sz=self.kernel_sz[i], stride=self.stride[i], padding=self.padding[i], dilation=self.dilation[i], groups=self.groups[i], droprate=self.droprate[i],
                                    drop_mode=self.drop_mode[i], use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, norm_mode=self.norm_mode, device='cuda'))
            else:
                self.stages.append(Stage(transition_flag=True, num_blocks=self.num_blocks[i], img_hw=self.img_hw[i], in_channels=self.main_channels[i], out_channels=self.expansion_dim[i],
                                        kernel_sz=self.kernel_sz[i], stride=self.stride[i], padding=self.padding[i], dilation=self.dilation[i], groups=self.groups[i], droprate=self.droprate[i],
                                        drop_mode=self.drop_mode[i], use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, transition_channels=self.main_channels[i-1], transition_kersz=self.transition_kersz[i],
                                        transition_stride=self.transition_stride[i], norm_mode=self.norm_mode, device='cuda'))
        self.stages = torch.nn.ModuleList(self.stages)
    
    # as same as u-net encoder design, each stage outputs final block output in each stage
    def forward(self, x):
        stage_output = []
        x = self.stem(x)
        for i in range(self.num_stages):
            x = self.stages[i](x)
            stage_output.append(x)
        return stage_output




# FPN: Feature Pyramid Network (hierarchical structure of each feature output from each backbone stage)
# Input: Output from DetectorBackbone forward()
# Output: Feature Structure constructed from Input. Information from all stage output are reinforced by top<->down feature sharing
# This Module will be connected to head of detector which outputs the bounding box location and classifying result.
class FPN(torch.nn.Module):
    def __init__(self, nstage_to_use, bidirectional, in_channels, out_channels, img_hw, downsample_kersz, upsample_kersz, downsample_stride, upsample_stride, 
                 downsample_dilation, upsample_dilation, downsample_groups, upsample_groups, upsample_padding, upsample_out_padding, norm_mode, device, dtype):
        super().__init__()
        # all list are orderd as low to high feature layer (down - up ordering) (example: [(56, 168),(28,84),(14,42),(7,21)])
        self.nstage_to_use = nstage_to_use # number of stages in FPN structure
        self.bidirectional = bidirectional # whether to share information bidirectional way (top->down, down->top)
        self.in_channels = in_channels # number of chanels from DetectorBackbone output #(example: [48, 96, 192, 384])
        self.out_channels = out_channels # number of channels that FPN outputs
        self.img_hw = img_hw # list of image (h, w) shape, this parameter is only used when norm_mode=='layer_norm' otherwise it notifies the user how the shape of input tensors are changed.
        
        self.downsample_kersz = downsample_kersz # kernel size used for top-down transform convolution
        self.upsample_kersz = upsample_kersz # kernel size used for down-top transform transposed convolution
        self.downsample_stride = downsample_stride # stride used for top-down transform convolution (mostly 2)
        self.upsample_stride = upsample_stride # stride size used for down-top transform transposed convolution
        
        self.downsample_dilation = downsample_dilation # dilation used for top-down transform convolution
        self.upsample_dilation = upsample_dilation # dilation used for down-top transform transposed convolution
        self.downsample_groups = downsample_groups # number of groups used for top-down transform convolution
        self.upsample_groups = upsample_groups # number of groups used for down-top transform transposed convolution
        
        self.upsample_padding = upsample_padding # padding used for down-top transform transposed convolution
        self.upsample_out_padding = upsample_out_padding # output padding used for down-top transform transposed convolution (this is used for adjusting output shape of transposed convolution)
        
        if norm_mode not in ['batch_norm', 'layer_norm']:
            raise Exception('norm_mode must be one of "batch_norm" or "layer_norm"')
        
        self.norm_mode = norm_mode
        self.device = device
        self.dtype = dtype
        
        if self.nstage_to_use > len(self.in_channels) or self.nstage_to_use <= 0:
            raise Exception("number of stage to use must be equal or smaller than total number of stages. Also it must be bigger than 0")
        
        # number of transform must be equal to nstage_to_use-1 (transfrom applies to intervals of each stages)
        if self.nstage_to_use-1 != len(self.downsample_kersz):
            raise Exception("number of transforms must be self.nstage_to_use-1!")
        
        self.num_stages = len(self.in_channels) # total number of channels
        
        # only bidirection FPN use down to up convolution
        if self.bidirectional:
            self.down_up_transform = []
            # if nstage_to_use is not same as num_stages it means that lower feature pyramid is excluded
            # for example, if nstage_to_use=3, num_stages=4, we will use top-3 stages (top 3 small feature map) to consturcture fpn. // inputs: (0,1,2,3) -> outputs: (1,2,3)
            for i, stage_i in enumerate(range(self.num_stages-self.nstage_to_use, self.num_stages-1), 0): 
                self.down_up_transform.append(torch.nn.Conv2d(in_channels=self.in_channels[stage_i], out_channels=self.in_channels[stage_i+1], kernel_size=self.downsample_kersz[i], 
                                                        stride=self.downsample_stride[i], padding=adjust_padding_for_strided_output(self.downsample_kersz[i], self.downsample_stride[i]),
                                                            dilation=self.downsample_dilation[i], groups=self.downsample_groups[i], device=self.device, dtype=self.dtype))
            self.down_up_transform = torch.nn.ModuleList(self.down_up_transform)

        self.up_down_transform = []
        for i, stage_i in enumerate(range(self.num_stages-1, self.num_stages-self.nstage_to_use, -1), 0):
            self.up_down_transform.append(torch.nn.ConvTranspose2d(in_channels=self.in_channels[stage_i], out_channels=self.in_channels[stage_i-1], kernel_size=self.upsample_kersz[i],
                                                              stride=self.upsample_stride[i], padding=self.upsample_padding[i], output_padding=self.upsample_out_padding[i],
                                                              groups=self.upsample_groups[i], dilation=self.upsample_dilation[i], device=self.device, dtype=self.dtype))
        self.up_down_transform = torch.nn.ModuleList(self.up_down_transform)
        
        # number of downsampled_output and upsampled_output must be sample
        if self.bidirectional and len(self.down_up_transform) != len(self.up_down_transform):
            raise Exception("number of down_up_transform and up_down_transform must be same!")
        
        self.post_conv = []
        self.normalize = []
        # transform of post_processing is pointwise (1x1) convolution. this transforms adjust the number of channels of each image to self.out_channels
        # through pointwise convolution, normalization method (layer_norm or batch_norm) is followed to increase the stability of training.
        for stage_i in range(self.num_stages-1, self.num_stages-self.nstage_to_use-1, -1):
            self.post_conv.append(torch.nn.Conv2d(in_channels=self.in_channels[stage_i], out_channels=self.out_channels, kernel_size=(1,1),
                                                  stride=(1,1), padding=(0,0), dilation=1, groups=1, device=self.device, dtype=self.dtype))
            if self.norm_mode=='layer_norm':
                self.normalize.append(torch.nn.LayerNorm(normalized_shape=[self.out_channels, self.img_hw[stage_i][0], self.img_hw[stage_i][1]], device=self.device, dtype=self.dtype))
            elif self.norm_mode=='batch_norm':
                self.normalize.append(torch.nn.BatchNorm2d(num_features=self.out_channels, device=self.device, dtype=self.dtype))

        self.post_conv = torch.nn.ModuleList(self.post_conv)
        self.normalize = torch.nn.ModuleList(self.normalize)
        
    
    # fpn inputs stage output of backbone and constructure fpn structure
    def forward(self, backbone_stage_o): # backbone_stage_o: (example: [(56, 168),(28,84),(14,42),(7,21)])
        fpn_o = [] # output of fpn module must output nstage_to_use number of features
        
        for i, stage_i in enumerate(range(self.num_stages-1, self.num_stages-self.nstage_to_use, -1), 0):
            if i==0:
                fpn_o.append(backbone_stage_o[stage_i])
            x = self.up_down_transform[i](backbone_stage_o[stage_i]) # [(7,21)->(14,42), (14,42)->(28,84), (28,84)->(56,168)]
            x = x + backbone_stage_o[stage_i-1]
            fpn_o.append(x)
        
        # only bidirectional fpn uses down-to-up (common conv2d) transforms
        if self.bidirectional:
            #fpn_o: [(7,21), (14,42), (28,84), (56,168)]
            for i, stage_i in enumerate(range(self.num_stages-self.nstage_to_use, self.num_stages-1), 0):
                x = self.down_up_transform[i](x) # [(56,168)->(28,84), (28,84)->(14,42), (14,42)->(7,21)]
                x = x + fpn_o[self.nstage_to_use-2-i]
                fpn_o[self.nstage_to_use-2-i] = x
        
        for i in range(self.nstage_to_use):
            fpn_o[i] = self.post_conv[i](fpn_o[i])
            fpn_o[i] = self.normalize[i](fpn_o[i])
            
        # output of fpn module are reinfored by top-down, down-top feature construction 
        # ex) ordering of modules is top down: [(7, 21), (14,42), (28, 84), (56, 168)]
        return fpn_o



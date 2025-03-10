from model.backbone.convnext_se.convnext_se import ConvNextV1, Stage, Stem
from model.utils.conv_2d import adjust_padding_for_strided_output, DepthWiseSepConv
from model.utils.stochastic_depth_drop import create_linear_p, create_uniform_p

import torch
import torch.nn as nn
import torchvision

# convnext v1 + sequeeze and excitation module + Unet Encoder (returns each stage output)
class Encoder(torch.nn.Module):
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
                                    drop_mode=self.drop_mode[i], use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, norm_mode=self.norm_mode, device=self.device))
            else:
                self.stages.append(Stage(transition_flag=True, num_blocks=self.num_blocks[i], img_hw=self.img_hw[i], in_channels=self.main_channels[i], out_channels=self.expansion_dim[i],
                                        kernel_sz=self.kernel_sz[i], stride=self.stride[i], padding=self.padding[i], dilation=self.dilation[i], groups=self.groups[i], droprate=self.droprate[i],
                                        drop_mode=self.drop_mode[i], use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, transition_channels=self.main_channels[i-1], transition_kersz=self.transition_kersz[i],
                                        transition_stride=self.transition_stride[i], norm_mode=self.norm_mode, device=self.device))
        self.stages = torch.nn.ModuleList(self.stages)
        
    def forward(self, x):
        stage_o = []
        x = self.stem(x)
        for i in range(self.num_stages):
            x = self.stages[i](x)
            stage_o.append(x)
        return x, stage_o

        
# should carefully consider the transition layer, because when up-convolution (transposed convolution) is used, output size is doubled.
class DBlock(torch.nn.Module):
    def __init__(self, img_hw, in_channels, out_channels, kernel_sz, stride, padding, groups, dilation, droprate, drop_mode,
                 use_se, squeeze_ratio, is_fuse=False, fused_channels=-1, transition=False, transition_channels=-1, transition_kersz=-1,
                 transition_stride=-1, transition_padding=-1, transition_out_padding=-1, norm_mode='layer_norm', device='cuda'):
        super().__init__()
        self.img_h = img_hw[0]
        self.img_w = img_hw[1]
        
        # residual path parameters
        self.block_channels = in_channels
        self.block_out_channels = out_channels
        self.kernel_sz = kernel_sz
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.droprate = droprate
        if drop_mode not in ['row', 'batch']:
            raise Exception("drop_mode must be either 'row' or 'batch'")
        self.drop_mode = drop_mode
        
        self.use_se = use_se
        self.squeeze_r = squeeze_ratio
        
        if transition and is_fuse:
            raise Exception("Upsampling and encoder stage output concat cannot happen same time!")
        
        self.is_fuse = is_fuse # whether to decide encoder stage output is concatenated to decoder path
        self.fused_channels = fused_channels # number of channels from encoder output
        
        if norm_mode not in ['layer_norm', 'batch_norm']:
            raise Exception(f"UnSupported normalization method: {norm_mode}. Use either 'layer_norm' or 'batch_norm'")
        self.norm_mode = norm_mode
        self.device = device
        
        # deconvolution path parameters
        self.transition = transition
        self.transition_channels = transition_channels
        self.transition_kersz = transition_kersz
        self.transition_stride = transition_stride
        self.transition_padding = transition_padding
        self.transition_out_padding = transition_out_padding
        
        if self.transition:
            # unlike encoder, kernel_size for self.conv_1 is not same as self.kernel_sz because of multiple possible choice of padding sizes
            self.conv_1 = torch.nn.ConvTranspose2d(in_channels=self.transition_channels, out_channels=self.block_channels, kernel_size=self.transition_kersz, 
                                                stride=self.transition_stride, padding=self.transition_padding, output_padding=self.transition_out_padding, groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=None)
            self.transition_conv = torch.nn.ConvTranspose2d(in_channels=self.transition_channels, out_channels=self.block_channels, kernel_size=self.transition_kersz, 
                                                stride=self.transition_stride, padding=self.transition_padding, output_padding=self.transition_out_padding, groups=1, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=None)
        else:
            if self.is_fuse:
                self.conv_1 = DepthWiseSepConv(in_channels=(self.block_channels+self.fused_channels), out_channels=self.block_channels, kernel_sz=self.kernel_sz,
                                                        stride=self.stride, padding=self.padding, dilation=self.dilation, device=self.device)
                # transition_conv is needed to adjust the number of output filters (pointwise convolution is used to fit the size)
                self.transition_conv = torch.nn.Conv2d(in_channels=self.block_channels+self.fused_channels, out_channels=self.block_channels, kernel_size=(1,1),
                                                       stride=(1,1), padding=(0,0), dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device)
            else:
                self.conv_1 = DepthWiseSepConv(in_channels=self.block_channels, out_channels=self.block_channels, kernel_sz=self.kernel_sz,
                                                        stride=self.stride, padding=self.padding, dilation=self.dilation, device=self.device)
            
        self.pointwise_1 = torch.nn.Conv2d(in_channels=self.block_channels, out_channels=self.block_out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=1, groups=self.groups, bias=True, padding_mode='zeros', device=self.device)
        self.pointwise_2 = torch.nn.Conv2d(in_channels=self.block_out_channels, out_channels=self.block_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=1, groups=self.groups, bias=True, padding_mode='zeros', device=self.device)
        
        if self.use_se:
            self.fc1 = torch.nn.Conv2d(in_channels=self.block_channels, out_channels=int(self.block_channels/self.squeeze_r), kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=1, groups=1, bias=True, device=self.device)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Conv2d(in_channels=int(self.block_channels/self.squeeze_r), out_channels=self.block_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), dilation=1, groups=1, bias=True, device=self.device)
            self.sigmoid = torch.nn.Sigmoid()
        
        # unlike batch normalization, layer normalization consumes more gpu memory as input spatial size increases.
        # while decoder head restore image size to original, it necessarily process large spatial size input tensor. it consumes much more gpu memory
        # than using batch normalization as normalization method. (4M vs 256M) In this case, use batch normalization instead of layer normalization.
        if self.norm_mode == 'layer_norm':
            self.normalization = torch.nn.LayerNorm([self.block_channels, self.img_h, self.img_w], device=self.device)
        elif self.norm_mode == 'batch_norm':
            self.normalization = torch.nn.BatchNorm2d(num_features=self.block_channels, device=self.device)
        self.gelu = torch.nn.GELU()
    
    # encoder stage output concatenation is not happend in the block-level, do that on the stage-level
    def forward(self, x):
        highway = x
    
        x = self.conv_1(x)
        x = self.normalization(x)
        x = self.pointwise_1(x)
        x = self.gelu(x)
        x = self.pointwise_2(x)
        
        if self.use_se:
            squeeze = torch.nn.functional.avg_pool2d(x, x.size()[2:])
            excitation = self.fc1(squeeze)
            excitation = self.relu(excitation)
            excitation = self.fc2(excitation)
            attention = self.sigmoid(excitation)
            x = x * attention
        
        if self.transition or self.is_fuse:
            highway = self.transition_conv(highway)
        
        highway += torchvision.ops.stochastic_depth(x, p=self.droprate, mode=self.drop_mode, training=self.training)
        return highway
    

# # should carefully consider the transition layer, because when up-convolution (transposed convolution) is used, output size is doubled.
# class DBlock(torch.nn.Module):
#     def __init__(self, img_hw, in_channels, out_channels, kernel_sz, stride, padding, groups, dilation, droprate, drop_mode,
#                  use_se, squeeze_ratio, is_fuse=False, fused_channels=-1, transition=False, transition_channels=-1, transition_kersz=-1,
#                  transition_stride=-1, transition_padding=-1, transition_out_padding=-1, device='cuda'):

class DStage(torch.nn.Module):
    def __init__(self, transition_flag, num_blocks:int, img_hw:list, in_channels, out_channels, kernel_sz, stride, padding, dilation, groups,
                 droprate, drop_mode:list, use_se:bool, squeeze_ratio:int, encoder_channels:int, transition_channels=-1, transition_kersz=-1, transition_stride=-1, 
                 transition_padding=-1, transition_out_padding=1, norm_mode='layer_norm', device='cuda'):
        
        super().__init__()
        self.transition_flag = transition_flag
        self.num_blocks = num_blocks
        self.img_h = img_hw[0]
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
        self.drop_mode = drop_mode
        self.use_se = use_se
        self.squeeze_r = squeeze_ratio
        self.encoder_channels = encoder_channels
        self.device = device
        
        self.transition_channels = transition_channels
        self.transition_kersz = transition_kersz
        self.transition_stride = transition_stride
        self.transition_padding = transition_padding
        self.transition_out_padding = transition_out_padding
        
        if norm_mode not in ['layer_norm', 'batch_norm']:
            raise Exception(f"Unsupported normalization method: {norm_mode}. Must be either 'layer_norm' 'batch_norm'")
        self.norm_mode = norm_mode
        
        self.blocks = []
        for i in range(num_blocks):
            if i == 0 and self.transition_flag: # upsampling and encoder stage output concatenation cannot happen same time!
                self.blocks.append(DBlock([self.img_h, self.img_w], in_channels=self.stage_channels, out_channels=self.stage_out_channels,
                                          kernel_sz=self.kernel_sz, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation,
                                          droprate=self.droprate[i], drop_mode=self.drop_mode[i], use_se=self.use_se, squeeze_ratio=self.squeeze_r,
                                          transition=True, transition_channels=self.transition_channels, transition_kersz=self.transition_kersz, 
                                          transition_stride=self.transition_stride, transition_padding=self.transition_padding, 
                                          transition_out_padding=self.transition_out_padding, norm_mode=self.norm_mode, device=self.device))
            else:
                if i==1: # after upsampling is dones, concatenation is applied to the decoder output
                    is_fused = True # self.encoder_channels are only valid when is_fused==True
                else:
                    is_fused = False
                self.blocks.append(DBlock([self.img_h, self.img_w], in_channels=self.stage_channels, out_channels=self.stage_out_channels,
                                          kernel_sz=self.kernel_sz, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation,
                                          droprate=self.droprate[i], drop_mode=self.drop_mode[i], use_se=self.use_se, squeeze_ratio=self.squeeze_r,
                                          is_fuse=is_fused, fused_channels=self.encoder_channels, norm_mode=self.norm_mode, device=self.device))
        self.blocks = torch.nn.ModuleList(self.blocks)
                
    def forward(self, x, enc_stage_o):
        for i in range(self.num_blocks):
            if i == 1: # after upconvolution are applied, concatenation with encoder stage output happens
                x = torch.concat([x, enc_stage_o], dim=1) # concatenate between decoder first block output and corresponding encoder output
            x = self.blocks[i](x)
        return x


# apply one transpose convolution layer (increase spatial dimension n times)
class DStemNx(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 dilation=1, device='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.device = device
        
        self.stem_conv = torch.nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                                                  stride=self.stride, padding=self.padding, output_padding=self.out_padding, groups=self.groups,
                                                  bias=True, dilation=self.dilation, padding_mode='zeros', device=self.device)
    
    def forward(self, x):
        x = self.stem_conv(x)
        return x

# apply stacked transpose convolution layer (increase spatial dimension n * num_stacked times)
class DStemStacked(torch.nn.Module):
    def __init__(self, in_channels:list, out_channels:list, kernel_size:list, stride:list, padding:list, output_padding:list, groups:list,
                 dilation:list, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.device = device
        
        self.num_stacked = len(self.in_channels)
        
        self.stacks = []        
        for i in range(self.num_stacked):
            self.stacks.append(torch.nn.ConvTranspose2d(in_channels=self.in_channels[i], out_channels=self.out_channels[i], kernel_size=self.kernel_size[i],
                                                    stride=self.stride[i], padding=self.padding[i], output_padding=self.out_padding[i], groups=self.groups[i],
                                                    bias=True, dilation=self.dilation[i], padding_mode='zeros', device=self.device))
        self.stacks = torch.nn.ModuleList(self.stacks)
        
        self.out_conv = torch.nn.Conv2d(in_channels=self.out_channels[-1], out_channels=self.out_channels[-1], kernel_size=(1,1), stride=1,
                                        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=self.device)
    
    def forward(self, x):
        for i in range(len(self.stacks)):
            x = self.stacks[i](x)
        x = self.out_conv(x)
        return x

# continues convnext style decoder block
class DStemStaged(torch.nn.Module):
    def __init__(self, num_blocks:list, img_hw:list, input_channels, main_channels, expansion_channels, kernel_sz, stride, padding, dilation, groups, droprate, drop_mode,
                 use_se, squeeze_ratio, transition_kersz, transition_stride, transition_padding, transition_out_padding, norm_mode, device='cuda'):
        super().__init__()
        self.num_blocks = num_blocks
        self.img_hw = img_hw
        
        self.input_channels = input_channels # number of last stage channels
        self.main_channels = main_channels
        self.expansion_channels = expansion_channels
        
        self.kernel_sz = kernel_sz
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.droprate = droprate
        self.drop_mode = drop_mode
        
        self.use_se = use_se
        self.squeeze_ratio = squeeze_ratio
        self.transition_kersz = transition_kersz
        self.transition_stride = transition_stride
        self.transition_padding = transition_padding
        self.transition_out_padding = transition_out_padding
        
        if norm_mode not in ['batch_norm', 'layer_norm']:
            raise Exception(f"Unsupported normalization method: {norm_mode}. Must be either 'layer_norm', 'batch_norm'")
        self.norm_mode = norm_mode
        
        self.device = device
        
        self.stages = []
        self.num_stages = len(self.num_blocks)
        for i in range(self.num_stages):
            stage = []
            for j in range(self.num_blocks[i]):
                if j == 0: # for the first block
                    if i==0:
                        transition_channels = self.input_channels
                    elif i!=0:
                        transition_channels = self.main_channels[i-1]
                    stage.append(DBlock(img_hw=self.img_hw[i], in_channels=self.main_channels[i], out_channels=self.expansion_channels[i], kernel_sz=self.kernel_sz[i], 
                                        stride=self.stride[i], padding=self.padding[i], groups=self.groups[i], dilation=self.dilation[i], droprate=self.droprate[i][j],
                                        drop_mode=self.drop_mode[i][j], use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, transition=True,
                                        transition_channels=transition_channels, transition_kersz=self.transition_kersz[i], transition_stride=self.transition_stride[i],
                                        transition_padding=self.transition_padding[i], transition_out_padding=self.transition_out_padding[i], norm_mode=self.norm_mode, device=self.device))
                else: # transition (up-sampling) does not happen if it is not first block
                    stage.append(DBlock(img_hw=self.img_hw[i], in_channels=self.main_channels[i], out_channels=self.expansion_channels[i], kernel_sz=self.kernel_sz[i], 
                                        stride=self.stride[i], padding=self.padding[i], groups=self.groups[i], dilation=self.dilation[i], norm_mode=self.norm_mode, droprate=self.droprate[i][j],
                                        drop_mode=self.drop_mode[i][j], use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, device=self.device))
        
            self.stages.append(torch.nn.ModuleList(stage))
        self.stages = torch.nn.ModuleList(self.stages)
    

    def forward(self, x):
        for i in range(len(self.stages)):
            for j in range(len(self.stages[i])):
                x = self.stages[i][j](x)
        return x

# decoder head (this does not need to be symmetrical with encoder)
class Decoder(torch.nn.Module):
    def __init__(self, num_blocks:list, img_hw:list, main_channels:list, expansion_dim:list, kernel_sz:list, 
                 stride:list, padding:list, dilation:list, groups:list, droprate:list, drop_mode:list, 
                 use_se:list, squeeze_ratio:int, encoder_channels:list,  transition_kersz:list, transition_stride:list, 
                 transition_padding:list, transition_out_padding:list, norm_mode:str, head:torch.nn.ModuleList, device='cuda'):
        super().__init__()
        
        self.num_blocks = num_blocks # list that contains number of blocks for each stage
        self.img_hw = img_hw # list that contains representative (img_height, img_width) for each stage
        self.main_channels = main_channels
        self.expansion_dim = expansion_dim
        self.kernel_sz = kernel_sz
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.droprate = droprate # stochastic depth drop probability ex) [[], [], []] -> outer: number of stages, inner: number of blocks
        self.drop_mode = drop_mode # stochastic depth drop mode
        
        self.use_se = use_se  # list that contains whether to use sequeeze and excitation module in the stage
        self.squeeze_ratio = squeeze_ratio
        
        self.encoder_channels = encoder_channels # list that contains number of channels for each encoder stage output (this excludes final encoder stage output)
        
        self.transition_kersz = transition_kersz
        self.transition_stride = transition_stride
        self.transition_padding = transition_padding
        self.transition_out_padding = transition_out_padding
        
        if norm_mode not in ['layer_norm', 'batch_norm']:
            raise Exception(f"Unsupported normalization method: {norm_mode}. Must be either 'layer_norm', 'batch_norm'")
        self.norm_mode = norm_mode
        
        # head is intentionally created outside of Decoder class because many variations and img size can be required in various situation.
        self.head = head # decoder head for further convolution

        self.device = device
        
        self.num_stages = len(self.num_blocks) # length of list indicates the number of stages
        self.stages = []        
        
        for i in range(self.num_stages):
            if i == 0: # start of decoder first inputs the number of last encoder stage output channels
                transition_channels = self.encoder_channels[0]
            else:
                transition_channels = self.main_channels[i-1]
            
            # unlike encoder, decoder always upsampling the input (transition_flag should be always True)
            self.stages.append(DStage(transition_flag=True, num_blocks=self.num_blocks[i], img_hw=self.img_hw[i], in_channels=self.main_channels[i],
                                        out_channels=self.expansion_dim[i], kernel_sz = self.kernel_sz[i], stride=self.stride[i], padding=self.padding[i],
                                        dilation = self.dilation[i], groups=self.groups[i], droprate=self.droprate[i], drop_mode=self.drop_mode[i],
                                        use_se=self.use_se[i], squeeze_ratio=self.squeeze_ratio, encoder_channels=self.encoder_channels[i+1], transition_channels=transition_channels, 
                                        transition_kersz=self.transition_kersz[i], transition_stride=self.transition_stride[i], transition_padding=self.transition_padding[i],
                                        transition_out_padding=self.transition_out_padding[i], norm_mode=self.norm_mode, device=self.device))
        self.stages = torch.nn.ModuleList(self.stages)
        
    def forward(self, x, enc_stage_out):
        encoder_idx = len(enc_stage_out)-2 # -1 from zero indexing, -1 from skipping last stage output
        for i in range(self.num_stages):
            if encoder_idx >= 0:
                x = self.stages[i](x, enc_stage_out[encoder_idx])
            encoder_idx -= 1

        x = self.head(x)
    
        return x


# container encoder and decoder
class UNet(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, num_cls:int, output_mode:str):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_cls = num_cls # number of segmentation class
        
        if output_mode not in ['logits', 'probs']:
            raise Exception(f"Unsupported output mode: {output_mode}. Must be either 'logits' or 'probs'")
        self.output_mode = output_mode
        
        if self.output_mode == 'probs':
            self.softmax = torch.nn.Softmax2d()
        
        if isinstance(self.decoder.head, DStemNx):
            in_channels = self.decoder.head.out_channels
        elif isinstance(self.decoder.head, DStemStacked):
            in_channels = self.decoder.head.out_channels[-1]
        elif isinstance(self.decoder.head, DStemStaged):
            in_channels = self.decoder.head.main_channels[-1]
        else:
            raise Exception("Not Implemented: Currently Head supports only DStemNx, DStemStacked, DStemStaged")
        
        self.cls_head = torch.nn.Conv2d(in_channels=in_channels, out_channels=self.num_cls, kernel_size=(1,1),
                                        stride=(1,1), padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=encoder.device)
    
    def forward(self, x):
        x, stage_o = self.encoder(x)
        x = self.decoder(x, stage_o)
        x = self.cls_head(x)  # (B, num_cls, ori_h, ori_w)
        
        if self.output_mode=='probs':
            x = self.softmax(x) # return probabilities
        
        del stage_o
        
        return x
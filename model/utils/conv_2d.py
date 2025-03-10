# Author: Yang Seung Yu
# calculate padding size for strided convolution output size to be 1/(stride) of the input size
# return padding size (height, width)
import numpy as np
import torch

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
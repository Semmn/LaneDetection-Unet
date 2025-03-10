import sys
import yaml

from model.head.convnext_se_unet import Encoder, DBlock, DStage, DStemNx, DStemStacked, DStemStaged, Decoder, UNet
from model.utils.conv_2d import adjust_padding_for_strided_output, DepthWiseSepConv
from model.utils.stochastic_depth_drop import create_linear_p, create_uniform_p

###################################################################################################
# class Encoder(torch.nn.Module):
#     def __init__(self, num_blocks:list, input_channels:int, stem_kersz:tuple, stem_stride:tuple, 
#                  img_hw:list, main_channels:list, expansion_dim:list, kernel_sz:list, stride:list, padding:list, dilation:list, groups:list, droprate:list, drop_mode:list, use_se:list, squeeze_ratio:int,
#                  transition_kersz:list, transition_stride:list, norm_mode:str, device='cuda'):
###################################################################################################

###################################################################################################
# class DStemNx(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
#                  dilation=1, device='cuda'):
###################################################################################################

###################################################################################################
# class DStemStacked(torch.nn.Module):
#     def __init__(self, in_channels:list, out_channels:list, kernel_size:list, stride:list, padding:list, output_padding:list, groups:list,
#                  dilation:list, device):
###################################################################################################

###################################################################################################
# class DStemStaged(torch.nn.Module):
#     def __init__(self, num_blocks:list, img_hw:list, input_channels, main_channels, expansion_channels, kernel_sz, stride, padding, dilation, groups, droprate, drop_mode,
#                  use_se, squeeze_ratio, transition_kersz, transition_stride, transition_padding, transition_out_padding, norm_mode, device='cuda'):
###################################################################################################

###################################################################################################
# class Decoder(torch.nn.Module):
#     def __init__(self, num_blocks:list, img_hw:list, main_channels:list, expansion_dim:list, kernel_sz:list, 
#                  stride:list, padding:list, dilation:list, groups:list, droprate:list, drop_mode:list, 
#                  use_se:list, squeeze_ratio:int, encoder_channels:list,  transition_kersz:list, transition_stride:list, 
#                  transition_padding:list, transition_out_padding:list, norm_mode:str, head:torch.nn.ModuleList, device='cuda'):
###################################################################################################

###################################################################################################
# if num_cls==2 -> binary segmentation -> classifying lane position on the road separately
# we need to at least make 5 masks for 4 lanes each presenting the lane position of the road. (0: background, 1: lane1, 2: lane2, 3: lane3, 4: lane4)

# class UNet(torch.nn.Module):
#     def __init__(self, encoder: Encoder, decoder: Decoder, num_cls:int, output_mode:str):
###################################################################################################


# creates unet instance from yaml configuration file
# config_path: model configuration file path
def create_unet(config_path):
    
    with open(config_path) as f:
        model_configs = yaml.load(f, Loader=yaml.Loader)
    
    device_config = model_configs['UNet']['device'] # 'cuda' or 'cpu'
    num_cls_config = model_configs['UNet']['num_cls']
    output_mode_config = model_configs['UNet']['output_mode']
    
    encoder_configs = model_configs['UNet']['encoder']
    
    encoder_num_blocks = encoder_configs['num_blocks'] # number of blocks in each stage
    encoder_dp_prob = encoder_configs['stochastic_dp_p'] # stochastic depth drop probability
    encoder_dp_mode = encoder_configs['stochastic_mode'] # 'batch' or 'row'
    encoder_dp_schedule = encoder_configs['stochastic_dp_schedule'] # 'linear' or 'uniform'
    
    # creates linearly decaying stochastic depth drop probability
    if encoder_dp_schedule == 'linear':
        dp_list, dp_mode = create_linear_p(encoder_num_blocks, encoder_dp_mode, encoder_dp_prob) 
    elif encoder_dp_schedule == 'uniform':
        dp_list, dp_mode = create_uniform_p(encoder_num_blocks, encoder_dp_mode, encoder_dp_prob) # create constant stochastic depth drop probability
        
    encoder_input_channels = encoder_configs['input_channels'] # number of input channels
    encoder_stem_kersz = encoder_configs['stem_kernel_size'] # stem layer kernel size
    encoder_stem_stride = encoder_configs['stem_stride'] # stem layer stride
    encoder_img_hw = encoder_configs['img_hw'] # image height and width for each stage input
    encoder_main_channels = encoder_configs['main_channels'] # number of channels for convnext blocks (not expanded channels)
    encoder_expansion_ratio = encoder_configs['expansion_ratio']
    encoder_expansion_dim = [encoder_main_channels[i] * encoder_expansion_ratio[i] for i in range(len(encoder_expansion_ratio))]
    encoder_kernel_sz = encoder_configs['kernel_size'] # kernel size for each stage (not pointwise conv)
    encoder_stride = encoder_configs['stride'] # stride for each stage
    encoder_padding = encoder_configs['padding'] # padding for each stage
    encoder_dilation = encoder_configs['dilation'] # dilation for each stage
    encoder_groups = encoder_configs['groups'] # number of groups for each stage
    encoder_use_se = encoder_configs['use_se'] # whether to use squeeze and excitation for each stage
    encoder_squeeze_ratio = encoder_configs['squeeze_ratio']
    encoder_transition_kersz = encoder_configs['transition_kernel_size'] # transition layer kernel size
    encoder_transition_stride = encoder_configs['transition_stride'] # transition layer stride
    encoder_norm_mode = encoder_configs['norm_mode'] # normalization mode ('batch_norm' or 'layer_norm')
        

    # intialize encoder
    unet_encoder = Encoder(num_blocks=encoder_num_blocks, input_channels=encoder_input_channels, stem_kersz=encoder_stem_kersz, 
                           stem_stride=encoder_stem_stride, img_hw=encoder_img_hw, main_channels=encoder_main_channels, 
                           expansion_dim=encoder_expansion_dim, kernel_sz=encoder_kernel_sz, stride=encoder_stride, padding=encoder_padding,
                           dilation=encoder_dilation, groups=encoder_groups, droprate=dp_list, drop_mode=dp_mode,
                           use_se=encoder_use_se, squeeze_ratio=encoder_squeeze_ratio, transition_kersz=encoder_transition_kersz,
                           transition_stride=encoder_transition_stride, norm_mode=encoder_norm_mode, device=device_config)
    
    decoder_configs = model_configs['UNet']['decoder']
    decoder_num_blocks = decoder_configs['num_blocks']
    decoder_img_hw = decoder_configs['img_hw']
    decoder_main_channels = decoder_configs['main_channels']
    decoder_expansion_ratio = decoder_configs['expansion_ratio']
    decoder_expansion_dim = [decoder_main_channels[i] * decoder_expansion_ratio[i] for i in range(len(decoder_expansion_ratio))]
    decoder_kernel_sz = decoder_configs['kernel_size']
    decoder_stride = decoder_configs['stride']
    decoder_padding = decoder_configs['padding']
    decoder_dilation = decoder_configs['dilation']
    decoder_groups = decoder_configs['groups']
    decoder_use_se = decoder_configs['use_se']
    decoder_squeeze_ratio = decoder_configs['squeeze_ratio']
    decoder_transition_kersz = decoder_configs['transition_kernel_size']
    decoder_transition_stride = decoder_configs['transition_stride']
    decoder_transition_padding = decoder_configs['transition_padding']
    decoder_transition_out_padding = decoder_configs['transition_out_padding']
    decoder_norm_mode = decoder_configs['norm_mode']
    
    
    decoder_dp_prob = decoder_configs['stochastic_dp_p']
    decoder_dp_mode = decoder_configs['stochastic_mode']
    decoder_dp_schedule = decoder_configs['stochastic_dp_schedule']
    
    head_type = decoder_configs['head']['head_type'] # one of 'head_nx', 'head_stacked', 'head_staged'    
    head_configs = decoder_configs['head'][head_type]
    
    if head_type in ['head_nx', 'head_stacked']:
        head_in_channels = head_configs['in_channels']
        head_out_channels = head_configs['out_channels']
        head_kernel_sz = head_configs['kernel_size']
        head_stride = head_configs['stride']
        head_padding = head_configs['padding']
        head_output_padding = head_configs['output_padding']
        head_groups = head_configs['groups']
        head_dilation = head_configs['dilation']
        
        # intialize head module in decoder
        if head_type =='head_nx':
            head_module = DStemNx(in_channels=head_in_channels, out_channels=head_out_channels, kernel_size=head_kernel_sz, stride=head_stride, padding=head_padding, output_padding=head_output_padding, groups=head_groups, 
                                  dilation=head_dilation, device=device_config)
        elif head_type =='head_stacked':
            head_module = DStemStacked(in_channels=head_in_channels, out_channels=head_out_channels, kernel_size=head_kernel_sz, stride=head_stride, padding=head_padding, output_padding=head_output_padding, 
                          groups=head_groups, dilation=head_dilation, device=device_config)
            
        if decoder_dp_schedule =='linear':
            dp_list, dp_mode = create_linear_p(decoder_num_blocks, dp_mode=decoder_dp_mode, last_p=decoder_dp_prob)
        elif decoder_dp_schedule == 'uniform':
            dp_list, dp_mode = create_uniform_p(decoder_num_blocks, dp_mode=decoder_dp_mode, last_p=decoder_dp_prob)
            
    elif head_type == 'head_staged':
        head_num_blocks = head_configs['num_blocks']
        head_img_hw = head_configs['img_hw']
        head_input_channels = head_configs['input_channels']
        head_main_channels = head_configs['main_channels']
        head_expansion_ratio = head_configs['expansion_ratio']
        head_expansion_dim = [head_main_channels[i] * head_expansion_ratio[i] for i in range(len(head_expansion_ratio))]
        head_kernel_sz = head_configs['kernel_size']
        head_stride = head_configs['stride']
        head_padding = head_configs['padding']
        head_dilation = head_configs['dilation']
        head_groups = head_configs['groups']
        head_use_se = head_configs['use_se']
        head_squeeze_ratio = head_configs['squeeze_ratio']
        head_transition_kersz = head_configs['transition_kernel_size']
        head_transition_stride = head_configs['transition_stride']
        head_transition_padding = head_configs['transition_padding']
        head_transition_out_padding = head_configs['transition_out_padding']
        head_norm_mode = head_configs['norm_mode']
        
        if decoder_dp_schedule =='linear':
            dp_list, dp_mode = create_linear_p(decoder_num_blocks+head_num_blocks, dp_mode=decoder_dp_mode, last_p=decoder_dp_prob)
        elif decoder_dp_schedule == 'uniform':
            dp_list, dp_mode = create_uniform_p(decoder_num_blocks+head_num_blocks, dp_mode=decoder_dp_mode, last_p=decoder_dp_prob)
        
        head_module = DStemStaged(num_blocks=head_num_blocks, img_hw=head_img_hw, input_channels=head_input_channels, 
                                 main_channels=head_main_channels, expansion_channels=head_expansion_dim,
                                 kernel_sz=head_kernel_sz, stride=head_stride, padding=head_padding, dilation=head_dilation, groups=head_groups, 
                                 droprate=dp_list[3:], drop_mode=dp_mode[3:],
                                 use_se=head_use_se, squeeze_ratio=head_squeeze_ratio, transition_kersz=head_transition_kersz, 
                                 transition_stride=head_transition_stride, transition_padding=head_transition_padding,
                                 transition_out_padding=head_transition_out_padding, norm_mode=head_norm_mode, device=device_config)

    unet_decoder = Decoder(num_blocks=decoder_num_blocks, img_hw=decoder_img_hw, main_channels=decoder_main_channels,
                           expansion_dim=decoder_expansion_dim, kernel_sz=decoder_kernel_sz, stride=decoder_stride, 
                           padding=decoder_padding, dilation=decoder_dilation, groups=decoder_groups, droprate=dp_list, drop_mode=dp_mode,
                           use_se=decoder_use_se, squeeze_ratio=decoder_squeeze_ratio, encoder_channels=list(reversed(encoder_main_channels)),
                           transition_kersz=decoder_transition_kersz, transition_stride=decoder_transition_stride,
                           transition_padding=decoder_transition_padding, transition_out_padding=decoder_transition_out_padding, norm_mode=decoder_norm_mode,
                           head=head_module, device=device_config)

    # intialize unet model
    convnext_unet = UNet(encoder=unet_encoder, decoder=unet_decoder, num_cls=num_cls_config, output_mode=output_mode_config)
    
    return convnext_unet
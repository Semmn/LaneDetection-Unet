import numpy as np
import torch

# get dice score for multi-class segmentation
# pred_mask: (B,C,H,W), mask: (B,C,H,W)
# mask must be one-hot encoded features.
def get_dice_score(pred_mask, mask):
    intersection = np.sum(pred_mask * mask)
    union = np.sum(pred_mask) + np.sum(mask)
    dice = intersection*2 / union
    return dice

# get iou score for multi-class segmentation
# pred_mask: (B,C,H,W), mask: (B,C,H,W)
# mask must be one-hot encoded features.
def get_iou_score(pred_mask, mask):
    intersection = np.sum(pred_mask * mask)
    union = np.sum(pred_mask) + np.sum(mask) - intersection
    iou = intersection / union
    return iou

# metrics for Lane IoU calculation
# use this metric when model outputs background information as 0th channel
# this metric calculates only IoU of lane pixels between prediction and ground truth mask
# if your model outputs channels excluding background, use get_iou_score, get_dice_score instead
def get_lane_score(pred_mask, mask):
    pred_mask = np.argmax(pred_mask, axis=1) # (B, H, W)
    mask = np.argmax(mask, axis=1)           # (B, H, W)
    intersection = np.sum(np.logical_and(pred_mask!=0, pred_mask==mask)) # exclude background pixels calculating IoU with mask pixels
    union = np.sum(pred_mask!=0) + np.sum(mask!=0) - intersection
    lane_iou = intersection / union
    return lane_iou


# metrics for masked region pixel reconstruction
# it returns mse value between output of model and ground truth label for masked region
# shape of image and label are same as (B, C, H, W)
# mask_location is binary tensor which 0 means background and 1 means masked region whoose shape (B, H, W)
# this function can be used for both unsupervised masked pixel reconstruction and self-supervised masked pixel reconstruction
def get_masked_pixel_reconst(image, label, mask_location):
    # mask key returns corresponding flatten pixel values
    mask_location = mask_location.unsqueeze(1).expand(-1, image.shape[1], -1, -1) # add channel dimension and expand dimension of channel to match with image, label
    
    mask_output = image[mask_location]
    mask_label = label[mask_location] 
    masked_pixel_reconst = ((mask_output - mask_label)**2).mean()
    return masked_pixel_reconst

# this functions returns mse values of between model output and label (normalized original image)
def get_pixel_reconst(image, label):
    pixel_reconst = ((image-label)**2).mean()
    return pixel_reconst
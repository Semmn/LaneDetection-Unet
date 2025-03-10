import torch

# Dice loss for multi-class segmentation (cls>=2)
# this loss calculates dice coefficient between prediction and ground truth mask and returns dice_loss
# in forward() method, pred_mask and mask should have shape (B, cls, H, W) and both must be same shape (mask must have one-hot encoded form)
# When background channel is included in mask and imbalance between foreground and background is severe you can use LaneLoss instead
class DICELoss(torch.nn.Module):
    def __init__(self, weights=None, num_cls=4):
        super(DICELoss, self).__init__()
        if weights!=None and len(weights) != num_cls:
            raise Exception("number of weights should be equal to number of classes")
        self.weights=weights # when weights are None, all classes have same weight (treated equally)
        self.num_cls = num_cls
    
    def forward(self, pred_mask, mask):
        intersection = torch.tensor(0.0, requires_grad=True)
        union = torch.tensor(0.0, requires_grad=True)
        if self.weights != None:
            for i in range(self.num_cls):
                intersection = intersection + (pred_mask[:, i, :, :] * mask[:, i, :, :]).sum() * self.weights[i]
                union = union + (torch.sum(pred_mask[:, i, :, :]) + torch.sum(mask[:, i, :, :])) * self.weights[i]
                
            dice = intersection * 2 / union
            dice_loss = 1 - dice
        else:
            intersection = (pred_mask * mask).sum()
            union = torch.sum(pred_mask) + torch.sum(mask)
            
            dice = intersection*2 / union
            dice_loss = 1 - dice
        return dice_loss

# use this when you have background channel in your mask
# background channel should be encoded as 0th channel
# this loss ignores background information and calculates only IoU of lane part
# when the background and foreground pixel imbalance is severe, this loss can be useful
class LaneDICELoss(torch.nn.Module):
    def __init__(self, num_cls=5):
        super(LaneDICELoss, self).__init__()
        self.num_cls = num_cls
    def forward(self, pred_mask, mask):
        intersection = (pred_mask[:, 1:, :, :] * mask[:, 1:, :, :]).sum()
        union = (torch.sum(pred_mask[:, 1:, :, :]) + torch.sum(mask[:, 1:, :, :]))
            
        dice = intersection * 2 / union
        dice_loss = 1 - dice
        return dice_loss


# this loss return only masked pixel reconstruction loss (mse)
# instead of calculating all the pixel regions, this loss calculate only masked region
# image, label: (B, C, H, W), mask_location: (B, H, W)
# mask_location only indicates the masked spatial location on the image
# identically works same as metrics.get_masked_pixel_reconst() function
class MaskedPixelReconstructLoss(torch.nn.Module):
    def __init__(self):
        super(MaskedPixelReconstructLoss, self).__init__()
    
    def forward(self, image, label, mask_location):
        mask_location = mask_location.unsqueeze(1).expand(-1, image.shape[1], -1, -1) # add channel dimension and expand dimension of channel to match with image, label
    
        masked_output = image[mask_location]
        masked_label = label[mask_location]
        masked_pixel_reconstruct = ((masked_output - masked_label)**2).mean()
        return masked_pixel_reconstruct
from PIL import Image
import numpy as np
import torch

# Dataset class for CULane segmentation dataset
# this inputs image and segmentation mask path list and return image and one-hot encoded mask
# if is_test=True, it return only image as test dataset doesn't have segmentation label
# transforms should be applied to both image and mask
# num_classes should include background class. you can decide whether to exclude background class in label mask or not
# remember that mask_onehot and image shape must be same. if you exclude background class in mask, your model should output num_classes-1 channels
class CULaneSegDataset(torch.utils.data.Dataset):
    def __init__(self, segmentation_img_list, segmentation_mask_list, transforms=None, num_classes=5, is_test=False):
        self.segmentation_img_list = segmentation_img_list
        self.segmentation_mask_list = segmentation_mask_list
        self.transforms = transforms # transforms that are only related to source image
        self.is_test = is_test
        self.num_classes = num_classes # number of classes for one-hot encoding
        
    def __len__(self):
        return len(self.segmentation_img_list)
    
    def __getitem__(self, idx):
        if self.is_test:
            image = Image.open(self.segmentation_img_list[idx])
            image = image.convert('RGB')
            image = np.array(image)
            image = image.astype(np.float32)
            image = image/255. # normalize image via simply dividing by 255
            
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed['image']
            return image
        else:
            image = Image.open(self.segmentation_img_list[idx])
            image = image.convert('RGB') # to ensure all image have 3 RGB channels format
            mask = Image.open(self.segmentation_mask_list[idx])
            image = np.array(image)
            mask = np.array(mask)
            
            image = image.astype(np.float32)
            image = image/255.
            
            if self.transforms:
                # albumentation transforms applies to both src image and segmentation mask
                transformed = self.transforms(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            mask_onehot = torch.nn.functional.one_hot(mask.long(), num_classes=self.num_classes)
            mask_onehot = mask_onehot.permute(2,0,1) # (H,W,C) -> (C, H, W)
            return image, mask_onehot
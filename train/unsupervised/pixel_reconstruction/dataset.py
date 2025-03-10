import torch
import numpy as np
import albumentations as A
from PIL import Image

# Masked Autoencoders Are Scalable Vision Learners (https://arxiv.org/pdf/2111.06377)
# this applies smoothly with transformer settings but we can also apply this to CNN based models
# however, as ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders refer, directly applying it
# can degrade the performance of CNN based models. (how to predict using only mask patches are not clear)
class MaskedCULaneDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, img_size, mask_window_size, mask_ratio):
        self.img_list = img_list
        self.img_size = img_size # default to (224, 672)
        self.mask_window_size = mask_window_size # (H, W) format
        self.mask_ratio = mask_ratio
        
        if img_size[0] % mask_window_size[0] !=0:
            raise Exception("height must be divisible by height of mask_window_size (H,W) format!")
        if img_size[1] % mask_window_size[1] != 0:
            raise Exception("width must be divisible by width of mask_window_size (H,W) format!")
        
        self.transforms = A.Compose([
            A.Resize(img_size[0], img_size[1]),
        ])
    
    def __len__(self):
        return len(self.img_list)
    
    # input: transformed (resized) images
    # output: masked images
    # this functions masks images randomly with given label ratio. masks are done in patch-wise manner 
    # and each patch is rectangular shape which its shape defined by mask_window_size parameter
    def _mask_images(self, transformed_images):
        H, W, C = transformed_images.shape
        transformed_images = torch.reshape(transformed_images, (H//self.mask_window_size[0], self.mask_window_size[0], 
                                                              W//self.mask_window_size[1], self.mask_window_size[1], C))
        transformed_images = transformed_images.swapaxes(1, 2).reshape(-1, self.mask_window_size[0], self.mask_window_size[1], C)
        N = H//self.mask_window_size[0] * W//self.mask_window_size[1] # (N, mask_window_size[0], mask_window_size[1], C)
        
        random_keys = torch.randperm(N)
        _, original_keys = torch.sort(random_keys) # get keys that keeps the original shuffle ordering
        
        transformed_images = torch.index_select(transformed_images, dim=0, index=random_keys) # random shuffle patches
        transformed_images[:int(N*self.mask_ratio), :, :] = 0 # masking
        masked_image = torch.index_select(transformed_images, dim=0, index=original_keys)
        
        masked_image = torch.reshape(masked_image, (H//self.mask_window_size[0], W//self.mask_window_size[1], self.mask_window_size[0],
                                                    self.mask_window_size[1], C)).swapaxes(1, 2)
        masked_image = torch.reshape(masked_image, (H, W, C)) # restore to original image shape
        
        return masked_image
        
    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]) # read image
        image = image.convert('RGB') # to ensure all image have 3 RGB channels format
        image = np.array(image)      # convert to numpy array
    
        transformed = self.transforms(image=image)
        image = transformed['image']
        
        image = image/255. # normalize image to 0~1 (simply divide it by 255)
        image = torch.tensor(image)
        label = image.clone() # copy the image tensors
        
        masked_image = self._mask_images(image)
        masked_image = masked_image.permute(2, 0, 1) # change to chanel first format
        label = label.permute(2, 0, 1) # change to channel first format
        
        return masked_image.float(), label.float()


# in this case, the task is to predict the pixel values of the lane-masked images. (but the loss will be calculated with whole image pixel values)
class LaneMaskedCULaneDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, mask_list, img_size, masking_width):
        self.img_list = img_list
        self.mask_list = mask_list
        self.img_size = img_size
        self.masking_width = masking_width # width of masking that expands horizontally to lane masks
        self.transforms = A.Compose([
            A.Resize(img_size[0], img_size[1]),
        ])
    
    def __len__(self):
        return len(self.img_list)
    
    def _mask_lanes(self, transformed_images, masks):
        H, W, C = transformed_images.shape # H, W, C
        transformed_images[masks!=0, :] = 0 # masking lane markings
        
        return transformed_images
    
    # masking_width are not applied constant qantity, rather it applies as stochastic range.
    def _mask_lanes_w(self, image, masks):
        H, W, C = image.shape
        random_noise = np.arange(self.masking_width) 
        skip_flag = False # flag for skip intialized as False
        
        for i in range(H):
            for j in range(W):
                if skip_flag:
                    if masks[i, j]==0: # until background pixels re-appears
                        p_range = j + np.random.choice(random_noise)
                        if p_range < W: # must not exceed image boundary
                            masks[i, j:p_range] = -1 # -1 indicates it is masked and should not be considered as lanes
                        skip_flag=False # update skip_flag to False
                    else:    
                        continue
                    
                # when search finds non-background (lane pixels) it removes the horizontally left-part of the image with random range
                if masks[i, j] != 0 and masks[i,j]!=-1: # -1 cannot be interpreted as lanes
                    m_range = j - np.random.choice(random_noise)
                    if m_range >=0: # must not exceed image boundary
                        masks[i, m_range:j] = masks[i, j]
                        skip_flag = True # skip_flag on, skipping until background pixels appears in horizontal direction.
                        
        masking = masks!=0
        image[masking, :] = 0
        return image
    
    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx])
        lane_mask = Image.open(self.mask_list[idx])
        
        image = image.convert('RGB')
        
        image = np.array(image)
        lane_mask = np.array(lane_mask)
        
        transformed = self.transforms(image=image, mask=lane_mask) # transforms both image and mask to specific size
        image = transformed['image']
        lane_mask = transformed['mask']
        
        image = image/255. # images are simply normalized by dividing by 255
        image = torch.tensor(image)
        
        label = image.clone() # original non-masked images are used as labels
        label = label.permute(2, 0, 1) # convert to channel first format
        
        if self.masking_width == 0:
            masked_image = self._mask_lanes(image, lane_mask)
        else:
            masked_image = self._mask_lanes_w(image, lane_mask)
        masked_image = masked_image.permute(2, 0, 1) # convert to channel first format
        return masked_image.float(), label.float()
    

# Masked Unsupervised Dataset for multi-dataset (culane, bdd100k, tusimple, llamas)
# order of each parameter list must be matched with the each dataset you want to use
# in case of using convnext model, img size must be fixed because of layer_norm operation.
class MaskedMultiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict:dict, img_size:list, target_size:list, mask_window_size:list, mask_ratio:float):
        self.data_dict = data_dict # dictionary that matches name of dataset and images
        self.img_size = img_size # img_size is given as list, indicating multi-scale, multi-ratio pre-training is possible.
        self.target_size = target_size # usually output from model must be same as input, but it can be different for some cases
        self.mask_window_size = mask_window_size
        self.mask_ratio = mask_ratio
        
        # check if each given img_size is divisible by mask_window_size
        for i, size in enumerate(img_size, 0):
            if size[0] % mask_window_size[i][0] != 0:
                raise Exception("height must be divisible by height of mask_window_size")
            if size[1] % mask_window_size[i][1] != 0:
                raise Exception("width must be divisible by width of mask_window_size")
        
        # note that each image from different dataset must be restored by decoder as same size for each dataset
        self.size_transforms = []
        for size in img_size:
            self.size_transforms.append(A.Compose([
                A.Resize(size[0], size[1]),
            ]))
        
        self.mask_size_transforms = []
        for size in target_size:
            self.mask_size_transforms.append(A.Compose([
                A.Resize(size[0], size[1]),
            ]))
        
        # mapper which dataset name corresponds to mask_window size, size_transforms (just simply following sequential order)
        self.indicators = {k:i for i, k in enumerate(data_dict.keys(), 0)}
        
        self.datasets = []
        for key, value in self.data_dict.items():
            for img_path in value:
                self.datasets.append((img_path, key)) # key must be present to indicate which image belongs to which dataset

    def __len__(self):
        return len(self.datasets)
    
    # input: transformed (resized) images
    # output: masked images
    # this functions masks images randomly with given label ratio. masks are done in patch-wise manner 
    # and each patch is rectangular shape which its shape defined by mask_window_size parameter
    def _mask_images(self, transformed_images, key):
        H, W, C = transformed_images.shape
        mask_window_h = self.mask_window_size[self.indicators[key]][0]
        mask_window_w = self.mask_window_size[self.indicators[key]][1]
        
        transformed_images = torch.reshape(transformed_images, (H//mask_window_h, mask_window_h, 
                                                              W//mask_window_w, mask_window_w, C))
        transformed_images = transformed_images.swapaxes(1, 2).reshape(-1, mask_window_h, mask_window_w, C)
        N = H//mask_window_h * W//mask_window_w # (N, mask_window_h, mask_window_w, C)
        
        random_keys = torch.randperm(N)
        _, original_keys = torch.sort(random_keys) # get keys that keeps the original shuffle ordering
        
        transformed_images = torch.index_select(transformed_images, dim=0, index=random_keys) # random shuffle patches
        transformed_images[:int(N*self.mask_ratio), :, :] = 0 # masking
        masked_image = torch.index_select(transformed_images, dim=0, index=original_keys)
        
        masked_image = torch.reshape(masked_image, (H//mask_window_h, W//mask_window_w, mask_window_h,
                                                    mask_window_w, C)).swapaxes(1, 2)
        masked_image = torch.reshape(masked_image, (H, W, C)) # restore to original image shape
        return masked_image
    
    def __getitem__(self, idx):
        path, key = self.datasets[idx]
        image = Image.open(path) # read image
        image = image.convert('RGB') # to ensure all image have 3 RGB channels format
        image = np.array(image)      # convert to numpy array
        label = np.copy(image)       # copy original image for labels
    
        transformed = self.size_transforms[self.indicators[key]](image=image)
        image = transformed['image']
        
        transformed_mask = self.mask_size_transforms[self.indicators[key]](image=label)
        label = transformed_mask['image']
        
        image = image/255. # normalize image to 0~1 (simply divide it by 255)
        label = label/255.
        
        image = torch.tensor(image)
        label = torch.tensor(label)
        
        masked_image = self._mask_images(image, key)
        masked_image = masked_image.permute(2, 0, 1) # change to chanel first format
        label = label.permute(2, 0, 1) # change to channel first format
        
        return masked_image.float(), label.float() # last convert to float type tensor

# Lane centered masked supervised dataset for multi-dataset (culane, bdd100k, tusimple, llamas)
class LaneMaskedMultiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict:dict, mask_dict:dict, img_size:list, target_size:list, masking_width:int):    
        self.data_dict = data_dict
        self.mask_dict = mask_dict
        self.img_size = img_size
        self.target_size = target_size
        self.masking_width = masking_width
        
        self.size_transforms = []
        for size in img_size:
            self.size_transforms.append(A.Compose([
                A.Resize(size[0], size[1]), # H, W format
            ]))
        
        self.mask_size_transforms = []
        for size in target_size:
            self.mask_size_transforms.append(A.Compose([
                A.Resize(size[0], size[1]) # H, W format
            ]))
        
        self.indicators = {k:i for i, k in enumerate(data_dict.keys(), 0)}
        
        self.datasets = []
        self.masks = []
        for key, value in self.data_dict.items():
            for path in value:
                self.datasets.append((path, key))
        for key, value in self.mask_dict.items():
            for path in value:
                self.masks.append((path, key))
    
    def __len__(self):
        return len(self.datasets)
    
    def _mask_lanes(self, transformed_images, masks):
        transformed_images[masks!=0, :] = 0 # masking lane markings
        return transformed_images
    
    # masking_width are not applied constant qantity, rather it applies as stochastic range.
    def _mask_lanes_w(self, image, masks):
        H, W, C = image.shape
        random_noise = np.arange(self.masking_width) 
        skip_flag = False # flag for skip intialized as False
        
        for i in range(H):
            for j in range(W):
                if skip_flag:
                    if masks[i, j]==0: # until background pixels re-appears
                        p_range = j + np.random.choice(random_noise)
                        if p_range < W: # must not exceed image boundary
                            masks[i, j:p_range] = 255 # 255 indicates it is masked and should not be considered as lanes
                        skip_flag=False # update skip_flag to False
                    else:    
                        continue
                    
                # when search finds non-background (lane pixels) it removes the horizontally left-part of the image with random range
                if masks[i, j] != 0 and masks[i,j]!=255: # 255 cannot be interpreted as lanes
                    m_range = j - np.random.choice(random_noise)
                    if m_range >=0: # must not exceed image boundary
                        masks[i, m_range:j] = masks[i, j]
                        skip_flag = True # skip_flag on, skipping until background pixels appears in horizontal direction.
                        
        masking = masks!=0
        image[masking, :] = 0
        return image
    
    def __getitem__(self, idx):
        img_path, key = self.datasets[idx]
        mask_path, _ = self.masks[idx]
        
        image = Image.open(img_path)
        image = image.convert('RGB')
        lane_mask = Image.open(mask_path)
        
        # in case of bdd100k (in case of color_maps labels) convert to RGB format ensures background informations are added. (H, W, C)
        # if you use bdd100k (masks labels), you just get (H, W) format, where pixel value 255 indicates background information, other values indicates various lane markings.
        if key=='bdd':
            lane_mask = lane_mask.convert('RGB') 
        
        image = np.array(image)
        lane_mask = np.array(lane_mask)
        label = np.copy(image) # copy image before convert to label
        
        # as we will treat culane mask as standard, we will process only llamas and tusimple, bdd dataset
        # culane format : (H, W) where each pixel value indicates lane+background information (0:background, 1~: lane)
        if key in ['llamas', 'bdd', 'tusimple']:
            lane_mask = np.max(lane_mask, axis=2) # channel-wise max operation makes the same format as culane mask format
            
        # input image and lane_mask must have same size
        transformed = self.size_transforms[self.indicators[key]](image=image, mask=lane_mask) # transforms both image and mask to specific size
        image = transformed['image']
        lane_mask = transformed['mask']
        
        # images are simply normalized by dividing by 255
        image = image/255.
        
        # transform label image to target_size
        transformed_label = self.mask_size_transforms[self.indicators[key]](image=label)
        label = transformed_label['image']
        label = label/255.
        
        image = torch.tensor(image)
        label = torch.tensor(label)
        label = label.permute(2, 0, 1) # convert to channel first format
        
        if self.masking_width == 0:
            masked_image = self._mask_lanes(image, lane_mask)
        else:
            masked_image = self._mask_lanes_w(image, lane_mask)
        masked_image = masked_image.permute(2, 0, 1) # convert to channel first format

        return masked_image.float(), label.float()
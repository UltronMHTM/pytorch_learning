from torch.utils.data import Dataset
from skimage import io
import os
import torch
from config import *
import numpy as np
#   one image contain only one class, class num are 3
class CardiacSet(Dataset):
    def __init__(self, root_dir):
        img_path = os.path.join(root_dir, "cardiac/image/train/A2C")
        mask_path = os.path.join(root_dir, "cardiac/label/train/A2C")
        img_names = os.listdir(img_path)
        self.images = []
        self.masks = []
        for name in img_names:
            self.images.append(io.imread(os.path.join(img_path, name)))
            self.masks.append(io.imread(os.path.join(mask_path, name)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # img
        img = self.images[index]
        img = torch.tensor(img, dtype=torch.float32)
        h = img.shape[0]
        w = img.shape[1]
        img = torch.reshape(img, (1,h,w))
        # mask
        mask = self.masks[index]
        out_mask = torch.zeros((class_num,h,w), dtype=torch.float)
        for i in range(class_num):
            tmp = out_mask[i, :, :]
            tmp[mask == i] = 1
            out_mask[i, :, :] = tmp
        sample = [img, out_mask]
        return sample

#     all classes are in the same image, class num are 3 + 1(background) = 4
class CardiacSet_update(Dataset):
    def __init__(self, root_dir):
        img_path = os.path.join(root_dir, "cardiac/image/train/A2C")
        mask_path = os.path.join(root_dir, "cardiac/label/train/A2C")
        img_names = os.listdir(img_path)
        self.images = []
        self.masks = []
        for name in img_names:
            # img
            img = torch.tensor(io.imread(os.path.join(img_path, name)), dtype=torch.float32)
            h = img.shape[0]
            w = img.shape[1]
            img = torch.reshape(img, (1,h,w))
            img = torch.reshape(img, (1,h,w))
            # mask
            mask = io.imread(os.path.join(mask_path, name)).astype(np.int16)
            mask = torch.LongTensor(mask)
            count = mask.flatten().bincount()
            labels = torch.nonzero(count)
            if len(labels) != class_num:
                # self.images.append(self.images[-1])
                # self.masks.append(self.masks[-1])
                continue
            for i in range(len(labels)):
                mask[mask==labels[i]] = i
                
            self.images.append(img)
            self.masks.append(mask)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]
        sample = [img, mask]
        return sample

        
        # img = torch.tensor(img, dtype=torch.float32)
        # h = img.shape[0]
        # w = img.shape[1]
        # img = torch.reshape(img, (1,h,w))
        # # mask
        # mask = self.masks[index].astype(np.int16)
        # mask = torch.LongTensor(mask)
        # count = mask.flatten().bincount()
        # labels = torch.nonzero(count)
        # for i in range(len(labels)):
        #     mask[mask==labels[i]] = i
        # # sample = [img, mask]
        # # return sample
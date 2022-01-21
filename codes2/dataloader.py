from torch.utils.data import Dataset
from skimage import io
import os
from config import *
import torch

class CardiacSet(Dataset):
    def __init__(self, root_dir):
        img_path = os.path.join(root_dir, "cardiac/image/train/A2C")
        mask_path = os.path.join(root_dir, "cardiac/label/train/A2C")
        img_names = os.listdir(img_path)
        self.imgs = []
        self.masks = []
        for name in img_names:
            self.imgs.append(io.imread(os.path.join(img_path, name)))
            self.masks.append(io.imread(os.path.join(mask_path, name)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # img
        img = self.imgs[index]
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
from torch.utils.data import Dataset
from skimage import io
import os
import torch

class MnistData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        img_list = []
        label_list = os.listdir(self.root_dir)
        for label in label_list:
            file_names = os.listdir(os.path.join(self.root_dir, label))
            for name in file_names:
                img_list.append(label + "&" + name)
        self.images = img_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_label = image_index.split("&")[0]
        img_name = image_index.split("&")[1]
        img_path = os.path.join(self.root_dir,img_label)
        img_path = os.path.join(img_path, img_name)
        img = io.imread(img_path)
        sample = [img, img_label]
        return sample
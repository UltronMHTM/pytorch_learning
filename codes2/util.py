import torch
import matplotlib.pyplot as plt
import numpy as np
import collections
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from config import *

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# inputs must be 2 binary tensors [N, C, H, W] with same size
def compute_IOU(mask1_batches, mask2_batches):
    iou_rate = torch.tensor([0],dtype=torch.float)
    for mask1_batch, mask2_batch in zip(mask1_batches, mask2_batches):
        for mask1, mask2 in zip(mask1_batch,mask2_batch):
            U_region = mask1 + mask2
            I_region = mask1 * mask2
            U_r = U_region.sum()
            I_r = I_region.sum()
            iou_rate += I_r/(U_r-I_r)

    return iou_rate/(mask1_batches.shape[0]*mask1_batches.shape[1])

def binarization(mask_batches):
    mask_batches[mask_batches <= threshold] = 0
    mask_batches[mask_batches > threshold] = 1
    return mask_batches

# input mask_batches -> [N, C, H, W], true_batches -> [N, H, W]
def binarization_update(mask_batches,true_batches):
    out_true_batches = torch.zeros(mask_batches.shape, dtype=torch.float)
    for i in range(len(mask_batches)):
        mask_c = mask_batches[i]
        maxval, channel = torch.max(mask_c,0)
        for c in range(len(mask_c)):
            mask_c[c][mask_c[c]==maxval] = 1
            mask_c[c][mask_c[c]!=1] = 0
        mask_batches[i] = mask_c      
        h = true_batches.shape[-2]
        w = true_batches.shape[-1]
        for j in range(class_num):
            out_true_batches[i, j][true_batches[i] == j] = 1
        # print("check true")
        # print("true shape = ", out_true_batches.shape)
        # print("channel bin = ", channel.flatten().long().bincount())
        # # print("mask_c bin1 = ", channel.flatten().long().bincount())
        # # print("mask_c bin2 = ", mask_c[0].flatten().long().bincount())
        # # print("mask_c bin3 = ", mask_c[0].flatten().long().bincount())
        # print("---------------------------------------------------------------------------")
    return mask_batches, out_true_batches

# def binarization_update_old(mask_batches):
#     for i in range(len(mask_batches)):
#         mask_c = mask_batches[i]
#         maxval, channel = torch.max(mask_c,0)
#         print(maxval.shape)
#         for c in range(len(mask_c)):
#             mask = mask_c[c]
#             print(mask.shape)
#             mask[mask==maxval] = 1
#             mask[mask!=maxval] = 0
#             mask_c[c] = mask
#             print(mask)
#         mask_batches[i] = mask_c 
#     return mask_batches

# [C, H, W]
def visulization(img, predict_masks, true_masks, index):
    _, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img.astype(np.uint8))
    height, width = img.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    # loop in 3 classes
    for i in range(1,len(predict_masks)):
        pred_m = predict_masks[i]
        true_m = true_masks[i]
        # pred_m
        padded_mask = np.zeros(
            (pred_m.shape[0] + 2, pred_m.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = pred_m
        contours = find_contours(padded_mask, 0.5)
        max_len = 0
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            if max_len < len(verts):
                mask_contours = verts
        mask_contours = np.fliplr(mask_contours) - 1
        p = Polygon(mask_contours, facecolor="none", edgecolor=pred_color[i-1])
        ax.add_patch(p)

        # true_m
        padded_mask = np.zeros(
            (true_m.shape[0] + 2, true_m.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = true_m
        contours = find_contours(padded_mask, 0.5)
        max_len = 0
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            if max_len < len(verts):
                mask_contours = verts
        mask_contours = np.fliplr(mask_contours) - 1
        p = Polygon(mask_contours, facecolor="none", edgecolor=true_color[i-1])
        ax.add_patch(p)
        
    plt.savefig("../visualization/visual%d.png" % index)
    plt.show()
    return ax

    # ax.imshow(masked_image.astype(np.uint8))
    # plt.show()
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
from matplotlib.patches import Polygon

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
    iou_rate = torch.tensor([0])
    for mask1_batch, mask2_batch in zip(mask1_batches, mask2_batches):
        for mask1, mask2 in zip(mask1_batch,mask2_batch):
            U_region = mask1 + mask2
            I_region = mask1 * mask2
            U_r = U_region.sum()
            I_r = I_region.sum()
            iou_rate += I_r/U_r
    return iou_rate/(mask1_batches.shape[-1]*mask1_batches.shape[-2])

def binarization(mask_batches):
    mask_batches[mask_batches <= threshold] = 0
    mask_batches[mask_batches > threshold] = 1
    return mask_batches

# [C, H, W]
def visulization(img, predict_masks, true_masks):
    _, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img.astype(np.uint8))
    height, width = img.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    # loop in 3 classes
    for pred_m, true_m in zip(predict_masks, true_masks):
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
        p = Polygon(mask_contours, facecolor="none", edgecolor=color)
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
        p = Polygon(mask_contours, facecolor="none", edgecolor=color)
        ax.add_patch(p)
    plt.show()
    return ax

    # ax.imshow(masked_image.astype(np.uint8))
    # plt.show()
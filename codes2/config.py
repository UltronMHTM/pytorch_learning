import torch

use_GPU = torch.cuda.is_available() 
input_channel = 1
class_num = 4
threshold = 0.5

val_rate = 0.1
test_rate = 0.1
n_epochs = 200
batch_size = 16
batch_size_test = 1
gpu0_bsz = 4
acc_grad = 1
worker_num = 0

GPU_UNUSE_ID = [2, 3, 7]
device = torch.device("cuda:0" if use_GPU else "cpu")
save_path = "models/running_test"
log_path = "Logs/running_test"
root_dir = "../"

visual_num = 10
figsize = (16,16)
title = "title"
true_color = ["yellow", "lime", "blue"]
pred_color = ["darkorange", "darkgreen", "darkblue"]
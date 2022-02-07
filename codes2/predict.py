import os
import torch
from util import *
from config import *
from model import U_net
from dataloader import CardiacSet_update
from data_parallel import BalancedDataParallel


def predict(root = root_dir):
    test_set = CardiacSet_update(root)
    indices = torch.tensor(range(len(test_set)))
    test_indices = indices[len(indices) - int(len(indices)*test_rate):]
    test_set = torch.utils.data.Subset(test_set, test_indices)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True,
                                               pin_memory=use_GPU, num_workers=0)
    model = U_net()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()

    path = os.path.join(root, save_path)
    last_epoch = 0
    for file in os.listdir(path):
        if last_epoch < int((file.split('_')[1]).split('.')[0]):
            last_epoch = int((file.split('_')[1]).split('.')[0]) 
    if os.path.exists(os.path.join(path, 'models_%d.dat' % last_epoch)):
        model.load_state_dict(torch.load(os.path.join(path, 'models_%d.dat' % last_epoch)))
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, mask) in enumerate(test_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                mask = mask.cuda()

            output = model(input)

            b_output, mask_output = binarization_update(output.cpu(),mask.cpu())
            t_error = compute_IOU(b_output.cpu(), mask_output)
            visulization(np.squeeze(input.cpu().numpy()), np.squeeze(b_output.cpu().numpy()), np.squeeze(mask_output.cpu().numpy()), batch_idx)
            if batch_idx >= visual_num:
                break
        
if __name__ == '__main__':
    predict()
    print("Done")
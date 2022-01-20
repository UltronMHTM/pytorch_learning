from config import *
from train import *
import time
import torch
import numpy as np
from torch.utils.data import Dataset

def init_model():
    model = MyDenseNet()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.dat')))
    model.eval()
    return model

def predict(input_imgs):
    if np.array(input_imgs).shape!=4:
        print("input images not follows: (index, channel, width, height)")
    model = init_model()
    input = torch.tensor(input, dtype=torch.float32)
    c = input.shape[0]
    w = input.shape[1]
    h = input.shape[2]
    input = torch.reshape(input, (c, 1, w, h))
    output = model(input)
    _, pred = output.data.cpu.topk(1,dim=1)
    return pred

def display_res(predict_path = root_dir+"/test"):
    test_set = MnistData(test_path)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    batch_time = AverageMeter()
    error = AverageMeter()
    model = init_model()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(test_loader):
            input = torch.tensor(input, dtype=torch.float32)
            c = input.shape[0]
            w = input.shape[1]
            h = input.shape[2]
            input = torch.reshape(input, (c, 1, w, h))
            target = [int(i) for i in target]
            target = torch.tensor(target)

            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)

            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

    return batch_time.avg, error.avg




if __name__ == '__main__':
    display_res()
    print("Done")


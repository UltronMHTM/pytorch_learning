import re
import os
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util import *
from config import *
from model import U_net
from dataloader import CardiacSet_update
from data_parallel import BalancedDataParallel
# # cuda_num =[5,6]
# os.environ["CUDA_DEVICES_ORDER"] ="PCI_BUS_ID"
# # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_num))
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train_epoch(model, loader, optimizer):
    # initialization
    batch_time = AverageMeter()
    losses = AverageMeter()
    TotalError = AverageMeter()
    class_error = []
    for i in range(class_num):
        class_error.append(AverageMeter())
    model.train()
    end = time.time()

    for batch_idx, (input, mask) in enumerate(loader):
        if use_GPU:
            input = input.cuda()
            # input.to(device)
            mask = mask.cuda()
            # mask.to(device)
        output = model(input)
        # print(len(mask[0].flatten().bincount()))
        loss = F.nll_loss(output, mask).cuda()
        # .to(device).cuda()
        # print("Loss=nan:",any(torch.isnan(loss.flatten())))

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss and iou
        b_output, mask_output = binarization_update(output.cpu(),mask.cpu())
        t_error = compute_IOU(b_output.cpu(), mask_output)
        TotalError.update(t_error/class_num, batch_size)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
    return batch_time, losses, TotalError, class_error
def test_epoch(model, loader):
    # initialization
    batch_time = AverageMeter()
    losses = AverageMeter()
    TotalError = AverageMeter()
    class_error = []
    for i in range(class_num):
        class_error.append(AverageMeter())
    model.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, mask) in enumerate(loader):
            if use_GPU:
                input = input.cuda()
                # input.to(device)
                mask = mask.cuda()
                # mask.to(device)
            output = model(input)
            loss = F.nll_loss(output, mask).cuda()
            # .to(device)
            # # softmax and BCE
            # shape = output[0][0].shape
            # for i in range(len(output)):
            #     for j in range(len(output[0])):
            #         tmp = F.softmax(output[i][j].flatten())
            #         output[i][j] = torch.reshape(tmp, shape)
            # loss = F.binary_cross_entropy(output, mask)

            # record loss and iou
            # softmax

            # shape = output[0][0].shape
            # for i in range(len(output)):
            #     for j in range(len(output[0])):
            #         tmp = F.softmax(output[i][j].flatten())
            #         output[i][j] = torch.reshape(tmp, shape)

            b_output, mask_output = binarization_update(output.cpu(),mask.cpu())
            t_error = compute_IOU(b_output.cpu(), mask_output)
            TotalError.update(t_error/class_num, batch_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
    return batch_time, losses, TotalError, class_error

def train(root = root_dir, continu = True):
    
    # if use_GPU:
    #     torch.cuda.set_device(0)

    # train, validate, test set init
    train_val_set = CardiacSet_update(root)
    test_set = CardiacSet_update(root)
    indices = torch.tensor(range(len(train_val_set)))
    # split train and validate set with the test set
    indices_train_val = indices[:len(indices) - int(len(indices)*test_rate)]
    # train_val_set = torch.utils.data.Subset(train_val_set, indices_train_val)
    # shuffle the train and validate set
    indices_train_val = torch.randperm(len(train_val_set))
    train_indices = indices_train_val[:len(indices) - int(len(indices)*val_rate) - int(len(indices)*test_rate)]
    valid_indices = indices_train_val[len(indices) - int(len(indices)*val_rate) - int(len(indices)*test_rate):
                                      len(indices) - int(len(indices)*test_rate)]
    test_indices = indices[len(indices) - int(len(indices)*test_rate):]
    train_set = torch.utils.data.Subset(train_val_set, train_indices)
    valid_set = torch.utils.data.Subset(train_val_set, valid_indices)
    test_set = torch.utils.data.Subset(test_set, test_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=use_GPU, num_workers=worker_num)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=use_GPU, num_workers=worker_num)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=use_GPU, num_workers=worker_num)
    # models init
    model = U_net()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
        model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()
        # model.to(device)
    
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of model parameter: %d -> %.2fM" % (total,total/1e6))
    path = os.path.join(root, save_path)
    
    last_epoch = 0
    if continu:
        for file in os.listdir(path):
            if last_epoch < int((file.split('_')[1]).split('.')[0]):
                last_epoch = int((file.split('_')[1]).split('.')[0]) 
        if os.path.exists(os.path.join(path, 'models_%d.dat' % last_epoch)):
            model.load_state_dict(torch.load(os.path.join(path, 'models_%d.dat' % last_epoch)))
        last_epoch = last_epoch - 1
    optimizer = torch.optim.Adam(model.parameters(),
                lr=0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.2 * n_epochs, 0.4 * n_epochs, 0.6 * n_epochs, 0.8 * n_epochs],
                                                     gamma=0.12, verbose=True)
    scheduler.last_epoch=last_epoch  # have to write like this 
    # scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=last_epoch, verbose=True)
    log_save_path = os.path.join(root,log_path)
    writer = SummaryWriter(log_save_path)
    best_iou = 0
    for epoch in range(last_epoch, n_epochs):
        scheduler.step()
        train_time, train_loss, train_iou_t, train_iou_c = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer
        )
        val_time, valid_loss, valid_iou_t, test_iou_c = test_epoch(
            model=model,
            loader=valid_loader,
        )
        test_time, test_loss, test_iou_t, test_iou_c = test_epoch(
            model=model,
            loader=test_loader,
        )

        # Determine if models is the best
        if valid_loader:
            if valid_iou_t.avg > best_iou:
                best_iou = valid_iou_t.avg
                print('New best error: %.4f' % best_iou)
                bestModel_state = model.state_dict()
                
        else:
            bestModel_state = model.state_dict()

        if epoch%20 == 0 or any(torch.isnan(train_loss.avg)):  
            torch.save(bestModel_state, os.path.join(path, 'models_%d.dat' % epoch))

        # Log results
        writer.add_scalar('train_loss', train_loss.avg, epoch)
        writer.add_scalar('train_iou', train_iou_t.avg, epoch)
        writer.add_scalar('valid_loss', valid_loss.avg, epoch)
        writer.add_scalar('valid_iou', valid_iou_t.avg, epoch)
        writer.add_scalar('test_loss', test_loss.avg, epoch)
        writer.add_scalar('test_iou', test_iou_t.avg, epoch)
        # if epoch%10 == 0:
        res = '\n\t'.join([
            'epoch = %d' % epoch,
            'train_loss = %.4f; train_iou = %.4f; train_time = %.4f' % (train_loss.avg, train_iou_t.avg, train_time.avg),
            'valid_loss = %.4f; valid_iou = %.4f; val_time = %.4f' % (valid_loss.avg, valid_iou_t.avg, val_time.avg),
            'test_loss = %.4f; test_iou = %.4f; test_time = %.4f' % (test_loss.avg, test_iou_t.avg, test_time.avg),
            '-------------------------------------------------------------------------------------------------'
        ])
        print(res)
        with open(os.path.join(log_save_path, 'results.csv'), 'a+') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_loss.avg,
                train_iou_t.avg,
                valid_loss.avg,
                valid_iou_t.avg,
                test_loss.avg,
                test_iou_t.avg
            ))

if __name__ == '__main__':
    train()
    print("Done")
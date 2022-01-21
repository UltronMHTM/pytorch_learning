import os
import time
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from config import *
from dataloader import *
from model import U_net

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

def compute_IOU(masks1, masks2):
    iou_rate = 0
    for mask1, mask2 in zip(masks1,masks2):
        U_region = mask1 + mask2
        I_region = mask1 * mask2
        U_r = sum(sum(i) for i in U_region)
        I_r = sum(sum(i) for i in I_region)
        iou_rate += I_r/U_r
    return iou_rate/len(masks1)

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
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = torch.nn.functional.cross_entropy(F.softmax(output), F.softmax(target))

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss and iou
        batch_size = target.size(0)
        pred = output.data.cpu()
        pred = pred.transpose(0,1)
        mask = mask.transpose(0,1)
        t_error = 0
        for i in range(class_num):
            iou = compute_IOU(pred, mask)
            class_error[i].update(iou, batch_size)
            t_error += iou
        TotalError.update(t_error/class_num, batch_size)
        losses.update(loss.item(), batch_size)
        # measure elapsed time
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

    for batch_idx, (input, mask) in enumerate(loader):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = torch.nn.functional.cross_entropy(F.softmax(output), F.softmax(target))

        # record loss and iou
        batch_size = target.size(0)
        pred = output.data.cpu()
        pred = pred.transpose(0,1)
        mask = mask.transpose(0,1)
        t_error = 0
        for i in range(class_num):
            iou = compute_IOU(pred, mask)
            class_error[i].update(iou, batch_size)
            t_error += iou
        TotalError.update(t_error/class_num, batch_size)
        losses.update(loss.item(), batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return batch_time, losses, TotalError, class_error

def train(root = root_dir):
    # train, validate, test set init
    train_set = CardiacSet(root)
    val_set = CardiacSet(root)
    test_set = CardiacSet(root)
    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices) - int(len(indices)*val_rate) - int(len(indices)*test_rate)]
    valid_indices = indices[len(indices) - int(len(indices)*val_rate) - int(len(indices)*test_rate):
                            len(indices) - int(len(indices)*test_rate)]
    test_indices = indices[len(indices) - int(len(indices)*test_rate):]

    train_set = torch.utils.data.Subset(train_set, train_indices)
    valid_set = torch.utils.data.Subset(val_set, valid_indices)
    test_set = torch.utils.data.Subset(test_set, test_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)

    # models init
    model = U_net()
    if os.path.exists(os.path.join(save_path, 'models.dat')):
        path = os.path.join(root, save_path)
        model.load_state_dict(torch.load(os.path.join(path, 'models.dat')))
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.25 * n_epochs, 0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.2)

    best_iou = 0
    for epoch in range(n_epochs):
        _, train_loss, train_iou_t, train_iou_c = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer
        )
        scheduler.step()
        _, valid_loss, valid_iou_t, test_iou_c = test_epoch(
            model=model,
            loader=valid_loader,
        )
        _, test_loss, test_iou_t, test_iou_c = test_epoch(
            model=model,
            loader=test_loader,
        )

        #
        # *******************************
        if epoch%10 == 0:
            res = '\n\t'.join([
                'train_loss = %.4f; train_iou = %.4f' % (train_loss.avg, train_iou_t.avg),
                'valid_loss = %.4f; valid_iou = %.4f' % (valid_loss.avg, valid_iou_t.avg),
                'test_loss = %.4f; test_iou = %.4f' % (test_loss.avg, test_iou_t.avg),
            ])
            print(res)

        # Determine if models is the best
        if valid_loader:
            if valid_iou_t.avg > best_iou:
                best_iou = valid_iou_t.avg
                print('New best error: %.4f' % best_iou)
                torch.save(model.state_dict(), os.path.join(save_path, 'models.dat'))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, 'models.dat'))

        # Log results
        with open(os.path.join(save_path, 'results.csv'), 'a') as f:
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
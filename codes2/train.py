import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from config import *
from dataloader import CardiacSet
from util import *
from model import U_net
from torch.utils.tensorboard import SummaryWriter
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
            mask = mask.cuda()
        
        output = model(input)
        # softmax and BCE
        shape = output[0][0].shape
        for i in range(len(output)):
            for j in range(len(output[0])):
                tmp = F.softmax(output[i][j].flatten())
                output[i][j] = torch.reshape(tmp, shape)
        loss = F.binary_cross_entropy(output, mask)
        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record loss and iou
        batch_size = mask.shape[0]
        b_output = binarization(output.cpu())
        mask = mask.cpu()
        t_error = compute_IOU(b_output, mask)
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

    for batch_idx, (input, mask) in enumerate(loader):
        if torch.cuda.is_available():
            input = input.cuda()
            mask = mask.cuda()

        output = model(input)
        # softmax and BCE
        shape = output[0][0].shape
        for i in range(len(output)):
            for j in range(len(output[0])):
                tmp = F.softmax(output[i][j].flatten())
                output[i][j] = torch.reshape(tmp, shape)
        loss = F.binary_cross_entropy(output, mask)
        # record loss and iou
        batch_size = mask.shape[0]
        b_output = binarization(output.cpu())
        mask = mask.cpu()
        t_error = compute_IOU(b_output, mask)
        TotalError.update(t_error / class_num, batch_size)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

    return batch_time, losses, TotalError, class_error

def train(root = root_dir,continu = True):
    # train, validate, test set init
    train_val_set = CardiacSet(root)
    test_set = CardiacSet(root)
    indices = torch.tensor(range(len(train_val_set)))
    # split train and validate set with the test set
    indices_train_val = indices[:len(indices) - int(len(indices)*test_rate)]
    train_val_set = torch.utils.data.Subset(train_val_set, indices_train_val)
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
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)

    # models init
    model = U_net()
    path = os.path.join(root, save_path)
    last_epoch = 0
    if continu:
        for file in os.listdir(path):
            if last_epoch < int(file.split('_')[1]):
                last_epoch = int(file.split('_')[1])
        if os.path.exists(os.path.join(path, 'models_%d.dat' % last_epoch)):
            model.load_state_dict(torch.load(os.path.join(path, 'models_%d.dat' % last_epoch)))
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.25 * n_epochs, 0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.2)

    writer = SummaryWriter(os.path.join(root,log_path))
    best_iou = 0
    for epoch in range(last_epoch, n_epochs):
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

        # Determine if models is the best
        if valid_loader:
            if valid_iou_t.avg > best_iou:
                best_iou = valid_iou_t.avg
                print('New best error: %.4f' % best_iou)
                torch.save(model.state_dict(), os.path.join(path, 'models_%d.dat' % epoch))
        else:
            torch.save(model.state_dict(), os.path.join(path, 'models_%d.dat' % epoch))

        # Log results
        writer.add_scalar('train_loss', train_loss.avg, epoch)
        writer.add_scalar('train_iou', train_iou_t.avg, epoch)
        writer.add_scalar('valid_loss', valid_loss.avg, epoch)
        writer.add_scalar('valid_iou', valid_iou_t.avg, epoch)
        writer.add_scalar('test_loss', test_loss.avg, epoch)
        writer.add_scalar('test_iou', test_iou_t.avg, epoch)
        if epoch%10 == 0:
            res = '\n\t'.join([
                'train_loss = %.4f; train_iou = %.4f' % (train_loss.avg, train_iou_t.avg),
                'valid_loss = %.4f; valid_iou = %.4f' % (valid_loss.avg, valid_iou_t.avg),
                'test_loss = %.4f; test_iou = %.4f' % (test_loss.avg, test_iou_t.avg),
            ])
            print(res)
        with open(os.path.join(log_path, 'results.csv'), 'a') as f:
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
import os
import time
import torch
from torch.utils.data import DataLoader
from config import *
from dataloader import *
from model import MyDenseNet

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

def train_epoch(model, loader, optimizer):
    # initialization
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    model.train()
    end = time.time()

    for batch_idx, (input, target) in enumerate(loader):
        input = torch.tensor(input,dtype=torch.float32)
        c = input.shape[0]
        w = input.shape[1]
        h = input.shape[2]
        input = torch.reshape(input, (c,1,w,h))
        target = [int(i) for i in target]
        target = torch.tensor(target)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return batch_time.avg, losses.avg, error.avg

def test_epoch(model, loader):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    model.eval()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
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
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return batch_time.avg, losses.avg, error.avg

def train(train_path = root_dir+"/train", test_path = root_dir+"/test"):
    # train, validate, test set init
    train_set = MnistData(train_path)
    val_set = MnistData(train_path)
    test_set = MnistData(test_path)
    indices = torch.randperm(len(train_set))
    train_indices = indices[:len(indices) - int(len(indices)*valid_rate)]
    valid_indices = indices[len(indices) - int(len(indices)*valid_rate):]
    train_set = torch.utils.data.Subset(train_set, train_indices)
    valid_set = torch.utils.data.Subset(val_set, valid_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)

    # model init
    model = MyDenseNet()
    if os.path.exists(os.path.join(save_path, 'model.dat')):
        model.load_state_dict(torch.load(os.path.join(save_path, 'model.dat')))
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    best_error = 1
    for epoch in range(n_epochs):
        _, train_loss, train_error = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer
        )
        scheduler.step()
        _, valid_loss, valid_error = test_epoch(
            model=model,
            loader=valid_loader,
        )
        _, test_loss, test_error = test_epoch(
            model=model,
            loader=test_loader,
        )

        if epoch%10 == 0:
            res = '\t'.join([
                'train_loss = %.4f; train_error = %.4f' % (train_loss, train_error),
                'valid_loss = %.4f; valid_error = %.4f' % (valid_loss, valid_error),
            ])
            print(res)

        # Determine if model is the best
        if valid_loader:
            if valid_error < best_error:
                best_error = valid_error
                print('New best error: %.4f' % best_error)
                torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))

        # Log results
        with open(os.path.join(save_path, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
                test_loss,
                test_error
            ))


if __name__ == '__main__':
    train()
    print("Done")
import os
import argparse
import warnings
import pathlib
import numpy as np
import torch
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import utils
import tqdm
from metric.separate_iou import siou

def validate_with_metrics(net, device, dl, metric_fns):
    accum = [0.0]*len(metric_fns)
    if len(dl) == 0:
        warnings.warn("validate_with_metric: ds is empty. You might want to look into that.")
        return accum

    net.eval()
    total = 0
    with torch.no_grad():
        for x, *y in dl:
            x = x.to(device)
            logits = net(x)
            probs = torch.softmax(logits, 1)
            yhat = probs.argmax(1)
            yy = y[0].cpu().numpy()
            yy[yy == 2] = 0
            yhat = yhat.cpu().numpy()
            for j in range(yhat.shape[0]):
                total += 1
                for i, metric_fn in enumerate(metric_fns):
                    accum[i] += metric_fn(yy[j], yhat[j])

    net.train()
    return [float(val)/total for val in accum]

def step_fn(elem, net, optim, loss_fn, device):
    x, y = elem
    x = x.to(device)
    y = y.to(device)
    optim.zero_grad()
    logits = net(x)
    loss = loss_fn(logits, y)
    loss.backward()
    optim.step()
    return float(loss.item())

class Trainer:
    def __init__(self, net, optim, device, tr_dl,
                 vl_dl, save_to, loss_fn, step_fn=step_fn, metric_fns=siou, **kwargs):
        self.tr_dl = tr_dl
        self.vl_dl = vl_dl
        self.save_to = pathlib.Path(save_to)
        self.net = net
        self.optim = optim
        self.device = device
        self.loss_fn = loss_fn
        self.step_fn = step_fn
        if isinstance(metric_fns, list):
            self.metric_fns = metric_fns
        else:
            self.metric_fns = [metric_fns]
        self.border_class = kwargs.get('border_class', 2)
        self.post_epoch = kwargs.get('post_epoch', None)
        self.sched = kwargs.get('sched', None)
        os.makedirs(self.save_to/'Checkpoints', exist_ok=True)
        os.makedirs(self.save_to/'Validation', exist_ok=True)
        os.makedirs(self.save_to/'Outputs', exist_ok=True)
        os.makedirs(self.save_to/'Log', exist_ok=True)

    def validate(self):
        return validate_with_metrics(self.net, self.device, self.vl_dl, self.metric_fns)

    def step(self, elem):
        return self.step_fn(elem, self.net, self.optim, self.loss_fn, self.device)

    def dry_train(self, epochs=3):
        """Check whether data is there"""
        print("Length of training dataloader: %d" % len(self.tr_dl))
        print("Length of validation dataloader: %d" % len(self.vl_dl))
        for i in range(epochs):
            for j, x in enumerate(self.tr_dl):
                print("Finished step %d" % j)
            print("Finished epoch %d" % i)
            for _ in self.vl_dl:
                pass
            print("Finished validation for epoch %d"%i)

    def train_for(self, epochs, print_every=5, save_every=20, begin_at=0):
        print("Printing every %d" % print_every)
        print("Saving every %d" % save_every)
        epochs += begin_at
        global_step = begin_at * len(self.tr_dl)
        writer = SummaryWriter(str(self.save_to/'Log'), purge_step=global_step)
        best_epoch_validation = [0.]*len(self.metric_fns)
        for epoch in tqdm.trange(begin_at, epochs):
            total_loss = 0
            for elem in tqdm.tqdm(self.tr_dl, "Step"):
                loss = self.step(elem)
                total_loss += loss
                global_step += 1

            total_loss /= len(self.tr_dl)
            writer.add_scalar('train/epoch_loss', total_loss, epoch)

            metrics = self.validate()
            for i in range(len(metrics)):
                metric = float(metrics[i])
                if metric > best_epoch_validation[i]:
                    best_epoch_validation[i] = metric
                    name = self.metric_fns[i].__doc__
                    if not name:
                        name = 'metric_%d'%i
                    fname = name+'_best.pt'
                    torch.save(self.net.state_dict(), self.save_to/'Checkpoints'/fname)
                    writer.add_scalar('validation/'+name, metric, epoch)

            tqdm.tqdm.write('Epoch {}/{}: Loss = {}'.format(epoch, epochs, total_loss))

            if (epoch+1) % save_every == 0:
                torch.save(self.net.state_dict(),
                           self.save_to/'Checkpoints'/'Net-{}.pt.tar'.format(epoch))
                torch.save(self.optim.state_dict(),
                           self.save_to/'Checkpoints'/'Optim-{}.pt.tar'.format(epoch))
            if self.post_epoch:
                self.post_epoch(self.net, self.device, epoch)
            if self.sched:
                self.sched.step()
                
def get_argparser():
    parser = argparse.ArgumentParser(description="Generic Argument Parser")
    # parser.add_argument('--eval', help="Evaluation mode", flag=True)
    parser.add_argument('--dry_run', help="Print size of dataloaders", action='store_true')
    parser.add_argument('--prenet', help="Path to pretrained model")
    parser.add_argument('--preopt', help="Path to pretrained optim")
    parser.add_argument('--dropout', help="Dropout ratio tobe used", type=float, default=.5)
    parser.add_argument('--momentum', help="Momentum to be used", type=float, default=0.9)
    parser.add_argument('--print_every', help="Print every x steps", type=int, default=5)
    parser.add_argument('--save_every', help="Save every x epochs", type=int, default=2)
    parser.add_argument('--workers', help="Num workers for dataloader", type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', help="Run for x epochs", type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', help="Which device to use. i.e. cuda:0, cpu", default='cuda')
    parser.add_argument('--begin', type=int, help="Begin epoch at", default=0)
    parser.add_argument('input', help="Where is the input stored?. Read dataset doc for more info")
    parser.add_argument('output', help="Where to store output?")
    return parser

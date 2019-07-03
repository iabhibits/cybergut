import torch
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from torch.nn import functional as F

import os
import argparse
import warnings

import utils

def mIoU(net, device, dl, save_as=None):
    """
    :param net: Network to be evaluated
    :param device: torch device to be used
    :param dl: torch dataloader
    :param save_as: where to save first batch?
        provide None if you don't want it to be saved.
    :param border_class: label of the class (i.e. which channel of nex(x) is border)
        provide None if network give two layer output(i.e. no border)
    """
    iou = 0.0
    nelems = 0
    if len(dl) == 0:
        warnings.warn("mIoU: dl is empty. You might want to look into that.")
        return 0.0
    
    net.eval()
    with torch.no_grad():
        for x, y in dl:
            nelems += len(x)
            x = x.to(device)
            y = y.to(device)
            logits = net(x)
            yhat = logits.argmax(1)
            iou += utils.IoUv2(yhat, y).sum().item()
            if save_as:
                y = y[:, None, :, :]
                yhat = yhat[:, None, :, :]
                y = torch.cat([y, yhat], dim=0)
                save_image(y, save_as, nrow=x.size(0))
                save_as = None
    net.train()
    return float(iou)/nelems

def loss_fn(yp, y, w):
    losses = F.cross_entropy(yp, y, reduction='none')
    losses *= w
    return losses.mean()

class Trainer:
    def __init__(self, net, optim, device, tr_dl, vl_dl, save_to, loss_fn=loss_fn, mIoU=mIoU):
        self.tr_dl = tr_dl
        self.vl_dl = vl_dl
        self.save_to = save_to
        self.net = net
        self.optim = optim
        self.device = device
        self.loss_fn = loss_fn
        self.mIoU = mIoU
        os.makedirs(os.path.join(save_to, 'Checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_to, 'Outputs'), exist_ok=True)
        os.makedirs(os.path.join(save_to, 'Log'), exist_ok=True)

    def validation_mIoU(self, save_as=None):
        return self.mIoU(self.net, self.device, self.vl_dl, save_as)


    def train_for(self, epochs, print_every=5, save_every=1, begin_at=0):
        print("Printing every %d" % print_every)
        print("Saving every %d" % save_every)
        epochs += begin_at
        running_loss = 0
        total_loss = 0
        global_step = begin_at * len(self.tr_dl)
        writer = SummaryWriter(os.path.join(self.save_to, 'Log'), purge_step=global_step)
        for i in range(begin_at, begin_at+epochs):
            for j, (x, y, w, t) in enumerate(self.tr_dl):
                self.optim.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)
                logits = self.net(x)
                loss = self.loss_fn(logits, y, w)
                del w
                loss.backward()
                self.optim.step()
                l = float(loss)
                total_loss += l
                running_loss += l
                
                if (j+1)%print_every == 0:
                    running_loss /= print_every
                    writer.add_scalar('train/step_loss', running_loss, global_step)
                    print('Step {}/{}, Epoch {}/{}: Loss = {}'.format(
                                        j, len(self.tr_dl), 
                                        i, epochs,
                                        running_loss))
                    running_loss = 0
            
                global_step += 1
            
            miou = self.validation_mIoU(os.path.join(self.save_to, 
                                        'Outputs', 
                                        'Epoch-{}.png'.format(i)))
            miou = float(miou)
            writer.add_scalar('validation/mIoU', miou, i)
            print("Epoch: {}, mIoU: {}".format(i, miou))
            total_loss /= len(self.tr_dl)
            writer.add_scalar('train/epoch_loss', total_loss, i)
            print("Epoch: {}, Loss: {}".format(i, total_loss))
            
            if (i+1) % save_every == 0:
                torch.save(self.net.state_dict(),
                            os.path.join(self.save_to, 'Checkpoints', 'Net-{}.pt.tar'.format(i)))
                torch.save(self.optim.state_dict(), 
                            os.path.join(self.save_to, 'Checkpoints', 'Optim-{}.pt.tar'.format(i)))

def get_argparser():
    parser = argparse.ArgumentParser(description="Generic Argument Parser")
    parser.add_argument('--prenet', help="Path to pretrained model")
    parser.add_argument('--preopt', help="Path to pretrained optim")
    parser.add_argument('--dropout', help="Dropout ratio tobe used", type=float, default=.5)
    parser.add_argument('--momentum', help="Momentum to be used", type=float, default=0.9)
    parser.add_argument('--print_every', help="Print every x steps", type=int, default=5)
    parser.add_argument('--save_every', help="Save every x epochs", type=int, default=2)
    parser.add_argument('--workers', help="Num workers for dataloader", type=int ,default=2)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', help="Run for x epochs", type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', help="Which device to use. i.e. cuda:0, cpu", required=True)
    parser.add_argument('--begin', type=int, help="Begin epoch at", default=0)
    parser.add_argument('input', help="Where is the input stored?. Read dataset doc for more info")
    parser.add_argument('output', help="Where to store output?")
    return parser
    

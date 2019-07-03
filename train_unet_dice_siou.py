"""with dice loss, separate iou as validation metric"""
import os
import glob

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.measure import label
from wand.image import Image as WImage

import utils
from utils import wand_transforms
from utils import dataset, ari
# import unet_v1 as unet
import unet
import trainer
from train_unet_dice import RuntimeDS
from metric.separate_iou import siou

def opener(x):
    return WImage(filename=x)

def validation_metric_fn(yhat, y):
    """compatible with validationdsv2"""
    cells = y[1].cpu().numpy()[0]
    yhat = yhat.cpu().numpy()[0].astype(np.uint8)
    return siou(yhat, cells)

def siou_metric_fn(true, pred):
    """sIoU"""
    lbl = label(true, background=0, connectivity=2)
    cells = np.array([lbl == i for i in range(lbl.max()+1)])
    pred = pred.astype(np.uint8)
    return siou(pred, cells, ignore_on_border=True)

def iou_metric_fn(true, pred):
    """IoU"""
    true, pred = true.astype(bool), pred.astype(bool)
    return (true&pred).sum()/(true|pred).sum()

def step_fn(elem, net, optim, loss_fn, device):
    x, y = elem
    x = x.to(device)
    y = y.to(device)
    optim.zero_grad()
    logits = net(x)
    probs = F.softmax(logits, dim=1)[:, 1]
    y[y == 2] = 0
    y = y.to(torch.float32)
    loss = loss_fn(probs, y)
    loss.backward()
    optim.step()
    return float(loss.item())

def loss_fn(y, t):
    N = t.size(0)
    y = y.contiguous().view(N, -1)
    t = t.contiguous().view(N, -1)
    smooth = 1.
    score = (2*(y*t).sum(1)+ smooth)/((y**2).sum(1) + t.sum(1) + smooth)
    return (1 - score).mean()

def main():
    argparser = trainer.get_argparser()
    args = argparser.parse_args()

    tr_ds = dataset.AnnotatedImages(
        os.path.join(args.input, 'train', ''),
        transforms.Compose([
            wand_transforms.Clone(),
            wand_transforms.RandomCrop(516),
            wand_transforms.RandomGamma(0.7, 1.3),
            wand_transforms.RandomHorizontalFlip(.5),
            wand_transforms.RandomVerticalFlip(.5),
            wand_transforms.Distort(4, 516, 12),
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            wand_transforms.Equalize('rescale'),
            wand_transforms.ToTensor(),
            wand_transforms.Three2One(),
        ]),
        image_opener=opener,
        mask_opener=opener
    )
    print(len(tr_ds))
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers)

    # vl_ds = dataset.ValidationDatasetv2(os.path.join(args.input, 'valid', 'npz'))
    # vl_dl = DataLoader(vl_ds)
    vl_ds = dataset.AnnotatedImages(
        os.path.join(args.input, 'valid', ''),
        transforms.Compose([
            wand_transforms.Clone(),
            # wand_transforms.RandomGamma(0.6, 1.5),
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            # wand_transforms.Equalize('no', 'rescale', 'hist', 'adapt'),
            wand_transforms.Equalize('rescale'),
            wand_transforms.ToTensor(),
            wand_transforms.Three2One(),
        ]),
        image_opener=opener,
        mask_opener=opener
    )
    vl_dl = DataLoader(vl_ds, args.batch_size, num_workers=args.workers)

    test_ds = RuntimeDS(os.path.join(args.input, 'test', ''), '*tif', ['rescale'])
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    print("Length of training dataloader: %d" % len(tr_dl))
    print("Length of validation dataloader: %d" % len(vl_dl))
    print("Length of test dataloader: %d" % len(test_dl))
    print()
    
    device = torch.device(args.device)

    net = unet.UNet(1, 2, 98, [2, 2, 1, 1, 1], [1, 1, 1, 1], args.dropout)
    net = nn.DataParallel(net)

    test_dir = os.path.join(args.output, 'Test', '')
    os.makedirs(test_dir, exist_ok=True)
    post_epoch = utils.post_epoch(test_dl, test_dir)

    if args.dry_run:
        atrainer = trainer.Trainer(net, None, device, tr_dl, vl_dl,
                                   args.output, loss_fn, step_fn)
        atrainer.dry_train()
        return
    if args.prenet:
        net.load_state_dict(torch.load(args.prenet))
    net.to(device)

    #optim = torch.optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=0.05)
    # optim = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.005)
    optim = torch.optim.SGD(net.parameters(), lr=args.lr,
                            weight_decay=0.005, momentum=args.momentum, nesterov=True)
    # optim = torch.optim.RMSprop(net.parameters(), weight_decay=0.005)
    if args.preopt:
        optim.load_state_dict(torch.load(args.preopt))

    metric_fns = [siou_metric_fn, iou_metric_fn]
    metric_fns.extend([ari.get_ari_fn(x) for x in [-1, 0, 1, 2]])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=0.0001)
    atrainer = trainer.Trainer(net, optim, device, tr_dl, vl_dl,
                               args.output, loss_fn, step_fn,
                               metric_fns=metric_fns,
                               post_epoch=post_epoch,
                               sched=sched)
    atrainer.train_for(args.epochs, args.print_every, args.save_every, args.begin)

if __name__ == '__main__':
    main()

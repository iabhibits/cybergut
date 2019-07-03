import os
import glob

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from wand.image import Image as WImage
from skimage.measure import label

import utils
from utils import wand_transforms
from utils import wmap_dataset, dataset, ari
import unet_v1 as unet
import trainer
from train_unet_dice import RuntimeDS
from metric.separate_iou import siou

def opener(x):
    return WImage(filename=x)
    
def step_fn(elem, net, optim, loss_fn, device):
    x, y, w, t = elem
    x = x.to(device)
    y = y.to(device)
    w = w.to(device)
    optim.zero_grad()
    logits = net(x)
    loss = loss_fn(logits, y, w)
    loss.backward()
    optim.step()
    return float(loss.item())

def loss_fn(yp, y, w):
    losses = F.cross_entropy(yp, y, reduction='none')
    losses *= w
    return losses.mean()

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

def main():
    argparser = trainer.get_argparser()
    args = argparser.parse_args()

    tr_ds = wmap_dataset.WithWmap(
        os.path.join(args.input, 'train', ''),
        transforms.Compose([
            # wand_transforms.Clone(),
            wand_transforms.RandomGamma(0.8, 1.2),
            wand_transforms.RandomHorizontalFlip(0.5),
            wand_transforms.RandomVerticalFlip(0.5),
            wand_transforms.Distort(4, 516, 12),
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
        ]),
        equalize=['no', 'rescale', 'hist', 'adapt'],
        image_opener=opener,
        mask_opener=opener,
    )
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers)

    # vl_ds = dataset.ValidationDatasetv2(os.path.join(args.input, 'valid', 'npz'))
    # vl_dl = DataLoader(vl_ds)
    vl_ds = dataset.AnnotatedImages(
        os.path.join(args.input, 'valid', ''),
        transforms.Compose([
            wand_transforms.Clone(),
            # wand_transforms.RandomGamma(0.6, 1.5),
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            wand_transforms.Equalize('no', 'rescale', 'hist', 'adapt'),
            wand_transforms.ToTensor(),
            wand_transforms.Three2One(),
        ]),
        image_opener=opener,
        mask_opener=opener
    )
    vl_dl = DataLoader(vl_ds, args.batch_size, num_workers=args.workers)

    test_ds = RuntimeDS(os.path.join(args.input, 'test', ''), '*tif')
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)


    device = torch.device(args.device)

    net = unet.UNet(4, 2, 98, [2, 2, 1, 1, 1], [1, 1, 1, 1], args.dropout)
    net = nn.DataParallel(net)

    if args.dry_run:
        atrainer = trainer.Trainer(net, None, device, tr_dl, vl_dl,
                                   args.output, loss_fn, step_fn, validation_metric_fn)
        atrainer.dry_train()
        return

    if args.prenet:
        net.load_state_dict(torch.load(args.prenet))

    net.to(device)
    test_dir = os.path.join(args.output, 'Test', '')
    os.makedirs(test_dir, exist_ok=True)
    post_epoch = utils.post_epoch(test_dl, test_dir)
    #optim = torch.optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=0.05)
    # optim = torch.optim.Adam(net.parameters(), weight_decay=0.007)
    optim = torch.optim.SGD(net.parameters(), lr=args.lr,
                            weight_decay=0.005, momentum=args.momentum, nesterov=True)
    if args.preopt:
        optim.load_state_dict(torch.load(args.preopt))

    metric_fns = [siou_metric_fn, iou_metric_fn]
    metric_fns.extend([ari.get_ari_fn(x) for x in [-1, 0, 1, 2]])

    atrainer = trainer.Trainer(net, optim, device, tr_dl, vl_dl,
                               args.output, loss_fn, step_fn,
                               metric_fns=metric_fns,
                               post_epoch=post_epoch)
    atrainer.train_for(args.epochs, args.print_every, args.save_every, args.begin)

if __name__ == '__main__':
    main()

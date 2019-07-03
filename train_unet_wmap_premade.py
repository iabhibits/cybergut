"""
Same as train_unet_wmap, except now images
are not perturbed each iteration. They are
pre generated
"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from wand.image import Image as WImage

import unet_v1 as unet
import trainer
from utils import premade_dataset, dataset, wand_transforms

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

def opener(x):
    return WImage(filename=x)

def main():
    argparser = trainer.get_argparser()
    args = argparser.parse_args()

    tr_ds = premade_dataset.PremadeWmap(
        os.path.join(args.input, 'train', '')
    )
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers)

    vl_ds = dataset.AnnotatedImages(
        os.path.join(args.input, 'valid', ''),
        transforms.Compose([
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            wand_transforms.Equalize('no', 'rescale', 'hist', 'adapt'),
            wand_transforms.ToTensor(),
            wand_transforms.Three2One()
        ]),
        channels=['TL-Brightfield - dsRed'],
        image_opener=opener,
        mask_opener=opener
    )
    vl_dl = DataLoader(vl_ds, 4, shuffle=False)

    device = torch.device(args.device)

    net = unet.UNet(4, 2, 98, [2, 2, 1, 1, 1], [1, 1, 1, 1], args.dropout)
    if args.dry_run:
        atrainer = trainer.Trainer(net, None, device, tr_dl, vl_dl,
                                   args.output, loss_fn, step_fn)
        atrainer.dry_train()
        return


    net = nn.DataParallel(net)
    if args.prenet:
        net.load_state_dict(torch.load(args.prenet))
    net.to(device)

    #optim = torch.optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=0.05)
    optim = torch.optim.SGD(net.parameters(), lr=args.lr,
                            weight_decay=0.005, momentum=args.momentum, nesterov=True)
    if args.preopt:
        optim.load_state_dict(torch.load(args.preopt))

    atrainer = trainer.Trainer(net, optim, device, tr_dl, vl_dl,
                               args.output, loss_fn, step_fn)
    atrainer.train_for(args.epochs, args.print_every, args.save_every, args.begin)

if __name__ == '__main__':
    main()
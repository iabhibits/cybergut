import os
import glob

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from wand.image import Image as WImage

import utils
from utils import wand_transforms
from utils import dataset
import unet
import trainer

def opener(x):
    return WImage(filename=x)

def crop_mask(imgs):
    last = imgs[-1]
    last = last[94:610, 94:610]
    imgs[-1] = last
    return imgs

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

class RuntimeDS(Dataset):
    """Raw images"""
    def __init__(self, root, glob_ptrn='*Brightfield*', equalize=['no', 'rescale', 'hist', 'adapt']):
        self.root = root
        self.files = glob.glob(os.path.join(root, glob_ptrn))
        self.equalize = wand_transforms.Equalize(*equalize, all=True)
        self.tonp = wand_transforms.ToNumpyArray(['gray'])
        self.totensor = wand_transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = [WImage(filename=self.files[i])]
        imgs = self.tonp(img)
        imgs = self.equalize(imgs)
        imgs = self.totensor(imgs)
        img = torch.cat(imgs)
        return img, os.path.basename(self.files[i])[:-4]

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
            wand_transforms.RandomHorizontalFlip(.5),
            wand_transforms.RandomVerticalFlip(.5),
            wand_transforms.Distort(4, 516, 12),
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            wand_transforms.Equalize('no', 'rescale', 'hist', 'adapt'),
            crop_mask,
            wand_transforms.ToTensor(),
            wand_transforms.Three2One(),
        ]),
        image_opener=opener,
        mask_opener=opener
    )
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers)

    vl_ds = dataset.AnnotatedImages(
        os.path.join(args.input, 'valid', ''),
        transforms.Compose([
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            wand_transforms.Equalize('no', 'rescale', 'hist', 'adapt'),
            crop_mask,
            wand_transforms.ToTensor(),
            wand_transforms.Three2One(),
        ]),
        with_name=True,
        channels=['TL-Brightfield - dsRed'],
        image_opener=opener,
        mask_opener=opener
    )
    vl_dl = DataLoader(vl_ds, args.batch_size, shuffle=False, num_workers=args.workers)

    test_ds = RuntimeDS(os.path.join(args.input, 'test', ''), '*tif')
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)
    device = torch.device(args.device)

    net = unet.UNet(4, 2, 0, [2, 1, 1, 1, 1], [1, 1, 1, 1], args.dropout)
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
    if args.preopt:
        optim.load_state_dict(torch.load(args.preopt))

    atrainer = trainer.Trainer(net, optim, device, tr_dl, vl_dl,
                               args.output, loss_fn, step_fn, post_epoch=post_epoch)
    atrainer.train_for(args.epochs, args.print_every, args.save_every, args.begin)

if __name__ == '__main__':
    main()

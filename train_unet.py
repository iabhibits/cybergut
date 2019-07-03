"""Normal Three class classifier"""

import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from wand.image import Image as WImage
import utils
from utils import wand_transforms
from utils import dataset
import unet
import trainer

def opener(x):
    return WImage(filename=x)

if __name__ == '__main__':
    argparser = trainer.get_argparser()
    args = argparser.parse_args()

    tr_ds = dataset.AnnotatedImages(
        os.path.join(args.input, 'train', ''),
        transforms.Compose([
            wand_transforms.RandomHorizontalFlip(.5),
            wand_transforms.RandomVerticalFlip(.5),
            wand_transforms.Distort(5, 516, 15),
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            wand_transforms.Equalize('adapt'), 
            wand_transforms.ToTensor(),
            wand_transforms.Three2One()
        ]),  
        channels=['TL-Brightfield - dsRed'],
        image_opener = opener,
        mask_opener = opener
    )
    tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True, num_workers=args.workers)

    vl_ds = dataset.AnnotatedImages(
        os.path.join(args.input, 'valid', ''),
        transforms.Compose([
            wand_transforms.Resize((516, 516)),
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            wand_transforms.Equalize('adapt'), 
            wand_transforms.ToTensor(),
            wand_transforms.Three2One()
        ]),  
        channels=['TL-Brightfield - dsRed'] ,
        image_opener = opener,
        mask_opener = opener
    )
    vl_dl = DataLoader(vl_ds, 6, shuffle=False)

    device = torch.device(args.device)

    net = unet.UNet(1, 3, 98, [2, 2, 1, 1, 1], [1, 1, 1, 1], 0.2)
    if args.prenet: net.load_state_dict(torch.load(args.prenet))
    net.to(device)

    #optim = torch.optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=0.05)
    optim = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.005, momentum=args.momentum, nesterov=True)
    if args.preopt: optim.load_state_dict(torch.load(args.preopt))

    weights = torch.tensor([.10, .25, .65]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weights).to(device)

    atrainer = trainer.Trainer(net, optim, device, tr_dl, vl_dl, args.output, loss_fn, 3)
    atrainer.train_for(args.epochs, args.print_every, args.save_every, args.begin)

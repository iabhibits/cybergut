import os
import glob
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from wand.image import Image as WImage

import unet_v1 as unet
from utils import dataset
from utils import wand_transforms
import eval

def opener(x):
    return WImage(filename=x)

class RuntimeDS(Dataset):
    """Raw images"""
    def __init__(self, root, glob_ptrn='*tif'):
        self.root = root
        self.files = glob.glob(os.path.join(root, glob_ptrn))
        self.equalize = wand_transforms.Equalize('no', 'rescale', 'hist', 'adapt', all=True)
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

def runtime_eval(net, args):
    os.makedirs(os.path.join(args.output, "logits"), exist_ok=True)
    ds = RuntimeDS(args.input)
    dl = DataLoader(ds, args.batch_size, shuffle=False)
    print("Size of dataloader is", len(dl))
    os.makedirs(os.path.join(args.output, "logits"), exist_ok=True)
    eval.eval_runtime(net, args.device, dl, args.output, only_fg_probs=True)

def validation_eval(net, args):
    os.makedirs(os.path.join(args.output, "Images"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "Generated"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "Masks"), exist_ok=True)
    vl_ds = dataset.AnnotatedImages(
        os.path.join(args.input, ''),
        transforms.Compose([
            wand_transforms.ToNumpyArray(['gray', 'bgr']),
            wand_transforms.Equalize('no', 'rescale', 'hist','adapt'),
            wand_transforms.ToTensor(),
            wand_transforms.Three2One()
        ]),
        channels=['TL-Brightfield - dsRed'],
        with_name=True,
        image_opener=opener,
        mask_opener=opener
    )
    vl_dl = DataLoader(vl_ds, 1, shuffle=False)

    eval.eval_validation(net, args.device, vl_dl, args.output)

def main():
    parser = eval.get_argparser()
    args = parser.parse_args()
    net = unet.UNet(4, 2, 98, [2, 2, 1, 1, 1], [1, 1, 1, 1], 0.5)
    if args.device.startswith('cpu'):
        state_dict = torch.load(args.checkpoint, map_location=lambda storage, location: storage)
    else:
        state_dict = torch.load(args.checkpoint)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net = net.to(args.device)
    if args.r:
        runtime_eval(net, args)
    else:
        validation_eval(net, args)

if __name__ == '__main__':
    main()
    
from torch.utils.data import Dataset
import glob
import os
import torch
import re

from PIL import Image
try:
    import cv2
    from skimage import io, img_as_ubyte, morphology
except:
    pass

import numpy as np

from . import roi, weightmap
from . import cv_transforms as cvtsfms
from . import transforms

class AnnotatedImages(Dataset):
    def __init__(self, root, transform=None,
                 channels=["TL-Brightfield - dsRed"],
                 with_name=False, exts=['tif', 'png'], **kwargs):
        """
        :param root str: Full path to root directory where Images are located.
            there should be two subdirectories in it. One Images containing
            original images. Other Masks containing masks.
        :param transform function: This function should take an iterable
            of images last of which is mask and transform it. Return
            must be of a type torch.Tensor.
        :param channels: Look at documentation of WithWmapFromROI.
        :param with_name boolean: getitem also returns the name of file
            without extension.
        :param exts: tuple or list with len(exts) == len(channels)+1 or
            len(exts) == 2. exts[-1] is extention for mask.
        :param image_opener: a callable f(x) to open an image. Default is
            Image.open that returns PIL image.
            x is path to file open
        :param mask_opener: see image_opener.
        """
        files = glob.glob(os.path.join(root, 'Masks', '*.png'))
        files = [os.path.basename(p) for p in files]
        self.with_name = with_name
        self.titles = [os.path.splitext(p)[0] for p in files]
        self.img_path = os.path.join(root, 'Images', '{}.{}')
        self.msk_path = os.path.join(root, 'Masks', '{}.{}')
        self.transform = transform
        self.re = re.compile(r'wv (?:.*\))')
        self.channels = channels
        self.img_opener = kwargs.get('image_opener', Image.open)
        self.msk_opener = kwargs.get('mask_opener', Image.open)
        if len(channels) + 1 == len(exts):
            self.exts = exts
        else:
            self.exts = [exts[0] for i in channels]
            self.exts.append(exts[-1])

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, i):
        imgs = []
        for c, e in zip(self.channels, self.exts):
            filename = self.re.sub('wv {})'.format(c), self.titles[i])
            filepath = self.img_path.format(filename, e)
            img = self.img_opener(filepath)
            imgs.append(img)
        filepath = self.msk_path.format(self.titles[i], self.exts[-1])
        msk = self.msk_opener(filepath)
        imgs.append(msk)

        if self.transform:
            imgs = self.transform(imgs)
            img = torch.cat(imgs[:-1], dim = 0)

        if self.with_name:
            return img, imgs[-1], self.titles[i]
        else:
            return img, imgs[-1]

class RuntimeDataset(Dataset):
    """Runtime dataset returns images and it's name, but no mask"""
    def __init__(self, roots, glb_ptrn='*.tif', transform=None, **kwargs):
        self.transform = transform
        self.img_opener = kwargs.get('image_opener', io.imread)
        self.files = []
        self.transform_expects_list = kwargs.get('transform_expects_list', False)
        for directory in roots:
            files = glob.glob(os.path.join(directory, glb_ptrn))
            self.files.extend(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self.img_opener(self.files[i])
        if self.transform_expects_list:
            img = self.do_transform([img])
            img = img[0]
        else:
            img = self.do_transform(img)
        img = img[None]
        return img, self.files[i]

    def do_transform(self, img):
        if self.transform:
            for fun, kwargs in self.transform:
                img = fun(img, **kwargs)
            #img = torch.tensor(img, dtype=torch.float32)
        return img

class RuntimeDatasetv2(RuntimeDataset):
    """Version 2 can work with more than one channles in input"""
    def __init__(self, roots, ptrn='TL-Brightfield - dsRed', transform=None,
                 channels=['TL-Brightfield - dsRed', 'UV - DAPI'],
                ):
        super().__init__(roots, glb_ptrn='*{}*'.format(ptrn), transform=transform)
        self.all_files = [[] for c in channels]
        reg = re.compile(ptrn)
        for i, c in enumerate(channels):
            for f in self.files:
                self.all_files[i].append(reg.sub(c, f))

    def __getitem__(self, i):
        imgs = [cv2.imread(f[i], cv2.IMREAD_ANYDEPTH) for f in self.all_files]
        imgs = [self.do_transform(img) for img in imgs]
        img = torch.stack(imgs)
        return img, self.files[i]

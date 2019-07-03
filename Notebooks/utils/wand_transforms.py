#  transforms.py
#  
#  Author: Dhruv Patel <dhruvhp01 at gmail dot com>
#  
#  This file contains classes that are callable with more than
#  one arguments. They are subclasses of transform classes.
#  see their documentation for details.

import numpy as np
from numpy import random
import torch
from wand.image import Image
from wand.color import Color
from . import transforms
from skimage import exposure
from torchvision.transforms import ToTensor as TorchToTensor

class ToNumpyArray(object):
    def __init__(self, formats=['gray', 'rgb']):
        """
        :param formats: a list of formats.
            allowed formats are 'gray', 'rgb', 'bgr'.
            if len(formats) is less than len(args) 
            when object is called, formats[0] will
            be assumed for all images but last. Last
            image will assumed to have format[-1].
        """
        self.formats = formats

    def __call__(self, args):
        if len(args) > len(self.formats):
            formats = [self.formats[0]]*(len(args))
            formats[-1] = self.formats[-1]
        else:
            formats = self.formats
        ret = []
        for oimg, fmt in zip(args, formats):
            bin_img = oimg.make_blob(format=fmt)
            if fmt == 'gray':
                N = len(bin_img)
                n = oimg.size[0]*oimg.size[1]
                if (N//n == 1): typ = np.uint8
                elif (N//n == 2): typ = np.uint16
                else: raise ValueError('Image has gray value more than what 16bit can hold, %d' % (N//n))
                img = np.frombuffer(bin_img, dtype=typ)
                img = img.reshape(*oimg.size)
            else:
                img = np.frombuffer(bin_img, np.uint8)
                img = img.reshape(*oimg.size, 3)
            ret.append(img)
        return ret

class ToTensor(object):
    def __call__(self, args):
        """Converts list of images (ndarrays) to pytorch tensor
        Make sure that each element is either float, int32 or int64
        and not int16
        """
        ret = []
        for img in args:
            if len(img.shape) == 2:
                img = img[None]
            ret.append(torch.from_numpy(img))
        return ret
        
class Resize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise ValueError("Size must be either integer or tupble")

    def __call__(self, args):
        for img in args:
            img.resize(*self.size)
        return args
        
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, args):
        if random.random() < self.p:
            for img in args:
                img.flip()
        return args
    
class RandomVerticalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, args):
        if random.random() < self.p:
            for img in args:
                img.flop()
        return args

class Equalize(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.all = kwargs.get('all', False)
    def __call__(self, args):
        ret = []
        if self.all:
            lst = args
        else:
            lst = args[:-1]
        for img in lst:
            for method in self.args:
                if   method == 'no':
                    tmp = exposure.rescale_intensity(1.*img, (0, 2**16-1)) 
                    tmp = tmp.astype(np.float32)
                    ret.append(tmp)
                elif method == 'rescale':
                    tmp = exposure.rescale_intensity(1.*img)
                    tmp = tmp.astype(np.float32)
                    ret.append(tmp)
                elif method.startswith('hist'):
                    tmp = exposure.equalize_hist(img)
                    tmp = tmp.astype(np.float32)
                    ret.append(tmp)
                elif method.startswith('adapt'):
                    tmp = exposure.equalize_adapthist(img)
                    tmp = tmp.astype(np.float32)
                    ret.append(tmp)
                else:
                    raise ValueError("Unknown equalize method %s" % method)
        if not self.all:
            ret.append(args[-1]) 
        return ret   
        
class EqualizeHistogram(object):
    def __call__(self, args):
        """
        :param args: Numpy images [img1, img2, img3, ..., msk]
        :returns: All images are equalized but msk using global
            histogram equalization
        """
        ret = [exposure.equalize_hist(img).astype(np.float32) for img in args[:-1]]
        ret.append(args[-1])
        return ret
        

class Distort(object):
    def __init__(self, grid=3, size=512, magnitude=10, bg_color='blue'):
        """
        :param grid: rows and columns of grid (integer)
        :param size: width and height of image (integer)
        :param magnitude: magnitude of distortion
        """
        self.magnitude = magnitude
        self.bg_color = Color(bg_color)
        width = size//grid
        center = width//2
        xs = center + np.arange(center, size, width)
        ys = xs
        xs, ys = np.meshgrid(xs, ys)
        self.xs = xs.reshape(-1)
        self.ys = ys.reshape(-1)
        
    def __call__(self, images):
        dx = np.random.randint(-self.magnitude, self.magnitude, len(self.xs))
        dy = np.random.randint(-self.magnitude, self.magnitude, len(self.ys))
        args = []
        
        new_x = dx+self.xs
        new_y = dy+self.ys
        for tmp in zip(self.xs, self.ys, new_x, new_y):
            args.extend(tmp)
        for img in images:
            img.virtual_pixel = 'mirror'
            img.distort('shepards', args)
        # last = images[-1]
        # last.background_color = self.bg_color
        # last.virtual_pixel = 'background'
        # last.distort('shepards', args)
        return images

class Three2One(object):
    """Convert three channel mask image into
        one channel target image
    """
    def __init__(self, c_dim=-1):
        self.c_dim = c_dim

    def __call__(self, args):
        msk = args[-1]
        msk = msk.argmax(self.c_dim)
        args[-1] = msk
        return args

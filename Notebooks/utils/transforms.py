#  transforms.py
#  
#  Author: Dhruv Patel <dhruvhp01 at gmail dot com>
#  
#  This file contains classes that are callable with more than
#  one arguments. They are subclasses of torchvisions' classes.
#  see their documentation for details.
#
#  Distort is not provided by torchvision.
#  It's originaly written by
#  Marcus D. Bloice <https://github.com/mdbloice>

import random

import torch
from torchvision import transforms as tf
from torchvision.transforms import functional as F

from numpy import random
import numpy as np
from PIL import Image, ImageDraw

import skimage
from skimage import exposure
import warnings

class ToTensor(tf.ToTensor):
    def __call__(self, imgs):
        return [F.to_tensor(img) for img in imgs]

class Convert(object):
    def __init__(self, mode):
        self.mode = mode
    
    def __call__(self, imgs):
        ret = [img.convert(self.mode) for img in imgs[:-1]]
        ret.append(imgs[-1])
        return ret

class Resize(tf.Resize):
    def __call__(self, imgs):
        return tuple(F.resize(a, self.size, self.interpolation) for a in imgs)
        
class RandomHorizontalFlip(tf.RandomHorizontalFlip):
    def __call__(self, args):
        if random.random() < self.p:
            return tuple(F.hflip(a) for a in args)
        return args
    
class RandomVerticalFlip(tf.RandomVerticalFlip):
    def __call__(self, args):
        if random.random() < self.p:
            return tuple(F.vflip(a) for a in args)
        return args

class Distort(object):
    """A callable class. See __call__ for more information"""
    
    def __init__(self, grid_width, grid_height, magnitude):
        """
        To choose good values experiment with different paramenters.
        
        :param grid_width int: How many columns?
        :param grid_height int: How many rows?
        :param magnitude int: How much to distort?
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude
        
    def __call__(self, images):
        """
        Distorts single image.
        
        :param images List: List of Pillow images. Each image
         is distorted using same arguments.
        :return: transformed images.
        """
        if isinstance(self.magnitude, tuple):
            magnitude = random.randint(*self.magnitude)
        else:
            magnitude = self.magnitude
        w, h = images[0].size
        
        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = w // horizontal_tiles
        height_of_square = h // vertical_tiles
        
        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))
        
        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                tmp = [horizontal_tile * width_of_square,
                       vertical_tile * height_of_square,
                       horizontal_tile * width_of_square,
                       vertical_tile * height_of_square]
                if horizontal_tile == horizontal_tiles - 1:
                    tmp[2] += width_of_last_square
                else:
                    tmp[2] += width_of_square
                if vertical_tile == vertical_tiles -1:
                    tmp[3] += height_of_last_square
                else:
                    tmp[3] += height_of_square
                dimensions.append(tmp)
        
        #indices of last_*
        last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_row = list(range(horizontal_tiles*vertical_tiles - horizontal_tiles, 
                              horizontal_tiles*vertical_tiles))
        polygons = [[x1,y1, x1,y2, x2,y2, x2,y1] for x1,y1, x2,y2 in dimensions]
        
        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])
        
        for a, b, c, d in polygon_indices:
            dx = random.randint(-magnitude, magnitude)
            dy = random.randint(-magnitude, magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

            generated_mesh = []
            for i in range(len(dimensions)):
                generated_mesh.append([dimensions[i], polygons[i]])

        return tuple(im.transform(im.size, Image.MESH, generated_mesh, 
                             resample=Image.BILINEAR) for im in images)

class DrawGrid(object):
    #TODO: Implement multichannel input
    def __init__(self, w):
        self.w = w
    
    def __call__(self, args):
        img = ImageDraw.Draw(args[0])
        msk = ImageDraw.Draw(args[1])
        h, w = args[0].size
        for i in range(0, w, self.w):
            img.line([i, 0, i, h-1], 0)
            msk.line([i, 0, i, h-1], 255)
        for i in range(0, h, self.w):
            img.line([0, i, w-1, i], 0)
            msk.line([0, i, w-1, i], 255)
        args[0].save('/tmp/image.png')
        args[1].save('/tmp/mask.png')
        return args

class RescaleIntensity(object):
    """Rescale to make it in range (0, 1)"""
    def __init__(self, mx='image'):
        """
        :param mx: If int values of images will be clipped to mx,
            if mx = 'image' whatever is maximum in image will be mx value.
        """
        self.max = mx
        
    def __call__(self, imgs):
        mask = imgs[-1]
        imgs = imgs[:-1]
    
        if isinstance(self.max, int): irange = (0, self.max)
        else:                         irange = self.max
        
        
        imgs =  [exposure.rescale_intensity(i.float().numpy(), irange, (0, 1)) for i in imgs]
        imgs = [torch.from_numpy(img) for img in imgs]
        imgs.append(mask)
        return imgs
        
class RandomCrop(object):
    """Crop Image at random location"""
    
    def __init__(self, hw):
        """
        :param hw: Int or tuple. If tuple, h = w is is uniform(*hw)
            result is square image
        """
        self.hw = hw
        
    def __call__(self, imgs):
        nh = nw = self.hw
        if isinstance(self.hw, tuple): nh = nw = random.randint(*self.hw)
        h, w = imgs[0].size
        if nh == h:
            i, j = 0, 0
        else:
            i = random.randint(0, h - nh)
            j = random.randint(0, w - nw)
        return [F.crop(img, i, j, nh, nw) for img in imgs]
        
        
class Zoom(object):
    def __init__(self, mx=.1, interpolation=Image.NEAREST):
        """
        Crops images at random with new width = (1 - mx)*old_width 
        and same for height. Rescales images to original size.
        """
        #TODO: Implement multichannel input
        self.factor = 1 - mx
        self.interpolation = interpolation
        
    def __call__(self, imgs):
        h, w = imgs[0].size
        nh = int(self.factor * h)
        nw = int(self.factor * w)
        params = tf.RandomCrop.get_params(args[0], (nh, nw))
        imgs = [F.crop(img, *params) for img in imgs]
        imgs = [F.resize(img, (h, w), self.interpolation) for img in imgs]
        return imgs

class ConvertTarget(object):
    """Rightnow target is [c, h, w] where c denotes 
        number of classes.
        returns [h, w]
    """
    def __init__(self, channel_dim = 0):
        self.channel_dim = channel_dim

    def __call__(self, args):
        if args[-1].ndimension() == 3 and args[-1].size(self.channel_dim) == 1:
            args[-1] = args[-1].squeeze()
        
        if len(args[-1].shape) > 2:
            mask = args[-1].argmax(self.channel_dim)
        else:
            mask = args[-1].clamp_(0, 1).long()
        args[-1] = mask
        return args

class EqualizeCLAHE(object):
    def __call__(self, args):
        imgs = args[:-1]
        msk= args[-1]
        ret = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for img in imgs:
                img = img.squeeze()
                img = exposure.equalize_adapthist(img.numpy())
                img = torch.from_numpy(img.astype(np.float32))
                ret.append(img)
        ret.append(msk)
        return ret

import os
import re
import glob
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import roi
from . import weightmap
from . import wand_transforms as wtsfms


class WithWMapFromRoi(Dataset):
    def __init__(self, root_dir, roi_dir, shape, 
                 channels=["TL-Brightfield - dsRed"], 
                 dist_grid=25, dist_mag=(2, 6)):
        """
        :param root_dir: directory should contain Images and Masks 
            subdirectories. Masks should be RGB with Green as foreground.
        :param roi_dir: directory containing zip files. For each xyz.zip 
            there should be xyz.tif in root_dir/Images and xyz.png in 
            root_dir/Masks.
        :param shape: either tuple (h, w) or integer h for (h, h)
            images and masks will be resized to (h, w)
        :param channels: Return image will have len(channels) channels.
            Each item in a list is a suffix that changes for different 
            channels for same image.
            i.e. 
            D - 2(fld 01 wv `Blue - FITC')
            D - 2(fld 01 wv `Green - dsRed')
            D - 2(fld 01 wv `TL-Brightfield - dsRed')
            D - 2(fld 01 wv `UV - DAPI')     
        :param dist_grid: Grid height and width parameter tobe used
            in Distort. Integer. 
            Note: Subject to change.
        :param dist_mag: magnitude for Distort. Tuple or integer.
            Note: Subject to change.
        """
        self.root_dir = root_dir
        self.roi_dir = roi_dir
        
        self.re = re.compile(r'wv (?:.*\)).*')
        
        self.channels = channels
        
        files = glob.glob(os.path.join(roi_dir, '*.zip'))
        files = [os.path.basename(p) for p in files]
        self.titles = [os.path.splitext(p)[0] for p in files]
        
        self.img_path = os.path.join(root_dir, 'Images', '{}.tif')
        self.msk_path = os.path.join(root_dir, 'Masks', '{}.png')
        self.roi_path = os.path.join(roi_dir,'{}.zip')
        
        if isinstance(shape, int): self.shape = (shape, shape)
        else:                      self.shape = shape
        
        self.dist = transforms.Distort(dist_grid, dist_grid, dist_mag)

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, i):
        polies = roi.get_polygons(self.roi_path.format(self.titles[i]))
        borders, _ = weightmap.generate_images(polies, *self.shape)
        
        imgs = []
        for c in self.channels:
            filename = self.re.sub(f'wv {c})', self.titles[i])
            img = cv2.imread(self.img_path.format(filename), cv2.IMREAD_ANYDEPTH)
            img = cv2.resize(img, self.shape, interpolation=cv2.INTER_AREA)
            imgs.append(img)
        n = len(imgs)
        
        msk = cv2.imread(self.msk_path.format(self.titles[i]))
        msk = cv2.resize(msk, self.shape, interpolation=cv2.INTER_AREA)
        
        cat = imgs + borders
        cat.append(msk)
        cat = cvtsfms.flip(cat)
        cat = cvtsfms.normalize(cat, idx=list(range(n)))

        #TODO: Code distortions for cv2. Make it true elastic if possible.
        cat = [Image.fromarray(img) for img in cat]
        cat = self.dist(cat)
        cat = [np.array(img) for img in cat]
        msk = cat[-1]
        msk = msk[:, :, 1] #RGB or BGR G is at index 1, which is forground
        msk = np.greater(msk, 127).astype(np.int64)
        
        wmap = weightmap.imgs2wmap(cat[n:-1], msk)

        img = np.stack(cat[:n])
        img = torch.from_numpy(img)
        
        k = morphology.disk(2)
        msk = morphology.opening(msk, k)
        msk = torch.from_numpy(msk)
        
        wmap = torch.from_numpy(wmap)
        return img.float(), msk, wmap.float(), self.titles[i]

class WithWmap(Dataset):
    def __init__(self, root, transform, equalize=['no'], **kwargs):
        """
        :param transform: It should return numpy array.
            Don't include equalize in transform. Use
            equalize argument for that.
        :param equalize: List of equalizations to be performed.
            expected argumets are = 'no', 'rescale', 'hist', 'adapt'
        """
        self.root = root

        basenames = glob.glob(os.path.join(root, 'Masks', '*.png'))
        basenames = [os.path.basename(b)[:-4] for b in basenames]
        self.basenames = basenames

        self.img_opener = kwargs.get('image_opener', Image.open)
        self.msk_opener = kwargs.get('mask_opener', Image.open)
        self.img_path = os.path.join(root, "Images", "{}.tif")
        self.msk_path = os.path.join(root, "Masks", "{}.png")

        self.transform = transform
        self.equalize = wtsfms.Equalize(*equalize)

        self.w_0 = kwargs.get('w_0', .25)
        self.w_1 = kwargs.get('w_1', .75)
        self.w_2 = kwargs.get('w_2', 1)
        self.sigma = kwargs.get('sigma', 25)

        self.to_tensor = wtsfms.ToTensor()

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, i):
        name = self.basenames[i]
        img = self.img_opener(self.img_path.format(name))
        msk = self.msk_opener(self.msk_path.format(name))
        boundary_names = glob.glob(os.path.join(self.root, "Masks", "boundries", name, '*.png'))
        boundries = [self.msk_opener(bname) for bname in boundary_names]
        imgs = [img]+boundries+[msk]
        imgs = self.transform(imgs)
        img, *boundries, msk = imgs
        *img, msk = self.equalize([img, msk]) #img now is a list of atleast 1 elem.

        wmaps = []
        for b in boundries:
            if 0 not in np.unique(b): 
                continue
            wmap = distance_transform_edt(b)
            wmaps.append(wmap)
        if len(wmaps) > 1:
            wmaps = np.stack(wmaps)
            wmaps.partition(1, axis=0)
            dists = wmaps[0]+wmaps[1]

        y = msk.argmax(-1)
        y_new = np.clip(y, None, 2) #treat border as foreground
        w_c = y_new*self.w_1 + (1-y_new)*self.w_0
        if len(wmaps) > 1:
            dists = np.square(dists)
            dists /= self.sigma**2
            dists *= -1.0
            w = self.w_2 * np.exp(dists)
            w += w_c
        else:
            w = w_c
        y[y == 2] = 0 #treat border as background
        img = np.stack(img) #now we will have (c, h, w) image.
        x, y, w = self.to_tensor([img, y, w])
        return x, y[0], w[0].float(), name
import numpy as np
from skimage import draw, transform
import cv2

def generate_images(polygons, h, w):
    imgs = []
    oimg= np.zeros((2048, 2048), dtype=np.uint8)
    kernel = np.zeros((5, 5), dtype=np.uint8)
    kernel[:, 2] = 1
    kernel[2, :] = 1
    for poly in polygons:
        img = np.zeros((2048, 2048), dtype=np.uint8)
        xy = np.array(poly).T
        cv2.fillPoly(oimg, [xy], 1)
        xy = xy.reshape(-1, 1, 2)
        img = cv2.polylines(img, xy, True, 255, 5)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        img  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img[img < 127] = 0
        img[img > 127] = 255
        img = 255 - img
        imgs.append(img)

    oimg = cv2.resize(oimg, (512, 512), interpolation=cv2.INTER_AREA)
    return imgs, oimg

def imgs2wmap(imgs, mask, sigma=4, w_f=None ,w_0=50):
    """
    :param imgs: List of HxW np.uint8 arrays. Each
        array is black and white image where black
        is border of single roi.
        There shold be only one roi per image.
    :param mask: single image HxW type np.uint8
        should be filled with ones inside polygons.
    :param sigma: see paper by Ronneberger et al.
    :param w_f: class weight for foreground pixels.
        if None it will be 1 - (foreground_area)/(total_area)
    :param w_0: see paper.
    :returns: an array A of dim HxW with dtype np.float32
        A[i, j] = exp^(- (d_1(i, j)^2 + d_2(i, j)^2)/(2*sigma^2))
        See unet paper for formula.
    """
    N = len(imgs)
    h, w = imgs[0].shape
    
    if w_f is None:
        area = mask.sum()
        mh, mw = mask.shape
        total = mh*mw
        w_c = 1 - (area/total)
    else:
        w_c = w_f
    wmapt = mask*w_c
    wmapt += (1 - mask)* (1 - w_c)
    if N == 1:
        return wmapt

    dists = []
    for img in imgs:
        dst = cv2.distanceTransform(img, cv2.DIST_L2, 3)
        dists.append(dst)
        
    stk = np.stack(dists, -1)
    stk.partition(1)
    ws = stk[:, :, [0, 1]] #find two borders with min distances
    wmap = ws.sum(-1)
    wmap = np.square(wmap, out=wmap)
    wmap /= 2*sigma**2
    wmap *= -1
    wmap = w_0*np.exp(wmap)
    wmap += wmapt
    return wmap

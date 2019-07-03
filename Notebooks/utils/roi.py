import struct
import numpy as np
import zipfile
from skimage import draw, io
import cv2

def parse_roi(roi):
    """roi: bytestream object containing ImageJ roi formatted data
       https://github.com/imagej/imagej1/blob/master/ij/io/RoiDecoder.java
       returns: { tl, br: (x, y), coords: Nx2 matrix
    """
    top, left, bottom, right, N = struct.unpack_from('>5h', roi, 8)

    r = {'tl': (top, left), 'br': (bottom, right), 'N': N}

    x = left + np.array(struct.unpack_from(f'>{N}h', roi, 64))
    y = top + np.array(struct.unpack_from(f'>{N}h', roi, 64+N*2))
    r['x'] = x
    r['y'] = y
    return r

def get_polygons(roi_zip, return_raw=False):
    """Get polygons from roi_zip file
    :param roi_zip string: path to zip file
    :returns: A list of tuples. [(xs, ys), ...]
    """
    z = zipfile.ZipFile(roi_zip)
    def file_to_roi(fname):
        f = z.open(fname).read()
        d = parse_roi(f)
        if return_raw:
            return d
        else:
            xy = d['x'], d['y']
            return xy
    return list(map(file_to_roi, z.namelist()))

def create_mask(polygons, size, save_to=None):
    img = np.zeros((size, size), dtype=np.uint8)
    for x, y in polygons:
        rr, cc = draw.polygon(y, x)
        img[rr, cc] = 255
        rr, cc = draw.polygon_perimeter(y, x, shape=(size, size))
        img[rr, cc] = 0
    if save_to:
        io.imsave(save_to, img)
    return img

def create_maskv2(polygons, size, separate=False, width=1, **kwargs):
    """
    New version of create_mask. Doesn't save file.
    :param polygons: [(xs, ys) ...] list of polygon tuples
    :param size: returned image will be of (size, size) ndarray
    :separate: If true alsro return boundary of each polygon as
        a separate image. this will be white image with black outline.
    :width: Width of outline.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    bdrs = []
    for x, y in polygons:
        pts = list(zip(x, y))
        pts = np.array(pts)
        pts = pts.reshape(-1, 2)
        cv2.fillPoly(img, [pts], 255)
        if separate:
            nimg = 255*np.ones_like(img)
            cv2.polylines(img, [pts], True, 0, width)
            cv2.polylines(nimg, [pts], True, 0, width)
            bdrs.append(nimg)
    if separate:
        return img, bdrs
    else:
        return bdrs

def create_color_mask(polygons, size, border=3, bg=0, fg=1, bd=2, aa=True):
    """Generates 3 channel mask
    :param polygons: [(xs, ys) ...] list of polygon tuples
    :param size: returned image will be of (size, size, 3) ndarray
    :param border: Width of border
    :param bg: Index of background channel
    :param fg: Index of forground channel
    :param bd: Index of border channel
    :returns: (size, size, 3) ndarray
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    color = [0,0,0]
    color[bg] = 255
    img[:] = color

    polies = []
    for x, y in polygons:
        pts = list(zip(x, y))
        pts = np.array(pts)
        pts = pts.reshape(-1, 2)
        polies.append(pts)

    color = [0, 0, 0]
    color[fg] = 255
    if aa: img = cv2.fillPoly(img, polies, color, cv2.LINE_AA)
    else: cv2.fillPoly(img, polies, color)

    color = [0,0,0]
    color[bd] = 255
    for poly in polies:
        poly = poly.reshape(-1, 1, 2)
        if aa: img = cv2.polylines(img, [poly], True, color, border, cv2.LINE_AA)
        else: img = cv2.polylines(img, [poly], True, color, border)
    return img


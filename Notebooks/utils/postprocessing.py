from skimage.morphology import binary_closing, disk, binary_opening, star
from skimage import morphology as morph
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage import io, draw
from skimage.color import label2rgb
import numpy as np

import cv2

import os

CROP_PAD = 5

def extract_cells(images, masks, names, output_directory):
    """Extracts cells from images and saves them into output_directory.
        A new directory is made for each image.
        within that directory, files will be saves as name_i.png and name_i_mask.png
        
        for debugging purpose image and it's overlay will also be saved as
            original.png and overlay.png
            
    :param images: [n, [c,], h, w] ndarray, original images. where n is batch_size
        if c is present [i, 0, :, :] is the original ith image. Other channels are
        assumed to be auxilary stained images.
    :param masks: [n, h, w] binary masks. 1 is where cell is present.
    :param names: iterable of length n. File name for each n.
    :param output_directory: root where subdirectories to be made.
    """
    images = images.clip(0, 1)
    for i in range(len(names)):
        oimg = images[i]
        if oimg.ndim > 2:
            oimg = oimg[0]
        name = os.path.basename(names[i])
        name, ext = os.path.splitext(name)
        print("Processing %s" % name)
        mask = masks[i]
        sel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, sel)
        
        labels = label(mask, connectivity=2, background=0)
        
        os.makedirs(os.path.join(output_directory, name),exist_ok=True)
        overlay = label2rgb(labels, oimg)
        
        oimg *= 2**16 - 1
        oimg = oimg.astype(np.uint16)
        mask *= 255
        cv2.imwrite(os.path.join(output_directory, name, 'mask.png'), mask)
        cv2.imwrite(os.path.join(output_directory, name, 'original.png'), oimg)
        
        
        path = os.path.join(output_directory, name, '{}.png')
        for j, prop in enumerate(regionprops(labels)):
            if prop.area > 700 and prop.solidity > 0.4 and prop.euler_number <= 1:
                minr, minc, maxr, maxc = prop.bbox
                minr = max(0,minr-CROP_PAD)
                minc = max(0, minc-CROP_PAD)
                maxr += CROP_PAD
                maxc += CROP_PAD
                cropped_img = oimg[minr:maxr, minc:maxc]
                cropped_msk = clear_border(mask[minr:maxr, minc:maxc])
                cv2.imwrite(path.format(j), cropped_img)
                cv2.imwrite(path.format(f'{j}_mask'), cropped_msk)
                cv2.rectangle(overlay, (minc, minr), (maxc, maxr), (1.,0,0), 3)
        io.imsave(os.path.join(output_directory, name, 'overlay.png'), overlay)

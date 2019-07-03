import numpy as np
import cv2
import torch

def siou(new, orig, ignore_on_border=False, 
         lb=0.2, ub=0.8, normalize='uniform'):
    """siou counts IoU over individual cells 
        and averages them.

    Args:
        orig: CxHxW where C is no of cells. all binary
        new : HxW predicted mask, again binary.
        lb: lower bound 
        ub: upper bound
        normalize: 'uniform', 'proportional' 'inv'
            uniform gives iou of each cell equal weight
            proportinoal gives iou of each cell weight proportional
                to it's area
            inv gives smaller cells more importance
    Returns: a scalar
    """
    if isinstance(new, torch.Tensor):
        new = new.cpu().numpy()
    if isinstance(orig, torch.Tensor):
        orig = orig.cpu().numpy()

    if orig.max() > 1:
        orig = (orig > 127).astype(np.uint8)
    if new.max() > 1:
        new = (new > 127).astype(np.uint8)
    h, w = new.shape
    if ignore_on_border:
        for i in range(w):
            if new[0, i] == 1:
                cv2.floodFill(new, None, (i, 0), 0)
            if new[h-1, i] == 1:
                cv2.floodFill(new, None, (i, h-1), 0)
        for i in range(h):
            if new[i, 0] == 1:
                cv2.floodFill(new, None, (0, i), 0)
            if new[i, w-1] == 1:
                cv2.floodFill(new, None, (w-1, i), 0)

    # orig = orig.astype(bool)
    # new = new.astype(bool)
    intersects = orig*new

    ious = []
    areas = []
    
    flags = 8 | (1<<8) | cv2.FLOODFILL_MASK_ONLY
    for cell, inter in zip(orig, intersects):
        conts, hier = cv2.findContours(inter, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        areas.append(cell.sum())
        #if there is a cell but wasnt
        #detected it will contribute 0
        #to final iou
        if len(conts) == 0:
            ious.append(0)
            continue

        max_area = 0
        for cont in conts:
            moments = cv2.moments(cont)
            area = moments['m00']
            if area > max_area:
                max_area = area
                max_x = int(moments['m10']/area)
                max_y = int(moments['m01']/area)
        if max_area == 0:
            #it seems inter is either line or a point
            ious.append(0)
            continue
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(new, mask, (max_x, max_y), None,  flags=flags)

        mask = mask[1:-1, 1:-1]
        intersect = mask*inter
        union = (mask+inter).clip(0, 1)
        iou = intersect.sum()/union.sum()
        if iou < lb:
            iou = 0.
        elif iou > ub:
            iou = 1.
        ious.append(iou)

    if isinstance(normalize, str):
        normz = [normalize]
    else:
        normz = normalize

    scores = []
    for norm in normz:
        norm = norm.lower()
        if norm[0] == 'i':
            weights = [1/a for a in areas]
        elif norm[0] == 'p':
            weights = areas
        else:
            weights = [1 for i in areas]
        W = sum(weights)
        weights = [w/W for w in weights]
        scores.append(sum(w*i for w, i in zip(weights, ious)))
    if isinstance(normalize, str):
        return scores[0]
    else:
        return scores
import torch
import numpy as np

def IoU(image, label):
    """
    :param image: torch tensor with values between 0 and 1.
        thresold is 0.5. i.e. if pixel is > .5 it's counted as 1 
        otherwise 0. shape [batch, hight, width]
    :param label: same as image but target. shape [batch, hight, width]
    :returns 1x1 tensor: IoU of two images.
    """
    union = image + label
    inter = (1 - image) + (1 - label)
    union = union.clamp_(0, 1)
    inter = inter.clamp_(0, 1)
    inter = 1 - inter
    u = union.view(image.size(0), -1)
    i = inter.view(image.size(0), -1)
    return i.sum(1).to(torch.float32)/u.sum(1).to(torch.float32)
    
def IoUv2(img, lbl):
    """
    :param img: resultant image, must be binary.
    :param lbl: original mask to be compared with. Must be binary.
    """
    img = img.reshape(len(img), -1)
    lbl = lbl.reshape(len(lbl), -1)
    tp = (img*lbl).sum(1)
    fp = ((1-lbl)*img).sum(1)
    fn = ((1-img)*lbl).sum(1)
    if isinstance(img, torch.Tensor):
        return tp.float()/(tp+fp+fn).float()
    else:
        return tp / (tp+fp+fn)

def create_image(mask, channel_dim=0):
    """From two channel logits or three channel logits
        create image"""
    if mask.shape[channel_dim] == 2:
        img = mask.argmax(channel_dim).to(torch.uint8).cpu().numpy()
    else:
        z = mask.argmax(channel_dim).cpu()
        c, h, w = mask.shape
        if channel_dim == 2:
            c, h, w = w, c, h
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (z == 0)
        img[:, :, 1] = (z == 1)
        img[:, :, 2] = (z == 2)
    img *= 255
    return img

def create_image_from_labels(labels, nclasses):
    """Must be two dimensional image. Each pixel must be 
        either 0, 1 or 2.
    :param labels: hxw image.
    :param nclasses: number of classes. 2 or 3.
    :param bgr: True if background is blue and border is red.
    :returns: Gray image if nclassses=2,
    """
    if nclasses == 2:
        img = labels.astype(np.uint8)
    else:
        h,w = labels.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (labels == 0)
        img[:, :, 1] = (labels == 1)
        img[:, :, 2] = (labels == 2)
    return img*255
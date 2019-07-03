import random
import cv2

def flip(imgs):
    """Flips image horizontaly or verticaly"""
    x = random.choice([-1, 0, 1, 2])
    if x == 2:
        return imgs
    else:
        return [cv2.flip(img, x) for img in imgs]

def normalize(imgs, idx=[0]):
    """Normalizes images[idx] between 0 and 1"""
    tmp = [img for img in imgs]
    for i in idx:
        tmp[i] = cv2.normalize(imgs[i], None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    return tmp

import os
import argparse

import numpy as np
from skimage import io
import torch
from torch.nn import functional as F

import utils
import tqdm

def eval_validation(net, device, ds, save_to):
    net.eval()
    with torch.no_grad():
        for x, y, t in ds:
            t = t[0]
            xx = x.to(device)
            yy = y.to(device)
            logits = net(xx)
            orig = xx[0]
            mask = utils.create_image_from_labels(yy[0], 3)
            gend = utils.create_image(logits[0], 0)

            logits = logits.cpu().numpy()
            io.imsave(os.path.join(save_to, "Images", f'{t}.png'), orig[2])
            io.imsave(os.path.join(save_to, "Masks", f'{t}.png'), mask)
            io.imsave(os.path.join(save_to, "Generated", f'{t}.png'), gend)
            np.save(os.path.join(save_to, "logits", f'{t}.npy'), logits)

def eval_runtime(net, device, ds, save_to, only_fg_probs=False):
    net.eval()
    with torch.no_grad():
        for x, t in tqdm.tqdm(ds):
            x = x.to(device)
            logits = net(x)
            if only_fg_probs:
                logits = F.softmax(logits, 1)[:, 1, :, :]
            cpulogits = logits.cpu().numpy()
            for i in range(len(x)):
                print("Processing %s" % t[i])
                np.save(os.path.join(save_to, "logits", f'{t[i]}.npy'), cpulogits[i])

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', help="Runtime", action="store_true")
    parser.add_argument('-d', '--device', default='cuda:0')
    parser.add_argument('-b', '--batch_size', default=5, type=int)
    parser.add_argument('checkpoint')
    parser.add_argument('input')
    parser.add_argument('output')
    return parser

import torch
import torch.nn.functional as functional
import cv2
from PIL import Image
import numpy as np
import os
import argparse
import torch
import torch.jit as jit

def pad_to_target(t, w, h):
    pd = (0, w - t.shape[3], 0, h - t.shape[2], 0, 0, 0, 0)
    return functional.pad(input = t, pad = pd)

def details_to_tensor(x):
        h, w = max(d.shape[2] for d in x), max(d.shape[3] for d in x)
        lu = []
        for t in x:
            t = pad_to_target(t, w, h)
            lu.append(t)
        return torch.cat(lu, dim = 1)

def tensor_to_details(x):
        lp = []
        num_channels = x.shape[1] // 3
        for ch in range(3):
            t = x[:, ch : ch + num_channels, :, :]
            lp.append(t)
        return lp

# convert tensors to [0, 1] range
def data_to_image(data, should_crop: bool = False):
    if should_crop: return torch.clip(data, 0, 1)
    mx = torch.max(data)
    mn = torch.min(data)
    if mn >= 0 and mx <= 1: return data
    rang = mx - mn
    if rang == 0: return 0.5 * torch.ones_like(data, device = data.device)
    return (data - mn) / rang

def tensors_to_pngs(vis, path, batch_num, should_crop):
    for i, pics in vis.items():
        pics = pics.cpu()
        for j, pic in enumerate(pics):
            save_tensor(pic, os.path.join(path, i, f'{batch_num}' + str(j) + '.png'), should_crop)

def save_tensor(tensor, path: str, should_crop: bool = False):
    pic = data_to_image(tensor, should_crop)
    pic = pic.numpy() * 255.
    pic = Image.fromarray(np.uint8(pic[0]), mode = 'L')
    pic.save(path)

# load image into tensor
def pngs_to_tensors(path, y = False):
    img = cv2.imread(path)
    if y: img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img = np.float32(img[:,:,0]) / 255.
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 1)
    img = torch.Tensor(img)
    if img.size(2) % 2 == 1: img = img[:, :, 0:img.size(2) - 1, :]
    if img.size(3) % 2 == 1: img = img[:, :, :, 0:img.size(3) - 1]
    return img

# update dictionary d1
def update_fyi_dict(d1, d2):
    if d1:
        for k2, v2 in d2.items():
            d1[k2] = d1[k2] + v2.detach()
    else:
        d1 = d2
    return d1

# boolean arguments in cmd
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class CharbonnierLoss(jit.ScriptModule):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps = 1e-3):
        super().__init__()
        self.eps = eps

    @jit.script_method
    def forward(self, x, y):
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps ** 2))
        return loss / torch.numel(y)
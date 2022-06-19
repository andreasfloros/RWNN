import os
import os.path
import glob
import numpy as np
import random
import torch
import torch.utils.data as udata
import cv2
import h5py

def image_to_patch(img, win, stride = 1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0 : endw - win + 1 : stride, 0 : endh - win + 1 : stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i : endw - win + i + 1 : stride, j : endh - win + j + 1 : stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k += 1
    return Y.reshape([endc, win, win, total_pat_num])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degrees
        out = np.rot90(out, k = 2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k = 2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k = 3)
    elif mode == 7:
        # rotate 270 degrees and flip
        out = np.rot90(out, k = 3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))

# prepare h5 files for training and validation
def prepare_data(data_path, stride = 10, aug_times = 0, patch_size = None):

    files = glob.glob(os.path.join(data_path, '*.png'))
    files.sort()

    val_max_count = 100

    for name in 'training', 'validation':

        h5f = h5py.File(f'h5/{name}_data.h5', 'w')
        count = 0
        scales = [0.18] if name == 'training' and patch_size is None else [1]
        skip = val_max_count if name == 'training' else 0

        for i in range(skip, len(files)):
            if count == val_max_count and name == 'validation': break
            img = cv2.imread(files[i])
            h, w, _ = img.shape
            for k in range(len(scales)):
                iimg = cv2.resize(img, (round(h * scales[k]), round(w * scales[k])), interpolation = cv2.INTER_CUBIC)
                iimg = np.expand_dims(iimg[:, :, 0].copy(), 0)
                iimg = np.float32(iimg / 255.)

                if patch_size is None or name != 'training':
                    h5f.create_dataset(str(count), data = iimg)
                    count += 1
                    continue

                patches = image_to_patch(iimg, win = patch_size, stride = stride)
                for n in range(patches.shape[3]):
                    data = patches[:, :, :, n].copy()
                    h5f.create_dataset(str(count), data = data)
                    count += 1
                    for m in range(aug_times):
                        data_aug = data_augmentation(data, np.random.randint(1,8))
                        h5f.create_dataset(str(count)+"_aug_%d" % (m + 1), data = data_aug)
                        count += 1
        h5f.close()
        print(f'{name} set # samples {count}')
    files.clear()

# definition of object to be passed to DataLoader
class Dataset(udata.Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.data_path = data_path
        h5f = h5py.File(self.data_path, 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        h5f = h5py.File(self.data_path, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

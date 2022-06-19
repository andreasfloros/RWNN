import numpy as np
import os
from glob import glob
import hdf5storage
from coordinator.wrapper import Wrapper
import torch
from helpers.pnp import pre_calculate, data_solution
from helpers.misc import pngs_to_tensors, save_tensor
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description = "Deblurring")
parser.add_argument("--epoch", type = int, default = -1, help = 'Select the epoch to load (make sure the file exists).')
parser.add_argument("--test_nl", type = float, default = 2.55, help = 'Set the testing noise level (0 to 255).')
parser.add_argument("--decode_depth", type = int, default = 1, help = 'Set the recursion depth for the transform.')
opt = parser.parse_args()

@torch.no_grad()
def main():
    wrapper_state_dict = os.path.join('logs', f'wrapper_net_epoch_{opt.epoch}.pth')

    device_ids = [0]
    wrapper_net = Wrapper(dae = False)
    model = torch.nn.DataParallel(wrapper_net, device_ids = device_ids)
    torch.backends.cudnn.benchmark = True
    model.load_state_dict(torch.load(wrapper_state_dict), strict = False)
    model.eval()

    lambd = 0.23
    model.module.transform_net.nenet.multiplier = 2

    files_source = glob(os.path.join('data', 'set12', '*.png'))
    files_source.sort()

    kernels = hdf5storage.loadmat(os.path.join('kernels_masks', 'Levin09.mat'))['kernels']
    # for each kernel
    for k_index in range(kernels.shape[1]):
        psnr_test = 0
        k = kernels[0, k_index].astype(np.float64)
        k_tensor = torch.Tensor(k).unsqueeze(0).unsqueeze(0).cuda()

        k_conv = torch.nn.Conv2d(1, 1, k_tensor.shape[-1], padding = 'same', bias = False, padding_mode = 'circular')
        k_conv.weight.data = torch.flip(k_tensor, [2, 3])

        img_index = 0
        # for each file
        for f in files_source:
            img_index += 1
            # image
            img_H_tensor = pngs_to_tensors(f).cuda()

            # degrade
            x = k_conv(img_H_tensor) + (opt.test_nl / 255.) * torch.randn_like(img_H_tensor)

            # precalc
            FB, FBC, F2B, FBFy = pre_calculate(x, k_tensor, sf = 1)

            # initialize iteration count and constants
            i = 0
            stdNv_ = model.module.transform_net.nenet(x)
            stdNv = 10 * stdNv_

            save_tensor(x[0, ...].cpu(), f"examples/pnp/deblur/{img_index}_degraded_{k_index}_kernel.png")

            while stdNv > stdNv_:

                # step 1, data fidelity term
                tau = lambd * (stdNv_**2 / stdNv**2)
                xd = data_solution(x, FB, FBC, F2B, FBFy, tau, sf = 1)
                # step 2, denoiser
                x, stdNv = model.module.fusion_denoise(xd, show_ne = True, layer = opt.decode_depth)
                i += 1

            save_tensor(x[0, ...].cpu(), f"examples/pnp/deblur/{img_index}_out_{k_index}_kernel.png")
            psnr = -10. * torch.log10(torch.nn.functional.mse_loss(x, img_H_tensor))
            print(i, psnr)
            psnr_test += psnr

        print(f'PSNR for kernel {k_index}:', psnr_test / img_index)

if __name__ == '__main__':
    main()
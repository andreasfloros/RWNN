import torch
from coordinator.wrapper_algorithms import WrapperAlgorithms
from models.transform_model import ImageTransform
from helpers.misc import CharbonnierLoss
from helpers.perlin import gen_noise_map

class Wrapper(WrapperAlgorithms):
    def __init__(self, dae, num_channels = 1, test_nl = 0, decode_depth = 1):
        self.alg = self.dae if dae else self.fusion_denoise
        super().__init__(num_channels)
        self.transform_net = ImageTransform(num_channels = num_channels)

        self.max_nrm_train_nl = 55. / 255.

        self.nrm_test_nl = test_nl / 255.
        self.decode_depth = decode_depth

        self.mse = torch.nn.MSELoss()
        # smooth l1
        self.l1 = CharbonnierLoss()

    def get_training_loss(self, x):
        l = {}
        nrm_sigma = torch.rand(1, device = x.device) * self.max_nrm_train_nl
        inp = x + nrm_sigma * torch.randn_like(x)
        res, ne = self.alg(inp, layer = self.decode_depth, show_ne = True)
        l['mse'] = self.mse(res, x)
        l['psnr'] = -10. * torch.log10(l['mse'])
        l['l1'] = self.l1(res, x)
        l['ne'] = self.l1(ne, nrm_sigma)
        if (self.alg == self.fusion_denoise):
            l['oloss'] = self.dn_net.orthogonal_loss()
        l['total loss'] = l['l1'] + l['ne'] + 10. * (l['oloss'] if (self.alg == self.fusion_denoise) else 0.)
        return l, l['total loss']

    def test(self, x, vis = False, map = False):
        l = {}
        map = gen_noise_map(x.shape, self.nrm_test_nl) if map else None
        inp = x + (self.nrm_test_nl if map is None else map) * torch.randn_like(x)
        res = self.alg(inp, layer = self.decode_depth, map = map).clip(0., 1.)
        l['mse'] = self.mse(res, x)
        l['psnr'] = -10. * torch.log10(l['mse'])
        l['l1'] = self.l1(res, x)
        l['total loss'] = l['l1']
        if not vis: return l, l['total loss']
        to_vis = {}
        to_vis['original'] = x
        xf, _ = self.transform_net(x, J = 1)
        to_vis['original transformed'] = self.transform_net.collect_pieces(xf)
        to_vis['input'] = inp
        inpf, _ = self.transform_net(inp, J = 1)
        to_vis['input transformed'] = self.transform_net.collect_pieces(inpf)
        to_vis['result'] = res
        return l, l['total loss'], to_vis

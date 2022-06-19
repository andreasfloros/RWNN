import torch
from models.ita import FusionNet
from helpers.misc import details_to_tensor, tensor_to_details

class WrapperAlgorithms(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        dn_p = 3
        dn_latent_dim = 256
        dn_iterations = 7

        self.dn_net = FusionNet(num_channels, dn_p, dn_latent_dim, dn_iterations)

    def dae(self, x, layer = 1, map = None, show_ne = False):
        [cn, dn], nes = self.transform_net.forward(x, map = map, J = layer)
        dn = [[torch.zeros_like(d) for d in dn[j]] for j in range(len(dn))]
        res = self.transform_net.inverse_from_pieces([cn, dn])
        if show_ne:
            return res, nes[-1]
        return res

    def fusion_denoise(self, x, layer = 1, map = None, show_ne = False):
        x, nes = self.transform_net.forward(x, map = map, J = layer)
        yc = x[0] - self.transform_net.offset
        for ds, ne in zip(x[1], nes):
            dn = details_to_tensor(ds)
            dn = self.dn_net(dn, yc, ne)
            dn = tensor_to_details(dn[:,:3,...])
            for j in range(3):
                ds[j] = dn[j][..., : ds[j].shape[-2], : ds[j].shape[-1]]
            yc = self.transform_net.inn.inverse((yc, ds), self.transform_net.default_ne)
            yc = self.transform_net.split.inverse(yc)
        if show_ne:
            return yc + self.transform_net.offset, nes[-1]
        else:
            return yc + self.transform_net.offset
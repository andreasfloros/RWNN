import torch
from helpers.misc import pad_to_target

# lazy invertible transformation (even and odd split)
class LazyWaveletTransform(torch.nn.Module):
    def __init__(self, undecimated):
        super().__init__()
        self.undecimated = undecimated

    def forward(self, x):
        ec, oc = self._1d_forward(x, dim = 3)
        ecer, ecor = self._1d_forward(ec, dim = 2)
        ocer, ocor = self._1d_forward(oc, dim = 2)
        return ecer, [ecor, ocer, ocor]

    def _1d_forward(self, x, dim):
        if self.undecimated:
            return x, x
        e = x[..., : : 2] if dim == 3 else x[..., : : 2, :]
        o = x[..., 1 : : 2] if dim == 3 else x[..., 1 : : 2, :]
        return e, o

    def inverse(self, x):
        ecer, [ecor, ocer, ocor] = x
        oc = self._1d_inverse((ocer, ocor), dim = 2)
        ec = self._1d_inverse((ecer, ecor), dim = 2)
        return self._1d_inverse((ec, oc), dim = 3)

    def _1d_inverse(self, x, dim):
        e, o = x
        if self.undecimated:
            return e
        sz = (e.shape[0], e.shape[1], e.shape[2], e.shape[3] + o.shape[3]) if dim == 3 \
             else (e.shape[0], e.shape[1], e.shape[2] + o.shape[2], e.shape[3])
        ix = torch.zeros(sz, device = e.device)
        if dim == 3:
            ix[..., : : 2] = e
            ix[..., 1 : : 2] = o
        else:
            ix[..., : : 2, :] = e
            ix[..., 1 : : 2, :] = o
        return ix
import torch
from helpers.misc import details_to_tensor, tensor_to_details
from models.punet import ResNet, DepthwiseSeparableConv
from models.nonlinear import AdaptiveReLU

# a series of lifting steps
class InvertibleNet(torch.nn.Module):
    def __init__(self, num_lifts, num_channels):
        super().__init__()
        self.lifts = torch.nn.ModuleList(LiftNet(num_channels = num_channels) for _ in range(num_lifts))
        
    def forward(self, x, sigma):
        for lift in self.lifts:
            x = lift(x, sigma)
        return x

    def inverse(self, x, sigma):
        for lift in reversed(self.lifts):
            x = lift.inverse(x, sigma)
        return x

# implements the lifting scheme
class LiftNet(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        latent_dim = 32
        p = 3
        iterations = 7
        nltyp = AdaptiveReLU
        convtyp = DepthwiseSeparableConv
        self.p = PredictorNet(num_channels, latent_dim, p, iterations, nltyp, convtyp)
        self.u = UpdaterNet(num_channels, latent_dim, p, iterations, nltyp, convtyp)

    def forward(self, x, sigma):
        c, d = x
        pd = self.p(c, sigma)
        nd = []
        for i in range(len(d)):
            nd.append(d[i] + pd[i][..., : d[i].shape[-2], : d[i].shape[-1]])
        c = c + self.u(nd, sigma)
        return c, nd

    def inverse(self, x, sigma):
        c, d = x
        c = c - self.u(d, sigma)
        pd = self.p(c, sigma)
        nd = []
        for i in range(len(d)):
            nd.append(d[i] - pd[i][..., : d[i].shape[-2], : d[i].shape[-1]])
        return c, nd

class PredictorNet(torch.nn.Module):
    def __init__(self, num_channels, latent_dim, p, iterations, nltyp, convtyp):
        super().__init__()
        self.net = ResNet(num_channels, latent_dim, 3 * num_channels, p, iterations, nltyp, convtyp)
        
    def forward(self, x, sigma):
        x = self.net(x, sigma)
        return tensor_to_details(x)

class UpdaterNet(torch.nn.Module):
    def __init__(self, num_channels, latent_dim, p, iterations, nltyp, convtyp):
        super().__init__()
        self.net = ResNet(3 * num_channels, latent_dim, num_channels, p, iterations, nltyp, convtyp)
        
    def forward(self, x, sigma):
        x = details_to_tensor(x)
        return self.net(x, sigma)

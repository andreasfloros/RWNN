import torch
from models.nonlinear import SoftThreshold

# Convolutional Learned ITA
class ITANet(torch.nn.Module):
    def __init__(self, in_dim, p, latent_dim, iterations):
        super().__init__()
        self.in_dim = in_dim
        self.iterations = iterations
        self.latent_dim = latent_dim

        self.synthesis_kernel = torch.nn.Conv2d(self.latent_dim, in_dim, p, padding = 'same', bias = False, padding_mode = 'circular')
        self.analysis_kernel = torch.nn.Conv2d(in_dim, self.latent_dim, p, padding = 'same', bias = False, padding_mode = 'circular')
        self.in_thresh = SoftThreshold(in_dim = latent_dim)

        self.thresholds = torch.nn.ModuleList(SoftThreshold(in_dim = latent_dim) for _ in range(iterations))
        self.intermediate_kernels = torch.nn.ModuleList(torch.nn.Conv2d(in_dim, self.latent_dim, p, padding = 'same', bias = False) for _ in range(iterations))

        torch.nn.init.xavier_normal_(self.synthesis_kernel.weight)
        self.analysis_kernel.weight.data = torch.flip(self.synthesis_kernel.weight.data.transpose(0, 1), [2, 3])
        for kernel in self.intermediate_kernels:
            kernel.weight.data = self.analysis_kernel.weight.data

        self.mse = torch.nn.MSELoss()
        self.delta = torch.empty(in_dim, in_dim, p, p, device = 'cuda')
        torch.nn.init.dirac_(self.delta)

    def forward(self, x, sigma):
        g = self.in_thresh(self.analysis_kernel(x), aug = sigma)
        for thresh, kernel in zip(self.thresholds, self.intermediate_kernels):
            g = thresh(g + kernel(x - self.synthesis_kernel(g)), aug = sigma)
        return self.synthesis_kernel(g)

    def orthogonal_loss(self):
        return self.mse(self.synthesis_kernel(self.analysis_kernel(self.delta)), self.delta)

class FusionNet(torch.nn.Module):
    def __init__(self, in_dim, p, latent_dim, iterations):
        super().__init__()
        self.in_dim = in_dim
        self.iterations = iterations
        self.latent_dim = latent_dim
        num_f = 3

        self.proj_kernel = torch.nn.Conv2d(in_dim, num_f * in_dim, p, padding = 'same', bias = False)

        self.ita_net = ITANet((3 + num_f) * in_dim, p, latent_dim, iterations)

    def forward(self, x, prior, sigma):
        prior = self.proj_kernel(prior)
        x = torch.cat((x, prior), dim = 1)
        return self.ita_net(x, sigma)[:,:3 * self.in_dim,...]
        
    def orthogonal_loss(self):
        return self.ita_net.orthogonal_loss()
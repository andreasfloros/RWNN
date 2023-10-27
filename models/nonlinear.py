import torch

# noise adaptive nonlinearity
class AdaptiveReLU(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.threshold = torch.nn.Parameter(-0.1 * torch.rand(1, in_dim, 1, 1))

    def forward(self, x, aug):
        return torch.relu(x + self.threshold * aug)

class SoftThreshold(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta = 20)
        self.threshold = torch.nn.Parameter(-0.1 * torch.rand(1, in_dim, 1, 1))

    def forward(self, x, aug):
        return torch.sign(x) * torch.relu(torch.abs(x) - self.softplus(self.threshold) * aug)
import torch
import torch.jit as jit

class DepthwiseSeparableConv(jit.ScriptModule):
    def __init__(self, in_dim, out_dim, p, bias = False, padding = 'same'):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(in_dim, in_dim, kernel_size = p, padding = padding, groups = in_dim, bias = False)
        self.pointwise = torch.nn.Conv2d(in_dim, out_dim, kernel_size = 1, bias = bias)

    @jit.script_method
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class TConvBlock(jit.ScriptModule):
    def __init__(self, in_dim, out_dim, p, nltyp, convtyp):
        super().__init__()
        self.conv = convtyp(in_dim, out_dim, p, padding = 'same', bias = False)
        self.thresh = nltyp(out_dim)

    @jit.script_method
    def forward(self, x, aug):
        return self.thresh(self.conv(x), aug = aug)

class ResBlock(jit.ScriptModule):
    def __init__(self, dim, p, nltyp, convtyp):
        super().__init__()
        self.tconva = TConvBlock(dim, dim, p, nltyp, convtyp)
        self.convb = convtyp(dim, dim, p, padding = 'same', bias = False)
        self.threshb = nltyp(dim)

    @jit.script_method
    def forward(self, x, aug):
        y = self.tconva(x, aug = aug)
        return self.threshb(self.convb(y) + x, aug = aug)

# PUNet
class ResNet(jit.ScriptModule):
    def __init__(self, in_dim, latent_dim, out_dim, p, iterations, nltyp, convtyp):
        super().__init__()
        self.convin = torch.nn.Conv2d(in_dim, latent_dim, p, padding = 'same', bias = False)
        self.blocks = torch.nn.ModuleList(ResBlock(latent_dim, p, nltyp, convtyp) for _ in range(iterations))
        self.convout = torch.nn.Conv2d(latent_dim, out_dim, p, padding = 'same', bias = False)

    @jit.script_method
    def forward(self, x, sigma):
        x = self.convin(x)
        for block in self.blocks:
            x = block(x, sigma)
        return self.convout(x)
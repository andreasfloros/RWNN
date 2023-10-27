import torch

class NENet(torch.nn.Module):
    def __init__(self, psz = 8, stride = 1, num_layers = 8, f_ch = 16):
        super().__init__()
        self.psz = psz
        self.stride = stride
        self.fsz = 5
        self.num_layers = num_layers
        self.f_ch = f_ch
        self.unfold = torch.nn.Unfold(kernel_size = (self.psz, self.psz), stride = self.stride)
        layers = []
        layers.append(torch.nn.Conv1d(in_channels = 1, out_channels = f_ch, kernel_size = self.fsz, padding = 0, bias = False))
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers):
            layers.append(torch.nn.Conv1d(in_channels = f_ch, out_channels = f_ch, kernel_size = self.fsz, padding = 0, bias = False))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv1d(in_channels = f_ch, out_channels = 1, kernel_size = self.fsz, padding = 0, bias = False))
        self.net = torch.nn.Sequential(*layers)
        self.sigmoid = torch.nn.Sigmoid()

        self.multiplier = 1.
        self.override_flag = False
        self.override = 5. / 255.

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_uniform_(m.weight, a = 0.0, mode = 'fan_out', nonlinearity= 'relu')

    def forward(self, img):
        if self.override_flag: return self.override * torch.ones((1), device = 'cuda')
        '''Unfold input image into patches'''
        P = self.unfold(img)
        P = P.permute(0, 2, 1)
        '''Reshape patches to vectors'''
        V = torch.reshape(P, (-1, 1, P.size(2)))
        V = self.net(V)
        W = self.sigmoid(torch.mean(V, 2))
        W = torch.reshape(W, (P.size(0), P.size(1), -1))
        pw = P * W
        pw = torch.reshape(pw, (1, -1, self.psz * self.psz))
        p = torch.reshape(P, (1, -1, self.psz * self.psz))
        '''Implement SVD(P x diag(w) x P^T)'''
        PWPT = torch.matmul(p.transpose(-2, -1), pw)
        S = torch.linalg.svdvals(PWPT)
        out = S[:, -1]
        out = (out / torch.sum(W)) ** 0.5
        out = torch.mean(out)
        return self.multiplier * out.unsqueeze(0)
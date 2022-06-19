from collections import deque
from helpers.misc import data_to_image
from models.noise_estimation import NENet
from models.split import LazyWaveletTransform
from models.inn import InvertibleNet
import torch

# the transformation network
class ImageTransform(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.num_lifts = 4
        self.split = LazyWaveletTransform(undecimated = False)
        self.inn = InvertibleNet(num_lifts = self.num_lifts, num_channels = self.num_channels)

        self.nenet = NENet()
        self.default_ne = torch.ones(1, device = 'cuda') * 0
        self.offset = 0.5
        
    # returns tuple t with t[0] = coarse part, t[1] = a list of lists containing the details
    # the elements of t[1] are lists of tensors with a length of 3 (bottom left, top right, bottom right)
    # the details are ordered wrt to the scale, i.e. scale K details are at t[1][0], scale K - 1 at t[1][1] etc
    # a list of the noise estimates is returned in the same order as the details
    def forward(self, x, J = float('inf'), map = None):
        x = x - self.offset
        d = deque()
        sigma = deque()
        while x.shape[2] > 7 and x.shape[3] > 7 and J > 0:
            if map is None:
                ne = self.nenet(x)
            else:
                ne = torch.nn.functional.interpolate(map, size = ((x.shape[2] + 1) // 2, (x.shape[3] + 1) // 2), mode = 'bicubic')
            x = self.split(x)
            x, yd = self.inn(x, ne)
            d.appendleft(yd)
            sigma.appendleft(ne)
            J -= 1
        x = x + self.offset
        return [x, d], sigma

    def inverse_from_pieces(self, x, sigma = None):
        yc = x[0] - self.offset
        for k in range(len(x[1])):
            yc = self.inn.inverse((yc, x[1][k]), self.default_ne if sigma is None else sigma[k])
            yc = self.split.inverse(yc)
        return yc + self.offset

    # collects the forward transformed pieces into a tensor for image visualisation
    def collect_pieces(self, x, should_crop = False, bandwise = True):
        yc = x[0]
        if bandwise: yc = data_to_image(yc, should_crop = True)
        for yd in x[1]:
            bl, tr, br = yd
            if bandwise:
                tr, bl, br = data_to_image(tr, should_crop = should_crop), data_to_image(bl, should_crop = should_crop), \
                             data_to_image(br, should_crop = should_crop)
            t = torch.cat((yc, tr), 3)
            b = torch.cat((bl, br), 3)
            yc = torch.cat((t, b), 2)
        if not bandwise: return data_to_image(yc, should_crop = should_crop)
        return yc

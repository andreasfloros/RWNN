import os
from time import time
import torch
from torch.utils.data import DataLoader
from helpers.misc import pngs_to_tensors, tensors_to_pngs, update_fyi_dict

# training and testing boilerplate
class Coordinator:
    def __init__(self, wrapper_net, batched_datasets, datasets, batch_size = 16, optimizer = torch.optim.Adam, wrapper_state_dict = None, num_workers = 6,
                 device = torch.device('cuda'), device_ids = [0], epoch_start = 0):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.cur_epoch = epoch_start -1
        self.loaders = {name : DataLoader(dataset = batched_datasets[name], num_workers = num_workers,
                        batch_size = batch_size if name == 'Training' else 1, shuffle = True, pin_memory = True) for name in batched_datasets}
        self.datasets = datasets
        self.model = torch.nn.DataParallel(wrapper_net, device_ids = device_ids)
        torch.backends.cudnn.benchmark = True
        self.optimizer = optimizer(self.model.parameters())
        if wrapper_state_dict: self.model.load_state_dict(torch.load(wrapper_state_dict), strict = False) #

        transform_params = sum(p.numel() for p in wrapper_net.transform_net.parameters())
        print('# transform parameters: ' + str(transform_params))
        total_params = sum(p.numel() for p in self.model.parameters())
        print('# total model parameters: ' + str(total_params))

    # set the path to see the transformed images
    @torch.no_grad()
    def test(self, path = None, should_crop = True, map = False):
        self.model.eval()
        info = {}
        t1 = time()
        for name in self.datasets:
            torch.cuda.synchronize()
            sum_fyi = {}
            num_batches = 0
            for idx, f in enumerate(self.datasets[name]):
                data = pngs_to_tensors(f)
                data = data.to(self.device)
                if path is None:
                    fyi, loss = self.model.module.test(data, map = map)
                else:
                    fyi, loss, vis = self.model.module.test(data, map = map, vis = True)
                    tensors_to_pngs(vis, os.path.join(path, 'rwnn', name), idx, should_crop)
                if torch.isnan(loss) or torch.isinf(loss): raise Exception("nan / inf loss")
                sum_fyi = update_fyi_dict(sum_fyi, fyi)
                num_batches += 1

            torch.cuda.synchronize()
            for k in sum_fyi: sum_fyi[k] /= num_batches
            fyistr = []
            for i, inf in sum_fyi.items(): fyistr.append(" " + i + f": {inf: .6e},")
            info[name] = ''.join(fyistr), sum_fyi
        dtmins = (time() - t1) / 60
        return dtmins, info

    # a single epoch over the specified (batched) datasets
    def train(self):
        info = {}
        self.cur_epoch += 1
        t1 = time()

        # training set
        torch.cuda.synchronize()
        self.model.train()
        sum_fyi = {}
        for iter, data in enumerate(self.loaders['Training'], 0):
            self.model.zero_grad(set_to_none = True)
            data = data.to(self.device)
            fyi, loss = self.model.module.get_training_loss(data)
            if torch.isnan(loss) or torch.isinf(loss): raise Exception("nan / inf loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1, norm_type = 2)
            
            self.optimizer.step()
            sum_fyi = update_fyi_dict(sum_fyi, fyi)
        torch.cuda.synchronize()
        for k in sum_fyi: sum_fyi[k] /= (iter + 1)
        fyistr = []
        for i, inf in sum_fyi.items(): fyistr.append(" " + i + f": {inf: .6e},")
        info['Training'] = ''.join(fyistr), sum_fyi

        # validation set
        with torch.no_grad():
            torch.cuda.synchronize()
            self.model.eval()
            sum_fyi = {}
            for iter, data in enumerate(self.loaders['Validation'], 0):
                data = data.to(self.device)
                fyi, loss = self.model.module.test(data)
                if torch.isnan(loss) or torch.isinf(loss): raise Exception("nan / inf loss")
                sum_fyi = update_fyi_dict(sum_fyi, fyi)
            torch.cuda.synchronize()
        for k in sum_fyi: sum_fyi[k] /= (iter + 1)
        fyistr = []
        for i, inf in sum_fyi.items(): fyistr.append(" " + i + f": {inf: .6e},")
        info['Validation'] = ''.join(fyistr), sum_fyi

        dtmins = (time() - t1) / 60

        return dtmins, info

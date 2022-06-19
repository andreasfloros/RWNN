import os
import argparse
import torch
from coordinator.wrapper import Wrapper
from coordinator.coordinator import Coordinator
from helpers.prepare_datasets import prepare_data, Dataset
from helpers.misc import str2bool

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description = "RWNN Training")
parser.add_argument("--should_prepare", type = str2bool, default = False, help = 'Set to True to run prepare_data.')
parser.add_argument("--epochs", type = int, default = 1000, help = 'Set the number of epochs.')
parser.add_argument("--max_patience", type = int, default = 5, help = 'Set the maximum patienece.')
parser.add_argument("--num_workers", type = int, default = 12, help = 'Set the number of workers.')
parser.add_argument("--batch_size", type = int, default = 32, help = 'Set the batch size.')
parser.add_argument("--epoch_start", type = int, default = 0, help = 'Load the state dict of epoch_start - 1.')
parser.add_argument("--dae", type = str2bool, default = True, help = 'RWNN-F or RWNN-DAE setting (train DAE first).')
parser.add_argument("--lr", type = float, default = 1e-3, help = 'Set the learning rate.')
parser.add_argument("--decode_depth", type = int, default = 2, help = 'Set the recursion depth for the transform.')
opt = parser.parse_args()

if __name__ == "__main__":
    
    print("\nPyTorch is running on " + torch.cuda.get_device_name(0) + "\n")
    
    if opt.should_prepare: prepare_data(data_path='data/bsd400', patch_size = 64, stride = 10, aug_times = 0)

    wrapper_net = Wrapper(
                          dae = opt.dae,
                          decode_depth = opt.decode_depth,
                          test_nl = 25.
                         )
    optimizer = lambda x : torch.optim.Adam(x, lr = opt.lr, weight_decay = 0)
    wrapper_state_dict = os.path.join('logs', f'wrapper_net_epoch_{opt.epoch_start - 1}.pth') if opt.epoch_start else None
    batched_datasets = {'Training' : Dataset('h5/training_data.h5'),
                        'Validation' : Dataset('h5/validation_data.h5')}
    datasets = {}

    coordinator = Coordinator(epoch_start = opt.epoch_start, wrapper_net = wrapper_net, batch_size = opt.batch_size, optimizer = optimizer,
                              wrapper_state_dict = wrapper_state_dict, num_workers = opt.num_workers, batched_datasets = batched_datasets, datasets = datasets)

    min_val_loss = float('inf')
    patience = opt.max_patience
    epoch_star = None

    for epoch in range(opt.epoch_start, opt.epochs + opt.epoch_start):
        print('')
        dtmins, info = coordinator.train()
        print('Epoch 'f'{epoch}, time: 'f'{dtmins: .3e} mins')
        for dataset, fyi in info.items():
            print(dataset + ' dataset,' + fyi[0])
        torch.save(coordinator.model.state_dict(), os.path.join('logs', 'wrapper_net_epoch_{}.pth'.format(epoch)))
        if info['Validation'][1]['total loss'] < min_val_loss:
            min_val_loss = info['Validation'][1]['total loss']
            epoch_star = epoch
            patience = opt.max_patience
        else:
            patience -= 1
            if patience == 0:
                coordinator.optimizer.param_groups[0]['lr'] /= 10.
                print('')
                print('Reducing learning rate to ', coordinator.optimizer.param_groups[0]['lr'])
                patience = opt.max_patience
                print('')
                if coordinator.optimizer.param_groups[0]['lr'] < 1e-6:
                    print('Learning rate dropped below 1e-6, best validation results on epoch ' + str(epoch_star))
                    break
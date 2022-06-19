import os
import argparse

from coordinator.coordinator import Coordinator
from glob import glob

from coordinator.wrapper import Wrapper
from helpers.misc import str2bool

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description = "RWNN Testing")
parser.add_argument("--epoch", type = int, default = -1, help = 'Select the epoch to load (make sure the file exists).')
parser.add_argument("--test_nl", type = float, default = 0., help = 'Set the testing noise level (0 to 255).')
parser.add_argument("--decode_depth", type = int, default = 1, help = 'Set the recursion depth for the transform.')
parser.add_argument("--map", type = str2bool, default = False, help = 'Set to True for a spatially varying noise map.')
parser.add_argument("--dae", type = str2bool, default = True, help = 'RWNN-F or RWNN-DAE setting (make sure you load the correct weights).')
opt = parser.parse_args()

if __name__ == "__main__":
    
    assert (opt.decode_depth > 1 and opt.map) == False, "Non-uniform maps are only supported for a single level."

    wrapper_net = Wrapper(
                          dae = opt.dae,
                          test_nl = opt.test_nl,
                          decode_depth = opt.decode_depth
                         )
    batch_size = 1
    wrapper_state_dict = os.path.join('logs', f'wrapper_net_epoch_{opt.epoch}.pth') if opt.epoch >= 0 else None
    num_workers = 6
    datasets = {
                'set12': glob(os.path.join('data', 'set12', '*.png'))
                }

    batched_datasets = {}
    
    print('')

    coordinator = Coordinator(wrapper_net = wrapper_net, batch_size = batch_size, wrapper_state_dict = wrapper_state_dict,
                              num_workers = num_workers, datasets = datasets, batched_datasets = batched_datasets)

    print('')

    dtmins, info = coordinator.test(path = 'examples/rwnn', map = opt.map)
    print('Time: 'f'{dtmins: .3e} mins')
    for dataset, fyi in info.items():
        print(dataset + ' dataset,' + fyi[0])
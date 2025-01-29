import os
import torch
import random
import itertools
import numpy as np
from argparse import ArgumentParser

def construct_parser():
    parser = ArgumentParser()

    parser.add_argument(
        '--device',
        metavar='--d',
        type=int,
        default=None,
        help='Device to be used'
    )
    
    parser.add_argument(
        '--num_workers',
        metavar='--w',
        type=int,
        default=0,
        help='Number of workers to use'
    )    

    parser.add_argument(
        '--num_replicates',
        metavar='--w',
        type=int,
        default=1,
        help='Number of experimental replicates'
    )    

    parser.add_argument(
        '--output_dir',
        metavar='--o',
        type=str,
        default='./outputs/',
        help='Name of the output directory'
    )

    parser.add_argument(
        '--random_seed',
        metavar='--s',
        type=int,
        default=10,
        help='Random seed'
    )    
    return parser


def set_seed(seed):
    # Python seed
    random.seed(seed)
    
    # Numpy seed
    np.random.seed(seed)
    
    # Torch seed
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device(device):
    # Select cpu if missing
    device = 'cpu' if device is None else device
    
    # Set device
    if device != 'cpu':    
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)    
    print('Device:\t', device)
    return device 


def remove_logs(dir_name):        
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".log"):
            os.remove(os.path.join(dir_name, item))      


def create_hypermarameters_grid(dict_params_lists, drop_params = []):
    
    # Remode hyper parameters to be dropped
    D = {
        k: v for k, v in dict_params_lists.items() 
        if v not in drop_params
    }
    
    # Create list of dictionary of hyper-params candidates
    L = [
        dict(zip(D.keys(), p)) for p in itertools.product(*D.values())
    ]

    return L


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

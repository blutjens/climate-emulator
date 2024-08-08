import os
import yaml
import numpy as np
import random
from pprint import pprint
from pathlib import Path
import torch

def set_all_seeds(seed, device='cpu',
                  use_deterministic_algorithms=False,
                  warn_only=False):
    """
    sets all seeds. 
    Copied from hrmelt.utils.utils on 4/4/2024
    See src: https://github.com/pytorch/pytorch/issues/7068
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        print('in utils.py -> set_all_seeds cuda')
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic=True
    # sets, e.g., nn.ConvTranspose2d to deterministic
    torch.use_deterministic_algorithms(mode=use_deterministic_algorithms, warn_only=warn_only)

def generate_dicts_recursive(input_dict, current_dict=None, depth=0):
    '''
    Recursively generates a list of dictionaries containing
    all possible combinations of the values
    source: chat gpt-3.5
    Copied from hrmelt.utils.utils on 4/4/2024
    Args:
        keys list(): List of dictionary keys
        input_dict dict(): Input dictionary
    '''
    keys = list(input_dict.keys())

    if current_dict is None:
        current_dict = {}

    if depth == len(keys):
        return [current_dict]

    key = keys[depth]
    values = input_dict[key]
    result_dicts = []

    for value in values:
        new_dict = current_dict.copy()
        new_dict[key] = value
        result_dicts.extend(generate_dicts_recursive(input_dict, new_dict, depth + 1))

    return result_dicts

def init_sweep_config(cfg, path_sweep_cfg, task_id=1, num_tasks=1):
    '''
    Updates the cfg with a randomly drawn combination of 
    hyperparameters from the sweep config. 
    Copied from hrmelt.utils.utils on 4/4/2024
    '''
    # Update logging paths
    data_keys = ['data_root', 'data_path', 'data_path_interim']
    for data_key in data_keys:
        if data_key in cfg:
            if '/d3/' in cfg[data_key]:
                print(f'Prepending /net/fs06/ to data path {data_key}.')
                cfg[data_key] = '/net/fs06/' + cfg[data_key]

    # Update config with sweep parameters
    sweep_cfg = yaml.safe_load(open(path_sweep_cfg, 'r'))
    # Initialize list of all possible cfg combinations
    list_of_sweep_cfgs = generate_dicts_recursive(sweep_cfg)
    print(f'Running {num_tasks}/{len(list_of_sweep_cfgs)} random sweep configurations on all tasks.')
    # Randomly shuffle the combinations and then draw the element with index
    # task.id. This is necessary because all tasks run on different GPUs, but
    # share the same random seed.
    random.seed(cfg['seed'])
    random.shuffle(list_of_sweep_cfgs)
    current_sweep_cfg = list_of_sweep_cfgs[task_id-1] # minus 1 switches from 1 to zero indexing

    # Update the main config with the parameters chosen for this sweep
    cfg.update(current_sweep_cfg)
    print('Choosing sweep configuration:')
    pprint(current_sweep_cfg)

    return cfg
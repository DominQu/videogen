from pathlib import Path
import json
import logging
import random

import torch
from numpy.random import default_rng

def set_global_seed(seed=777):
    """Set global seed for modules that have random generators"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np_rng = default_rng(seed)
    random.seed(seed)
    return np_rng

def load_params(config_file: Path) -> dict:
        """Load network parameters from json file"""
        if not config_file.is_file():
            raise ValueError("Specified config file is not valid")

        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.info("Config file loaded succesfully")
        return config
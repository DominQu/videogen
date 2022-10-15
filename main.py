from pathlib import Path
import logging
import random

import torch
import numpy as np
from numpy.random import default_rng

from src.models import AutoEncoder, RecurrentReversiblePredictor
from src.crev_net import CrevNet

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S', level=logging.DEBUG)

def set_global_seed(seed=777):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np_rng = default_rng(seed)
    random.seed(seed)
    return np_rng

if __name__ == "__main__":

    # Seed all modules, numpy random generator is returned to pass to other functions
    np_rng = set_global_seed()

    config = Path("configs/mnist_config.json")
    network = CrevNet(config)
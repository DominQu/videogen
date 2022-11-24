from pathlib import Path
import json
import logging
import random

import torch
from numpy.random import default_rng
from PIL import Image
import torchvision.transforms as transforms


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

def load_sequence(img_dir: Path, seq_len: int=20, img_size: int=64) -> list:
    """Load frame sequence from directory"""
    if not img_dir.is_dir():
        raise ValueError("Given path is not a directory!")
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    
    frames = img_dir.glob('*')
    seq = []
    for frame in frames:
        img = Image.open(str(frame))
        seq.append(transform(img))

    if len(seq) < seq_len+2:
        raise ValueError("There isn't enough frames in specified directory")

    return seq[:seq_len+2]
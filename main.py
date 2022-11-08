from pathlib import Path
import logging
import random
import json

import torch
import numpy as np
from numpy.random import default_rng
import wandb

from src.crev_net import CrevNet
from datasets.mnist import MovingMnistDataset
from datasets.pennaction import PennActionDataset

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S', level=logging.INFO)
wandb.init(project="inzynierka", entity="dominqu", mode='online')

def set_global_seed(seed=777):
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

def train_penn_action(np_rng):
    config = Path("configs/pennaction_config.json")
    params = load_params(config)

    dataset = PennActionDataset(
        batch_size = params["batch_size"], 
        dataset_path ='data/Penn_Action',
        sequence_length = params["warmup_steps"] + params["prediction_steps"] + 2,
        np_rng=np_rng,
        img_size = params["autoencoder"]["input_shape"][1],
    )
    print(dataset.get_size())
    network = CrevNet(params, dataset, device)
    network.load("models/pennaction1/checkpoint_epoch_2_21_49_07.tar")
    network.train()
    # network.eval()

def train_mmnist():
    config = Path("configs/mnist_config.json")
    params = load_params(config)

    dataset = MovingMnistDataset(
        batch_size = params["batch_size"], 
        dataset_path ='data',
        sequence_length = params["warmup_steps"] + params["prediction_steps"] + 2,
        image_size = params["autoencoder"]["input_shape"][1],
        num_digits = params["num_digits"],
        channels = params["autoencoder"]["input_shape"][0]
    )
    
    network = CrevNet(params, dataset, device)
    # network.load("models/run_03_11_2022_00_17_52/final_epoch_29_13_09_00.tar")
    network.train()
    network.eval()

if __name__ == "__main__":

    # Seed all modules, numpy random generator is returned to pass to other functions
    np_rng = set_global_seed(123)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_penn_action(np_rng)
    # train_mmnist()
    
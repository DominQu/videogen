from pathlib import Path
import logging

import torch
import numpy as np
import wandb

from src.crev_net import CrevNet
from src.utils import load_params, load_sequence, set_global_seed
from datasets.mnist import MovingMnistDataset
from datasets.pennaction import PennActionDataset

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %I:%M:%S', level=logging.INFO)
wandb.init(project="inzynierka", entity="dominqu", mode='disabled')

def train_penn_action(np_rng, device):
    config = Path("configs/pennaction_config.json")
    params = load_params(config)

    dataset = PennActionDataset(
        batch_size = params["batch_size"], 
        dataset_path ='data/Penn_Action',
        sequence_length = params["warmup_steps"] + params["prediction_steps"] + 2,
        np_rng=np_rng,
        img_size = params["autoencoder"]["input_shape"][1],
    )

    network = CrevNet(params, dataset, device)
    network.train()

def test_penn_action(np_rng, device):
    config = Path("configs/pennaction_config.json")
    params = load_params(config)

    dataset = PennActionDataset(
        batch_size = params["batch_size"], 
        dataset_path ='data/Penn_Action',
        sequence_length = params["warmup_steps"] + params["prediction_steps"] + 2,
        np_rng=np_rng,
        img_size = params["autoencoder"]["input_shape"][1],
    )

    network = CrevNet(params, dataset, device)
    network.load("samples/penn_action_model.tar")
    network.eval()

def test_real_data(frames_dir, np_rng, device):
    config = Path("configs/pennaction_config_custom_data.json")
    params = load_params(config)

    dataset = PennActionDataset(
        batch_size = params["batch_size"], 
        dataset_path ='data/Penn_Action',
        sequence_length = params["warmup_steps"] + params["prediction_steps"] + 2,
        np_rng=np_rng,
        img_size = params["autoencoder"]["input_shape"][1],
    )

    network = CrevNet(params, dataset, device)
    network.load("samples/penn_action_model.tar")

    frames_dir = Path(frames_dir)
    input_sequence = load_sequence(frames_dir)
    network.test_forward(input_sequence)

def train_mmnist(device):
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
    network.train()


def test_mmnist(device):
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
    network.load("samples/mnist_model")
    network.eval()

if __name__ == "__main__":

    # Seed all modules, numpy random generator is returned to pass to other functions
    np_rng = set_global_seed(777)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # test_mmnist(device)
    train_mmnist(device)
    # test_real_data("wpisz_sciezke_do_danych", np_rng, device)
    
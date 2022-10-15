from pathlib import Path
import json
import logging

import torch.nn as nn
import torch.optim as optim

from .models import AutoEncoder, RecurrentReversiblePredictor

class CrevNet:
    """Class implementing CrevNet architecture.
    It uses modules from models.py file.
    Enables model training and inference
    """
    def __init__(self, config_file: Path):

        self.config = CrevNet._load_params(config_file)

        # Instantiate neural network components and things needed for training

        self.auto_encoder = AutoEncoder(**self.config["autoencoder"])
        self.recurrent_module = RecurrentReversiblePredictor(**self.config["recurrent"], batch_size=self.config["batch_size"])

        self.ae_optimizer = optim.Adam(self.auto_encoder.parameters(), lr=self.config["lr"])
        self.recurrent_module_optimizer = optim.Adam(self.recurrent_module.parameters(), lr=self.config["lr"])
        
        self.criterion = nn.MSELoss()
    def __call__(self, x):
        """Take inputs and generate video sequence""" 
        pass


    def train(self):
        """Train the network"""


        pass

    def load(self, model_path: Path):
        """Load trained model from file"""
        pass

    @staticmethod
    def _load_params(config_file: Path) -> dict:
        """Load network parameters from json file"""
        if not config_file.is_file():
            raise ValueError("Specified config file is not valid")

        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.info("Config file loaded succesfully")
        return config

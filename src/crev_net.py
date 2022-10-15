from pathlib import Path
import json
import logging
import pip._vendor.rich.progress as progress 

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from .models import AutoEncoder, RecurrentReversiblePredictor

class CrevNet:
    """Class implementing CrevNet architecture.
    It uses modules from models.py file.
    Enables model training and inference
    """
    def __init__(self, config: dict, dataset, device: str='cpu'):
        
        self.config = config
        self.dataset = dataset
        self.device = device

        # Get training params from config
        self.epochs = self.config["epochs"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config["lr"]
        self.warmup_steps = self.config["warmup_steps"]
        self.prediction_steps = self.config["prediction_steps"]
        try:
            self.iterations = self.config["iterations"]
        except KeyError:
            # If number of iterations wasn't given in json than iterate over whole dataset
            self.iterations = self.dataset.get_size()
        self.input_shape = self.config["autoencoder"]["input_shape"]

        # Instantiate neural network components and things needed for training
        self.auto_encoder = AutoEncoder(**self.config["autoencoder"])
        self.auto_encoder.to(device)
        self.recurrent_module = RecurrentReversiblePredictor(**self.config["recurrent"], batch_size=self.config["batch_size"], device=self.device)
        self.recurrent_module.to(device)
        logging.info(f"Neural network modules intitialized and transfered to device: {device}")

        self.ae_optimizer = optim.Adam(self.auto_encoder.parameters(), lr=self.lr)
        self.recurrent_module_optimizer = optim.Adam(self.recurrent_module.parameters(), lr=self.lr)

        self.lr_recurrent_scheduler = optim.lr_scheduler.StepLR(self.ae_optimizer, step_size=self.config["lr_step_size"], gamma=self.config["lr_gamma"])
        self.lr_ae_scheduler = optim.lr_scheduler.StepLR(self.recurrent_module_optimizer, step_size=self.config["lr_step_size"], gamma=self.config["lr_gamma"])
        
        self.criterion = nn.MSELoss()

    def __call__(self, x):
        """Take inputs and generate video sequence""" 
        pass

    def prepare_input_sequence(self, input: list, sequence_length: int):
        """Prepare input sequence"""

        input_sequence = []
        for i in range(sequence_length):
            # Select only the number of channels given in the input shape
            frame1 = torch.unsqueeze(input[i][:, :self.input_shape[0]], 2)
            frame2 = torch.unsqueeze(input[i+1][:, :self.input_shape[0]], 2)
            frame3 = torch.unsqueeze(input[i+2][:, :self.input_shape[0]], 2)
            # Stack three consecutive frames in temporal dimension
            input_sequence.append(torch.cat((frame1, frame2, frame3), dim=2).to(self.device))

        return input_sequence

    def forward(self, input_sequence):
        """Pass data forward through the network
        
        Parameters
        ----------
        input_sequence [list]
            sequence of batched input images
        
        Returns
        -------
        loss 
            loss for the forward pass through the whole sequence
        """

        self.recurrent_module.init_hidden()

        loss = 0
        memory = self.recurrent_module.get_empty_memory()
        for i in range(1, self.warmup_steps + self.prediction_steps):
            state = self.auto_encoder(input_sequence[i-1])
            state, memory = self.recurrent_module((state, memory))
            next_frames = self.auto_encoder(state, encode=False)
            loss += self.criterion(next_frames, input_sequence[i])

        return loss

    def train(self):
        """Train the network"""
        for epoch in range(self.epochs):
            self.auto_encoder.train()
            self.recurrent_module.train()

            logging.info(f"Starting epoch: {epoch+1}")
            epoch_loss = 0
            for iter in progress.track(range(self.iterations)):
                # Zero the gradients for every batch
                self.ae_optimizer.zero_grad()
                self.recurrent_module_optimizer.zero_grad()

                input = next(self.dataset.get_train_batch())
                input_sequence = self.prepare_input_sequence(input, self.warmup_steps + self.prediction_steps)

                loss = self.forward(input_sequence)
                wandb.log({"loss": loss})
                epoch_loss += loss.item()
                loss.backward()

                self.ae_optimizer.step()
                self.recurrent_module_optimizer.step()
            
            logging.info(f"Training loss for epoch: {epoch+1} is: {epoch_loss/self.iterations:0.3f}")
            self.lr_ae_scheduler.step()
            self.lr_recurrent_scheduler.step()

    def load(self, model_path: Path):
        """Load trained model from file"""
        pass

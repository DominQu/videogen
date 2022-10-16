from calendar import c
from pathlib import Path
import logging
import pip._vendor.rich.progress as progress 

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from .models import AutoEncoder, RecurrentReversiblePredictor

def calculate_ssim(prediction, target):
    """Calculate structural similarity index"""
    batch_ssim = []
    for img_pred, img_target in zip(prediction.cpu().numpy(), target.cpu().numpy()):
        ssim_val = ssim(img_pred.squeeze()[2],
                        img_target.squeeze()[2],
                        gaussian_weights=True, 
                        sigma=1.5, 
                        use_sample_covariance=False)
        batch_ssim.append(ssim_val)
        # plt.imshow(img_pred.squeeze()[2].reshape((64, 64, 1)))
        # plt.savefig("pred.png")
        # plt.imshow(img_target.squeeze()[2].reshape((64, 64, 1)))
        # plt.savefig("target.png")

    return torch.mean(torch.tensor(batch_ssim))
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
        self.eval_iterations = int(0.2 * self.iterations)
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

    def eval_forward(self, input_sequence):

        predicted_sequence = []
        self.recurrent_module.init_hidden()

        eval_ssim_loss = 0
        eval_mse_loss = 0

        memory = self.recurrent_module.get_empty_memory()
        input = input_sequence[0]
        for i in range(1, self.warmup_steps+self.prediction_steps):
            predicted_sequence.append(input)
            state = self.auto_encoder(input)
            new_state, memory = self.recurrent_module((state, memory))
            if i == self.warmup_steps:
                # The input is conditioned on two past frames 
                # so after the warmup we should change the current frame 
                # and leave two from warmup sequence
                input = self.auto_encoder(new_state, encode=False)
                input[:, :, :2] = input_sequence[i][:, :, :2]
            elif i == self.warmup_steps + 1:
                # Leave only one frame
                input = self.auto_encoder(new_state, encode=False)
                input[:, :, :1] = input_sequence[i][:, :, :1]
            elif i > self.warmup_steps + 1:
                # The whole frame is predicted now
                prediction = self.auto_encoder(new_state, encode=False)
                # Calculate loss here
                eval_ssim_loss += calculate_ssim(prediction, input_sequence[i-1])
                eval_mse_loss += self.criterion(prediction, input_sequence[i-1])
                input = prediction
            else:
                # Skip the decoding phase as we don't need the predicted image
                input = input_sequence[i]
        
        predicted_sequence.append(input)
        eval_loss = (eval_mse_loss/self.prediction_steps, eval_ssim_loss/self.prediction_steps)

        return predicted_sequence, eval_loss

    def eval(self):
        # Epoch evaluation
        self.auto_encoder.train(False)
        self.recurrent_module.train(False)

        epoch_eval_ssim_loss = 0
        epoch_eval_mse_loss = 0

        with torch.no_grad():
            for eval_iter in range((self.eval_iterations)):
                input = next(self.dataset.get_test_batch())
                input_sequence = self.prepare_input_sequence(input, self.warmup_steps + self.prediction_steps)

                predicted_sequence, eval_loss = self.eval_forward(input_sequence)
                epoch_eval_mse_loss += eval_loss[0]
                epoch_eval_ssim_loss += eval_loss[1]
                wandb.log({"eval_mse_loss": eval_loss[0], "eval_ssim": eval_loss[1]})


        return epoch_eval_mse_loss/self.eval_iterations, epoch_eval_ssim_loss/self.eval_iterations

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

        return loss/(self.warmup_steps + self.prediction_steps)

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
                wandb.log({"training_mse_loss": loss})
                epoch_loss += loss.item()
                loss.backward()

                self.ae_optimizer.step()
                self.recurrent_module_optimizer.step()
            
            self.lr_ae_scheduler.step()
            self.lr_recurrent_scheduler.step()

            e_loss = self.eval()
            t_loss = epoch_loss/self.iterations
            wandb.log({"epoch_eval_mse_loss": e_loss[0], "epoch_eval_ssim": e_loss[1], "epoch_train_mse_loss": t_loss})

            logging.info(
f"Finished epoch: {epoch+1}. \
Training mse loss: {t_loss*10**3:0.3f} [10e-3]. \
Eval mse loss: {e_loss[0]*10**3:0.3f} [10e-3]. \
Eval ssim: {e_loss[1]*10**3:0.3f} [10e-3]")

    def load(self, model_path: Path):
        """Load trained model from file"""
        pass

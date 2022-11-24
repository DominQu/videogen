from calendar import c
from pathlib import Path
import logging
import pip._vendor.rich.progress as progress 
import os
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from .models import AutoEncoder, RecurrentReversiblePredictor
from .models_utils import normalize_image

def calculate_ssim(prediction, target):
    """Calculate structural similarity index"""
    batch_ssim = []
    for img_pred, img_target in zip(prediction.cpu().numpy(), target.cpu().numpy()):
        ssim_val = ssim(img_pred.squeeze()[2],
                        img_target.squeeze()[2],
                        gaussian_weights=True, 
                        sigma=1.5, 
                        use_sample_covariance=False,
                        channel_axis=0)
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
        self.device = torch.device(device)
        time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        self.run_dir = Path(f"models/run_{time}")
        wandb.config.update(self.config)

        # Get training params from config
        self.epochs = self.config["epochs"]
        self.save_every = self.config["save_every"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config["lr"]
        self.warmup_steps = self.config["warmup_steps"]
        self.prediction_steps = self.config["prediction_steps"]
        try:
            self.iterations = self.config["iterations"]
            self.eval_iterations = int(0.2 * self.iterations)
        except KeyError:
            # If number of iterations wasn't given in json than iterate over whole dataset
            self.iterations = self.dataset.get_size()[0]
            self.eval_iterations = self.dataset.get_size()[1]
        self.input_shape = self.config["autoencoder"]["input_shape"]

        # Instantiate neural network components and things needed for training
        self.auto_encoder = AutoEncoder(**self.config["autoencoder"])
        self.auto_encoder.to(self.device)
        self.recurrent_module = RecurrentReversiblePredictor(**self.config["recurrent"], batch_size=self.config["batch_size"], device=self.device)
        self.recurrent_module.to(self.device)
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

    def save_outputs(self, input_sequence, output_sequence, epoch):
        batch_ind = torch.randint(self.batch_size, (1,)).item()
        output_dir = self.run_dir / "outputs" 
        output_img_dir = output_dir / "img"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_img_dir, exist_ok=True)
        suffix = f"_{epoch}" if epoch >= 0 else ""

        original_seq = []
        predicted_seq = []
        # Save every image of the input and predicted sequences
        for i, batch in enumerate(input_sequence):
            frame = batch[batch_ind].transpose(0, 1)[2]
            if frame.shape[0] == 1:
                original_img = frame.reshape((self.input_shape[1], self.input_shape[2])).cpu().numpy() * 255
                prediction_frame = output_sequence[i][batch_ind, 0, 2, :, :].reshape((self.input_shape[1], self.input_shape[2]))
            else:
                original_img = frame.moveaxis(0, 2).cpu().numpy()
                original_img = normalize_image(original_img, high=255, low=0)
                prediction_frame = output_sequence[i][batch_ind, :, 2, :, :].moveaxis(0, 2)

            plt.imshow(original_img.astype(np.uint8))
            plt.savefig("original.jpg")
            mode = 'L' if len(original_img.shape) == 2 else 'RGB'
            original_img = Image.fromarray(original_img.astype(np.uint8), mode)
            original_seq.append(original_img)
            
            # Add predicted image normalization
            prediction_img = normalize_image(prediction_frame.cpu().numpy(), high=255, low=0)
            plt.imshow(prediction_img.astype(np.uint8))
            plt.savefig("prediction.jpg")
            prediction_img = Image.fromarray(prediction_img.astype(np.uint8), mode)
            predicted_seq.append(prediction_img)
            prediction_img.save(output_img_dir / f"pred_epoch{suffix}_seq_{i}.jpg")
            original_img.save(output_img_dir / f"orig_epoch{suffix}_seq_{i}.jpg")

        # Save gifs with the whole sequence
        original_seq[0].save(self.run_dir / "outputs" / f"orig_epoch{suffix}.gif", format="GIF",
            append_images=original_seq, save_all=True, duration=150, loop=0)
        predicted_seq[0].save(self.run_dir / "outputs" / f"pred_epoch{suffix}.gif", format="GIF",
            append_images=predicted_seq, save_all=True, duration=150, loop=0)

        wandb.save(str(self.run_dir / "outputs" / "*"))

    def eval(self, epoch=-1):
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
                if eval_iter == 0:
                    self.save_outputs(input_sequence, predicted_sequence, epoch)

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
        try:
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

                e_loss = self.eval(epoch)
                t_loss = epoch_loss/self.iterations
                wandb.log({"epoch": epoch, 
                           "epoch_eval_mse_loss": e_loss[0], 
                           "epoch_eval_ssim": e_loss[1], 
                           "epoch_train_mse_loss": t_loss})

                logging.info(
    f"Finished epoch: {epoch+1}. \
    Training mse loss: {t_loss*10**3:0.3f} [10e-3]. \
    Eval mse loss: {e_loss[0]*10**3:0.3f} [10e-3]. \
    Eval ssim: {e_loss[1]*10**3:0.3f} [10e-3]")

                if epoch % self.save_every == 0:
                    time = datetime.now().strftime("%H_%M_%S")
                    self.save(self.run_dir / f"checkpoint_epoch_{epoch}_{time}.tar")
        except KeyboardInterrupt:
            pass
            # time = datetime.now().strftime("%H_%M_%S")
            # self.save(self.run_dir / f"interrupted_epoch_{epoch}_{time}.tar")
        finally:
            time = datetime.now().strftime("%H_%M_%S")
            self.save(self.run_dir / f"final_epoch_{epoch}_{time}.tar")

    def test_forward(self, input_sequence):
        """Pass one sequence through the model"""
        self.auto_encoder.train(False)
        self.recurrent_module.train(False)

        with torch.no_grad():
            input_sequence = torch.unsqueeze(torch.stack(input_sequence, dim=0), 1)
            if input_sequence.dim() == 4:
                input_sequence = torch.unsqueeze(input_sequence, 2)
            # input_sequence = input_sequence.transpose(4, 3).transpose(3, 2)
            input_sequence = self.prepare_input_sequence(input_sequence, self.warmup_steps + self.prediction_steps)

            predicted_sequence, _ = self.eval_forward(input_sequence)
            self.save_outputs(input_sequence, predicted_sequence, -1)

            return predicted_sequence

    def save(self, path: Path, **kwargs):
        """Save models' weigths"""
        state_dict= {
            "ae_state_dict": self.auto_encoder.state_dict(),
            "ae_optim_state_dict": self.ae_optimizer.state_dict(),
            "recurrent_state_dict": self.recurrent_module.state_dict(),
            "recurrent_optim_state_dict": self.recurrent_module_optimizer.state_dict(),
            **kwargs
        }
        path = Path(path)
        if not path.parent.exists():
            os.makedirs(path.parent)

        torch.save(state_dict, path)
        wandb.save(str(self.run_dir / "*"))
        logging.info("Models saved succesfully")

    def load(self, path: Path):
        """Load trained model from file"""
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Given path: {path} doesn't exist")
        state_dict = torch.load(path, map_location=self.device)
        self.auto_encoder.load_state_dict(state_dict["ae_state_dict"])
        self.ae_optimizer.load_state_dict(state_dict["ae_optim_state_dict"])
        self.recurrent_module.load_state_dict(state_dict["recurrent_state_dict"])
        self.recurrent_module_optimizer.load_state_dict(state_dict["recurrent_optim_state_dict"])

        self.auto_encoder.to(self.device)
        self.recurrent_module.to(self.device)

        logging.info("Models loaded succesfully")

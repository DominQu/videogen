from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import MNIST

class MovingMnist(Dataset):
    """Dataset object responsible for creation of the moving mnist data
    
    Parameters
    ----------
    data_path [Path]
        directory in which dataset will be placed or can be found
    train [bool]
        load training data or test data
    sequence_length [int]
        length of generated moving mnist sequence
    image_size [int]
        size for the moving mnist sequence, output will be a square
    num_digits [int]
        number of digits overlapping each other on the moving sequence
    channels [int]
        number of channels for the generated sequence
    starting_size [int]
        size of the static mnist image loaded from dataset
    max_move [int]
        absolute value for maximal movement of the digits, in pixels
    """
    def __init__(self,
                 data_path: Path, 
                 train: bool, 
                 sequence_length: int, 
                 image_size: int, 
                 num_digits: int, 
                 channels: int, 
                 starting_size: int = 32,
                 max_move: int = 4
                 ):
        super().__init__()
        self.mnist = MNIST(data_path, train=train, download=True, transform=Compose([Resize(starting_size), ToTensor()]))
        self.data_len = len(self.mnist)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.starting_size = starting_size
        self.num_digits = num_digits
        self.channels = channels
        self.max_move = max_move
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        """Magic method responsible for generation of the moving mnist sequence
        
        Returns
        -------
        data_sequence [torch.tensor]
            tensor with the generated moving mnist sequence
        """

        # Empty tensor for the sequence
        data_sequence = torch.zeros(
            (self.sequence_length, self.image_size, self.image_size, self.channels), 
            dtype=torch.float32
            )
        
        for _ in range(self.num_digits):
            ind = int(torch.randint(self.data_len, (1,)))
            img, _ = self.mnist[ind]

            # Get random position of the number and random movement
            pos = torch.randint(self.image_size - self.starting_size, (2,))
            mov = torch.randint(-self.max_move, self.max_move+1, (2,))

            for t in range(self.sequence_length):
                # Add rules for changing movement direction
                if pos[0] < 0:
                    pos[0] = 0
                    mov[0] = -mov[0]
                elif pos[0] >= self.image_size - self.starting_size:
                    pos[0] = self.image_size - self.starting_size - 1
                    mov[0] = -mov[0]
                if pos[1] < 0:
                    pos[1] = 0
                    mov[1] = -mov[1]
                elif pos[1] >= self.image_size - self.starting_size:
                    pos[1] = self.image_size - self.starting_size - 1
                    mov[1] = -mov[1]
                for c in range(self.channels):
                    data_sequence[t, pos[0]:pos[0]+self.starting_size, pos[1]:pos[1]+self.starting_size, c] += img.squeeze()
                pos += mov

        data_sequence[data_sequence>1] = 1
        return data_sequence

class MovingMnistDataset:
    """Class for Moving Mnist dataset.
    It is responsible for batching the data.
    Reshaping it to shape required by the CrevNet module and
    stacking frames in temporal dimension
    """
    def __init__(self, batch_size: int, dataset_path: Path) -> None:

        self.dataset = (MovingMnist(dataset_path, True), MovingMnist(dataset_path, False))
        self.train_dataloader = DataLoader(self.dataset[0], batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.dataset[1], batch_size, shuffle=True, drop_last=True)

    def get_test_batch(self):
        for batch in self.test_dataloader:
            batch = self._restructure_batch(batch)
            yield batch

    def get_train_batch(self):
        for batch in self.train_dataloader:
            batch = self._restructure_batch(batch)
            yield batch

    def get_size(self):
        return None
    
    @staticmethod
    def _restructure_batch(batch):
        """Input batch has shape:
        [batch_size, sequence_len, width, height, channels]
        Desired shape is:
        [sequence_len, batch_size, channels, width, height]
        """
        batch = batch.transpose(0, 1)
        batch = batch.transpose(2, 3)
        batch = batch.transpose(3, 4)
        return batch

if __name__ == "__main__":

    mnist = MovingMnist('data/MNIST', True, 20, 64, 2, 3)
    print(mnist[0])
from pathlib import Path
import random
import logging
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from numpy.random import default_rng
import cv2

class PennAction(Dataset):

    def __init__(self,
                 data_path: str, 
                 train: bool, 
                 sequence_length: int,
                 numpy_rng,
                 test_split: float=0.2,
                 img_size: int=256):
        """Basic version of the dataset loads only one sequence from given video"""
        self.data_path = Path(data_path) / "frames"
        self.train = train
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.data = list(self.data_path.glob("*"))
        self.data = np.array(self.data)
        first_set_len = int(self.data.shape[0] * (1 - test_split))
        choice = np.random.choice(range(self.data.shape[0]), size=(first_set_len,), replace=False)    
        ind = np.zeros(self.data.shape[0], dtype=bool)
        ind[choice] = True
        rest = ~ind

        self.train_data = self.data[ind]
        self.test_data = self.data[rest]

        self.transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor()])
        
    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, index):
        """Get one sequence from the dataset"""
        data_path = self.train_data[index] if self.train else self.test_data[index]
        return self._load_sequence(data_path)

    def _load_sequence(self, data_path: Path):
        """Load one sequence from the dataset
        
        Parameters
        ----------
        data_path [Path]
            path to directory with images for one action
            
        Returns
        -------
        sequence [torch.tensor]
            video sequence of one action, it has shape [seq_len, channels, height, width]
        """
        frames = list(data_path.glob("*"))
        
        if len(frames) < self.sequence_length:
            logging.error("Sequence length is higher than number of frames")
        
        sequence = torch.zeros((self.sequence_length, 3, self.img_size, self.img_size))

        for i in range(self.sequence_length):
            frame = frames[i]
            img = Image.open(frame)
            tensor_img = self.transform(img)
            sequence[i] = tensor_img
        return sequence


class PennActionDataset:
    def __init__(self, 
                 batch_size: int, 
                 dataset_path: Path,
                 sequence_length: int,
                 np_rng,
                 img_size: int=256):

        self.dataset = (
            PennAction(
                dataset_path, True, sequence_length, np_rng, img_size=img_size
                ), 
            PennAction(
                dataset_path, False, sequence_length, np_rng, img_size=img_size
                )
            )
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

    @staticmethod
    def _restructure_batch(batch):
        """Input batch has shape:
        [batch_size, sequence_len, channels, width, height]
        Desired shape is:
        [sequence_len, batch_size, channels, width, height]
        """
        return batch.transpose(0, 1)


if __name__ == "__main__":
    dataset = PennActionDataset(16, "data/Penn_Action", 12, default_rng(1))
    # batch = dataset.get_train_batch()
    for batch in dataset.get_train_batch():
        print(batch.shape)
        break
    # img = dataset[0][0]
    # print(img.shape)
    # cv2.imshow('img', img.moveaxis(0, 2).numpy())
    # # plt.show()
    # cv2.waitKey(0)
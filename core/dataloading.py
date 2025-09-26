from utilities import *
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class SimpleDataset(Dataset):
    def __init__(self, data, config):
        """
          Initialise the dataset by converting the data into tensors.
          This is the only method worth looking at.
        """
        self.config = config

        # process the data
        isosigs = [] # the pairs of numbers
        targets = [] # the things to predict

        for isosig in tqdm(data):
          # pad sequence
          padded = isosig + "\n"*(self.config.context_length-len(isosig))

          # tokenize sequence
          tokenized = tokenize_list(padded)

          # save shifted sequences
          isosigs.append(tokenized[:-1])
          targets.append(tokenized[1:])

          print(len(tokenized))

        # save it as tensors
        self.isosigs = torch.tensor(isosigs, dtype=int)
        self.targets = torch.tensor(targets, dtype=int)

    def __len__(self):
        """
          Returns the length of the dataset.
        """
        return len(self.isosigs)

    def __getitem__(self, index):
        """
          Returns an item in the dataset at the given index.
        """
        sample = (
            self.isosigs[index],
            self.targets[index]
        )

        return sample
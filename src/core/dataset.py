import numpy as np
import torch
from torch.utils.data import Dataset


class MarketDataset(Dataset):
    def __init__(self, npy_file_path: str) -> None:
        self.data = np.load(npy_file_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.data[index]

        sample_tensor = torch.from_numpy(sample)

        return sample_tensor.unsqueeze(0)

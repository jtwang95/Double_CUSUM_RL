from torch.utils.data import Dataset
import torch


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        if torch.is_tensor(x):
            self.x = x
        else:
            self.x = torch.tensor(x, dtype=torch.float32)
        if torch.is_tensor(y):
            self.y = y
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

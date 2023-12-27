from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import torch

class ModelingDataset:

    def __init__(self, train: Dataset, val: Dataset):
        self.train = train
        self.val = val

    def get_dataloaders(self, batch_size):
        return (
            DataLoader(self.train, batch_size=batch_size, shuffle=True),
            DataLoader(self.val, batch_size=batch_size, shuffle=True)
        )

def gen_lin_data(w: Tensor, b: float, num_samples: int, noise: float = 0.01) -> Tuple[Tensor, Tensor]:
    rand_x = torch.rand(num_samples, len(w))
    base_y = rand_x.matmul(w.reshape(-1, 1))
    noise_y = torch.randn(num_samples, 1) * noise

    return rand_x, base_y + b + noise_y


def fashion_mnist(root = "datasets/fashion_mnist", resize=(28, 28)) -> ModelingDataset:
    trans = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Lambda(torch.flatten)
    ])

    train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=trans, download=True
    )
    val = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=trans, download=True
    )

    return ModelingDataset(train, val)
from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import torch
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.model_selection import train_test_split

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

class DataFrameDataset(Dataset):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            target_col: str,
        ):
        self.dataframe = dataframe
        self.target_col = target_col
        self.feature_cols = [
            c
            for c in self.dataframe.columns
            if c != target_col
        ]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]

        features = torch.tensor(
            sample.loc[self.feature_cols].values,
            dtype=torch.float32
        )
        target = torch.tensor(
            sample.loc[self.target_col],
            dtype=torch.float32
        )

        return features, target

def kaggle_house_preprocessing(df):
    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

    # One-hot encode categorical variables
    df = pd.get_dummies(df, drop_first=True, dtype=float)

    # Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def kaggle_housing(root = "datasets/kaggle_housing", val_pct = 0.2):
    with open(Path(root) / "train.csv") as f:
        df = pd.read_csv(f)

    postprocessed_df = kaggle_house_preprocessing(df)

    train_df, val_df = train_test_split(postprocessed_df, test_size=val_pct)

    train = DataFrameDataset(train_df, "SalePrice")
    val = DataFrameDataset(val_df, "SalePrice")

    return ModelingDataset(train, val)
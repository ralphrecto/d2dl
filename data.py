from typing import Tuple, List
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import torch
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class ModelingDataset:

    def __init__(self, train: Dataset, val: Dataset, test = None):
        self.train = train
        self.val = val
        self.test = test

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
            feature_cols: str
        ):
        self.dataframe = dataframe
        self.target_col = target_col
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]

        features = torch.tensor(
            sample.loc[self.feature_cols].values,
            dtype=torch.float32
        )

        if self.target_col in self.dataframe:
            target = torch.tensor(
                sample.loc[[self.target_col]].values,
                dtype=torch.float32
            )
        else:
            target = None

        return features, target

KAGGLE_HOUSING_CATEGORICAL_VARS = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition"
]

def kaggle_house_preprocessing(
        raw_df: pd.DataFrame,
        target_col: str,
        non_features: List[str],
        # one_hot_encoder 
    ):
    df = raw_df.drop(columns=non_features)

    numerical_cols = [
        c
        for c in df.select_dtypes(include=['float64', 'int64']).columns
        if c != target_col and c not in KAGGLE_HOUSING_CATEGORICAL_VARS
    ]

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Standardize numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=KAGGLE_HOUSING_CATEGORICAL_VARS, dummy_na=True, dtype=float)

    df[non_features] = raw_df[non_features]

    feature_cols = [
        c
        for c in df.columns
        if (c != target_col) and (c not in non_features)
    ]

    return df, feature_cols

def kaggle_housing(root = "datasets/kaggle_housing", val_pct = 0.2):
    with open(Path(root) / "train.csv") as f:
        train_df = pd.read_csv(f)

    with open(Path(root) / "test.csv") as f:
        test_df = pd.read_csv(f)

    all_df = pd.concat([
        train_df.assign(split="train"),
        test_df.assign(split="test", SalePrice=np.nan),
    ])

    preprocessed_df, feature_cols = kaggle_house_preprocessing(all_df, target_col="SalePrice", non_features=["Id", "split"])

    preprocessed_train_df = preprocessed_df.query("split == 'train'").drop(columns=["split"])
    preprocessed_train_df["SalePrice"] = np.log(preprocessed_train_df["SalePrice"])

    preprocessed_test_df = preprocessed_df.query("split == 'test'").drop(columns=["split", "SalePrice"])

    train_df, val_df = train_test_split(preprocessed_train_df, test_size=val_pct)

    train = DataFrameDataset(train_df, "SalePrice", feature_cols)
    val = DataFrameDataset(val_df, "SalePrice", feature_cols)
    test = DataFrameDataset(preprocessed_test_df, "SalePrice", feature_cols)

    return ModelingDataset(train, val, test)
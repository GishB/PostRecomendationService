import sys

sys.path.append("..")

from typing import List
from info.schema import PostGet
import pandas as pd
from torch.utils.data import Dataset
import torch


class CustomImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.loc[idx].iloc[3:]
        return torch.Tensor(data).unsqueeze(0)


def data_to_json(data: pd.DataFrame) -> List[PostGet]:
    return [{"id": sample_of_data.id,
             "text": sample_of_data.text,
             "topic": sample_of_data.topic} for sample_of_data in data.itertuples()]

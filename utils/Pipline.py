import sys

sys.path.append("..")

from utils.DataConvertors import CustomImageDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd


class ModelInference:
    def __init__(self,
                 model: object = None,
                 df_users: pd.DataFrame = None,
                 df_posts: pd.DataFrame = None,
                 user_index: int = None,
                 limit: int = None):
        self.columns = ['post_id', "text", "topic", 'topic_covid', 'topic_entertainment', 'topic_movie',
                        'topic_politics', 'topic_sport',
                        'topic_tech', 'kmean_label', 'DistanceToCluster_0', 'DistanceToCluster_1',
                        'DistanceToCluster_2', 'DistanceToCluster_3', 'DistanceToCluster_4',
                        'DistanceToCluster_5', 'DistanceToCluster_6', 'DistanceToCluster_7', 'DistanceToCluster_8',
                        'DistanceToCluster_9', 'DistanceToCluster_10',
                        'DistanceToCluster_11', 'DistanceToCluster_12', 'DistanceToCluster_13', 'DistanceToCluster_14',
                        'gender', 'age', 'Belarus', 'Cyprus',
                        'Estonia', 'Finland', 'Kazakhstan', 'Latvia', 'Russia', 'Switzerland', 'Turkey', 'Ukraine',
                        'exp_1', 'exp_2', 'exp_3', 'exp_4', 'Android', 'iOS', 'ads', 'organic']
        self.df_users = df_users.copy()
        self.df_posts = df_posts.copy()
        self.user_index = user_index
        self.model = model
        self.limit = limit

    def create_cartesian_product(self) -> pd.DataFrame:
        """Create matrix product between user info and all posts via cross join by user id."""
        user_info_row = self.df_users[self.df_users.user_id == self.user_index]
        return pd.merge(user_info_row, self.df_posts, how='cross')[self.columns]

    def create_torch_dataset(self, data) -> object:
        return DataLoader(CustomImageDataset(df=data), batch_size=64, pin_memory=True, num_workers=4, shuffle=False)

    @torch.inference_mode()
    def model_inference(self, data) -> list[float]:
        self.model.eval()
        return list(self.model(data).view(data.shape[0]).detach().numpy())

    def get_probs(self, loader) -> list[float]:
        data = []
        for x in loader:
            data.extend(self.model_inference(x))
        return data

    def data_to_json(self, data: pd.DataFrame) -> list:
        return [{"id": sample_of_data.post_id,
                 "text": sample_of_data.text,
                 "topic": sample_of_data.topic} for sample_of_data in data.itertuples()]

    def predict(self) -> list:
        data = self.create_cartesian_product()
        loader = self.create_torch_dataset(data)
        probs = self.get_probs(loader)
        data["probs"] = probs
        data.sort_values(by=["probs"], ascending=False, inplace=True)
        return self.data_to_json(data.head(self.limit))

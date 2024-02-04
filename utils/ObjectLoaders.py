import torch
import os

import pandas as pd
from sqlalchemy import create_engine

from psycopg2.extensions import register_adapter, AsIs
import numpy as np

register_adapter(np.int64, AsIs)


def batch_load_sql(query: str, engine: object) -> pd.DataFrame:
    CHUNKSIZE = 200000
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for ind, chunk_dataframe in enumerate(pd.read_sql(query, conn, chunksize=CHUNKSIZE)):
        print(chunk_dataframe.shape)
        print(ind)
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features_post() -> pd.DataFrame:
    engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
    )
    df_post_data = batch_load_sql("""SELECT * FROM aleksandr_samofalov_features_lesson_22_post_october_2023""",
                                  engine)
    df_post_data = df_post_data.drop(['index'], axis=1)
    return df_post_data


def load_features() -> pd.DataFrame:
    engine = create_engine(
    "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
    "postgres.lab.karpov.courses:6432/startml"
    )
    df_user_data = batch_load_sql("""SELECT * FROM aleksandr_samofalov_features_lesson_22_users_october_2023""",
                                  engine)
    df_user_data = df_user_data.drop(['index'], axis=1)
    return df_user_data


def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path('../../RecomendationPostService/models/CustomResidualNN_2024_CPU.pth')
    return torch.jit.load(model_path)
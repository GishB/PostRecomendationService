import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from typing import List
from datetime import datetime

sys.path.append('..')

from info.schema import PostGet
from utils.Pipline import ModelInference
from utils.ObjectLoaders import load_features, load_features_post, load_models


df_user_features = load_features()
df_post_features = load_features_post()
user_list_ind = df_user_features.user_id.to_list()
model = load_models()
app = FastAPI()


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int = 200,
        time: datetime = None,
        limit: int = 5) -> List[PostGet]:
    if id in user_list_ind:
        result = ModelInference(df_users=df_user_features,
                                df_posts=df_post_features,
                                model=model,
                                user_index=id,
                                limit=limit).predict()
    else:
        raise HTTPException(404, "user dose not exist in database")
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")

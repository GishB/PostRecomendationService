from pydantic import BaseModel
from datetime import datetime


class UserGet(BaseModel):
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str

    class Config:
        orm_mode = True


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: list[PostGet]


class FeedGet(BaseModel):
    user_id: int
    post_id: int
    action: str
    time: datetime
    user: UserGet
    post: PostGet

    class Config:
        orm_mode = True

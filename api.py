from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_content
from src.predict import predict_collaborative


app = FastAPI(
    title="Music Recommendation System"
)


class SongInput(BaseModel):

    name: str


@app.get("/")
def home():

    return {

        "message":

        "Music Recommendation API working"

    }


@app.post("/recommend/content")
def content_api(data: SongInput):

    result = predict_content(

        data.name

    )

    return {

        "recommendations": result

    }


@app.post("/recommend/collaborative")
def collaborative_api(data: SongInput):

    result = predict_collaborative(

        data.name

    )

    return {

        "recommendations": result

    }